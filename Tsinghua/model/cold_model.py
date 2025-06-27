import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class KernelFeatureEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_type='fourier', degree=3, gamma=0.5):
        """
        Kernel Feature Encoder: Maps low-dimensional features x to a higher-dimensional space.
        """
        super(KernelFeatureEncoder, self).__init__()
        self.kernel_type = kernel_type
        self.degree = degree
        self.gamma = gamma
        self.out_dim = out_dim
        
        # Fourier Kernel
        if kernel_type == 'fourier':
            self.w = nn.Parameter(torch.randn(out_dim // 2, in_dim))
            self.b = nn.Parameter(torch.randn(out_dim // 2))

    def forward(self, x):
        if self.kernel_type == 'polynomial':
            return (torch.matmul(x, x.T) + 1) ** self.degree  # Polynomial Kernel
        
        elif self.kernel_type == 'rbf':
            x_norm = torch.sum(x**2, dim=1, keepdim=True)
            dist = x_norm + x_norm.T - 2 * torch.matmul(x, x.T)
            return torch.exp(-self.gamma * dist)  # RBF Gaussian Kernel

        elif self.kernel_type == 'fourier':
            transformed_x = torch.cat([
                torch.sin(torch.matmul(x, self.w.T) + self.b),
                torch.cos(torch.matmul(x, self.w.T) + self.b)
            ], dim=-1)
            return transformed_x  # Fourier Feature Mapping
        
        elif self.kernel_type == 'sigmoid':
            return torch.tanh(0.1 * torch.matmul(x, x.T) + 1)  # Sigmoid Kernel
        
        else:
            raise ValueError("Unsupported kernel type")

class Transformer_C(nn.Module):
    def __init__(self, n_users, n_apps, hid_dim, seq_len=4, num_heads=4, num_layers=2, dropout=0.2):
        super(Transformer_C, self).__init__()

        self.seq_len = seq_len

        self.kernel_encoder = KernelFeatureEncoder(2, hid_dim, kernel_type='fourier')
        self.user_emb = nn.Embedding(n_users, hid_dim)

        self.time_embedding = nn.Linear(24, hid_dim)  # Time encoding

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.combined_linear = nn.Linear(hid_dim, hid_dim, bias=True)

        self.classifier = nn.Linear(hid_dim, n_apps, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, u, t, mode='train'):

        # Kernel feature encoding
        x = self.kernel_encoder(x)  # (batch, hid_dim)

        # User embedding
        #user_vec = self.user_emb(u)  # (batch, hid_dim)

        batch_size = u.shape[0]  
        x = x.view(batch_size, self.seq_len, -1) 

        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, hid_dim)

        # Process time features
        time_weights = torch.sigmoid(self.time_embedding(t))  # (batch, hid_dim)

        # Aggregate sequence representations
        app_vector = torch.mean(x, dim=1)  # (batch, hid_dim)
        #user_vec = user_vec.squeeze(1) 
        #combined_vec = torch.cat([app_vector, user_vec], dim=-1)
        combined_vec = self.dropout(app_vector)

        user_vector = F.relu(self.combined_linear(combined_vec))
        
        time_weights = time_weights.squeeze(1) 

        # Time-weighted feature fusion
        final_out = F.relu(user_vector * time_weights)

        # Compute final predictions
        out = self.dropout(final_out)
        scores = self.classifier(out)

        if mode == 'predict':
            loss = torch.mean(F.cross_entropy(scores, y.view(-1), reduction='none'))
            return scores, loss
        else:
            loss = torch.mean(F.cross_entropy(scores, y.view(-1), reduction='none'))
            return loss
