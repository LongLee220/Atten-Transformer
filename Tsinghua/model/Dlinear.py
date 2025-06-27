import torch
import torch.nn as nn
import torch.nn.functional as F


class AppUsageDLinear(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs['seq_len']
        self.d_model = configs['d_model']
        self.app_emb = nn.Embedding(configs['app_vocab_size'], self.d_model)
        self.time_emb = nn.Linear(configs['time_feat_dim'], self.d_model)

        self.linear = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(configs['dropout'])
        self.norm = nn.LayerNorm(self.d_model)

        self.out_proj = nn.Linear(self.d_model + 24, configs['num_app'])


    def encode_input(self, x_app, x_time):
        # x_app: [B, L]          -> App ID
        # x_time: [B, L, D_time] -> Time features
        x_app = x_app.to(dtype=torch.long)
        app_emb = self.app_emb(x_app)          # [B, L, D]
        time_emb = self.time_emb(x_time)       # [B, L, D]
        return app_emb[:, -1, :]  + time_emb              # [B, L, D]

    def forward(self, x_app, x_time, time_vecs, targets, mode):
        x = self.encode_input(x_app, x_time)   # [B, L, D]
        x = self.linear(x)                     # [B, L, D]
        x = self.norm(F.gelu(x))               # [B, L, D]
        x = self.dropout(x)

        
        out = torch.cat((x, time_vecs[:, -1, :] ), dim=1)  # shape: [B, L_total, D] or [B, D_total]

        score = self.out_proj(out)         # [B, num_app]
        if mode == 'predict':
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return score
        else:
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return loss
    
