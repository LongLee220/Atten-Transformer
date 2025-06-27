import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ReformerLayer

class AppUsageReformer(nn.Module):
    def __init__(self, configs, bucket_size=4, n_hashes=4):
        super(AppUsageReformer, self).__init__()
        self.pred_len = configs['pred_len']        # typically 1 for next-app prediction
        self.seq_len = configs['seq_len']          # input sequence length
        self.d_model = configs['d_model']          # embedding dimension

        # Embedding layers
        self.app_emb = nn.Embedding(configs['app_vocab_size'], self.d_model)
        self.time_emb = nn.Linear(configs['time_feat_dim'], self.d_model)

        # Encoder with Reformer layers
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    ReformerLayer(
                        None,
                        d_model=self.d_model,
                        n_heads=configs['n_heads'],
                        bucket_size=bucket_size,
                        n_hashes=n_hashes
                    ),
                    d_model=self.d_model,
                    d_ff=configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                ) for _ in range(configs['e_layers'])
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )

        # Final projection to app vocabulary
        self.projection = nn.Linear(self.d_model + 24, configs['num_app'])

        
    def encode_input(self, x_app, x_time):
        """
        Embed and combine app ID and time features
        :param x_app: [B, L]         int64 tensor for app ID sequence
        :param x_time: [B, L, D_t]   float tensor for time features
        :return: [B, L, D]           embedded sequence
        """
        x_app = x_app.to(dtype=torch.long)
        app_embed = self.app_emb(x_app)               # [B, L, D]
        time_embed = self.time_emb(x_time)            # [B, L, D]
        return app_embed[:, -1, :]  + time_embed

    def forward(self, x_app, x_time, time_vec, targets, mode):
        """
        Forward pass for next app prediction
        :param x_app: [B, L] int64
        :param x_time: [B, L, time_feat_dim]
        :return: logits of shape [B, num_app] for classification
        """
        x_app = x_app.to(dtype=torch.long)
        x = self.encode_input(x_app, x_time)          # [B, L, D]
        x = x.unsqueeze(1)
        enc_out, _ = self.encoder(x)                  # [B, L, D]
        
                      
        out = torch.cat((time_vec[:, -1, :] , enc_out[:, -1, :]), 1)
        score = self.projection(out)         # [B, num_app]
        if mode == 'predict':
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return score
        else:
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return loss