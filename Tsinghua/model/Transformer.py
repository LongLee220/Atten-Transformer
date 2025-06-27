import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class AppUsageTransformer(nn.Module):
    def __init__(self, configs):
        super(AppUsageTransformer, self).__init__()
        self.seq_len = configs['seq_len']
        self.d_model = configs['d_model']
        self.num_app = configs['num_app']

        # Embedding layers
        self.app_emb = nn.Embedding(configs['app_vocab_size'], self.d_model)
        self.time_emb = nn.Linear(configs['time_feat_dim'], self.d_model)

        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs['factor'], attention_dropout=configs['dropout']),
                        configs['d_model'], configs['n_heads']
                    ),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                )
                for _ in range(configs['e_layers'])
            ],
            norm_layer=nn.LayerNorm(configs['d_model'])
        )

        # Classification head
        self.projection = nn.Linear(self.d_model+24, self.num_app)

    def encode_input(self, x_app, x_time):
        # x_app: [B, L]          -> App ID
        # x_time: [B, L, D_time] -> Time features
        x_app = x_app.to(dtype=torch.long)
        app_emb = self.app_emb(x_app)          # [B, L, D]
        time_emb = self.time_emb(x_time)       # [B, L, D]
        return app_emb[:, -1, :]  + time_emb              # [B, L, D]

    def forward(self, x_app, x_time, time_vecs, targets, mode):
        x = self.encode_input(x_app, x_time)           # [B, L, D]
        x = x.unsqueeze(1)
        enc_out, _ = self.encoder(x, attn_mask=None)   # [B, L, D]
        x_last = enc_out[:, -1, :]                     # [B, D]

        
        out = torch.cat((x_last, time_vecs[:, -1, :] ), dim=1)  # shape: [B, L_total, D] or [B, D_total]
        score = self.projection(out)         # [B, num_app]
        if mode == 'predict':
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return score
        else:
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return loss