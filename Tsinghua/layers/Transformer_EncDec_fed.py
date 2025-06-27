import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec_fed import Encoder, EncoderLayer
from layers.Autoformer_EncDec import my_Layernorm
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock
from layers.MultiWaveletCorrelation import MultiWaveletTransform


class AppUsageFEDformer(nn.Module):
    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        super(AppUsageFEDformer, self).__init__()
        self.seq_len = configs['seq_len']
        self.d_model = configs['d_model']
        self.version = version
        self.modes = modes
        self.mode_select = mode_select

        # --- App ID + Time embedding ---
        self.app_emb = nn.Embedding(configs['app_vocab_size'], self.d_model)
        self.time_emb = nn.Linear(1, self.d_model)

        # --- Select attention block (Fourier or Wavelet) ---
        if version == 'Wavelets':
            attn_block = MultiWaveletTransform(ich=self.d_model, L=1, base='legendre')
        else:
            attn_block = FourierBlock(in_channels=self.d_model,
                                      out_channels=self.d_model,
                                      n_heads=configs['n_heads'],
                                      seq_len=configs['seq_len'],
                                      modes=self.modes,
                                      mode_select_method=self.mode_select)

        # --- Encoder ---
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AutoCorrelationLayer(attn_block, configs['d_model'], configs['n_heads']),
                    configs['d_model'],
                    configs['d_ff'],
                    dropout=configs['dropout'],
                    activation=configs['activation']
                )
                for _ in range(configs['e_layers'])
            ],
            norm_layer=my_Layernorm(configs['d_model'])
        )

        # --- Output classification head ---
        self.projection = nn.Linear(self.d_model+24, configs['num_app'])  # predict next app

    def encode_input(self, x_app, x_time):
        # x_app: [B, L]          -> App ID
        # x_time: [B, L, D_time] -> Time features
        x_app = x_app.to(dtype=torch.long)
        x_time = x_time.unsqueeze(-1)
        app_emb = self.app_emb(x_app)          # [B, L, D]
        time_emb = self.time_emb(x_time)       # [B, L, D]
        return app_emb + time_emb              # [B, L, D]

    def forward(self, x_app, x_time, time_vecs, targets, mode):
        x = self.encode_input(x_app, x_time)          # [B, L, D]
        enc_out, _ = self.encoder(x, attn_mask=None)  # [B, L, D]
        last_token = enc_out[:, -1, :]                # [B, D]
        out = torch.cat((last_token, time_vecs[:, -1, :] ), dim=1)

        score = self.projection(out)          # [B, num_app]

        if mode == 'predict':
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return score
        else:
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return loss
