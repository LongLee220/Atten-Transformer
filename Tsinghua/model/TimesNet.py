import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Conv_Blocks import Inception_Block_V1
import torch.fft


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # remove DC component
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]  # [B, k]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.k = configs['top_k']
        self.conv = nn.Sequential(
            Inception_Block_V1(configs['d_model'], configs['d_ff'], configs['num_kernels']),
            nn.GELU(),
            Inception_Block_V1(configs['d_ff'], configs['d_model'], configs['num_kernels'])
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            total_len = self.seq_len
            if total_len % period != 0:
                pad_len = ((total_len // period) + 1) * period - total_len
                padding = torch.zeros([B, pad_len, N], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                out = x

            out = out.reshape(B, -1, period, N).permute(0, 3, 1, 2).contiguous()  # [B, N, n_period, period]
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])

        res = torch.stack(res, dim=-1)  # [B, T, D, k]
        period_weight = F.softmax(period_weight, dim=1)  # [B, k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, dim=-1)
        return res + x  # residual


class AppUsageTimesNet(nn.Module):
    def __init__(self, configs):
        super(AppUsageTimesNet, self).__init__()
        self.seq_len = configs['seq_len']
        self.d_model = configs['d_model']
        self.num_app = configs['num_app']

        # Embedding
        self.app_emb = nn.Embedding(configs['app_vocab_size'], self.d_model)
        self.time_emb = nn.Linear(1, self.d_model)

        # TimesNet Layers
        self.blocks = nn.ModuleList([TimesBlock(configs) for _ in range(configs['e_layers'])])
        self.norm = nn.LayerNorm(self.d_model)

        # Final classifier
        self.projection = nn.Linear(self.d_model+24, configs['num_app'])

    def encode_input(self, x_app, x_time):
        # x_app: [B, L], x_time: [B, L, time_feat_dim]
        x_app = x_app.to(dtype=torch.long)
        x_time = x_time.unsqueeze(-1)
        app_embed = self.app_emb(x_app)       # [B, L, D]
        time_embed = self.time_emb(x_time)    # [B, L, D]
        return app_embed + time_embed         # [B, L, D]

    def forward(self, x_app, x_time, time_vec, targets, mode):
        x = self.encode_input(x_app, x_time)  # [B, L, D]
        for block in self.blocks:
            x = self.norm(block(x))           # [B, L, D]


        out = torch.cat((time_vec[:, -1, :] , x[:, -1, :]), 1)

        score = self.projection(out)      # [B, num_app]

        if mode == 'predict':
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return score
        else:
            loss = torch.mean(F.cross_entropy(score, targets.view(-1), reduction='none'))
            return loss