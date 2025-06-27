import datetime
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm

# 1. 加载并预处理数据
df = pd.read_csv('data/App_usage_trace.txt', sep=' ', names=['user', 'time', 'location', 'app', 'traffic'])
df = df[['user', 'time', 'app']]
df['time'] = df['time'].astype(str).str[:-2]
df.drop_duplicates(inplace=True)
df = df[df.groupby('app')['app'].transform('count') >= 10]

# 2. 构造序列和最近使用记录
seq_length = 8
prev_user, prev_time = None, None
app_seq, recent_apps = [], []
all_app_seq, all_recent_apps = [], []

for i in tqdm(range(len(df))):
    row = df.iloc[i]
    user, app = row['user'], row['app']
    time = datetime.datetime.strptime(row['time'], '%Y%m%d%H%M')
    
    if user != prev_user:
        app_seq, recent_apps = [app], [app]
        all_app_seq.append([])
        all_recent_apps.append([])
    else:
        if (time - prev_time).total_seconds() / 60 <= 7:
            if len(app_seq) == seq_length:
                all_app_seq.append(app_seq.copy())
                app_seq = app_seq[1:] + [app]
            else:
                app_seq.append(app)
                all_app_seq.append([])
        else:
            app_seq = [app]
            all_app_seq.append([])
        all_recent_apps.append(recent_apps.copy())
        recent_apps.append(app)
        if len(recent_apps) > 10:
            recent_apps.pop(0)

    prev_user, prev_time = user, time

df['app_seq'] = all_app_seq
df['recent_apps'] = all_recent_apps
df = df[df['app_seq'].map(len) > 0]
df = df[df.groupby('user')['user'].transform('count') >= 50]

# 3. 编码 app
encoder = LabelEncoder()
df['app_encoded'] = encoder.fit_transform(df['app'])
num_classes = len(encoder.classes_)

# 4. 划分训练 / 测试
df['time'] = df['time'].astype(int)
df_train = df[df['time'] <= 201604251200]
df_test = df[df['time'] > 201604251200]

# 5. MFU 构建分数
user_counter = {}
for _, row in df_train.iterrows():
    user = row['user']
    user_counter.setdefault(user, Counter()).update(row['app_seq'])

mfu_scores, mfu_targets = [], []
for _, row in df_test.iterrows():
    user, app = row['user'], row['app']
    app_id = encoder.transform([app])[0]
    mfu_targets.append(app_id)
    score = torch.zeros(num_classes)
    if user in user_counter:
        for i, (a, _) in enumerate(user_counter[user].most_common(10)):
            idx = encoder.transform([a])[0]
            score[idx] = 1 / (i + 1)
    mfu_scores.append(score)

# 6. MRU 构建分数
mru_scores, mru_targets = [], []
for _, row in df_test.iterrows():
    app = row['app']
    app_id = encoder.transform([app])[0]
    mru_targets.append(app_id)
    score = torch.zeros(num_classes)
    for i, a in enumerate(row['recent_apps'][-10:][::-1]):
        idx = encoder.transform([a])[0]
        score[idx] = 1 / (i + 1)
    mru_scores.append(score)

# 7. 评估函数
def compute_metrics(scores, targets, top_k):
    hr, dcg, ndcg, mrr = [], [], [], []
    _, pred_idx = scores.topk(max(top_k), dim=1)
    targets = targets.view(-1, 1)

    for k in top_k:
        hit = (pred_idx[:, :k] == targets).any(dim=1).float()
        hr.append(hit.mean().item())

        rank = (pred_idx[:, :k] == targets).nonzero(as_tuple=True)[1] + 1
        dcg_k = (1.0 / torch.log2(rank.float() + 1)).sum().item() / len(scores)
        dcg.append(dcg_k)
        idcg_k = 1.0 / torch.log2(torch.tensor(2.0))
        ndcg.append(dcg_k / idcg_k.item())

        first_hits = (pred_idx[:, :k] == targets).nonzero(as_tuple=True)
        if len(first_hits[0]) > 0:
            mrr_k = (1.0 / (first_hits[1].float() + 1)).sum().item()
            mrr.append(mrr_k / len(scores))
        else:
            mrr.append(0.0)

    return hr, dcg, ndcg, mrr

# 8. 执行评估
topk = [1, 3, 5]
mfu_scores = torch.stack(mfu_scores)
mfu_targets = torch.tensor(mfu_targets)
mru_scores = torch.stack(mru_scores)
mru_targets = torch.tensor(mru_targets)

mfu_hr, mfu_dcg, mfu_ndcg, mfu_mrr = compute_metrics(mfu_scores, mfu_targets, topk)
mru_hr, mru_dcg, mru_ndcg, mru_mrr = compute_metrics(mru_scores, mru_targets, topk)

# 9. 输出结果
for i, k in enumerate(topk):
    print(f"[MFU]  HR@{k}: {mfu_hr[i]:.4f}, nDCG@{k}: {mfu_ndcg[i]:.4f}, MRR@{k}: {mfu_mrr[i]:.4f}")
    print(f"[MRU]  HR@{k}: {mru_hr[i]:.4f}, nDCG@{k}: {mru_ndcg[i]:.4f}, MRR@{k}: {mru_mrr[i]:.4f}")