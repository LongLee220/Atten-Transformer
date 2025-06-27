import os
import argparse
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from model.Transformer import AppUsageTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of AppUsage2Vec model")
    
    parser.add_argument('--epoch', type=int, default=10, help="The number of epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="The size of batch")
    parser.add_argument('--dim', type=int, default=64, help="The embedding size of users and apps")
    parser.add_argument('--seq_length', type=int, default=8, help="The length of previously used app sequence")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in DNN")
    parser.add_argument('--alpha', type=float, default=0.1, help="Discount oefficient for loss function")
    parser.add_argument('--topk', type=float, default=5, help="Topk for loss function")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed")
    
    return parser.parse_args()


def collate_fn(batch):
    return {
        "x": torch.stack([sample["x"] for sample in batch]),
        "y": torch.stack([sample["y"] for sample in batch]),
        "u": torch.stack([sample["u"] for sample in batch]),
        "t": torch.stack([sample["t"] for sample in batch])
    }


def compute_metrics(scores, targets, top_k):
    hr, dcg, ndcg, mrr = [], [], [], []
    
    _, predicted_indices = scores.topk(max(top_k), dim=1, largest=True, sorted=True)
    targets = targets.view(-1, 1)
    
    for k in top_k:
        hits_k = (predicted_indices[:, :k] == targets).any(1)
        hr.append(hits_k.float().mean().item())

        ranks_k = (predicted_indices[:, :k] == targets).nonzero(as_tuple=True)[1] + 1
        dcg_k = (1.0 / torch.log2(ranks_k.float() + 1)).sum().item() / predicted_indices.size(0)
        dcg.append(dcg_k)

        idcg_k = 1.0 / torch.log2(torch.tensor(2.0))
        ndcg.append(dcg_k / idcg_k if idcg_k > 0 else 0)

        first_hit_indices = (predicted_indices[:, :k] == targets).nonzero(as_tuple=True)
        if len(first_hit_indices) > 0:
            mrr_k = torch.sum(1.0 / (first_hit_indices[1] + 1).float())
            sum_k = predicted_indices.shape[0]
            mrr.append(mrr_k.item() / sum_k)
        else:
            mrr.append(0.0)

    return hr, dcg, ndcg, mrr


def training_step(model, optimizer, x, y, u, t, device):
    optimizer.zero_grad()
    loss = model(x, y, u, t, mode='train')
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, test_loader, device, top_k=[1, 2, 3, 4, 5]):
    model.eval()
    hr_list, dcg_list, ndcg_list, mrr_list = {k: [] for k in top_k}, {k: [] for k in top_k}, {k: [] for k in top_k}, {k: [] for k in top_k}
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data in test_loader:
            batch = {key: value.to(device) for key, value in data.items()}
            x, y, u, t = batch["x"], batch["y"], batch["u"], batch["t"]
            scores, loss = model(x, y, u, t, mode='predict')

            total_loss += loss.item()
            num_batches += 1

            hr, dcg, ndcg, mrr = compute_metrics(scores, y, top_k)
            for k in top_k:
                hr_list[k].append(hr[k-1])
                dcg_list[k].append(dcg[k-1])
                ndcg_list[k].append(ndcg[k-1])
                mrr_list[k].append(mrr[k-1])

    avg_loss = total_loss / num_batches
    return {k: np.mean(hr_list[k]) for k in top_k}, {k: np.mean(dcg_list[k]) for k in top_k}, {k: np.mean(ndcg_list[k]) for k in top_k}, {k: np.mean(mrr_list[k]) for k in top_k}, avg_loss



def main():
    args = parse_args()
    
    # random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
        torch.cuda.manual_seed_all(args.seed)
    
    # data load
    train_cache = f'./data/stand/graph_dataset_train.pt'
    test_cache = f'./data/stand/graph_dataset_test.pt'
    

    train_dataset = torch.load(train_cache)
    test_dataset = torch.load(test_cache)

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    num_users = len(open(os.path.join(f'./data/stand', 'user2id.txt'), 'r').readlines())
    num_apps = len(open(os.path.join(f'./data/stand', 'app2id.txt'), 'r').readlines())

    # model & optimizer
    configs = {
        'seq_len': 8,
        'd_model': 64,
        'd_ff': 128,
        'dropout': 0.1,
        'factor': 5,
        'n_heads': 4,
        'e_layers': 2,
        'activation': 'gelu',
        'app_vocab_size': num_apps,
        'time_feat_dim': 8,
        'num_app': num_apps
    }
    
    model = AppUsageTransformer(configs)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # train & evaluation
    total_loss = 0
    itr = 1
    p_itr = 500
    best_acc = 0
    Ks = [1,5,10]
    acc_history = [[0,0,0]]
    best_acc = 0
    top_k=[1, 2, 3, 4, 5]
    
    for e in range(args.epoch):
        
        model.train()
        for batch in train_loader:
            app_seqs, time_seqs, targets, users, time_vecs = batch["x"][:, :, 0].to(device),batch["x"][:, :, 1].to(device), batch["y"].to(device), batch["u"].to(device), batch["t"].to(device)
            #print(batch["x"].shape)       

            optimizer.zero_grad()
            loss = model(app_seqs, time_seqs, time_vecs, targets, 'train')
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if itr % p_itr == 0:
                print("[TRAIN] Epoch: {} / Iter: {} Loss - {}".format(e+1, itr, total_loss/p_itr))
                total_loss = 0
            itr += 1
            
        model.eval()
        hr_list, dcg_list, ndcg_list, mrr_list = {k: [] for k in top_k}, {k: [] for k in top_k}, {k: [] for k in top_k}, {k: [] for k in top_k}

        with torch.no_grad():
            for batch in test_loader:
                app_seqs, time_seqs, targets, users, time_vecs = batch["x"][:, :, 0].to(device),batch["x"][:, :, 1].to(device), batch["y"].to(device), batch["u"].to(device), batch["t"].to(device)     

                
                scores = model(app_seqs, time_seqs, time_vecs, targets,  'predict')

                
                
                hr, dcg, ndcg, mrr = compute_metrics(scores, targets, top_k=[1, 2, 3, 4, 5])
                for k in top_k:
                    hr_list[k].append(hr[k-1])
                    dcg_list[k].append(dcg[k-1])
                    ndcg_list[k].append(ndcg[k-1])
                    mrr_list[k].append(mrr[k-1])

        hr, _, ndcg, mrr, = {k: np.mean(hr_list[k]) for k in top_k}, {k: np.mean(dcg_list[k]) for k in top_k}, {k: np.mean(ndcg_list[k]) for k in top_k}, {k: np.mean(mrr_list[k]) for k in top_k}
        for k in [1, 2, 3, 4, 5]:
            print(f"HR@{k}: {hr[k]:.5f}, NDCG@{k}: {ndcg[k]:.5f}, MRR@{k}: {mrr[k]:.5f}")



if __name__ == "__main__":
    main()