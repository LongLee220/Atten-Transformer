#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:08:37 2025

@author: longlee
"""

import os
import argparse
import random
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.cold_model import Transformer_C
from model.stand_model import Transformer_S

from utils.graph import graph_seq
from utils.split_time import time_processed
from utils.split_stand import stand_processed
from utils.split_cold import cold_processed


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of AppUsage2Vec model")
    
    parser.add_argument('--model_name', type=str, default='Transformer', help="Model name")
    parser.add_argument('--epoch', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size")
    parser.add_argument('--dim', type=int, default=128, help="Embedding size for users and apps")
    parser.add_argument('--seq_length', type=int, default=8, help="Length of the app usage sequence")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--split', type=str, default='stand', help="Data split method: time, cold, or stand")
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

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

    print(f"Using device: {device}")

    # Data processing
    print(f"Processing {args.split} split dataset...")

    train_file = f'./data/{args.split}/train.txt'
    test_file = f'./data/{args.split}/test.txt'
    valid_file = f'./data/{args.split}/validation.txt'

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"Preprocessing required for split: {args.split}")
        if args.split == 'stand':
            stand_processed(args.seq_length)
        elif args.split == 'cold':
            cold_processed(args.seq_length)
        else:
            time_processed(args.seq_length)
    else:
        print(f"Found existing preprocessed files for split '{args.split}', skipping preprocessing.")

    print("Loading dataset...")
    train_cache = f'./data/{args.split}/graph_dataset_train.pt'
    test_cache = f'./data/{args.split}/graph_dataset_test.pt'
    valid_cache = f'./data/{args.split}/graph_dataset_valid.pt'

    #if os.path.exists(valid_cache):
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print("Found cached dataset. Loading from .pt files.")
        train_dataset = torch.load(train_cache)
        test_dataset = torch.load(test_cache)
        valid_dataset = torch.load(valid_cache)
    else:
        print("No cached dataset found. Processing raw data.")
        train_dataset = graph_seq(args.split, path=train_file, mode='train')
        test_dataset = graph_seq(args.split, path=test_file, mode='test')
        valid_dataset = graph_seq(args.split, path=test_file, mode='valid')

        torch.save(train_dataset, train_cache)
        torch.save(test_dataset, test_cache)
        torch.save(valid_dataset, valid_cache)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # Model selection
    num_users = len(open(os.path.join(f'./data/{args.split}', 'user2id.txt'), 'r').readlines()) + (1 if args.split == 'cold' else 0)
    num_apps = len(open(os.path.join(f'./data/{args.split}', 'app2id.txt'), 'r').readlines())

    print(f"Initializing {args.model_name} model...")
    if args.split == 'cold':
        model = Transformer_C(num_users, num_apps, args.dim, args.seq_length) if args.model_name == 'Transformer' else print('No model!!!')
    elif args.split == 'stand':
        model = Transformer_S(num_users, num_apps, args.dim, args.seq_length) if args.model_name == 'Transformer' else print('No model!!!')
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for e in range(args.epoch):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            x, y, u, t = batch["x"].to(device), batch["y"].to(device), batch["u"].to(device), batch["t"].to(device)
            train_loss = training_step(model, optimizer, x, y, u, t, device)
            total_train_loss += train_loss

        train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {e+1}/{args.epoch} - Train Loss: {train_loss:.5f}")

        # Use validation set here
        hr, _, ndcg, mrr, val_loss = evaluate_model(model, valid_loader, device)  # <- Use valid_loader, not test_loader
        print(f"Epoch {e+1}/{args.epoch} - Validation Loss: {val_loss:.5f}")
        for k in [1, 2, 3, 4, 5]:
            print(f"HR@{k}: {hr[k]:.5f}, NDCG@{k}: {ndcg[k]:.5f}, MRR@{k}: {mrr[k]:.5f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {e+1}")
                break

        hr, _, ndcg, mrr, test_loss = evaluate_model(model, test_loader, device)
        print(f"Epoch {e+1}/{args.epoch} - Test Loss: {test_loss:.5f}")
        for k in [1, 2, 3, 4, 5]:
            print(f"HR@{k}: {hr[k]:.5f}, NDCG@{k}: {ndcg[k]:.5f}, MRR@{k}: {mrr[k]:.5f}")

    print("Training complete!")
    #torch.save(model.state_dict(), 'model.pth')


if __name__ == "__main__":
    main()
