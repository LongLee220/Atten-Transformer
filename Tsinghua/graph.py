import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import ast

def generate_positional_encodings(hour_index, embedding_dim=24):
    embedding_size_half = embedding_dim // 2
    step = 2 * math.pi / embedding_dim
    theta = torch.linspace(0, 2 * math.pi, embedding_size_half) - step * hour_index
    sin_component = torch.sin(theta)
    cos_component = torch.cos(theta)
    encoding = torch.cat((sin_component, cos_component))
    return encoding.view(1, -1)  # reshape to (1, embedding_dim) for DataLoader compatibility


def graph_seq(split, path, mode):
    """ Load CSV data and convert it into PyTorch-compatible format """
    data = pd.read_csv(path, sep='\t')
    sequence_list = []

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing Data"):
        app_seq = ast.literal_eval(row["app_seq"])  # safer evaluation than eval()
        time_seq = ast.literal_eval(row["time_seq"])  

        target_app = torch.tensor([row["app"]], dtype=torch.long)
        user = torch.tensor([row["user"]], dtype=torch.long)

        # Parse time information
        time_str = str(row["time"])
        hour = int(time_str.split('_')[1])  # ensure the correct time string format
        global_time = generate_positional_encodings(hour)

        num_nodes = len(app_seq)

        # Construct input features for Transformer
        node_features = torch.tensor([[app_seq[i], time_seq[i]] for i in range(num_nodes)], dtype=torch.float)

        # Store as PyTorch dictionary
        sequence_data = {
            "x": node_features,  # Tensor (sequence_length, feature_dim) as Transformer input
            "y": target_app,  # Target application
            "u": user,  # User ID
            "t": global_time  # Global temporal information
        }

        sequence_list.append(sequence_data)

    # Save dataset
    save_path = f"./data/{split}/graph_dataset_train.pt" if mode == "train" else f"./data/{split}/graph_dataset_test.pt"
    torch.save(sequence_list, save_path)
    print(f"Sequence dataset {mode} saved!")
    return sequence_list
