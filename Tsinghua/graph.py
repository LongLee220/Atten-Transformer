import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import ast

def generate_positional_encodings(hour_index, embedding_dim=24):
    """
    Generate positional encodings for the given hour index.

    Parameters:
        hour_index (int): Hour of the day (0-23).
        embedding_dim (int): Dimension of the positional encoding.

    Returns:
        torch.Tensor: A (1, embedding_dim) tensor representing positional encodings.
    """
    embedding_size_half = embedding_dim // 2
    step = 2 * math.pi / embedding_dim
    theta = torch.linspace(0, 2 * math.pi, embedding_size_half) - step * hour_index
    sin_component = torch.sin(theta)
    cos_component = torch.cos(theta)
    encoding = torch.cat((sin_component, cos_component))
    return encoding.view(1, -1)  # Reshape to (1, embedding_dim) for DataLoader processing


def graph_seq(path, mode):
    """
    Process CSV data and convert it into PyTorch format.

    Parameters:
        path (str): Path to the dataset file.
        mode (str): Either 'train' or 'test' indicating the dataset type.

    Saves:
        Processed sequence dataset in PyTorch format.
    """
    data = pd.read_csv(path, sep='\t')
    sequence_list = []

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing Data"):
        app_seq = ast.literal_eval(row["app_seq"])  # Safely convert string representation to list
        time_seq = ast.literal_eval(row["time_seq"])  

        target_app = torch.tensor([row["app"]], dtype=torch.long)
        user = torch.tensor([row["user"]], dtype=torch.long)

        # Parse time information
        time_str = str(row["time"])
        hour = int(time_str.split('_')[1])  # Extract hour from the formatted string
        global_time = generate_positional_encodings(hour)

        num_nodes = len(app_seq)

        # **Construct Transformer input features**
        node_features = torch.tensor([[app_seq[i], time_seq[i]] for i in range(num_nodes)], dtype=torch.float)

        # **Store as PyTorch dictionary**
        sequence_data = {
            "x": node_features,  # Tensor (5, feature_dim) as Transformer input
            "y": target_app,  # Target app
            "u": user,  # User ID
            "t": global_time  # Global time encoding
        }

        sequence_list.append(sequence_data)

    # **Save the processed dataset**
    save_path = "./data/graph_dataset_train.pt" if mode == "train" else "./data/graph_dataset_test.pt"
    torch.save(sequence_list, save_path)
    print(f"Sequence dataset {mode} saved!")


if __name__ == '__main__':

    # Load processed dataset
    loaded_graph_list = torch.load("./data/graph_dataset.pt")

    # Create DataLoader
    batch_size = 8
    data_loader = DataLoader(loaded_graph_list, batch_size=batch_size, shuffle=True)

    # Iterate through DataLoader
    for batch in data_loader:
        print(batch)
        break  # Only print the first batch
