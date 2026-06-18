"""Load and prepare graph data for PyTorch Geometric (PyG) training.

Converts CSV-format graph records into PyG Data objects, handling node
attributes, edge indices, edge attributes, and graph-level attributes.
Supports optional positional encoding (RandomWalkPE) and fine-tune mode
(which includes per-graph labels).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops

import torch_geometric.transforms as T
from config import DEVICE, GRAPH_DATA_DIR

# Positional encoding transformer
transform = T.AddRandomWalkPE(walk_length=2, attr_name='pe')


def get_one_graph(id, data_df, fine_tune=False, PE=False):
    """Build a PyG Data object for a single graph by its ID.

    Args:
        id: Graph ID in the data DataFrame.
        data_df: DataFrame containing graph records.
        fine_tune: If True, include the graph label (y) for supervised fine-tuning.
        PE: If True, add RandomWalk positional encoding to nodes.

    Returns:
        torch_geometric.data.Data object (on CUDA if available).
    """

    input_graph = data_df.loc[data_df['id'] == int(id)].iloc[0]

    node_attr = json.loads(input_graph['node_attr'])
    x = torch.tensor(node_attr,dtype=torch.float)

    edge_index = json.loads(input_graph['edge_index'])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()

    edge_attr = json.loads(input_graph['edge_attr'])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    graph_attr = json.loads(input_graph['group_attr'])
    graph_attr = torch.tensor(graph_attr, dtype=torch.float)
    
    # Fine-tune mode: include graph label
    if fine_tune:
        # graph_label = eval(input_graph['group_label_easy'])
        # graph_label = list(graph_label)[0]
        graph_label = input_graph['label_id']
        y = int(graph_label)
        oneData = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, graph_attr=graph_attr).to(DEVICE)
    else:
        oneData = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, graph_attr=graph_attr).to(DEVICE)

    if PE:
        # Positional encoding
        oneData.edge_index,oneData.edge_attr = add_self_loops(edge_index,edge_attr)
        oneData = transform(oneData)
        #oneData.pe = oneData.pe.to(DEVICE)
        oneData.edge_index,oneData.edge_attr = remove_self_loops(oneData.edge_index, oneData.edge_attr)
        
        oneData = oneData.to(DEVICE)
        #print(oneData.edge_index)
    # Print non-zero indices
    # if non_zero_indices.numel() > 0:
    #     print("Non-zero elements and their indices:")
    #     for idx in non_zero_indices:
    #         print(f"Index: {idx}, Value: {oneData.pe[idx[0]]}")
    # else:
    #     #print("No non-zero elements in the tensor.")
    #print("oneData_edge_index={},\n,oneData_PE={}".format(oneData.edge_index,oneData.pe))
    return oneData


def get_data(data_index_path, data_all_path, PE=False):
    """Load pre-training triple-graph data from CSV index and graph files.

    Args:
        data_index_path: Path to CSV with triple graph IDs (g1_id, g2_id, g3_id).
        data_all_path: Path to CSV with all graph data records.
        PE: Whether to add positional encoding.

    Returns:
        List of PyG Data objects (3 per index row).
    """
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    data_index_df = pd.read_csv(data_index_path)
    data_df = pd.read_csv(data_all_path)

    train_data_index = data_index_df
    print(f"Number of training triples: {len(train_data_index)}")

    train_data = []
    for i in range(len(train_data_index)):
        g1_id, g2_id, g3_id = train_data_index.iloc[i, :]
        train_data.append(get_one_graph(g1_id, data_df, PE=PE))
        train_data.append(get_one_graph(g2_id, data_df, PE=PE))
        train_data.append(get_one_graph(g3_id, data_df, PE=PE))

    return train_data


def get_all_data(data_all_path, fine_tune=False, PE=False):
    """Load all graph data from a single CSV file.

    Args:
        data_all_path: Path to CSV with graph records (must have 'id' column).
        fine_tune: If True, include graph labels for supervised learning.
        PE: Whether to add positional encoding.

    Returns:
        List of PyG Data objects.
    """
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    data_df = pd.read_csv(data_all_path)
    print(f"Total data records: {len(data_df)}")

    all_data = []
    for i in range(len(data_df)):
        g1_id = data_df.loc[i, 'id']
        all_data.append(get_one_graph(g1_id, data_df, fine_tune, PE))
    return all_data


if __name__ == '__main__':
    all_data = get_all_data(str(GRAPH_DATA_DIR / "correct" / "graph_train.csv"))
