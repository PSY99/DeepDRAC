"""Generate graph embedding vectors using a trained GNN model.

Loads graph data, runs inference through the GNN encoder, and saves
per-graph embedding vectors to CSV for downstream clustering/evaluation.
"""

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from data_loader import get_all_data
from models.gps_global_model import NNConv_model
from config import DEVICE,  GRAPH_DATA_DIR, CHECKPOINT_DIR, EMBEDDING_DIR


def get_embedding(input_data: list, best_model_path: str, output_path: str,
                  node_out_feature: int = 30, graph_out_feature: int = 2):
    """Generate and save embedding vectors for each graph in input_data.

    Args:
        input_data: List of PyG Data objects.
        best_model_path: Path to trained model checkpoint ('' for random init).
        output_path: CSV output path.
        node_out_feature: GNN node output dimension.
        graph_out_feature: GNN graph output dimension.
    """
    if len(input_data) == 0:
        return

    node_in_feature = input_data[0].num_node_features
    edge_in_feature = input_data[0].num_edge_features
    graph_in_feature = input_data[0].graph_attr.shape[0]
    model = NNConv_model(
        node_in_feature, edge_in_feature, graph_in_feature,
        node_out_feature=node_out_feature, graph_out_feature=graph_out_feature
    ).to(DEVICE)

    if best_model_path:
        print(f"Loading model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Using randomly initialized model")
        model.apply(lambda m: m.reset_parameters()
                    if hasattr(m, 'reset_parameters') else None)

    model.eval()
    data_loader = DataLoader(input_data, batch_size=1)

    rows = []
    with torch.no_grad():
        for i, batchData in enumerate(data_loader):
            embedding_data = model(batchData)
            embedding_data = embedding_data.squeeze(0).tolist()
            rows.append({'id': i + 1, 'embedding_data': embedding_data})
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1} graphs")

    res_df = pd.DataFrame(rows, columns=['id', 'embedding_data'])
    res_df.to_csv(output_path, index=False)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    # Example usage — adjust paths as needed
    model_select = 'only_tune'

    if model_select == 'only_tune':
        best_model_path = str(CHECKPOINT_DIR / "fine-tune-model2/1/gps_model_only_tune.pt")
    elif model_select == 'pre_and_tune':
        best_model_path = str(CHECKPOINT_DIR / "fine-tune-model3/1/gps_model.pt")
    else:
        best_model_path = ''

    train_graph_path = str(GRAPH_DATA_DIR / "correct" / "graph_train.csv")
    output_path = str(EMBEDDING_DIR / "train_embedding.csv")
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    data = get_all_data(train_graph_path, PE=True)
    get_embedding(data, best_model_path, output_path)
