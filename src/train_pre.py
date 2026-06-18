"""DeepDRAC pre-training script.

Trains the GraphGPS+GINEConv model using triplet loss on graph triples
from the pre-training dataset. Saves model checkpoints and Fisher
information matrix for subsequent fine-tuning.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import wandb
import swanlab

from data_loader import get_data
from models.gps_global_model import NNConv_model
from config import DEVICE,  PRE_TRAIN_DIR, GRAPH_DATA_DIR, PRE_TRAIN_MODEL_DIR

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ── Hyperparameters ──────────────────────────────────────────────────────────
random.seed(4)
torch.manual_seed(4)

embedding_dim = 32
epochs = 100
batch_size = 3 * 16
lr = 0.001
triple_loss_Gamma = 0.10
model_label = "gps_global_model"
loss_label = "triplet_loss"

# ── Experiment tracking (optional) ──────────────────────────────────────────
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY")

if SWANLAB_API_KEY:
    swanlab.login(api_key=SWANLAB_API_KEY)
    swanlab.init(
        project="CIC-IDS2017-train",
        name=f"pre-train-{model_label}",
        config={
            "embedding_dim": embedding_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "Gamma": triple_loss_Gamma,
            "model": model_label,
            "loss": loss_label,
        },
    )


# ── Loss function ───────────────────────────────────────────────────────────
def cosdists(a: torch.Tensor, b: torch.Tensor):
    """Cosine distance between two tensors."""
    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    return 1 - cos_sim


def triple_loss(y1: torch.Tensor, y2: torch.Tensor, y3: torch.Tensor,
                Gamma: float = triple_loss_Gamma):
    """Triplet margin loss: ||y1 - y2|| - ||y1 - y3|| + Gamma."""
    dist = torch.nn.PairwiseDistance(p=2)
    loss_temp = dist(y1, y2) - dist(y1, y3) + Gamma
    loss = torch.maximum(loss_temp, torch.zeros_like(loss_temp))
    return torch.mean(loss)


# ── Training loop ───────────────────────────────────────────────────────────
def train(train_data: list, save_model_path: str, model_label: str,
          batch_size: int = batch_size, lr: float = lr):
    """Pre-train the GNN model with triplet loss."""
    node_in_feature = train_data[0].num_node_features
    edge_in_feature = train_data[0].num_edge_features
    graph_in_feature = train_data[0].graph_attr.shape[0]
    print(f"Node feature dim: {node_in_feature}, "
          f"Edge feature dim: {edge_in_feature}, "
          f"Graph feature dim: {graph_in_feature}")

    model = NNConv_model(
        node_in_feature, edge_in_feature, graph_in_feature,
        node_out_feature=30, graph_out_feature=2
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Fisher information matrix for EWC
    fisher_information = {name: torch.zeros_like(param)
                          for name, param in model.named_parameters()}

    model.train()
    for e in range(epochs):
        # Shuffle triples
        shuffle_list = []
        new_train_data = []
        for i in range(0, len(train_data), 3):
            temp = [train_data[i], train_data[i + 1], train_data[i + 2]]
            shuffle_list.append(temp)

        random.seed(e)
        random.shuffle(shuffle_list)
        for one in shuffle_list:
            new_train_data.append(one[0])
            new_train_data.append(one[1])
            new_train_data.append(one[2])

        train_loader = DataLoader(new_train_data, batch_size=batch_size)
        for i, batchData in enumerate(train_loader):
            optimizer.zero_grad()
            y_pre = model(batchData)
            y_pre_new = y_pre.view(-1, 3, embedding_dim)
            y1, y2, y3 = torch.split(y_pre_new, 1, dim=1)
            y1 = y1.squeeze(1)
            y2 = y2.squeeze(1)
            y3 = y3.squeeze(1)
            loss = triple_loss(y1, y2, y3)

            if SWANLAB_API_KEY:
                swanlab.log({"loss": loss.item()})

            if i % 20 == 0:
                print(f"epoch->{e}, iter->{i}, loss->{loss}")

            loss.backward()
            optimizer.step()

            # Accumulate Fisher information
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_information[name] += param.grad ** 2

    # Save model and Fisher matrix
    torch.save(model.state_dict(), save_model_path)
    torch.save(fisher_information, save_model_path.replace(".pt", "_fisher.pt"))
    print(f"Model saved to {save_model_path}")


# ── Main entry point ────────────────────────────────────────────────────────
if __name__ == '__main__':
    start_time = time.time()

    graph_id_path = PRE_TRAIN_DIR / "trple-graph-pre-all.csv"
    graph_data_path = GRAPH_DATA_DIR / "correct" / "graph_train.csv"

    train_data = get_data(
        data_index_path=str(graph_id_path),
        data_all_path=str(graph_data_path),
        PE=True,
    )
    get_data_time = time.time()
    print(f"Data loading time: {(get_data_time - start_time) / 60:.2f} min")

    PRE_TRAIN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model_path = str(PRE_TRAIN_MODEL_DIR / f"{model_label}.pt")
    train(train_data, save_model_path, model_label)

    train_time = time.time()
    print(f"Training time: {(train_time - get_data_time) / 60:.2f} min")

    if SWANLAB_API_KEY:
        swanlab.finish()
