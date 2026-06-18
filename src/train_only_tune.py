"""DeepDRAC fine-tuning from scratch (no pre-training).

Fine-tunes a randomly initialized GNN + MLP classifier on labeled alert
subgraphs. Used as an ablation baseline to measure the benefit of pre-training.
"""

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

import wandb

from data_loader import get_all_data
from models.gps_global_model import NNConv_model
from models.fine_tune_mlp import Fine_tune_MLP
from config import DEVICE,  CHECKPOINT_DIR, GRAPH_DATA_DIR

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

random.seed(1)
torch.manual_seed(1)
embedding_dim = 32
epochs = 100
num_classes = 14

# ── Class weights ───────────────────────────────────────────────────────────
# Computed by data_process/compute_class_weights.py
# Formula: weight[i] = total_samples / count_per_class[i]
# These values are from the paper's CIC-IDS2017 experiments.
# Run compute_class_weights.py on YOUR graph data to get correct weights.
weights = torch.tensor([
    1.01, 372.33, 992.89, 992.89, 1276.57, 1489.33, 1489.33,
    1787.2, 1787.2, 1787.2, 2234.0, 2978.67, 4468.0, 4468.0
]).to(DEVICE)

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


def train(GNN_model, mlp_model, train_data, optimizer, batch_size=1):
    GNN_model.train(); mlp_model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    loss_list = []
    for batchData in DataLoader(train_data, batch_size=batch_size, shuffle=True):
        optimizer.zero_grad()
        labels = batchData.y.to(DEVICE)
        outputs = mlp_model(GNN_model(batchData))
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        loss.backward(); optimizer.step()
    return sum(loss_list) / len(loss_list)


def validate(GNN_model, mlp_model, val_data, batch_size=1):
    GNN_model.eval(); mlp_model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_predictions, val_labels, val_losses = [], [], []
    with torch.no_grad():
        for batchData in DataLoader(val_data, batch_size=batch_size, shuffle=True):
            labels = batchData.y.to(DEVICE)
            outputs = mlp_model(GNN_model(batchData))
            val_losses.append(criterion(outputs, labels).item())
            val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    nonzero = list(range(1, num_classes))
    appeared = [c for c in nonzero if c in set(val_labels)]
    if not appeared:
        return avg_val_loss, val_accuracy, np.nan, np.nan
    val_recall = recall_score(val_labels, val_predictions, labels=appeared, average='macro')
    val_f1 = f1_score(val_labels, val_predictions, labels=appeared, average='macro')
    return avg_val_loss, val_accuracy, val_recall, val_f1


def test(GNN_model, mlp_model, test_data, batch_size=1):
    GNN_model.eval(); mlp_model.eval()
    test_predictions, test_labels = [], []
    with torch.no_grad():
        for batchData in DataLoader(test_data, batch_size=batch_size, shuffle=True):
            labels = batchData.y
            outputs = mlp_model(GNN_model(batchData))
            test_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_labels.extend(labels.numpy())
    test_accuracy = accuracy_score(test_labels, test_predictions)
    nonzero = list(range(1, num_classes))
    appeared = [c for c in nonzero if c in set(test_labels)]
    if not appeared:
        return test_accuracy, np.nan, np.nan
    test_recall = recall_score(test_labels, test_predictions, labels=appeared, average='macro')
    test_f1 = f1_score(test_labels, test_predictions, labels=appeared, average='macro')
    return test_accuracy, test_recall, test_f1


def fine_tune_model(GNN_model, mlp_model, train_data, val_data,
                    epochs=100, lr=0.00001, batch_size=1):
    optimizer = torch.optim.Adam(
        list(GNN_model.parameters()) + list(mlp_model.parameters()), lr=lr
    )
    best_val_acc, best_val_f1 = 0, 0.0
    for epoch in range(epochs):
        train_loss = train(GNN_model, mlp_model, train_data, optimizer, batch_size)
        val_loss, val_acc, val_recall, val_f1 = validate(GNN_model, mlp_model, val_data)
        if WANDB_API_KEY:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                       "val_accuracy": val_acc, "val_recall": val_recall, "val_f1": val_f1})
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.2f}, "
              f"Val Acc: {val_acc:.2%}, Val F1: {val_f1:.2%}")
        if not np.isnan(val_f1) and val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_gnn_params = GNN_model.state_dict()
            best_mlp_params = mlp_model.state_dict()
        elif np.isnan(val_f1) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_gnn_params = GNN_model.state_dict()
            best_mlp_params = mlp_model.state_dict()
    GNN_model.load_state_dict(best_gnn_params)
    mlp_model.load_state_dict(best_mlp_params)
    return GNN_model, mlp_model


if __name__ == '__main__':
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)

    batch_size = 32; lr = 0.0001; epochs_val = 100
    validate_ratio = 0.1; test_ratio = 0.1
    node_out_feature = 30; graph_out_feature = 2
    fine_tune_ver = 3; data_path_ver = fine_tune_ver - 1
    model_label = '_only_tune'

    ratio_list = [0.01, 0.05, 0.10, 0.20, 0.50, 0.70, 1.00]
    for ratio in ratio_list:
        fine_tune_num = int(ratio * 100)
        print(f"Only-tune sample ratio: {ratio} ({fine_tune_num}%)")

        fine_tune_graph_path = str(
            GRAPH_DATA_DIR / f"fine-tune/fine-tune-sample{ratio}.csv"
        )

        ckpt_dir = CHECKPOINT_DIR / f"fine-tune-model{fine_tune_ver}" / str(fine_tune_num)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_gnnModel_path = str(ckpt_dir / f"gps_model{model_label}.pt")
        save_mlpModel_path = str(ckpt_dir / f"mlp_model{model_label}.pt")

        if WANDB_API_KEY:
            wandb.init(
                project="CIC-IDS2017",
                name=f"only-tune-ver{fine_tune_ver}-sample-{fine_tune_num}%",
                tags=['only-tune'],
                config={
                    "embedding_dim": embedding_dim, "epochs": epochs_val,
                    "batch_size": batch_size, "lr": lr, "num_class": num_classes,
                    "average": "only-attack-macro", "loss": "CrossEntropyLoss",
                    "validate_ratio": validate_ratio, "test_ratio": test_ratio,
                    "node_out_feature": node_out_feature, "graph_out_feature": graph_out_feature,
                },
            )

        data = get_all_data(data_all_path=fine_tune_graph_path, fine_tune=True, PE=True)

        train_data, temp_data = train_test_split(
            data, test_size=validate_ratio + test_ratio, random_state=88
        )
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=88)

        node_in_feature = train_data[0].num_node_features
        edge_in_feature = train_data[0].num_edge_features
        graph_in_feature = train_data[0].graph_attr.shape[0]
        print(f"Node dim: {node_in_feature}, Edge dim: {edge_in_feature}, Graph dim: {graph_in_feature}")

        GNN_model = NNConv_model(
            node_in_feature, edge_in_feature, graph_in_feature,
            node_out_feature=node_out_feature, graph_out_feature=graph_out_feature
        ).to(DEVICE)
        mlp_model = Fine_tune_MLP(embedding_dim, embedding_dim * 2, num_classes).to(DEVICE)

        GNN_model, mlp_model = fine_tune_model(
            GNN_model, mlp_model, train_data, val_data,
            epochs=epochs_val, lr=lr, batch_size=batch_size
        )

        torch.save(GNN_model.state_dict(), save_gnnModel_path)
        torch.save(mlp_model.state_dict(), save_mlpModel_path)

        test_accuracy, test_recall, test_f1 = test(GNN_model, mlp_model, test_data)
        if WANDB_API_KEY:
            wandb.log({"Test Accuracy": test_accuracy, "Test Recall": test_recall, "Test F1": test_f1})
        print(f"Test Accuracy: {test_accuracy:.2%}, Test Recall: {test_recall:.2%}, Test F1: {test_f1:.2%}")

        if WANDB_API_KEY:
            wandb.finish()
