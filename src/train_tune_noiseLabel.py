import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""DeepDRAC fine-tuning with noisy labels.

Evaluates DeepDRAC's robustness under label noise by fine-tuning on
datasets with controlled label noise ratios (0.01-1.0). Uses per-noise-ratio
class weights to handle imbalanced data.
"""

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
from config import DEVICE,  PRE_TRAIN_MODEL_DIR, CHECKPOINT_DIR, GRAPH_DATA_DIR

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ── Hyperparameters ──────────────────────────────────────────────────────────
random.seed(1)
torch.manual_seed(1)
embedding_dim = 32
epochs = 100
num_classes = 14

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

# ── Per-noise-ratio class weights ───────────────────────────────────────────
# Computed by data_process/generate_noise_train_data.py
# For each noise rate, that script loads graph data, applies label noise,
# samples 10% per class, then runs: weight[i] = total_samples / count_per_class[i]
# These values are from the paper's CIC-IDS2017 experiment (ver=5).
# Run generate_noise_train_data.py on YOUR data to get correct weights.
NOISE_RATE_WEIGHTS = {
    0.01: [1.02, 297.0, 891.0, 891.0, 891.0, 891.0, 891.0, 891.0, 891.0, 891.0, 891.0, 891.0, 891.0, 891.0],
    0.05: [1.05, 177.2, 221.5, 295.33, 221.5, 221.5, 295.33, 295.33, 295.33, 221.5, 295.33, 295.33, 221.5, 295.33],
    0.1: [1.11, 98.44, 147.67, 147.67, 110.75, 126.57, 126.57, 147.67, 126.57, 126.57, 147.67, 177.2, 98.44, 126.57],
    0.2: [1.26, 63.43, 68.31, 68.31, 63.43, 63.43, 68.31, 68.31, 55.5, 59.2, 52.24, 68.31, 74.0, 63.43],
    0.3: [1.43, 40.27, 44.3, 42.19, 40.27, 42.19, 40.27, 46.63, 44.3, 40.27, 42.19, 46.63, 46.63, 46.63],
    0.4: [1.68, 31.75, 32.93, 32.93, 29.63, 31.75, 32.93, 31.75, 30.66, 35.56, 31.75, 35.56, 31.75, 30.66],
    0.5: [2.01, 26.09, 26.88, 24.64, 24.64, 23.34, 26.09, 26.09, 26.88, 27.72, 27.72, 25.34, 26.88, 25.34],
    0.6: [2.5, 20.14, 21.61, 22.72, 18.08, 21.61, 22.72, 25.31, 23.32, 22.72, 19.26, 22.72, 22.15, 21.1],
    0.7: [3.34, 18.52, 18.91, 18.14, 18.91, 18.52, 20.2, 18.14, 17.1, 18.91, 17.78, 19.76, 18.14, 18.52],
    0.8: [5.03, 15.54, 15.54, 16.11, 17.72, 17.04, 15.28, 16.11, 14.52, 17.37, 16.41, 18.08, 15.54, 16.41],
    0.9: [10.07, 14.29, 13.63, 14.29, 13.84, 14.52, 15.02, 14.06, 14.06, 15.82, 15.54, 14.77, 13.84, 14.29],
    1.0: [0, 12.86, 13.44, 13.44, 12.86, 13.24, 12.86, 13.44, 11.99, 12.49, 13.65, 13.04, 13.04, 12.86],
}


def train(GNN_model, mlp_model, train_data, optimizer, weights, batch_size=1):
    GNN_model.train(); mlp_model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    loss_list = []
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for batchData in train_loader:
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
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        for batchData in val_loader:
            labels = batchData.y.to(DEVICE)
            outputs = mlp_model(GNN_model(batchData))
            val_losses.append(criterion(outputs, labels).item())
            val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    nonzero_classes = list(range(1, num_classes))
    appeared = [c for c in nonzero_classes if c in set(val_labels)]
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
    nonzero_classes = list(range(1, num_classes))
    appeared = [c for c in nonzero_classes if c in set(test_labels)]
    if not appeared:
        return test_accuracy, np.nan, np.nan
    test_recall = recall_score(test_labels, test_predictions, labels=appeared, average='macro')
    test_f1 = f1_score(test_labels, test_predictions, labels=appeared, average='macro')
    return test_accuracy, test_recall, test_f1


def fine_tune_model(GNN_model, mlp_model, train_data, val_data, weights,
                    epochs=100, lr=0.00001, batch_size=1):
    optimizer = torch.optim.Adam(
        list(GNN_model.parameters()) + list(mlp_model.parameters()), lr=lr
    )
    best_val_acc, best_val_f1 = 0, 0.0
    for epoch in range(epochs):
        train_loss = train(GNN_model, mlp_model, train_data, optimizer, weights, batch_size)
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

    batch_size = 32; lr = 0.00001; epochs_val = 100
    validate_ratio = 0.1; test_ratio = 0.1
    node_out_feature = 30; graph_out_feature = 2
    ver = 1

    GNN_model_pre_path = str(PRE_TRAIN_MODEL_DIR / "gps_global_model.pt")

    noise_ratio_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for noise_rate in noise_ratio_list:
        noise_label_num = int(noise_rate * 100)
        print(f"Noise rate: {noise_rate} ({noise_label_num}%)")

        fine_tune_graph_path = str(
            GRAPH_DATA_DIR / f"noise-lossWeights_label_data{ver}/train_sample/"
            f"noise_{noise_rate}_sample_0.1.csv"
        )

        ckpt_dir = CHECKPOINT_DIR / f"noise-lossWeights-fine-tune-model{ver}" / str(noise_label_num)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_gnnModel_path = str(ckpt_dir / "gps_model.pt")
        save_mlpModel_path = str(ckpt_dir / "mlp_model.pt")

        weights_now = torch.tensor(NOISE_RATE_WEIGHTS[noise_rate]).to(DEVICE)

        if WANDB_API_KEY:
            wandb.init(
                project="CIC-IDS2017",
                name=f"fine-tune-noise-lossWeights{ver}-{noise_label_num}%",
                tags=["noise", "fine-tune"],
                config={
                    "embedding_dim": embedding_dim, "epochs": epochs_val,
                    "batch_size": batch_size, "lr": lr, "num_class": num_classes,
                    "average": "only-attack-macro", "loss": "CrossEntropyLoss",
                    "noise_rate": noise_rate, "validate_ratio": validate_ratio,
                    "test_ratio": test_ratio, "node_out_feature": node_out_feature,
                    "graph_out_feature": graph_out_feature,
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
        GNN_model.load_state_dict(torch.load(GNN_model_pre_path, map_location=DEVICE))

        mlp_model = Fine_tune_MLP(embedding_dim, embedding_dim * 2, num_classes).to(DEVICE)

        GNN_model, mlp_model = fine_tune_model(
            GNN_model, mlp_model, train_data, val_data, weights_now,
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
