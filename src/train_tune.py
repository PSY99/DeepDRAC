"""DeepDRAC fine-tuning script.

Fine-tunes a pre-trained GraphGPS+GINEConv model with an MLP classifier
on labeled alert subgraphs at various sample ratios. Uses class-weighted
cross-entropy loss and wandb for experiment tracking.
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
from config import DEVICE,  PRE_TRAIN_MODEL_DIR, FINE_TUNE_MODEL_DIR, GRAPH_DATA_DIR

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ── Hyperparameters ──────────────────────────────────────────────────────────
random.seed(1)
torch.manual_seed(1)
embedding_dim = 32
epochs = 100
num_classes = 14
weights = torch.tensor([
    1.01, 372.33, 992.89, 992.89, 1276.57, 1489.33, 1489.33,
    1787.2, 1787.2, 1787.2, 2234.0, 2978.67, 4468.0, 4468.0
]).to(DEVICE)

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")


def train(GNN_model, mlp_model, train_data, optimizer, batch_size=1):
    """Train one epoch."""
    GNN_model.train()
    mlp_model.train()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    loss_list = []
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for batchData in train_loader:
        optimizer.zero_grad()
        labels = batchData.y.to(DEVICE)
        y_pre = GNN_model(batchData)
        outputs = mlp_model(y_pre)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(loss_list) / len(loss_list)


def validate(GNN_model, mlp_model, val_data, batch_size=1):
    """Validate on held-out set (attack-class macro metrics)."""
    GNN_model.eval()
    mlp_model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    val_predictions, val_labels, val_losses = [], [], []
    with torch.no_grad():
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        for batchData in val_loader:
            labels = batchData.y.to(DEVICE)
            y_pre = GNN_model(batchData)
            outputs = mlp_model(y_pre)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            predictions = torch.argmax(outputs, dim=1)
            val_predictions.extend(predictions.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    avg_val_loss = sum(val_losses) / len(val_losses)
    val_accuracy = accuracy_score(val_labels, val_predictions)

    nonzero_classes = list(range(1, num_classes))
    appeared_classes = [c for c in nonzero_classes if c in set(val_labels)]
    if not appeared_classes:
        val_recall = np.nan
        val_f1 = np.nan
        print("Warning: no attack samples in validation set")
    else:
        val_recall = recall_score(val_labels, val_predictions,
                                  labels=appeared_classes, average='macro')
        val_f1 = f1_score(val_labels, val_predictions,
                          labels=appeared_classes, average='macro')

    return avg_val_loss, val_accuracy, val_recall, val_f1


def test(GNN_model, mlp_model, test_data, batch_size=1):
    """Test on held-out set (attack-class macro metrics)."""
    GNN_model.eval()
    mlp_model.eval()
    test_predictions, test_labels = [], []

    with torch.no_grad():
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        for batchData in test_loader:
            labels = batchData.y
            y_pre = GNN_model(batchData)
            outputs = mlp_model(y_pre)
            predictions = torch.argmax(outputs, dim=1)
            test_predictions.extend(predictions.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_accuracy = accuracy_score(test_labels, test_predictions)

    nonzero_classes = list(range(1, num_classes))
    appeared_classes = [c for c in nonzero_classes if c in set(test_labels)]
    if not appeared_classes:
        test_recall = np.nan
        test_f1 = np.nan
        print("Warning: no attack samples in test set")
    else:
        test_recall = recall_score(test_labels, test_predictions,
                                   labels=appeared_classes, average='macro')
        test_f1 = f1_score(test_labels, test_predictions,
                           labels=appeared_classes, average='macro')

    return test_accuracy, test_recall, test_f1


def fine_tune_model(GNN_model, mlp_model, train_data, val_data,
                    epochs=100, lr=0.00001, batch_size=1):
    """Fine-tune GNN + MLP, tracking best model by val F1."""
    optimizer = torch.optim.Adam(
        list(GNN_model.parameters()) + list(mlp_model.parameters()), lr=lr
    )
    best_val_acc = 0
    best_val_f1 = 0.0

    for epoch in range(epochs):
        train_loss = train(GNN_model, mlp_model, train_data, optimizer,
                           batch_size=batch_size)
        val_loss, val_accuracy, val_recall, val_f1 = validate(
            GNN_model, mlp_model, val_data
        )
        if WANDB_API_KEY:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_recall": val_recall,
                "val_f1": val_f1,
            })
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.2f}, "
              f"Val Loss: {val_loss:.2f}, Val Acc: {val_accuracy:.2%}, "
              f"Val Recall: {val_recall:.2%}, Val F1: {val_f1:.2%}")

        if not np.isnan(val_f1):
            if val_f1 >= best_val_f1:
                best_val_f1 = val_f1
                best_gnn_params = GNN_model.state_dict()
                best_mlp_params = mlp_model.state_dict()
        else:
            if val_accuracy >= best_val_acc:
                best_val_acc = val_accuracy
                best_gnn_params = GNN_model.state_dict()
                best_mlp_params = mlp_model.state_dict()

    GNN_model.load_state_dict(best_gnn_params)
    mlp_model.load_state_dict(best_mlp_params)
    return GNN_model, mlp_model


if __name__ == '__main__':
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)

    batch_size = 32
    lr = 0.00001
    epochs_val = 100
    validate_ratio = 0.1
    test_ratio = 0.1
    node_out_feature = 30
    graph_out_feature = 2

    GNN_model_pre_path = str(PRE_TRAIN_MODEL_DIR / "gps_global_model.pt")

    ratio_list = [0.01, 0.05, 0.10, 0.20, 0.50, 0.70, 1.00]
    for ratio in ratio_list:
        fine_tune_num = int(ratio * 100)
        print(f"Fine-tune sample ratio: {ratio} ({fine_tune_num}%)")

        fine_tune_graph_path = str(
            GRAPH_DATA_DIR / f"fine-tune/fine-tune-sample{ratio}.csv"
        )

        ckpt_dir = FINE_TUNE_MODEL_DIR / str(fine_tune_num)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_gnnModel_path = str(ckpt_dir / "gps_model.pt")
        save_mlpModel_path = str(ckpt_dir / "mlp_model.pt")

        if WANDB_API_KEY:
            wandb.init(
                project="CIC-IDS2017",
                name=f"fine-tune-sample-{fine_tune_num}%",
                config={
                    "embedding_dim": embedding_dim,
                    "epochs": epochs_val,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_class": num_classes,
                    "average": "only-attack-macro",
                    "loss": "CrossEntropyLoss",
                    "validate_ratio": validate_ratio,
                    "test_ratio": test_ratio,
                    "node_out_feature": node_out_feature,
                    "graph_out_feature": graph_out_feature,
                },
            )

        data = get_all_data(
            data_all_path=fine_tune_graph_path, fine_tune=True, PE=True
        )

        train_data, temp_data = train_test_split(
            data, test_size=validate_ratio + test_ratio, random_state=88
        )
        val_data, test_data = train_test_split(
            temp_data, test_size=0.5, random_state=88
        )

        node_in_feature = train_data[0].num_node_features
        edge_in_feature = train_data[0].num_edge_features
        graph_in_feature = train_data[0].graph_attr.shape[0]
        print(f"Node dim: {node_in_feature}, Edge dim: {edge_in_feature}, "
              f"Graph dim: {graph_in_feature}")

        GNN_model = NNConv_model(
            node_in_feature, edge_in_feature, graph_in_feature,
            node_out_feature=node_out_feature, graph_out_feature=graph_out_feature
        ).to(DEVICE)
        GNN_model.load_state_dict(torch.load(GNN_model_pre_path, map_location=DEVICE))

        mlp_model = Fine_tune_MLP(
            embedding_dim, embedding_dim * 2, num_classes
        ).to(DEVICE)

        GNN_model, mlp_model = fine_tune_model(
            GNN_model, mlp_model, train_data, val_data,
            epochs=epochs_val, lr=lr, batch_size=batch_size
        )

        torch.save(GNN_model.state_dict(), save_gnnModel_path)
        torch.save(mlp_model.state_dict(), save_mlpModel_path)

        test_accuracy, test_recall, test_f1 = test(
            GNN_model, mlp_model, test_data
        )
        if WANDB_API_KEY:
            wandb.log({
                "Test Accuracy": test_accuracy,
                "Test Recall": test_recall,
                "Test F1": test_f1,
            })
        print(f"Test Accuracy: {test_accuracy:.2%}, "
              f"Test Recall: {test_recall:.2%}, Test F1: {test_f1:.2%}")

        if WANDB_API_KEY:
            wandb.finish()
