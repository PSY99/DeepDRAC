"""Compute class weights from graph CSV label distribution.

This script reads a graph CSV (produced by log2graph.py) and computes
per-class weights for cost-sensitive learning using the inverse-frequency
formula: weight[i] = total_samples / count_per_class[i].

The computed weights are used as the `CrossEntropyLoss(weight=...)` parameter
in the training scripts (train_tune.py, train_only_tune.py, etc.).

Usage:
    python data_process/compute_class_weights.py

This is extracted from the original research notebook:
    dataset/graph-format-data/generate_train_data.ipynb (cell edb06cf0)

After running, copy the printed weight list into the training scripts.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast
import pandas as pd
from config import GRAPH_DATA_DIR


# ── Helper: remove BENIGN from label set ────────────────────────────────────
# (Original code from generate_train_data.ipynb)

def parse_and_strip_benign(label_str):
    """Convert string to set and remove 'BENIGN'.
    If the set becomes empty after removal, keep the original."""
    label_set = ast.literal_eval(label_str)
    new_set = label_set - {'BENIGN'}
    return str(new_set) if new_set else str(label_set)


# ── Label-to-ID mapping ─────────────────────────────────────────────────────
# (Original code from generate_train_data.ipynb)
# These are the 14 attack types from the paper's CIC-IDS2017 experiments.
# For a different dataset, update this dict to match your attack types.

LABEL_MAP = {
    "{'BENIGN'}": 0,
    "{'Bot'}": 1,
    "{'PortScan'}": 2,
    "{'DDoS'}": 3,
    "{'SSH-Patator'}": 4,
    "{'DoS Hulk'}": 5,
    "{'Web Attack ?Brute Force'}": 6,
    "{'Web Attack ?XSS'}": 7,
    "{'DoS slowloris'}": 8,
    "{'Infiltration'}": 9,
    "{'DoS Slowhttptest'}": 10,
    "{'DoS GoldenEye'}": 11,
    "{'FTP-Patator'}": 12,
    "{'Web Attack ?Sql Injection'}": 13,
}


# ── Main: compute weights from graph data ───────────────────────────────────
# (Original code from generate_train_data.ipynb, cell edb06cf0)

if __name__ == '__main__':
    # Load the graph CSV (output of log2graph.py)
    graph_path = GRAPH_DATA_DIR / 'correct' / 'graph_train.csv'
    print(f"Reading graph data from: {graph_path}")
    graph_data = pd.read_csv(graph_path)

    # Clean labels: remove BENIGN from label sets
    graph_data['cleaned_group_label'] = graph_data['group_label'].apply(
        parse_and_strip_benign
    )

    # Map to integer class IDs
    graph_data['label_id'] = graph_data['cleaned_group_label'].map(LABEL_MAP)
    print(f"Total graphs: {len(graph_data)}")
    print(f"Label distribution:")

    # ── Compute class weights (original formula) ────────────────────────────
    # weight[i] = total_samples / count_per_class[i]
    counts = graph_data['cleaned_group_label'].value_counts().to_dict()
    total = sum(counts.values())

    weights = []
    for c in counts.keys():
        count = counts.get(c, 1)  # avoid division by zero
        weights.append(total / count)

    # Print results
    print("Label order:")
    print(list(counts.keys()))
    print()
    print("Class weights (total/count, rounded to 2 decimals):")
    rounded = [round(w, 2) for w in weights]
    print(rounded)
    print()
    print("Copy the list above into the `weights` variable in:")
    print("  src/train_tune.py")
    print("  src/train_only_tune.py")
    print()
    print(f"Also update `num_classes = {len(counts)}` in those files.")
