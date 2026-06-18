"""Generate fine-tune training samples at various label ratios.

Reads a graph CSV (with group labels), maps labels to numeric IDs,
and performs proportional stratified sampling at specified ratios
for supervised fine-tuning experiments.

Usage:
    python generate_finetune_train.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast
import math

import pandas as pd
from config import GRAPH_DATA_DIR


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


def parse_and_strip_benign(label_str):
    """Remove BENIGN from label set, keep original if empty after removal."""
    label_set = ast.literal_eval(label_str)
    new_set = label_set - {'BENIGN'}
    return str(new_set) if new_set else str(label_set)


def proportional_sample(df, label_col, ratio):
    """Proportionally sample each label group, keeping at least 1 per group."""
    sampled_dfs = []
    for label_value, group_df in df.groupby(label_col):
        sample_size = max(1, math.floor(len(group_df) * ratio))
        sampled = group_df.sample(n=sample_size, random_state=42)
        sampled_dfs.append(sampled)
    return pd.concat(sampled_dfs).reset_index(drop=True)


def generate_fine_tune_train_data(input_path, output_path, ratio=0.10):
    """Generate fine-tune data at the given sample ratio."""
    graph_data = pd.read_csv(input_path)
    graph_data['cleaned_group_label'] = graph_data['group_label'].apply(
        parse_and_strip_benign
    )
    graph_data['label_id'] = (
        graph_data['cleaned_group_label']
        .map(LABEL_MAP)
        .fillna(13)
        .astype(int)
    )

    sampled_df = proportional_sample(graph_data, 'cleaned_group_label', ratio=ratio)
    sampled_df.to_csv(output_path, index=False)
    print(f"Fine-tune data (ratio={ratio}): {len(sampled_df)} samples -> {output_path}")


if __name__ == "__main__":
    input_path = str(GRAPH_DATA_DIR / "correct" / "graph_train.csv")
    ratio_list = [0.01, 0.05, 0.10, 0.20, 0.50, 0.70, 1.00]

    for ratio in ratio_list:
        out_dir = GRAPH_DATA_DIR / "fine-tune"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"fine-tune-sample{ratio}.csv")
        generate_fine_tune_train_data(input_path, output_path, ratio=ratio)
