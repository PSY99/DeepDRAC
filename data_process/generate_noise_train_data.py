"""Generate noise-labeled fine-tune training data.

This script implements the noise-label experiments from the paper:
1. Loads the full graph training data
2. Applies random label noise at configurable ratios (0.01 to 1.0)
3. Creates proportional stratified samples (10% per class)
4. Computes per-noise-rate class weights via inverse-frequency formula

The generated CSV files are consumed by train_tune_noiseLabel.py.

Usage:
    python data_process/generate_noise_train_data.py

This is extracted from the original research notebook:
    dataset/graph-format-data/noise_label_data/generate_noise_train_data.ipynb

After running, copy the printed weight dictionaries into
train_tune_noiseLabel.py's NOISE_RATE_WEIGHTS dict.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast
import math
import random

import pandas as pd
from config import GRAPH_DATA_DIR


# ── Label mapping ───────────────────────────────────────────────────────────
# (Original code from generate_noise_train_data.ipynb, cell feeb5258)
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

# Reverse mapping
id_to_label = {v: k for k, v in LABEL_MAP.items()}


# ── Helper functions ────────────────────────────────────────────────────────
# (All three functions are direct copies from the original notebook)

def parse_and_strip_benign(label_str):
    """Remove 'BENIGN' from label set. Keep original if result is empty."""
    label_set = ast.literal_eval(label_str)
    new_set = label_set - {'BENIGN'}
    return str(new_set) if new_set else str(label_set)


def noise_label(train_df, ratio):
    """Randomly pollute labels at the given ratio.
    (Original code from generate_noise_train_data.ipynb, cell 2ef75490)

    - Label 0 (BENIGN): changed to a random class in 1..13
    - Labels 1..13: 50% chance to change to 0, 50% chance to random 1..13
    """
    rows_to_pollute = train_df.sample(frac=ratio)
    for index, row in rows_to_pollute.iterrows():
        label = row['label_id']
        if label == 0:
            new_label = random.randint(1, 13)
        else:
            if random.random() <= 0.5:
                new_label = 0
            else:
                new_label = random.randint(1, 13)
        train_df.loc[index, 'label_id'] = new_label
    return train_df


def proportional_sample(df, label_col, ratio):
    """Stratified sampling: keep `ratio` fraction of each label group,
    with at least 1 sample per group.
    (Original code from generate_noise_train_data.ipynb, cell 36b2e172)
    """
    sampled_dfs = []
    for label_value, group_df in df.groupby(label_col):
        sample_size = max(1, math.floor(len(group_df) * ratio))
        sampled = group_df.sample(n=sample_size, random_state=42)
        sampled_dfs.append(sampled)
    return pd.concat(sampled_dfs).reset_index(drop=True)


def get_weights(graph_data):
    """Compute per-class inverse-frequency weights.
    (Original code from generate_noise_train_data.ipynb, cell e14561fd)

    Formula: weight[i] = total_samples / count_per_class[i]
    """
    counts = graph_data['label_id'].value_counts().to_dict()
    total = sum(counts.values())
    counts = dict(sorted(counts.items()))

    weights = [0 for _ in range(len(id_to_label))]
    for c in counts.keys():
        count = counts.get(c, 0)
        if count == 0:
            continue
        now_weight = round(total / count, 2)
        weights[int(c)] = now_weight

    print(f"  Weights (total={total}): {[round(w, 2) for w in weights]}")
    return weights


# ── Main: generate noise-labeled data ───────────────────────────────────────
# (Original code from generate_noise_train_data.ipynb, cell 5420ca40)

if __name__ == '__main__':
    # Load full training graph data
    graph_path = GRAPH_DATA_DIR / 'correct' / 'graph_train.csv'
    print(f"Reading graph data from: {graph_path}")
    graph_data = pd.read_csv(graph_path)

    # Clean labels
    graph_data['cleaned_group_label'] = graph_data['group_label'].apply(
        parse_and_strip_benign
    )
    graph_data['label_id'] = graph_data['cleaned_group_label'].map(LABEL_MAP)
    print(f"Total graphs: {len(graph_data)}")
    print(f"Label distribution:\n{graph_data['cleaned_group_label'].value_counts()}\n")

    # Noise ratios and sample ratio
    noise_rate_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sample_ratio = 0.1  # take 10% of each class for fine-tuning
    ver = 1  # experiment version (increment for each run)
    weights_dic = {}

    out_dir = GRAPH_DATA_DIR / f'noise-lossWeights_label_data{ver}' / 'train_sample'
    out_dir.mkdir(parents=True, exist_ok=True)

    for noise_rate in noise_rate_list:
        print(f"\n=== Noise rate: {noise_rate} ===")
        train_df = graph_data.copy()
        noise_train_df = noise_label(train_df, noise_rate)

        # Map IDs back to label strings for proportional sampling
        noise_train_df['now_cleaned_group_label'] = noise_train_df['label_id'].map(id_to_label)

        # Sample 10% proportionally
        sampled_df = proportional_sample(
            noise_train_df, 'now_cleaned_group_label', ratio=sample_ratio
        )
        print(f"  Sampled {len(sampled_df)} graphs (from {len(noise_train_df)} total)")

        # Compute weights for this noise rate
        weights_dic[noise_rate] = get_weights(sampled_df)

        # Save
        output_path = out_dir / f'noise_{noise_rate}_sample_{sample_ratio}.csv'
        sampled_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")

    # Print the full weight dictionary for copy-paste into train_tune_noiseLabel.py
    print("\n" + "=" * 60)
    print("Copy the dict below into train_tune_noiseLabel.py's NOISE_RATE_WEIGHTS:")
    print("=" * 60)
    print(weights_dic)
