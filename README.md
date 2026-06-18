# DeepDRAC: Deep Alert Clustering for Network Intrusion Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Paper**: *DeepDRAC: Disposition Recommendation for Alert Clusters Based on Security Event Patterns* — IEEE Transactions on Information Forensics and Security (T-IFS), 2025.

DeepDRAC is a graph neural network (GNN) based framework for aggregating and
analyzing network intrusion detection alerts. It transforms raw IDS alerts into
graph-structured data, applies community detection and frequent pattern mining
for subgraph partitioning, then uses a pre-trained GraphGPS + GINEConv model
to produce graph embeddings. These embeddings enable incremental clustering
of alert subgraphs, dramatically reducing the workload for security analysts
while maintaining high attack detection accuracy.

## Overview

Modern Network Intrusion Detection Systems (NIDS) generate overwhelming volumes
of alerts — most of which are false positives or redundant notifications.
DeepDRAC addresses this alert fatigue problem through a multi-stage pipeline:

1. **Graph Construction** — Raw IDS alerts are converted into attributed
   graphs (IPs as nodes, alert interactions as edges) and partitioned via
   Louvain community detection with domain-specific frequent pattern rules.

2. **Self-Supervised Pre-training** — A GraphGPS+GINEConv encoder is pre-trained
   with triplet loss on graph triples to learn structural representations of
   alert subgraphs.

3. **Supervised Fine-tuning** — The pre-trained encoder is fine-tuned with an
   MLP classifier on a small set of labeled subgraphs (as little as 1% labeled
   data).

4. **Incremental Clustering** — Graph embeddings are grouped by base pattern
   profiles and clustered with IncrementalDBSCAN. Analysts only need to examine
   one representative sample per cluster.

## Directory Structure

```
deepdrac/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore
├── config.py                    # Centralized path configuration
├── data/
│   ├── alert_logs/              # Alert CSV files (user-provided)
│   │   └── correct_data/        # Labeled alert CSV files (see step 1)
│   ├── frequent_mining/         # Frequent edge pattern rules
│   ├── knowledge/               # Port-service & signature encodings
│   └── graph_data/              # Generated graph-format CSVs & pre-train data
├── data_process/                # Log-to-graph conversion & graph clustering
│   ├── log2graph.py             # Main pipeline: log → graph + Louvain partition
│   ├── log2graph_compare.py     # Alternative partition algorithms
│   ├── node_attribute_tools.py  # Node feature engineering
│   └── edge_attribute_tools.py  # Edge feature engineering
├── src/
│   ├── models/                  # GNN model definitions
│   │   ├── gps_global_model.py  # GraphGPS + GINEConv encoder
│   │   ├── nnconv_model.py      # NNConv baseline model
│   │   ├── fine_tune_mlp.py     # MLP classification head
│   │   └── nn/                  # Custom NN layers (EGAT, NNConv, etc.)
│   ├── data_loader.py           # PyG Data loading utilities
│   ├── train_pre.py             # Pre-training with triplet loss
│   ├── train_tune.py            # Fine-tuning with cross-entropy
│   ├── train_only_tune.py       # Fine-tuning from scratch (ablation)
│   ├── train_tune_noiseLabel.py # Fine-tuning with label noise
│   └── evaluation/
│       ├── evluation.py          # Main evaluation: embedding + clustering
│       ├── get_embedding.py     # Generate graph embeddings
│       └── get_base_pattern.py  # Base pattern profile extraction
├── notebooks/                   # Jupyter notebooks for analysis & plotting
└── checkpoints/                 # Model weights (gitignored — see below)
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)
- Linux or WSL2 environment

### Setup

```bash
# Clone the repository
git clone https://github.com/<username>/deepdrac.git
cd deepdrac

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch (match your CUDA version)
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

# Install PyTorch Geometric (match your PyTorch/CUDA version)
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv

# Install IncrementalDBSCAN
pip install incdbscan

# Verify the installation
python config.py
```

### Environment Variables (Optional)

For experiment tracking with Weights & Biases or SwanLab:

```bash
export WANDB_API_KEY="your-wandb-api-key"
export WANDB_ENTITY="your-wandb-username-or-team"
export SWANLAB_API_KEY="your-swanlab-api-key"
```

Without these, training and evaluation scripts will still run — just without
remote experiment logging.

## Quick Start

### 1. Prepare Alert Data

DeepDRAC processes **IDS alert logs** in CSV format. Each row represents one alert
generated by an IDS (e.g., Snort, Suricata). The expected input format is:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Alert sequence number |
| `real_time` | datetime | Alert timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `sip_str` | string | Source IP address |
| `dip_str` | string | Destination IP address |
| `sport` | int | Source port |
| `dport` | int | Destination port |
| `protocol` | string | Transport protocol (`TCP`, `UDP`, etc.) |
| `category_standard` | string | IDS alert category name |
| `category_standard_id` | int | Alert category/signature ID |
| `signature_priority` | int | Alert severity / priority level |
| `Label` | string | Ground-truth label (e.g., `BENIGN`, attack type name) |

Place your alert CSV files (e.g., `train.csv`, `test.csv`) under
`data/alert_logs/correct_data/`. See `samples/alert_sample.csv` and
`samples/README.md` for the complete format specification.

> **Why `correct_data/`?** In the original research, raw IDS alerts contained
> label errors (e.g., attacks mislabeled as BENIGN). The notebook
> `notebooks/alert_label_merge.ipynb` was used to apply label corrections,
> producing the "corrected" CSV files. The directory name is kept for
> compatibility. If your data already has accurate labels, simply place your
> CSV files there directly — no correction step is needed.
>
> The paper evaluates DeepDRAC on two benchmark datasets and one real-world
> enterprise dataset. DeepDRAC works with any IDS alert logs that conform to
> the schema above.

### 2. Mine Frequent Edge Patterns

```bash
# Step 2a: Extract community edge lists from alert CSVs
python data/frequent_mining/graph_partition.py

# Step 2b: Mine frequent association rules
python data/frequent_mining/generate_rules.py
```

Output: `data/frequent_mining/frequent_mining_edges.txt` and `frequent_combinations_output.txt`.

### 3. Build Alert Graphs

```bash
# For training set
python data_process/log2graph.py  # reads train.csv

# For test set (run programmatically with appropriate dates)
python -c "
from log2graph import Log2graph
from config import GRAPH_DATA_DIR
log2graph = Log2graph(csv_name='test.csv', start_time='<start>', end_time='<end>', out_path=str(GRAPH_DATA_DIR / 'correct/graph_test.csv'))
log2graph.format_data2csv()
"
```

This reads alert CSVs and produces `graph_train.csv` (~150MB) and `graph_test.csv` (~40MB)
in `data/graph_data/correct/`. Expected runtime: ~10 minutes per CSV on a modern CPU.

### 4. Generate Training Data

```bash
# Generate pre-training triples
python data_process/generate_pre_train.py

# Generate fine-tune samples at various ratios
python data_process/generate_finetune_train.py
```

Output:
- `data/graph_data/pre-train/trple-graph-pre-all.csv` — triplet index for pre-training
- `data/graph_data/fine-tune/fine-tune-sample{ratio}.csv` — labeled samples at 1%–100%

### 5. Pre-train the GNN Encoder

```bash
python src/train_pre.py
```

Trains the GraphGPS+GINEConv model using triplet loss. Checkpoints saved to
`checkpoints/pre-train-model/`. Set `WANDB_API_KEY` or `SWANLAB_API_KEY` for
optional experiment tracking.

**Note on hardware**: Training uses the device detected in `config.py` (GPU if
available, CPU otherwise). On CPU, pre-training with 100 epochs may take
several hours. Reduce `epochs` in `src/train_pre.py` for quick testing.

### 6. Fine-tune with Labeled Data

```bash
python src/train_tune.py
```

Fine-tunes the pre-trained model with an MLP classifier at various labeled
sample ratios (1%, 5%, 10%, 20%, 50%, 70%, 100%). Checkpoints saved to
`checkpoints/fine-tune-model3/`.

### 7. Evaluate

```bash
python src/evaluation/evluation.py
```

Generates graph embeddings, performs IncrementalDBSCAN clustering within each
base pattern group, and reports workload reduction and attack detection metrics.

## Model Checkpoints

Model weights (`.pt` files) are **not** included in this repository. Train them
from scratch following the Quick Start guide above. The expected checkpoint
directory structure after training is:

```
checkpoints/
├── pre-train-model/
│   └── gps_global_model.pt          # Pre-trained encoder
├── fine-tune-model2/                 # Only-tune (from scratch) checkpoints
│   ├── 1/gps_model_only_tune.pt
│   ├── 5/gps_model_only_tune.pt
│   └── ...
└── fine-tune-model3/                 # Pre-train + fine-tune checkpoints
    ├── 1/gps_model.pt
    ├── 5/gps_model.pt
    └── ...
```

## Dataset-Specific Configuration

The training scripts contain **example values** from the paper's CIC-IDS2017
experiments. When applying DeepDRAC to your own dataset, you need to update:

### Class Weights

`src/train_tune.py`, `src/train_only_tune.py`, and `src/train_tune_noiseLabel.py`
contain hardcoded per-class weights for cost-sensitive learning. These are
computed by **`data_process/compute_class_weights.py`**:

```bash
python data_process/compute_class_weights.py
```

This script reads the full graph CSV, counts labels per class, and prints
the weights. The formula is:

```
weight[i] = total_samples / count_per_class[i]
```

Copy the printed list into the training scripts and update `num_classes`.

### Label Mapping

The attack-type-to-class-ID mapping (`LABEL_MAP`) appears in:
- `data_process/generate_finetune_train.py`
- `data_process/compute_class_weights.py`
- `src/evaluation/evluation.py`

Update these dicts to match your dataset's attack type strings. The number of
entries determines `num_classes`.

### IP-to-Role Mapping

`data_process/node_attribute_tools.py` maps specific IP addresses to network
roles (Firewall, DNS Server, etc.) via a hardcoded dict. Replace with your
own network topology. Unknown IPs default to "Outsider."

### Knowledge Files

`data/knowledge/encoded_signature.csv` and `encoded_port_service.csv` are
used by `data_process/edge_attribute_tools.py` for edge feature construction.
To regenerate them from your alert data:

```bash
python data/knowledge/generate_encodings.py
```

The port_service mapping requires a hand-crafted `data/knowledge/port_service.csv`
describing your network's port-to-service assignments.

### Noise Label Experiments

`src/train_tune_noiseLabel.py` contains per-noise-rate class weights in the
`NOISE_RATE_WEIGHTS` dict. These are generated by:

```bash
python data_process/generate_noise_train_data.py
```

This script applies label noise, creates stratified fine-tune samples, and
computes weights — all following the original notebook logic from the paper.

### Date Ranges and Time Windows

The `__main__` blocks of `data_process/log2graph.py` and
`data/frequent_mining/graph_partition.py` contain example start/end timestamps.
Update these to match your data's time range. The default time window is
10 minutes (`time_interval=60*10`).

## Citation

If you use DeepDRAC in your research, please cite:

```bibtex
@ARTICLE{11037486,
  author={Liu, Yang and Ruan, Gaofei and Luo, Zian and Zhang, Shilong and Liu, Donghao and Fan, Xin and Zhou, Yadong and Liu, Ting},
  journal={IEEE Transactions on Information Forensics and Security},
  title={DeepDRAC: Disposition Recommendation for Alert Clusters Based on Security Event Patterns},
  year={2025},
  volume={20},
  pages={6443-6458},
  doi={10.1109/TIFS.2025.3580337}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.

## Contact

For questions or issues, please open a GitHub issue or contact the authors.
