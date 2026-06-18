# DeepDRAC 运行文档（中文）

DeepDRAC：基于图神经网络的网络入侵检测告警深度聚合方法。

> **论文**: *DeepDRAC: Disposition Recommendation for Alert Clusters Based on Security Event Patterns* — IEEE Transactions on Information Forensics and Security (T-IFS), 2025, vol. 20, pp. 6443-6458. DOI: 10.1109/TIFS.2025.3580337

## 1. 总体流程

DeepDRAC 的完整实验链路：

1. 原始告警日志整理为带标签的 CSV。
2. 使用图聚类/社区划分算法把告警日志切分成告警子图。
3. 将告警子图保存为 `graph_train.csv`、`graph_test.csv` 等图格式数据。
4. 基于图格式数据生成预训练三元组和微调数据。
5. 运行 DeepDRAC 预训练。
6. 运行 DeepDRAC 微调。
7. 生成 graph embedding，并执行 base pattern + 增量聚类评估。

如果仓库中已有 `data/graph_data/graph_train.csv`、`graph_test.csv`、`data/graph_data/pre-train/` 和 `checkpoints/`，可以直接从第 5 步或第 7 步开始。

## 2. 目录说明

```text
deepdrac/
├── config.py                    # 集中路径配置（自动检测项目根目录）
├── data/
│   ├── alert_logs/              # CIC-IDS2017 格式化告警 CSV
│   │   └── correct_data/        # 标签修正后的版本
│   ├── frequent_mining/         # 图划分规则、频繁边挖掘结果
│   ├── knowledge/               # 端口-服务、攻击特征编码
│   └── graph_data/              # DeepDRAC 训练/测试图格式数据
├── data_process/                # 告警日志转图、图聚类/社区划分代码
├── src/                         # DeepDRAC 训练和评估代码
│   ├── models/                  # GNN 模型定义
│   ├── evaluation/              # 评估脚本
│   ├── train_pre.py             # 预训练
│   ├── train_tune.py            # 微调
│   └── data_loader.py           # 数据加载
├── checkpoints/                 # 已训练模型（gitignored）
└── notebooks/                   # Jupyter 分析笔记本
```

## 3. 运行环境

建议使用 Linux 或 WSL2/服务器环境运行。

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：

```bash
pip install numpy pandas scikit-learn matplotlib tqdm wandb swanlab
pip install python-louvain igraph leidenalg mlxtend incdbscan
```

还需要安装与 CUDA/PyTorch 版本匹配的：

```bash
pip install torch
pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv
```

### 路径配置

所有路径通过 `config.py` 自动检测（基于项目根目录）。如需自定义根目录：

```bash
export DEEPDRA C_ROOT=/path/to/deepdrac
```

## 4. 图聚类/子图构建

主方法脚本：`data_process/log2graph.py`

该脚本按时间窗口读取告警日志，将 IP 作为节点、告警交互作为边构建告警图，使用 Louvain 社区划分得到初始子图，再结合 `data/frequent_mining/frequent_combinations_output.txt` 中的规则做二次拆分/合并，最终生成 DeepDRAC 图格式 CSV。

运行：

```bash
cd data_process
python log2graph.py
```

脚本当前默认处理 `data/alert_logs/correct_data/train.csv`，输出 `data/graph_data/correct/graph_train.csv`。如果要生成 `graph_test.csv`，需要把脚本末尾的 `csv_name = 'train.csv'` 改成 `test.csv` 再运行。

### 其他社区划分结果

```bash
python data_process/log2graph_compare.py
```

支持：Louvain / Leiden / GN / LPA / Second_Louvain。

## 5. DeepDRAC 预训练

```bash
cd src
python train_pre.py
```

输入：`data/graph_data/pre-train/trple-graph-pre-all.csv` + `data/graph_data/graph_train.csv`

输出：`checkpoints/pre-train-model/gps_global_model_ver4.pt` 和对应的 Fisher 信息矩阵。

## 6. DeepDRAC 微调

```bash
python src/train_tune.py
```

在 1% 到 100% 的不同标注比例下微调，输出保存至 `checkpoints/fine-tune-model3/`。

## 7. DeepDRAC 评估

```bash
python src/evaluation/evaluate.py
```

该脚本：1) 用训练好的 GNN 生成 graph embedding；2) 基于 base pattern 分组，用 IncrementalDBSCAN 做增量聚类和风险匹配评估。

如需切换模型版本（only_tune → pre_and_tune），修改脚本中的 `model_select_ver` 变量。

## 8. 快速复现实验

如果已有图数据和 checkpoint：

```bash
python src/evaluation/evaluate.py
```

完整从图聚类开始：

```bash
python data_process/log2graph.py
python src/train_pre.py
python src/train_tune.py
python src/evaluation/evaluate.py
```

## 9. 常见问题

- **路径不存在** — 代码通过 `config.py` 自动检测项目根目录，或设置 `DEEPDRA C_ROOT` 环境变量。
- **Checkpoint 不存在** — 模型权重未包含在仓库中（通过 `.gitignore` 排除）。运行训练脚本生成，或从外部链接下载预训练权重。
- **Wandb/Swanlab 登录失败** — 设置环境变量 `WANDB_API_KEY` / `SWANLAB_API_KEY`，或留空以跳过远程日志记录。
