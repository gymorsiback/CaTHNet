# CaTHNet: Constraint-Aware Type-Gated Heterogeneous Hypergraph Learning for Intelligent Model Deployment in Distributed Cloud-Edge Environments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/pytorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**CaTHNet** (**C**onstraint-**a**ware **T**ype-gated **H**ypergraph **Net**work) is a deep learning framework for intelligent server allocation in distributed large-model deployment scenarios. As large-scale AI models increasingly require distributed deployment across multiple coordinated nodes, selecting optimal server combinations becomes critical for balancing deployment quality, system reliability, and service latency. Traditional rule-based or heuristic methods struggle with the inherent heterogeneity of system entities (users, models, servers) and the high-order deployment relationships that cannot be decomposed into pairwise interactions.

CaTHNet addresses these challenges through:

- **Heterogeneous Hypergraph Modeling**: Unifies users, models, and servers into a single hypergraph structure where hyperedges explicitly capture group-level relationships (user-model workflows, model-server redundancy, server topology).
- **Type-Aware Gating Mechanism**: Dynamically balances aggregated hypergraph information with type-specific features, preventing semantic dilution during message passing.
- **Constraint-Aware Ranking Learning**: Integrates capacity and latency constraints directly into the training objective, eliminating the need for manual multi-objective weight tuning.
- **Residual Connections**: Mitigates over-smoothing in the hypergraph convolution layers to preserve discriminative node representations.

## Repository Structure

```
CaTHNet/
├── models/
│   ├── HGNN.py                    # CaTHNet model architecture
│   ├── baselines.py               # Baseline models (CE-GCN, CE-GAT, HAN, HGCN)
│   ├── losses.py                  # Constraint-aware ranking loss
│   └── layers.py                  # Hypergraph convolution layers
├── datasets/
│   ├── topk_placement_loader.py   # Dataset loader for deployment data
│   └── ...                        # Data utilities
├── utils/
│   ├── hypergraph_utils.py        # Hypergraph construction utilities
│   ├── metrics.py                 # Evaluation metrics (NDCG, MRR, MAP, etc.)
│   └── experiment_logger.py       # Experiment logging
├── datasetsnew1/                  # Semi-synthetic benchmark dataset
│   └── generate_dataset.py        # Dataset generation script
├── photo/train/
│   └── train_photo.py             # Visualization script for figures and tables
├── paper/                         # LaTeX source of the paper
├── train_v2.py                    # Main training script
├── train_ablation.py              # Ablation study training
├── eval_generalization.py         # Generalization evaluation
├── eval_inference.py              # Inference analysis (constraints, diversity)
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU acceleration)
- 8GB+ GPU memory

### Installation

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

### Training

```bash
# Train CaTHNet (default configuration)
python train_v2.py --model ours

# Train baseline models
python train_v2.py --model gat
python train_v2.py --model gcn
python train_v2.py --model han
python train_v2.py --model hypergcn
```

### Ablation Study

```bash
python train_ablation.py
```

### Generalization Evaluation

```bash
python eval_generalization.py
```

### Generate Visualizations

```bash
python photo/train/train_photo.py
```

## Dataset

The semi-synthetic benchmark dataset is grounded in real-world cloud infrastructure traces and includes:

| Component | Scale | Source |
|-----------|-------|--------|
| Servers | 1,500 nodes across 6 cloud regions | Azure GreenSKU framework |
| Models | 227 large models in 5 categories | LMSYS Chatbot Arena, etc. |
| Users | 10,000 per split (train/test) | Synthetic with geographic clustering |
| Generalization test sets | 6 scenarios (30%–80% intra-regional) | Bootstrap-resampled ground truth |

Ground truth deployment decisions are derived through multi-objective optimization jointly considering latency, capacity matching, network quality, and load balancing, with MMR-based selection for geographic diversity.

## Baselines

| Method | Type | Description |
|--------|------|-------------|
| Random | Heuristic | Uniform random server sampling |
| Popular | Heuristic | Rank by deployment frequency |
| User-Aware | Heuristic | Select geographically closest servers |
| Resource-Match | Heuristic | Cosine similarity on resource vectors |
| Load-Balanced | Heuristic | Greedy least-loaded selection |
| CE-GCN | Neural | Graph Convolutional Network |
| CE-GAT | Neural | Graph Attention Network |
| HAN | Neural | Hierarchical Attention Network |
| HGCN | Neural | Hypergraph Convolutional Network |

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.8+ |
| PyTorch | 1.12+ |
| CUDA | 11.0+ |
| NumPy | 1.21+ |
| Pandas | 1.3+ |
| scikit-learn | 1.0+ |
| Matplotlib | 3.5+ |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

**Last Updated**: April 2026
**Version**: 2.0.0
