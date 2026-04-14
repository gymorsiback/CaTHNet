"""
Comprehensive Training Script v2
================================
Improvements over train_experiment.py:
  1. Uses new multi-source semi-synthetic dataset (datasetsnew1)
  2. Constraint-aware loss (capacity + latency penalties)
  3. Supports multiple model architectures (ours + 4 neural baselines)
  4. Unified evaluation pipeline
  5. Exports all results for paper tables

Usage:
  python train_v2.py --model ours
  python train_v2.py --model gcn
  python train_v2.py --model gat
  python train_v2.py --model han
  python train_v2.py --model hypergcn
  python train_v2.py --model all   # run all models sequentially
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.HGNN import HGNN_ModelPlacement
from models.baselines import (
    GCN_Placement, GAT_Placement, HAN_Placement, HyperGCN_Placement,
    build_clique_adjacency,
)
from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_for_model_placement
from utils.metrics import RecommendationMetrics, compute_diversity_metrics


# ============================================================================
# Constraint-Aware Ranking Loss
# ============================================================================

class ConstraintAwareRankingLoss(nn.Module):
    """
    Softmax ranking loss with differentiable constraint penalties.

    L = L_ranking + lambda_cap * L_capacity + lambda_lat * L_latency

    Constraints are enforced via soft penalties on the softmax probability
    distribution, making them differentiable and end-to-end trainable.
    """

    def __init__(self, k=5, temperature=0.05,
                 lambda_cap=0.1, lambda_lat=0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.lambda_cap = lambda_cap
        self.lambda_lat = lambda_lat

    def forward(self, all_scores, positive_indices,
                model_resources=None, server_capacities=None,
                model_user_latency=None, latency_threshold=None):

        num_models = all_scores.size(0)
        ranking_losses = []

        for i in range(num_models):
            model_scores = all_scores[i]
            pos_idx = positive_indices[i]
            probs = torch.softmax(model_scores / self.temperature, dim=0)
            pos_probs = probs[pos_idx]
            loss = -torch.log(pos_probs + 1e-10).mean()
            ranking_losses.append(loss)

        ranking_loss = torch.stack(ranking_losses).mean()

        # Capacity constraint penalty
        capacity_penalty = torch.tensor(0.0, device=all_scores.device)
        if model_resources is not None and server_capacities is not None:
            probs_all = torch.softmax(all_scores / self.temperature, dim=1)
            expected_load = probs_all.T @ model_resources
            violation = F.relu(expected_load - server_capacities)
            capacity_penalty = violation.mean()

        # Latency constraint penalty
        latency_penalty = torch.tensor(0.0, device=all_scores.device)
        if model_user_latency is not None and latency_threshold is not None:
            probs_all = torch.softmax(all_scores / self.temperature, dim=1)
            expected_lat = (probs_all * model_user_latency).sum(dim=1)
            lat_violation = F.relu(expected_lat - latency_threshold)
            latency_penalty = lat_violation.mean()

        total_loss = (ranking_loss
                      + self.lambda_cap * capacity_penalty
                      + self.lambda_lat * latency_penalty)

        return total_loss, {
            'ranking': ranking_loss.item(),
            'capacity': capacity_penalty.item(),
            'latency': latency_penalty.item(),
        }


# ============================================================================
# Heuristic Baselines
# ============================================================================

def run_heuristic_baselines(dataset, metrics_calculator):
    """Run all heuristic baselines and return results."""
    results = {}
    model_indices, positive_servers_global = dataset.get_evaluation_pairs()
    pos_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in positive_servers_global
    ]

    # Random
    pred = torch.rand(dataset.num_models, dataset.num_servers)
    m = metrics_calculator.compute_all_metrics(pred, pos_local, prefix="")
    results['Random'] = {k: v for k, v in m.items()}
    print(f"  Random     - P@5: {m['precision@5']:.4f}  NDCG@5: {m['ndcg@5']:.4f}")

    # Popular (degree centrality)
    degree = torch.from_numpy(dataset.topology.sum(axis=0)).float()
    pred = degree.unsqueeze(0).expand(dataset.num_models, -1)
    m = metrics_calculator.compute_all_metrics(pred, pos_local, prefix="")
    results['Popular'] = {k: v for k, v in m.items()}
    print(f"  Popular    - P@5: {m['precision@5']:.4f}  NDCG@5: {m['ndcg@5']:.4f}")

    # User-Aware
    user_model_groups = defaultdict(list)
    for _, row in dataset.user_model_df.iterrows():
        mid = int(row['ModelID']) - 1
        uid = int(row['UserID']) - 1
        user_model_groups[mid].append(uid)

    user_locs = dataset.users_df[['Lo', 'La']].values
    server_locs = dataset.servers_df[['Lo', 'La']].values
    pred = torch.zeros(dataset.num_models, dataset.num_servers)
    for mid in range(dataset.num_models):
        if mid in user_model_groups:
            uids = user_model_groups[mid]
            avg_loc = user_locs[uids].mean(axis=0)
            dists = np.linalg.norm(server_locs - avg_loc, axis=1)
            scores = 1.0 - dists / (dists.max() + 1e-10)
            pred[mid] = torch.from_numpy(scores).float()
        else:
            pred[mid] = 0.5

    m = metrics_calculator.compute_all_metrics(pred, pos_local, prefix="")
    results['User-Aware'] = {k: v for k, v in m.items()}
    print(f"  User-Aware - P@5: {m['precision@5']:.4f}  NDCG@5: {m['ndcg@5']:.4f}")

    # Resource-Matching
    model_req = dataset.models_df[['Modelsize', 'Modelresource']].values.astype(float)
    model_req = (model_req - model_req.mean(0)) / (model_req.std(0) + 1e-8)
    server_cap = dataset.servers_df[['ComputationCapacity', 'StorageCapacity']].values.astype(float)
    server_cap = (server_cap - server_cap.mean(0)) / (server_cap.std(0) + 1e-8)
    pred = torch.zeros(dataset.num_models, dataset.num_servers)
    for mid in range(dataset.num_models):
        dists = np.linalg.norm(server_cap - model_req[mid], axis=1)
        scores = 1.0 - dists / (dists.max() + 1e-10)
        pred[mid] = torch.from_numpy(scores).float()
    m = metrics_calculator.compute_all_metrics(pred, pos_local, prefix="")
    results['Resource-Match'] = {k: v for k, v in m.items()}
    print(f"  Res-Match  - P@5: {m['precision@5']:.4f}  NDCG@5: {m['ndcg@5']:.4f}")

    # Load-Balanced
    server_loads = np.zeros(dataset.num_servers)
    pred = torch.zeros(dataset.num_models, dataset.num_servers)
    for mid in range(dataset.num_models):
        scores = 1.0 / (server_loads + 1.0)
        pred[mid] = torch.from_numpy(scores).float()
        top_servers = np.argsort(-scores)[:5]
        server_loads[top_servers] += 1
    m = metrics_calculator.compute_all_metrics(pred, pos_local, prefix="")
    results['Load-Balanced'] = {k: v for k, v in m.items()}
    print(f"  Load-Bal   - P@5: {m['precision@5']:.4f}  NDCG@5: {m['ndcg@5']:.4f}")

    return results


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_one_epoch(model, features, graph_matrix, positive_indices,
                    criterion, optimizer, dataset, device,
                    model_resources=None, server_capacities=None,
                    model_user_latency=None, latency_threshold=None):
    model.train()
    optimizer.zero_grad()
    all_scores = model(features, graph_matrix)

    loss, loss_components = criterion(
        all_scores, positive_indices,
        model_resources, server_capacities,
        model_user_latency, latency_threshold,
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item(), loss_components


def evaluate(model, features, graph_matrix, dataset, metrics_calc, device):
    model.eval()
    with torch.no_grad():
        scores = model(features, graph_matrix)
        _, pos_global = dataset.get_evaluation_pairs()
        pos_local = [
            [s - dataset.num_users - dataset.num_models for s in servers]
            for servers in pos_global
        ]
        metrics = metrics_calc.compute_all_metrics(scores, pos_local)
    return metrics


# ============================================================================
# Constraint Pre-computation
# ============================================================================

def precompute_constraints(dataset, device):
    """Pre-compute constraint tensors for differentiable penalty."""
    model_res = dataset.models_df['Modelresource'].values.astype(np.float32)
    model_resources = torch.from_numpy(model_res).to(device)

    server_cap = dataset.servers_df['ComputationCapacity'].values.astype(np.float32)
    server_cap_norm = server_cap / server_cap.max()
    server_capacities = torch.from_numpy(server_cap_norm).to(device)

    user_locs = dataset.users_df[['Lo', 'La']].values
    server_locs = dataset.servers_df[['Lo', 'La']].values

    user_model_groups = defaultdict(list)
    for _, row in dataset.user_model_df.iterrows():
        mid = int(row['ModelID']) - 1
        uid = int(row['UserID']) - 1
        if 0 <= mid < dataset.num_models and 0 <= uid < dataset.num_users:
            user_model_groups[mid].append(uid)

    latency_matrix = np.zeros((dataset.num_models, dataset.num_servers), dtype=np.float32)
    for mid in range(dataset.num_models):
        uids = user_model_groups.get(mid, [])
        if uids:
            sampled = uids[:200] if len(uids) > 200 else uids
            u_locs = user_locs[sampled]
            diffs = server_locs[np.newaxis, :, :] - u_locs[:, np.newaxis, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            latency_matrix[mid] = dists.mean(axis=0)
        else:
            latency_matrix[mid] = np.median(latency_matrix[:mid] if mid > 0
                                            else np.zeros((1, dataset.num_servers)))

    lat_max = latency_matrix.max() + 1e-10
    latency_matrix /= lat_max
    model_user_latency = torch.from_numpy(latency_matrix).to(device)

    latency_threshold = torch.tensor(0.5, device=device)

    return model_resources, server_capacities, model_user_latency, latency_threshold


# ============================================================================
# Model Factory
# ============================================================================

NEURAL_MODELS = {
    'ours': 'ReHGNN (Ours)',
    'gcn': 'GCN',
    'gat': 'GAT',
    'han': 'HAN',
    'hypergcn': 'HyperGCN',
}

USES_HYPERGRAPH = {'ours', 'hypergcn'}
USES_ADJACENCY = {'gcn', 'han'}
USES_SPARSE_ADJ = {'gat'}


def create_model(model_name, in_ch, n_hid, num_users, num_models, num_servers,
                 dropout):
    if model_name == 'ours':
        return HGNN_ModelPlacement(in_ch, n_hid, num_users, num_models,
                                   num_servers, dropout)
    elif model_name == 'gcn':
        return GCN_Placement(in_ch, n_hid, num_users, num_models,
                             num_servers, dropout)
    elif model_name == 'gat':
        return GAT_Placement(in_ch, n_hid, num_users, num_models,
                             num_servers, dropout)
    elif model_name == 'han':
        return HAN_Placement(in_ch, n_hid, num_users, num_models,
                             num_servers, dropout)
    elif model_name == 'hypergcn':
        return HyperGCN_Placement(in_ch, n_hid, num_users, num_models,
                                   num_servers, dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Main Training Loop
# ============================================================================

def _cuda_health_check(device):
    """Verify CUDA is functional before training."""
    if device.type != 'cuda':
        return True
    try:
        torch.cuda.empty_cache()
        gc.collect()
        a = torch.randn(64, 64, device=device)
        b = torch.randn(64, 64, device=device)
        c = torch.mm(a, b)
        _ = c.sum().item()
        del a, b, c
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        print(f"  !! CUDA health check FAILED: {e}")
        print(f"  !! Please restart Python / reboot the machine to reset GPU state.")
        return False


def _backup_model_dir(model_dir):
    """Backup critical files before training to prevent data loss on crash."""
    backup_dir = model_dir / '_backup'
    backup_dir.mkdir(parents=True, exist_ok=True)
    for fname in ['train_metrics.csv', 'summary.json', 'config.json',
                   'best_model.pt', 'score_matrix.npy', 'inference_topk.json']:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
    return backup_dir


def _restore_from_backup(model_dir):
    """Restore backed-up files after a training crash."""
    backup_dir = model_dir / '_backup'
    if not backup_dir.exists():
        return
    restored = []
    for fname in ['train_metrics.csv', 'summary.json', 'config.json',
                   'best_model.pt', 'score_matrix.npy', 'inference_topk.json']:
        bak = backup_dir / fname
        if bak.exists():
            shutil.copy2(bak, model_dir / fname)
            restored.append(fname)
    if restored:
        print(f"  >> Restored from backup: {', '.join(restored)}")
    shutil.rmtree(backup_dir, ignore_errors=True)


def train_single_model(model_name, config, dataset, H, G, A_norm, device):
    """Train a single model with comprehensive per-epoch data logging."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {NEURAL_MODELS[model_name]}")
    print(f"{'='*80}")

    if not _cuda_health_check(device):
        raise RuntimeError("CUDA is in a bad state. Please restart Python or reboot.")

    # --- Output directory for this model ---
    model_dir = Path(config['results_dir']) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Backup existing results to prevent data loss on crash ---
    _backup_model_dir(model_dir)

    features = torch.FloatTensor(dataset.node_features).to(device)

    if model_name in USES_HYPERGRAPH:
        graph_matrix = torch.FloatTensor(G).to(device)
    elif model_name in USES_SPARSE_ADJ:
        rows, cols = np.nonzero(A_norm)
        graph_matrix = torch.LongTensor(np.stack([rows, cols])).to(device)
        print(f"  Sparse edge_index: {graph_matrix.shape[1]:,} edges")
    else:
        graph_matrix = torch.FloatTensor(A_norm).to(device)

    model = create_model(
        model_name, features.shape[1], config['n_hid'],
        dataset.num_users, dataset.num_models, dataset.num_servers,
        config['dropout'],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    metrics_calc = RecommendationMetrics(k_list=config['eval_k_list'])

    pos_indices = []
    for mid in range(dataset.num_models):
        pos_indices.append(dataset.model_positive_servers[mid])
    pos_indices = torch.LongTensor(pos_indices).to(device)

    model_res, server_cap, model_lat, lat_thresh = precompute_constraints(
        dataset, device
    )

    criterion = ConstraintAwareRankingLoss(
        k=config['k_positive'], temperature=0.05,
        lambda_cap=config['lambda_cap'], lambda_lat=config['lambda_lat'],
    )

    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])

    lr_schedule = config.get('lr_schedule', 'cosine')
    if lr_schedule == 'cosine':
        cosine_epochs = config['max_epochs'] - config['warmup_epochs']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cosine_epochs, 1), eta_min=1e-6
        )
        print(f"  LR schedule: CosineAnnealing (T_max={cosine_epochs}, eta_min=1e-6)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, verbose=False
        )
        print(f"  LR schedule: ReduceLROnPlateau (factor=0.5, patience=15)")

    # --- Save config for this model ---
    model_config = {**config, 'model_name': model_name,
                    'display_name': NEURAL_MODELS[model_name],
                    'num_params': num_params,
                    'graph_type': 'hypergraph' if model_name in USES_HYPERGRAPH else 'adjacency',
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')}
    with open(model_dir / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    # --- Per-epoch CSV logger ---
    csv_path = model_dir / 'train_metrics.csv'
    csv_fields = [
        'epoch', 'loss_total', 'loss_ranking', 'loss_capacity', 'loss_latency',
        'learning_rate', 'score_min', 'score_max', 'score_mean', 'score_std',
        'precision@1', 'precision@3', 'precision@5', 'precision@10', 'precision@20',
        'recall@1', 'recall@3', 'recall@5', 'recall@10', 'recall@20',
        'f1@1', 'f1@3', 'f1@5', 'f1@10', 'f1@20',
        'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@20',
        'hit_rate@1', 'hit_rate@3', 'hit_rate@5', 'hit_rate@10', 'hit_rate@20',
    ]
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    best_ndcg = 0.0
    best_epoch = 0
    best_metrics = {}
    all_epoch_records = []
    patience = config.get('patience', 200)
    early_stopped = False
    start_time = time.time()

    for epoch in range(config['max_epochs']):
        if epoch < config['warmup_epochs']:
            lr = config['lr'] * (epoch + 1) / config['warmup_epochs']
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        loss, loss_parts = train_one_epoch(
            model, features, graph_matrix, pos_indices,
            criterion, optimizer, dataset, device,
            model_res, server_cap, model_lat, lat_thresh,
        )

        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if (epoch + 1) % config['print_freq'] == 0 or epoch == 0:
            metrics = evaluate(model, features, graph_matrix, dataset,
                               metrics_calc, device)
            metrics['loss'] = loss

            # Score statistics for analysis
            model.eval()
            with torch.no_grad():
                curr_scores = model(features, graph_matrix)
                s_min = curr_scores.min().item()
                s_max = curr_scores.max().item()
                s_mean = curr_scores.mean().item()
                s_std = curr_scores.std().item()
            model.train()

            current_lr = optimizer.param_groups[0]['lr']

            print(f"  Epoch {epoch+1:3d}/{config['max_epochs']} | "
                  f"Loss: {loss:.4f} (R:{loss_parts['ranking']:.4f} "
                  f"C:{loss_parts['capacity']:.4f} L:{loss_parts['latency']:.4f}) | "
                  f"P@5: {metrics['precision@5']:.4f} | "
                  f"NDCG@5: {metrics['ndcg@5']:.4f} | "
                  f"LR: {current_lr:.6f}")

            # Write to CSV
            row = {
                'epoch': epoch + 1,
                'loss_total': round(loss, 6),
                'loss_ranking': round(loss_parts['ranking'], 6),
                'loss_capacity': round(loss_parts['capacity'], 6),
                'loss_latency': round(loss_parts['latency'], 6),
                'learning_rate': round(current_lr, 8),
                'score_min': round(s_min, 6),
                'score_max': round(s_max, 6),
                'score_mean': round(s_mean, 6),
                'score_std': round(s_std, 6),
            }
            for k_val in config['eval_k_list']:
                for metric_name in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
                    key = f'{metric_name}@{k_val}'
                    row[key] = round(metrics.get(key, 0.0), 6)
            csv_writer.writerow(row)
            csv_file.flush()
            all_epoch_records.append(row)

            if metrics['ndcg@5'] > best_ndcg:
                best_ndcg = metrics['ndcg@5']
                best_epoch = epoch + 1
                best_metrics = metrics.copy()
                torch.save(model.state_dict(), model_dir / 'best_model.pt')
                print(f"    >> New best! NDCG@5: {best_ndcg:.4f}")

            if lr_schedule == 'plateau':
                scheduler.step(metrics['ndcg@5'])

            epochs_no_improve = (epoch + 1) - best_epoch
            if epochs_no_improve >= patience and (epoch + 1) >= config['warmup_epochs'] + patience:
                print(f"\n  ** Early stopping at epoch {epoch+1}: "
                      f"no improvement for {epochs_no_improve} epochs "
                      f"(best={best_ndcg:.4f} at epoch {best_epoch})")
                early_stopped = True
                break

        if lr_schedule == 'cosine' and epoch >= config['warmup_epochs']:
            scheduler.step()

    csv_file.close()
    total_time = time.time() - start_time
    actual_epochs = epoch + 1

    # --- Save best-epoch score matrix for inference analysis ---
    print(f"  Saving best model inference data...")
    best_state = torch.load(model_dir / 'best_model.pt', weights_only=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_scores = model(features, graph_matrix)
        # Top-K predictions per model for inference analysis
        topk_vals, topk_idx = torch.topk(best_scores, k=20, dim=1)
        inference_data = {
            'topk_server_indices': topk_idx.cpu().numpy().tolist(),
            'topk_scores': topk_vals.cpu().numpy().tolist(),
        }
        with open(model_dir / 'inference_topk.json', 'w') as f:
            json.dump(inference_data, f)

        # Full score matrix (for detailed analysis / heatmaps)
        np.save(model_dir / 'score_matrix.npy', best_scores.cpu().numpy())

    # --- Save training summary ---
    summary = {
        'model_name': model_name,
        'display_name': NEURAL_MODELS[model_name],
        'best_ndcg5': best_ndcg,
        'best_epoch': best_epoch,
        'total_epochs': actual_epochs,
        'max_epochs': config['max_epochs'],
        'early_stopped': early_stopped,
        'patience': patience,
        'lr_schedule': lr_schedule,
        'training_time_seconds': round(total_time, 2),
        'num_params': num_params,
        'best_metrics': {k: round(float(v), 6) for k, v in best_metrics.items()
                         if isinstance(v, (int, float))},
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    with open(model_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Training succeeded — remove backup
    backup_dir = model_dir / '_backup'
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)

    stop_reason = f"early stopped (patience={patience})" if early_stopped else "max epochs reached"
    print(f"\n  Best NDCG@5: {best_ndcg:.4f} at epoch {best_epoch}")
    print(f"  Actual epochs: {actual_epochs}/{config['max_epochs']} ({stop_reason})")
    print(f"  Training time: {total_time:.1f}s")
    print(f"  Saved to: {model_dir}/")
    print(f"    - train_metrics.csv  ({len(all_epoch_records)} records)")
    print(f"    - best_model.pt")
    print(f"    - score_matrix.npy")
    print(f"    - inference_topk.json")
    print(f"    - summary.json")
    print(f"    - config.json")

    return {
        'model_name': model_name,
        'display_name': NEURAL_MODELS[model_name],
        'best_ndcg5': best_ndcg,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'training_time': total_time,
        'num_params': num_params,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        choices=['ours', 'gcn', 'gat', 'han', 'hypergcn', 'all'])
    parser.add_argument('--data_root', type=str, default='datasetsnew1')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Maximum training epochs (early stopping may end sooner)')
    parser.add_argument('--patience', type=int, default=200,
                        help='Early stopping patience: stop if no NDCG@5 improvement for this many epochs')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'plateau'],
                        help='LR schedule: cosine=CosineAnnealing (guarantees convergence), '
                             'plateau=ReduceLROnPlateau (legacy)')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--n_hid', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lambda_cap', type=float, default=0.1)
    parser.add_argument('--lambda_lat', type=float, default=0.1)
    args = parser.parse_args()

    config = {
        'k_positive': 10,
        'n_hid': args.n_hid,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'max_epochs': args.epochs,
        'patience': args.patience,
        'lr_schedule': args.lr_schedule,
        'print_freq': 10,
        'warmup_epochs': 30,
        'eval_k_list': [1, 3, 5, 10, 20],
        'lambda_cap': args.lambda_cap,
        'lambda_lat': args.lambda_lat,
        'data_root': args.data_root,
        'results_dir': f'results_v2/{args.data_root}',
    }

    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data root: {config['data_root']}")

    # Load dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    dataset = TopKPlacementDataset(
        split='train', k_positive=config['k_positive'],
        data_root=config['data_root'],
    )
    dataset.prepare()

    # Build hypergraph
    print("\n" + "=" * 80)
    print("CONSTRUCTING GRAPHS")
    print("=" * 80)
    use_gpu = torch.cuda.is_available()
    H, G, edge_info = construct_H_for_model_placement(dataset, k_neig=10,
                                                       use_gpu=use_gpu)

    # Build clique adjacency for GCN/GAT/HAN baselines
    print("\nBuilding clique-expanded adjacency matrix...")
    A_norm = build_clique_adjacency(H)
    print(f"  Adjacency shape: {A_norm.shape}")

    # Run heuristic baselines
    print("\n" + "=" * 80)
    print("HEURISTIC BASELINES")
    print("=" * 80)
    metrics_calc = RecommendationMetrics(k_list=config['eval_k_list'])
    heuristic_results = run_heuristic_baselines(dataset, metrics_calc)

    # Determine which neural models to train
    if args.model == 'all':
        model_list = list(NEURAL_MODELS.keys())
    else:
        model_list = [args.model]

    # Train neural models
    neural_results = {}
    for model_name in model_list:
        try:
            result = train_single_model(
                model_name, config, dataset, H, G, A_norm, device
            )
            neural_results[model_name] = result
        except Exception as e:
            print(f"\n  ERROR training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            model_dir = Path(config['results_dir']) / model_name
            _restore_from_backup(model_dir)

    # ====================================================================
    # Consolidate: include previously trained models not in this session
    # ====================================================================
    results_dir = Path(config['results_dir'])
    all_model_keys = ['ours', 'gcn', 'gat', 'han', 'hypergcn']
    for mkey in all_model_keys:
        if mkey not in neural_results:
            summary_path = results_dir / mkey / 'summary.json'
            if summary_path.exists():
                with open(summary_path) as _f:
                    s = json.load(_f)
                neural_results[mkey] = {
                    'display_name': s['display_name'],
                    'best_ndcg5': s['best_ndcg5'],
                    'best_epoch': s['best_epoch'],
                    'training_time': s.get('training_time_seconds', 0),
                    'num_params': s['num_params'],
                    'best_metrics': s['best_metrics'],
                }

    # ====================================================================
    # Final Summary
    # ====================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<20} {'P@5':>8} {'R@5':>8} {'F1@5':>8} "
          f"{'NDCG@5':>8} {'Hit@5':>8}")
    print("-" * 72)

    for name, m in heuristic_results.items():
        print(f"{name:<20} {m.get('precision@5',0):>8.4f} "
              f"{m.get('recall@5',0):>8.4f} {m.get('f1@5',0):>8.4f} "
              f"{m.get('ndcg@5',0):>8.4f} {m.get('hit_rate@5',0):>8.4f}")

    print("-" * 72)
    for name, res in neural_results.items():
        m = res['best_metrics']
        tag = "** " if name == 'ours' else "   "
        print(f"{tag}{res['display_name']:<17} {m.get('precision@5',0):>8.4f} "
              f"{m.get('recall@5',0):>8.4f} {m.get('f1@5',0):>8.4f} "
              f"{m.get('ndcg@5',0):>8.4f} {m.get('hit_rate@5',0):>8.4f}")

    # ====================================================================
    # Save all results (JSON + CSV for visualization)
    # ====================================================================
    # 1. Comprehensive JSON
    all_results = {
        'config': config,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'heuristic': {k: {kk: float(vv) for kk, vv in v.items()}
                      for k, v in heuristic_results.items()},
        'neural': {},
    }
    for name, res in neural_results.items():
        all_results['neural'][name] = {
            'display_name': res['display_name'],
            'best_ndcg5': res['best_ndcg5'],
            'best_epoch': res['best_epoch'],
            'training_time': res['training_time'],
            'num_params': res['num_params'],
            'metrics': {k: float(v) for k, v in res['best_metrics'].items()
                        if isinstance(v, (int, float))},
        }
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # 2. Unified comparison CSV (all methods, easy for plotting)
    comparison_rows = []
    for name, m in heuristic_results.items():
        row = {'method': name, 'type': 'heuristic'}
        for k_val in config['eval_k_list']:
            for metric_name in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
                key = f'{metric_name}@{k_val}'
                row[key] = round(m.get(key, 0.0), 6)
        comparison_rows.append(row)
    for name, res in neural_results.items():
        m = res['best_metrics']
        row = {'method': res['display_name'], 'type': 'neural',
               'best_epoch': res['best_epoch'],
               'training_time': round(res['training_time'], 2),
               'num_params': res['num_params']}
        for k_val in config['eval_k_list']:
            for metric_name in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
                key = f'{metric_name}@{k_val}'
                row[key] = round(m.get(key, 0.0), 6)
        comparison_rows.append(row)

    if comparison_rows:
        comp_df_path = results_dir / 'method_comparison.csv'
        all_keys = list(dict.fromkeys(
            k for row in comparison_rows for k in row.keys()
        ))
        with open(comp_df_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(comparison_rows)

    # 3. Heuristic baselines detail CSV
    heuristic_path = results_dir / 'heuristic_baselines.csv'
    h_rows = []
    for name, m in heuristic_results.items():
        row = {'method': name}
        row.update({k: round(float(v), 6) for k, v in m.items()})
        h_rows.append(row)
    if h_rows:
        with open(heuristic_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=h_rows[0].keys())
            writer.writeheader()
            writer.writerows(h_rows)

    print(f"\nAll results saved to: {results_dir}/")
    print(f"  - all_results.json          (complete JSON)")
    print(f"  - method_comparison.csv     (unified table for plotting)")
    print(f"  - heuristic_baselines.csv   (heuristic detail)")
    for name in neural_results:
        print(f"  - {name}/train_metrics.csv  (per-epoch training curves)")
        print(f"  - {name}/score_matrix.npy   (inference score matrix)")
        print(f"  - {name}/inference_topk.json (top-K predictions)")
        print(f"  - {name}/best_model.pt      (model checkpoint)")
        print(f"  - {name}/summary.json       (training summary)")
        print(f"  - {name}/config.json        (hyperparameters)")


if __name__ == '__main__':
    main()
