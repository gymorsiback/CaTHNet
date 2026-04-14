"""
Ablation Study for ReHGNN
=========================
Trains ablation variants of ReHGNN to quantify the contribution of
each architectural component:

  1. w/o Type Projection   — shared linear replaces per-type projections
  2. w/o Residual          — remove skip connections
  3. w/o Gating            — remove type-aware gating
  4. w/o Constraint Loss   — set lambda_cap = lambda_lat = 0
  5. w/ Normalization       — add TypeAwareNorm (reverse ablation,
                              verifies that omitting norm is beneficial)

Usage:
  python train_ablation.py --data_root datasetsnew1 --epochs 300
  python train_ablation.py --variant no_gating   # run a single variant
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import torch
import torch.optim as optim
import gc
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.HGNN import HGNN_ModelPlacement
from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_for_model_placement
from utils.metrics import RecommendationMetrics

from train_v2 import (
    ConstraintAwareRankingLoss,
    train_one_epoch,
    evaluate,
    precompute_constraints,
)

# ============================================================================
# Ablation Variants
# ============================================================================

ABLATION_VARIANTS = {
    'full_model': {
        'display': 'Full Model (ReHGNN)',
        'ablation_config': None,
        'lambda_cap': None,
        'lambda_lat': None,
    },
    'no_type_proj': {
        'display': 'w/o Type Projection',
        'ablation_config': {'use_type_projection': False},
        'lambda_cap': None,
        'lambda_lat': None,
    },
    'no_residual': {
        'display': 'w/o Residual',
        'ablation_config': {'use_residual': False},
        'lambda_cap': None,
        'lambda_lat': None,
    },
    'no_gating': {
        'display': 'w/o Gating',
        'ablation_config': {'use_gating': False},
        'lambda_cap': None,
        'lambda_lat': None,
    },
    'no_constraint_loss': {
        'display': 'w/o Constraint Loss',
        'ablation_config': None,
        'lambda_cap': 0.0,
        'lambda_lat': 0.0,
    },
    'with_norm': {
        'display': 'w/ Type-aware Norm',
        'ablation_config': {'use_layer_norm': True},
        'lambda_cap': None,
        'lambda_lat': None,
    },
    'no_heterogeneous': {
        'display': 'w/o Heterogeneous Design',
        'ablation_config': {'use_type_projection': False, 'use_gating': False},
        'lambda_cap': None,
        'lambda_lat': None,
    },
    'no_residual_no_gating': {
        'display': 'w/o Residual + Gating',
        'ablation_config': {'use_residual': False, 'use_gating': False},
        'lambda_cap': None,
        'lambda_lat': None,
    },
    'minimal_arch': {
        'display': 'Minimal (Backbone Only)',
        'ablation_config': {'use_type_projection': False, 'use_residual': False, 'use_gating': False},
        'lambda_cap': None,
        'lambda_lat': None,
    },
}


def train_ablation_variant(variant_name, variant_cfg, config, dataset,
                           G, device):
    """Train one ablation variant and return results."""
    print(f"\n{'='*80}")
    print(f"ABLATION: {variant_cfg['display']}")
    print(f"{'='*80}")

    out_dir = Path(config['results_dir']) / 'ablation' / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    features = torch.FloatTensor(dataset.node_features).to(device)
    graph_matrix = torch.FloatTensor(G).to(device)

    abl_cfg = variant_cfg['ablation_config']
    model = HGNN_ModelPlacement(
        features.shape[1], config['n_hid'],
        dataset.num_users, dataset.num_models, dataset.num_servers,
        config['dropout'], ablation_config=abl_cfg,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    print(f"  Ablation config: {abl_cfg or 'Full model'}")

    metrics_calc = RecommendationMetrics(k_list=config['eval_k_list'])

    pos_indices = []
    for mid in range(dataset.num_models):
        pos_indices.append(dataset.model_positive_servers[mid])
    pos_indices = torch.LongTensor(pos_indices).to(device)

    model_res, server_cap, model_lat, lat_thresh = precompute_constraints(
        dataset, device
    )

    lam_cap = variant_cfg['lambda_cap'] if variant_cfg['lambda_cap'] is not None else config['lambda_cap']
    lam_lat = variant_cfg['lambda_lat'] if variant_cfg['lambda_lat'] is not None else config['lambda_lat']

    criterion = ConstraintAwareRankingLoss(
        k=config['k_positive'], temperature=0.05,
        lambda_cap=lam_cap, lambda_lat=lam_lat,
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

    var_config = {
        **config,
        'variant': variant_name,
        'display_name': variant_cfg['display'],
        'ablation_config': abl_cfg,
        'lambda_cap_actual': lam_cap,
        'lambda_lat_actual': lam_lat,
        'num_params': num_params,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(var_config, f, indent=2)

    csv_path = out_dir / 'train_metrics.csv'
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

            model.eval()
            with torch.no_grad():
                scores = model(features, graph_matrix)
                s_min = scores.min().item()
                s_max = scores.max().item()
                s_mean = scores.mean().item()
                s_std = scores.std().item()
            model.train()

            current_lr = optimizer.param_groups[0]['lr']

            print(f"  Epoch {epoch+1:3d}/{config['max_epochs']} | "
                  f"Loss: {loss:.4f} | "
                  f"P@5: {metrics['precision@5']:.4f} | "
                  f"NDCG@5: {metrics['ndcg@5']:.4f} | "
                  f"LR: {current_lr:.6f}")

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

            if metrics['ndcg@5'] > best_ndcg:
                best_ndcg = metrics['ndcg@5']
                best_epoch = epoch + 1
                best_metrics = metrics.copy()
                torch.save(model.state_dict(), out_dir / 'best_model.pt')
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

    # Save inference data
    best_state = torch.load(out_dir / 'best_model.pt', weights_only=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_scores = model(features, graph_matrix)
        topk_vals, topk_idx = torch.topk(best_scores, k=20, dim=1)
        inference_data = {
            'topk_server_indices': topk_idx.cpu().numpy().tolist(),
            'topk_scores': topk_vals.cpu().numpy().tolist(),
        }
        with open(out_dir / 'inference_topk.json', 'w') as f:
            json.dump(inference_data, f)
        np.save(out_dir / 'score_matrix.npy', best_scores.cpu().numpy())

    summary = {
        'variant': variant_name,
        'display_name': variant_cfg['display'],
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
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    stop_reason = f"early stopped (patience={patience})" if early_stopped else "max epochs reached"
    print(f"\n  Best NDCG@5: {best_ndcg:.4f} at epoch {best_epoch}")
    print(f"  Actual epochs: {actual_epochs}/{config['max_epochs']} ({stop_reason})")
    print(f"  Training time: {total_time:.1f}s")
    print(f"  Saved to: {out_dir}/")

    del model, optimizer, scheduler, features, graph_matrix, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'variant': variant_name,
        'display_name': variant_cfg['display'],
        'best_ndcg5': best_ndcg,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'training_time': total_time,
        'num_params': num_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='datasetsnew1')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Fixed training epochs for all variants')
    parser.add_argument('--patience', type=int, default=99999,
                        help='Early stopping patience (default: disabled for uniform curve length)')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'plateau'],
                        help='LR schedule: cosine=CosineAnnealing, plateau=ReduceLROnPlateau')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: auto-detect from tuning)')
    parser.add_argument('--n_hid', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout (default: auto-detect from tuning)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (default: auto-detect from tuning)')
    parser.add_argument('--lambda_cap', type=float, default=0.1)
    parser.add_argument('--lambda_lat', type=float, default=0.1)
    parser.add_argument('--variant', type=str, default='all',
                        choices=list(ABLATION_VARIANTS.keys()) + ['all'])
    args = parser.parse_args()

    lr = args.lr
    dropout = args.dropout
    weight_decay = args.weight_decay

    ours_config_path = Path(f'results_v2/{args.data_root}/ours/config.json')
    if ours_config_path.exists() and (lr is None or dropout is None or weight_decay is None):
        with open(ours_config_path) as f:
            ours_cfg = json.load(f)
        if lr is None:
            lr = ours_cfg.get('lr', 0.0005)
        if dropout is None:
            dropout = ours_cfg.get('dropout', 0.05)
        if weight_decay is None:
            weight_decay = ours_cfg.get('weight_decay', 0.0001)
        print(f"  Auto-detected ReHGNN hyperparams from main comparison:")
        print(f"    lr={lr}, dropout={dropout}, weight_decay={weight_decay}")
    else:
        lr = lr or 0.0005
        dropout = dropout or 0.05
        weight_decay = weight_decay or 0.0001

    config = {
        'k_positive': 10,
        'n_hid': args.n_hid,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': weight_decay,
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    dataset = TopKPlacementDataset(
        split='train', k_positive=config['k_positive'],
        data_root=config['data_root'],
    )
    dataset.prepare()

    print("\n" + "=" * 80)
    print("CONSTRUCTING HYPERGRAPH")
    print("=" * 80)
    use_gpu = torch.cuda.is_available()
    H, G, edge_info = construct_H_for_model_placement(dataset, k_neig=10,
                                                       use_gpu=use_gpu)

    if args.variant == 'all':
        variant_list = list(ABLATION_VARIANTS.keys())
    else:
        variant_list = [args.variant]

    all_results = {}
    for vname in variant_list:
        vcfg = ABLATION_VARIANTS[vname]

        if vname == 'full_model':
            ours_summary = Path(config['results_dir']) / 'ours' / 'summary.json'
            if ours_summary.exists():
                print(f"\n{'='*80}")
                print(f"ABLATION: {vcfg['display']}  [reusing main comparison result]")
                print(f"{'='*80}")
                with open(ours_summary) as f:
                    sm = json.load(f)
                abl_out = Path(config['results_dir']) / 'ablation' / 'full_model'
                abl_out.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(ours_summary, abl_out / 'summary.json')
                all_results[vname] = {
                    'variant': vname,
                    'display_name': vcfg['display'],
                    'best_ndcg5': sm['best_ndcg5'],
                    'best_epoch': sm['best_epoch'],
                    'best_metrics': sm['best_metrics'],
                    'training_time': sm.get('training_time_seconds', 0),
                    'num_params': sm.get('num_params', 0),
                }
                print(f"  Copied from {ours_summary}")
                print(f"  NDCG@5 = {sm['best_ndcg5']:.4f} (epoch {sm['best_epoch']})")
                continue

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        time.sleep(2)

        try:
            result = train_ablation_variant(
                vname, vcfg, config, dataset, G, device
            )
            all_results[vname] = result
        except Exception as e:
            print(f"\n  ERROR training {vname}: {e}")
            import traceback
            traceback.print_exc()
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                time.sleep(5)
                _ = torch.zeros(1, device=device)
                del _
                print(f"  GPU recovered, continuing to next variant...")
            except Exception:
                print(f"  WARNING: GPU may be in bad state, attempting next variant anyway...")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    print(f"\n{'Variant':<28} {'P@5':>8} {'R@5':>8} {'F1@5':>8} "
          f"{'NDCG@5':>8} {'Hit@5':>8} {'Params':>8}")
    print("-" * 90)
    for vname, res in all_results.items():
        m = res['best_metrics']
        tag = "** " if vname == 'full_model' else "   "
        print(f"{tag}{res['display_name']:<25} "
              f"{m.get('precision@5', 0):>8.4f} "
              f"{m.get('recall@5', 0):>8.4f} "
              f"{m.get('f1@5', 0):>8.4f} "
              f"{m.get('ndcg@5', 0):>8.4f} "
              f"{m.get('hit_rate@5', 0):>8.4f} "
              f"{res['num_params']:>8,}")

    # Save consolidated results
    abl_dir = Path(config['results_dir']) / 'ablation'
    abl_dir.mkdir(parents=True, exist_ok=True)

    abl_json = {
        'config': config,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'variants': {},
    }
    for vname, res in all_results.items():
        abl_json['variants'][vname] = {
            'display_name': res['display_name'],
            'best_ndcg5': res['best_ndcg5'],
            'best_epoch': res['best_epoch'],
            'training_time': res['training_time'],
            'num_params': res['num_params'],
            'metrics': {k: float(v) for k, v in res['best_metrics'].items()
                        if isinstance(v, (int, float))},
        }
    with open(abl_dir / 'ablation_results.json', 'w') as f:
        json.dump(abl_json, f, indent=2)

    rows = []
    for vname, res in all_results.items():
        m = res['best_metrics']
        row = {
            'variant': vname,
            'display_name': res['display_name'],
            'num_params': res['num_params'],
            'best_epoch': res['best_epoch'],
            'training_time': round(res['training_time'], 2),
        }
        for k_val in config['eval_k_list']:
            for mn in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
                key = f'{mn}@{k_val}'
                row[key] = round(m.get(key, 0.0), 6)
        row['mrr'] = round(m.get('mrr', 0.0), 6)
        row['map'] = round(m.get('map', 0.0), 6)
        rows.append(row)

    if rows:
        with open(abl_dir / 'ablation_comparison.csv', 'w', newline='') as f:
            all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nAblation results saved to: {abl_dir}/")


if __name__ == '__main__':
    main()
