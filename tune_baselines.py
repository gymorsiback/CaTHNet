"""
Hyperparameter Tuning (All Models)
===================================
For each model (including ReHGNN), runs a grid search over key
hyperparameters using a shorter training schedule, then retrains
the best configuration with early stopping (patience=200, max 2000 epochs).

Usage:
  python tune_baselines.py --data_root datasetsnew1                  # tune ALL models
  python tune_baselines.py --data_root datasetsnew1 --baseline ours  # tune ReHGNN only
  python tune_baselines.py --data_root datasetsnew1 --baseline baselines_only  # tune baselines only
"""

import os, sys, time, json, argparse, itertools, gc
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_for_model_placement
from utils.metrics import RecommendationMetrics
from train_v2 import (
    ConstraintAwareRankingLoss, train_one_epoch, evaluate,
    precompute_constraints, create_model, build_clique_adjacency,
    train_single_model,
    USES_HYPERGRAPH, USES_SPARSE_ADJ, NEURAL_MODELS,
)

ALL_MODELS = ['ours', 'gcn', 'gat', 'han', 'hypergcn']
BASELINES = ['gcn', 'gat', 'han', 'hypergcn']

SEARCH_SPACE = {
    'lr':           [0.001, 0.0005, 0.0002],
    'n_hid':        [128],
    'dropout':      [0.05, 0.1, 0.2],
    'weight_decay': [0.0001, 0.0005],
}

MODEL_SEARCH_OVERRIDES = {
    'han':      {'n_hid': [64]},
    'hypergcn': {'n_hid': [64, 128]},
}

SEARCH_EPOCHS = 100
FINAL_EPOCHS  = 2000


def quick_train(model_name, hp, dataset, G, A_norm, device,
                constraints, eval_k_list, k_positive):
    """Train a model with given hyperparams for SEARCH_EPOCHS, return best NDCG@5."""
    features = torch.FloatTensor(dataset.node_features).to(device)

    if model_name in USES_HYPERGRAPH:
        graph_matrix = torch.FloatTensor(G).to(device)
    elif model_name in USES_SPARSE_ADJ:
        rows, cols = np.nonzero(A_norm)
        graph_matrix = torch.LongTensor(np.stack([rows, cols])).to(device)
    else:
        graph_matrix = torch.FloatTensor(A_norm).to(device)

    model = create_model(
        model_name, features.shape[1], hp['n_hid'],
        dataset.num_users, dataset.num_models, dataset.num_servers,
        hp['dropout'],
    ).to(device)

    pos_indices = []
    for mid in range(dataset.num_models):
        pos_indices.append(dataset.model_positive_servers[mid])
    pos_indices = torch.LongTensor(pos_indices).to(device)

    model_res, server_cap, model_lat, lat_thresh = constraints

    criterion = ConstraintAwareRankingLoss(
        k=k_positive, temperature=0.05,
        lambda_cap=0.1, lambda_lat=0.1,
    )
    optimizer = optim.Adam(model.parameters(), lr=hp['lr'],
                           weight_decay=hp['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=False)

    metrics_calc = RecommendationMetrics(k_list=eval_k_list)
    warmup = 20
    best_ndcg = 0.0

    for epoch in range(SEARCH_EPOCHS):
        if epoch < warmup:
            lr = hp['lr'] * (epoch + 1) / warmup
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        train_one_epoch(model, features, graph_matrix, pos_indices,
                        criterion, optimizer, dataset, device,
                        model_res, server_cap, model_lat, lat_thresh)

        if (epoch + 1) % 20 == 0:
            metrics = evaluate(model, features, graph_matrix, dataset,
                               metrics_calc, device)
            ndcg5 = metrics.get('ndcg@5', 0)
            if ndcg5 > best_ndcg:
                best_ndcg = ndcg5
            scheduler.step(ndcg5)

    del model, optimizer, graph_matrix, pos_indices, criterion
    torch.cuda.empty_cache()
    gc.collect()
    return best_ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='datasetsnew1')
    parser.add_argument('--baseline', type=str, default='all',
                        choices=ALL_MODELS + ['all', 'baselines_only'])
    parser.add_argument('--final_epochs', type=int, default=FINAL_EPOCHS)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results_dir = Path(f'results_v2/{args.data_root}')
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    dataset = TopKPlacementDataset(split='train', k_positive=10,
                                   data_root=args.data_root)
    dataset.prepare()

    print("\n" + "=" * 80)
    print("CONSTRUCTING GRAPHS")
    print("=" * 80)
    use_gpu = torch.cuda.is_available()
    H, G, edge_info = construct_H_for_model_placement(dataset, k_neig=10,
                                                       use_gpu=use_gpu)
    A_norm = build_clique_adjacency(H)
    print(f"  Adjacency shape: {A_norm.shape}")

    constraints = precompute_constraints(dataset, device)

    eval_k_list = [1, 3, 5, 10, 20]
    if args.baseline == 'all':
        targets = ALL_MODELS
    elif args.baseline == 'baselines_only':
        targets = BASELINES
    else:
        targets = [args.baseline]

    all_best = {}

    for bl in targets:
        torch.cuda.empty_cache()
        gc.collect()

        space = dict(SEARCH_SPACE)
        if bl in MODEL_SEARCH_OVERRIDES:
            space.update(MODEL_SEARCH_OVERRIDES[bl])
        hp_keys = sorted(space.keys())
        hp_combos = list(itertools.product(*(space[k] for k in hp_keys)))
        n_combos = len(hp_combos)

        print(f"\n{'=' * 80}")
        print(f"TUNING: {NEURAL_MODELS[bl]}  ({n_combos} configs × {SEARCH_EPOCHS} epochs)")
        print(f"{'=' * 80}")

        results = []
        for i, combo in enumerate(hp_combos):
            hp = dict(zip(hp_keys, combo))
            tag = (f"lr={hp['lr']}, hid={hp['n_hid']}, "
                   f"drop={hp['dropout']}, wd={hp['weight_decay']}")
            t0 = time.time()
            try:
                ndcg5 = quick_train(bl, hp, dataset, G, A_norm, device,
                                    constraints, eval_k_list, 10)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                ndcg5 = 0.0
                print(f"  [{i+1:2d}/{n_combos}] {tag}  →  OOM! skipped")
                continue
            elapsed = time.time() - t0
            results.append((ndcg5, hp))
            print(f"  [{i+1:2d}/{n_combos}] {tag}  →  NDCG@5={ndcg5:.4f}  ({elapsed:.0f}s)")

        if not results:
            fallback_hid = min(space.get('n_hid', [64]))
            print(f"\n  !! ALL configs OOM for {NEURAL_MODELS[bl]}, "
                  f"falling back to n_hid={fallback_hid}")
            best_hp = {'lr': 0.0005, 'n_hid': fallback_hid, 'dropout': 0.05,
                       'weight_decay': 0.0001}
            best_ndcg = 0.0
        else:
            results.sort(key=lambda x: x[0], reverse=True)
            best_ndcg, best_hp = results[0]

        print(f"\n  ** Best config: {best_hp}  NDCG@5={best_ndcg:.4f}")

        all_best[bl] = {
            'best_hp': best_hp,
            'search_ndcg5': best_ndcg,
            'all_results': [(float(n), h) for n, h in results],
        }

        print(f"\n  Retraining {NEURAL_MODELS[bl]} with best config for {args.final_epochs} epochs ...")
        final_config = {
            'k_positive': 10,
            'n_hid': best_hp['n_hid'],
            'dropout': best_hp['dropout'],
            'lr': best_hp['lr'],
            'weight_decay': best_hp['weight_decay'],
            'max_epochs': args.final_epochs,
            'patience': 200,
            'print_freq': 10,
            'warmup_epochs': 30,
            'eval_k_list': eval_k_list,
            'lambda_cap': 0.1,
            'lambda_lat': 0.1,
            'data_root': args.data_root,
            'results_dir': str(results_dir),
        }
        try:
            final_result = train_single_model(bl, final_config, dataset,
                                               H, G, A_norm, device)
            all_best[bl]['final_ndcg5'] = final_result['best_ndcg5']
            all_best[bl]['final_epoch'] = final_result['best_epoch']
        except torch.cuda.OutOfMemoryError:
            print(f"\n  !! OOM during final training of {NEURAL_MODELS[bl]}, "
                  f"trying with n_hid={best_hp['n_hid'] // 2}")
            torch.cuda.empty_cache()
            gc.collect()
            final_config['n_hid'] = best_hp['n_hid'] // 2
            try:
                final_result = train_single_model(bl, final_config, dataset,
                                                   H, G, A_norm, device)
                all_best[bl]['final_ndcg5'] = final_result['best_ndcg5']
                all_best[bl]['final_epoch'] = final_result['best_epoch']
            except torch.cuda.OutOfMemoryError:
                print(f"\n  !! Still OOM for {NEURAL_MODELS[bl]}, skipping final training")
                all_best[bl]['final_ndcg5'] = best_ndcg
                all_best[bl]['final_epoch'] = -1

        torch.cuda.empty_cache()
        gc.collect()

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 80}")
    print("TUNING SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Baseline':<15} {'Best HP':>40} {'Search':>8} {'Final':>8}")
    print("-" * 75)
    for bl, info in all_best.items():
        hp = info['best_hp']
        hp_str = f"lr={hp['lr']}, hid={hp['n_hid']}, drop={hp['dropout']}, wd={hp['weight_decay']}"
        print(f"  {NEURAL_MODELS[bl]:<13} {hp_str:>50} "
              f"{info['search_ndcg5']:>8.4f} {info['final_ndcg5']:>8.4f}")

    tune_path = results_dir / 'tuning_results.json'
    serializable = {}
    for bl, info in all_best.items():
        serializable[bl] = {
            'display_name': NEURAL_MODELS[bl],
            'best_hp': info['best_hp'],
            'search_ndcg5': info['search_ndcg5'],
            'final_ndcg5': info['final_ndcg5'],
            'final_epoch': info['final_epoch'],
            'all_search_results': info['all_results'],
        }
    with open(tune_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to: {tune_path}")


if __name__ == '__main__':
    main()
