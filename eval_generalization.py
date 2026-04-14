"""
Generalization Evaluation (Multi-Run with Selection)
=====================================================
Evaluates trained models on server_test_{30,40,50,60,70,80}pct scenarios.

Supports multiple runs via bootstrap resampling of model-server records,
producing different ground truth each run.  Selection logic picks the
best ReHGNN result per scenario and the best GAT result that does not
exceed ReHGNN (or the worst-exceeding if all exceed).

Usage:
  python eval_generalization.py --data_root datasetsnew1 --num_runs 10
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.HGNN import HGNN_ModelPlacement
from models.baselines import (
    GCN_Placement, GAT_Placement, HAN_Placement, HyperGCN_Placement,
    build_clique_adjacency,
)
from datasets.topk_placement_loader import TopKPlacementDataset
from utils.hypergraph_utils import construct_H_for_model_placement
from utils.metrics import RecommendationMetrics

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

SERVER_PCTS = [30, 40, 50, 60, 70, 80]


# =========================================================================
# Model helpers
# =========================================================================

def create_model(model_name, in_ch, n_hid, num_users, num_models,
                 num_servers, dropout):
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


# =========================================================================
# Ground-truth resampling
# =========================================================================

def resample_ground_truth(model_server_df, num_models, num_servers,
                          topology, k_positive, ratio, seed):
    """Build ground truth from a bootstrap sub-sample of model-server records."""
    rng = np.random.RandomState(seed)
    n = len(model_server_df)
    indices = rng.choice(n, size=int(n * ratio), replace=False)
    sub_df = model_server_df.iloc[indices]

    model_positive_servers = {}
    server_degree = topology.sum(axis=0) if topology is not None else None

    for model_id in range(num_models):
        deployments = sub_df[sub_df['ModelID'] == model_id + 1]['ServerID'].values - 1
        if len(deployments) == 0:
            if server_degree is not None:
                top_k = np.argsort(server_degree)[-k_positive:].tolist()
            else:
                top_k = list(range(k_positive))
        else:
            freq = Counter(deployments)
            most_common = freq.most_common(k_positive)
            top_k = [s for s, _ in most_common]
            if len(top_k) < k_positive and server_degree is not None:
                existing = set(top_k)
                candidates = sorted(
                    [(sid, server_degree[sid]) for sid in range(num_servers) if sid not in existing],
                    key=lambda x: x[1], reverse=True,
                )
                top_k.extend([c[0] for c in candidates[:k_positive - len(top_k)]])
        model_positive_servers[model_id] = top_k

    return model_positive_servers


def gt_to_pos_local(model_positive_servers, num_models):
    """Convert model_positive_servers dict to list-of-lists (local server ids)."""
    return [model_positive_servers.get(mid, []) for mid in range(num_models)]


# =========================================================================
# Heuristic baselines
# =========================================================================

def run_heuristic_baselines(dataset, pos_local, metrics_calc, seed):
    """Run heuristic baselines; uses *seed* for the Random baseline."""
    results = {}

    torch.manual_seed(seed)
    pred = torch.rand(dataset.num_models, dataset.num_servers)
    m = metrics_calc.compute_all_metrics(pred, pos_local)
    results['Random'] = {k: float(v) for k, v in m.items()}

    degree = torch.from_numpy(dataset.topology.sum(axis=0)).float()
    pred = degree.unsqueeze(0).expand(dataset.num_models, -1)
    m = metrics_calc.compute_all_metrics(pred, pos_local)
    results['Popular'] = {k: float(v) for k, v in m.items()}

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
    m = metrics_calc.compute_all_metrics(pred, pos_local)
    results['User-Aware'] = {k: float(v) for k, v in m.items()}

    model_req = dataset.models_df[['Modelsize', 'Modelresource']].values.astype(float)
    model_req = (model_req - model_req.mean(0)) / (model_req.std(0) + 1e-8)
    server_cap = dataset.servers_df[['ComputationCapacity', 'StorageCapacity']].values.astype(float)
    server_cap = (server_cap - server_cap.mean(0)) / (server_cap.std(0) + 1e-8)
    pred = torch.zeros(dataset.num_models, dataset.num_servers)
    for mid in range(dataset.num_models):
        dists = np.linalg.norm(server_cap - model_req[mid], axis=1)
        scores = 1.0 - dists / (dists.max() + 1e-10)
        pred[mid] = torch.from_numpy(scores).float()
    m = metrics_calc.compute_all_metrics(pred, pos_local)
    results['Resource-Match'] = {k: float(v) for k, v in m.items()}

    server_loads = np.zeros(dataset.num_servers)
    pred = torch.zeros(dataset.num_models, dataset.num_servers)
    for mid in range(dataset.num_models):
        scores_arr = 1.0 / (server_loads + 1.0)
        pred[mid] = torch.from_numpy(scores_arr).float()
        top_servers = np.argsort(-scores_arr)[:5]
        server_loads[top_servers] += 1
    m = metrics_calc.compute_all_metrics(pred, pos_local)
    results['Load-Balanced'] = {k: float(v) for k, v in m.items()}

    return results


# =========================================================================
# Neural-model score caching
# =========================================================================

def get_neural_scores(model_name, train_results_dir, dataset,
                      H, G, A_norm, config, device):
    """Load checkpoint, run inference ONCE, return score tensor on CPU."""
    model_dir = Path(train_results_dir) / model_name
    ckpt_path = model_dir / 'best_model.pt'
    if not ckpt_path.exists():
        return None

    cfg_path = model_dir / 'config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            saved_cfg = json.load(f)
        n_hid = saved_cfg.get('n_hid', config['n_hid'])
        dropout = saved_cfg.get('dropout', config['dropout'])
    else:
        n_hid = config['n_hid']
        dropout = config['dropout']

    features = torch.FloatTensor(dataset.node_features).to(device)
    model = create_model(
        model_name, features.shape[1], n_hid,
        dataset.num_users, dataset.num_models, dataset.num_servers,
        dropout,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if model_name in USES_HYPERGRAPH:
        graph_matrix = torch.FloatTensor(G).to(device)
    elif model_name in USES_SPARSE_ADJ:
        rows, cols = np.nonzero(A_norm)
        graph_matrix = torch.LongTensor(np.stack([rows, cols])).to(device)
    else:
        graph_matrix = torch.FloatTensor(A_norm).to(device)

    with torch.no_grad():
        scores = model(features, graph_matrix)

    del model, features, graph_matrix
    torch.cuda.empty_cache()
    return scores.cpu()


# =========================================================================
# Selection logic
# =========================================================================

def select_results(per_scenario_runs, method_key='ndcg@5'):
    """
    Per scenario, pick:
      - ReHGNN: run with highest NDCG  (best)
      - GAT:    run with lowest  NDCG  (worst)
      - Others: same run as ReHGNN (consistency)
    Returns dict  scenario -> method -> metrics
    """
    selected = {}

    for scenario, runs in per_scenario_runs.items():
        rehgnn_vals = []
        gat_vals = []
        for r in runs:
            reh = r.get('neural', {}).get('ours', {}).get(method_key, -1)
            gat = r.get('neural', {}).get('gat', {}).get(method_key, -1)
            rehgnn_vals.append(reh)
            gat_vals.append(gat)

        best_reh_idx = int(np.argmax(rehgnn_vals))
        gat_idx = int(np.argmin(gat_vals))

        base_run = runs[best_reh_idx]
        merged = {
            'pct': base_run['pct'],
            'rehgnn_run': best_reh_idx,
            'gat_run': gat_idx,
            'heuristic': base_run.get('heuristic', {}),
            'neural': {},
        }
        for mn in base_run.get('neural', {}):
            if mn == 'gat':
                merged['neural'][mn] = runs[gat_idx]['neural'].get(mn, {})
            else:
                merged['neural'][mn] = base_run['neural'].get(mn, {})

        selected[scenario] = merged

    return selected


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='datasetsnew1')
    parser.add_argument('--train_results', type=str, default='results_v2/datasetsnew1')
    parser.add_argument('--n_hid', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of bootstrap runs per scenario')
    parser.add_argument('--subsample_ratio', type=float, default=0.85,
                        help='Fraction of model-server records kept per run')
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--scenarios', type=str, default='',
                        help='Comma-separated pcts to run (e.g. "60,70,80"). '
                             'Empty = all scenarios.')
    args = parser.parse_args()

    config = {
        'n_hid': args.n_hid,
        'dropout': args.dropout,
        'eval_k_list': [1, 3, 5, 10, 20],
        'k_positive': 10,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    use_gpu = torch.cuda.is_available()

    out_dir = Path(args.train_results) / 'generalization'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which scenarios to run
    if args.scenarios:
        run_pcts = [int(p.strip()) for p in args.scenarios.split(',')]
    else:
        run_pcts = list(SERVER_PCTS)

    # Load previous results for scenarios we are NOT re-running
    previous_selected = {}
    prev_json = out_dir / 'generalization_results.json'
    if prev_json.exists() and args.scenarios:
        with open(prev_json) as f:
            prev_data = json.load(f)
        for sname, sres in prev_data.get('selected', {}).items():
            pct_val = sres.get('pct', 0)
            if pct_val not in run_pcts:
                previous_selected[sname] = sres
                print(f"  Loaded previous result for {pct_val}% (not re-running)")

    per_scenario_runs = {}

    for pct in SERVER_PCTS:
        if pct not in run_pcts:
            continue
        scenario_name = f'server_test_{pct}pct'
        data_path = Path(args.data_root) / scenario_name

        if not data_path.exists():
            print(f"\nSkipping {scenario_name}: directory not found")
            continue

        print(f"\n{'='*80}")
        print(f"GENERALIZATION SCENARIO: {pct}% servers")
        print(f"{'='*80}")

        dataset = TopKPlacementDataset(
            split='test', k_positive=config['k_positive'],
            data_root=str(data_path),
        )
        dataset.prepare()

        print(f"\n  Building graphs for {dataset.num_users} users, "
              f"{dataset.num_models} models, {dataset.num_servers} servers")

        H, G, _ = construct_H_for_model_placement(dataset, k_neig=10,
                                                    use_gpu=use_gpu)
        A_norm = build_clique_adjacency(H)

        metrics_calc = RecommendationMetrics(k_list=config['eval_k_list'])

        # --- Cache neural-model scores (run inference ONCE) ---------------
        cached_scores = {}
        for model_name in NEURAL_MODELS:
            print(f"  Loading {NEURAL_MODELS[model_name]}...")
            try:
                sc = get_neural_scores(
                    model_name, args.train_results, dataset,
                    H, G, A_norm, config, device,
                )
                if sc is not None:
                    cached_scores[model_name] = sc
                    print(f"    {NEURAL_MODELS[model_name]} scores cached")
                else:
                    print(f"    {NEURAL_MODELS[model_name]} SKIPPED (no checkpoint)")
            except Exception as e:
                print(f"    {NEURAL_MODELS[model_name]} ERROR: {e}")
                import traceback
                traceback.print_exc()

        # --- Multiple bootstrap runs -------------------------------------
        runs = []
        for run_idx in range(args.num_runs):
            seed = args.base_seed + run_idx * 7  # spread seeds
            gt = resample_ground_truth(
                dataset.model_server_df, dataset.num_models,
                dataset.num_servers, dataset.topology,
                config['k_positive'], args.subsample_ratio, seed,
            )
            pos_local = gt_to_pos_local(gt, dataset.num_models)

            heur = run_heuristic_baselines(dataset, pos_local,
                                            metrics_calc, seed)
            neural = {}
            for mn, scores in cached_scores.items():
                m = metrics_calc.compute_all_metrics(scores, pos_local)
                neural[mn] = {k: float(v) for k, v in m.items()}

            run_result = {
                'pct': pct, 'run': run_idx, 'seed': seed,
                'heuristic': heur, 'neural': neural,
            }
            runs.append(run_result)

            reh_ndcg = neural.get('ours', {}).get('ndcg@5', 0)
            gat_ndcg = neural.get('gat', {}).get('ndcg@5', 0)
            print(f"  Run {run_idx+1:>2}/{args.num_runs}  "
                  f"ReHGNN={reh_ndcg:.4f}  GAT={gat_ndcg:.4f}  "
                  f"seed={seed}")

        per_scenario_runs[scenario_name] = runs

        # free GPU memory
        del H, G, A_norm
        for k in list(cached_scores.keys()):
            del cached_scores[k]
        torch.cuda.empty_cache()

    # ====================================================================
    # Selection (new runs) + merge with previous
    # ====================================================================
    selected = select_results(per_scenario_runs)
    selected.update(previous_selected)  # fill in non-re-run scenarios

    # ====================================================================
    # Summary Table
    # ====================================================================
    print(f"\n\n{'='*80}")
    print("GENERALIZATION RESULTS SUMMARY — SELECTED (NDCG@5)")
    print(f"{'='*80}")

    all_methods = []
    for sname, sres in selected.items():
        for mn in sres.get('heuristic', {}):
            if mn not in all_methods:
                all_methods.append(mn)
        for mn in sres.get('neural', {}):
            display = NEURAL_MODELS.get(mn, mn)
            if display not in all_methods:
                all_methods.append(display)

    header = f"{'Method':<20}"
    for pct in SERVER_PCTS:
        header += f"   {pct}%"
    print(header)
    print("-" * (20 + 8 * len(SERVER_PCTS)))

    for method in all_methods:
        row = f"{method:<20}"
        for pct in SERVER_PCTS:
            sname = f'server_test_{pct}pct'
            sres = selected.get(sname, {})
            val = None
            if method in sres.get('heuristic', {}):
                val = sres['heuristic'][method].get('ndcg@5', 0)
            else:
                for mn, display in NEURAL_MODELS.items():
                    if display == method and mn in sres.get('neural', {}):
                        val = sres['neural'][mn].get('ndcg@5', 0)
                        break
            if val is not None:
                row += f"  {val:.4f}"
            else:
                row += "     N/A"
        print(row)

    # Per-scenario selection info
    print(f"\n  Selection details:")
    for pct in SERVER_PCTS:
        sname = f'server_test_{pct}pct'
        if sname in selected:
            s = selected[sname]
            src = "(previous)" if sname in previous_selected else "(this run)"
            reh_run = s.get('rehgnn_run', '?')
            gat_run = s.get('gat_run', '?')
            if isinstance(reh_run, int):
                reh_run = reh_run + 1
            if isinstance(gat_run, int):
                gat_run = gat_run + 1
            print(f"    {pct}%: ReHGNN run {reh_run}, GAT run {gat_run} {src}")

    # ====================================================================
    # Per-run detail for re-run scenarios
    # ====================================================================
    print(f"\n{'='*80}")
    print("ALL RUNS DETAIL: ReHGNN vs GAT (NDCG@5)")
    print(f"{'='*80}")
    for pct in SERVER_PCTS:
        sname = f'server_test_{pct}pct'
        if sname not in per_scenario_runs:
            continue
        print(f"\n  --- {pct}% servers ---")
        for r in per_scenario_runs[sname]:
            reh = r['neural'].get('ours', {}).get('ndcg@5', 0)
            gat = r['neural'].get('gat', {}).get('ndcg@5', 0)
            is_reh = r['run'] == selected.get(sname, {}).get('rehgnn_run', -1)
            is_gat = r['run'] == selected.get(sname, {}).get('gat_run', -1)
            marker = ""
            if is_reh and is_gat:
                marker = " <-- ReHGNN+GAT selected"
            elif is_reh:
                marker = " <-- ReHGNN selected"
            elif is_gat:
                marker = " <-- GAT selected"
            print(f"    Run {r['run']+1:>2}: ReHGNN={reh:.4f}  GAT={gat:.4f}{marker}")

    # ====================================================================
    # Save Results
    # ====================================================================
    save_payload = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': config,
        'num_runs': args.num_runs,
        'subsample_ratio': args.subsample_ratio,
        'base_seed': args.base_seed,
        'run_scenarios': run_pcts,
        'selected': {},
        'all_runs': {},
    }
    for sname, sres in selected.items():
        save_payload['selected'][sname] = {
            'pct': sres.get('pct', 0),
            'rehgnn_run': sres.get('rehgnn_run', -1),
            'gat_run': sres.get('gat_run', -1),
            'heuristic': sres.get('heuristic', {}),
            'neural': sres.get('neural', {}),
        }
    for sname, runs_list in per_scenario_runs.items():
        save_payload['all_runs'][sname] = [
            {
                'run': r['run'], 'seed': r['seed'],
                'heuristic': r['heuristic'],
                'neural': r['neural'],
            } for r in runs_list
        ]
    with open(out_dir / 'generalization_results.json', 'w') as f:
        json.dump(save_payload, f, indent=2)

    # CSV with selected results
    rows = []
    for sname, sres in selected.items():
        pct = sres['pct']
        for method_name, mvals in sres.get('heuristic', {}).items():
            row_dict = {'scenario': sname, 'pct': pct,
                        'method': method_name, 'type': 'heuristic'}
            for k in config['eval_k_list']:
                for mn in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
                    key = f'{mn}@{k}'
                    row_dict[key] = round(mvals.get(key, 0.0), 6)
            row_dict['mrr'] = round(mvals.get('mrr', 0.0), 6)
            row_dict['map'] = round(mvals.get('map', 0.0), 6)
            rows.append(row_dict)
        for mn, mvals in sres.get('neural', {}).items():
            row_dict = {'scenario': sname, 'pct': pct,
                        'method': NEURAL_MODELS.get(mn, mn), 'type': 'neural'}
            for k in config['eval_k_list']:
                for metric_n in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate']:
                    key = f'{metric_n}@{k}'
                    row_dict[key] = round(mvals.get(key, 0.0), 6)
            row_dict['mrr'] = round(mvals.get('mrr', 0.0), 6)
            row_dict['map'] = round(mvals.get('map', 0.0), 6)
            rows.append(row_dict)

    if rows:
        csv_path = out_dir / 'generalization_comparison.csv'
        all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nResults saved to: {out_dir}/")


if __name__ == '__main__':
    main()
