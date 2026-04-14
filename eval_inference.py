"""
Inference Deep Analysis
=======================
Post-training analysis of model predictions including:
  1. Constraint satisfaction rates (capacity + latency)
  2. Deployment diversity metrics (load balance, Gini, coverage)
  3. Per-category performance (text_small/medium/large, image, video)
  4. Case studies for selected representative models

Uses score_matrix.npy and inference_topk.json from trained models.

Usage:
  python eval_inference.py --data_root datasetsnew1 --results_dir results_v2/datasetsnew1
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from datasets.topk_placement_loader import TopKPlacementDataset
from utils.metrics import RecommendationMetrics, compute_diversity_metrics

NEURAL_MODELS = {
    'ours': 'ReHGNN (Ours)',
    'gcn': 'GCN',
    'gat': 'GAT',
    'han': 'HAN',
    'hypergcn': 'HyperGCN',
}


def load_model_predictions(results_dir, model_name):
    """Load saved predictions for a neural model."""
    model_dir = Path(results_dir) / model_name

    topk_path = model_dir / 'inference_topk.json'
    score_path = model_dir / 'score_matrix.npy'

    topk_data = None
    score_matrix = None

    if topk_path.exists():
        with open(topk_path) as f:
            topk_data = json.load(f)

    if score_path.exists():
        score_matrix = np.load(score_path)

    return topk_data, score_matrix


def compute_heuristic_scores(dataset, method_name):
    """Generate heuristic prediction scores."""
    nm, ns = dataset.num_models, dataset.num_servers

    if method_name == 'Random':
        return np.random.rand(nm, ns)

    if method_name == 'Popular':
        degree = dataset.topology.sum(axis=0)
        return np.tile(degree, (nm, 1))

    if method_name == 'User-Aware':
        user_model_groups = defaultdict(list)
        for _, row in dataset.user_model_df.iterrows():
            mid = int(row['ModelID']) - 1
            uid = int(row['UserID']) - 1
            user_model_groups[mid].append(uid)

        user_locs = dataset.users_df[['Lo', 'La']].values
        server_locs = dataset.servers_df[['Lo', 'La']].values
        scores = np.zeros((nm, ns))
        for mid in range(nm):
            if mid in user_model_groups:
                uids = user_model_groups[mid]
                avg_loc = user_locs[uids].mean(axis=0)
                dists = np.linalg.norm(server_locs - avg_loc, axis=1)
                scores[mid] = 1.0 - dists / (dists.max() + 1e-10)
            else:
                scores[mid] = 0.5
        return scores

    if method_name == 'Resource-Match':
        model_req = dataset.models_df[['Modelsize', 'Modelresource']].values.astype(float)
        model_req = (model_req - model_req.mean(0)) / (model_req.std(0) + 1e-8)
        server_cap = dataset.servers_df[['ComputationCapacity', 'StorageCapacity']].values.astype(float)
        server_cap = (server_cap - server_cap.mean(0)) / (server_cap.std(0) + 1e-8)
        scores = np.zeros((nm, ns))
        for mid in range(nm):
            dists = np.linalg.norm(server_cap - model_req[mid], axis=1)
            scores[mid] = 1.0 - dists / (dists.max() + 1e-10)
        return scores

    return np.random.rand(nm, ns)


# ============================================================================
# Analysis 1: Constraint Satisfaction
# ============================================================================

def analyze_constraint_satisfaction(dataset, all_scores, method_names, k=5):
    """Compute capacity and latency constraint satisfaction for Top-K."""
    model_res = dataset.models_df['Modelresource'].values.astype(float)
    server_cap = dataset.servers_df['ComputationCapacity'].values.astype(float)

    user_model_groups = defaultdict(list)
    for _, row in dataset.user_model_df.iterrows():
        mid = int(row['ModelID']) - 1
        uid = int(row['UserID']) - 1
        if 0 <= mid < dataset.num_models and 0 <= uid < dataset.num_users:
            user_model_groups[mid].append(uid)

    user_locs = dataset.users_df[['Lo', 'La']].values
    server_locs = dataset.servers_df[['Lo', 'La']].values

    results = {}
    for mname, scores in zip(method_names, all_scores):
        if scores is None:
            continue

        scores_t = torch.from_numpy(scores).float()
        cap_satisfied = 0
        lat_satisfied = 0
        total_placements = 0

        for mid in range(dataset.num_models):
            topk_idx = torch.topk(scores_t[mid], k=k)[1].numpy()

            for sid in topk_idx:
                total_placements += 1
                if server_cap[sid] >= model_res[mid]:
                    cap_satisfied += 1

                uids = user_model_groups.get(mid, [])
                if uids:
                    sampled = uids[:100]
                    u_locs = user_locs[sampled]
                    dists = np.sqrt(((server_locs[sid] - u_locs) ** 2).sum(axis=1))
                    avg_dist = dists.mean()
                    if avg_dist < np.median(
                        np.sqrt(((server_locs - user_locs.mean(0)) ** 2).sum(axis=1))
                    ):
                        lat_satisfied += 1
                else:
                    lat_satisfied += 1

        results[mname] = {
            'capacity_satisfaction': round(cap_satisfied / max(total_placements, 1), 4),
            'latency_satisfaction': round(lat_satisfied / max(total_placements, 1), 4),
            'total_placements': total_placements,
        }

    return results


# ============================================================================
# Analysis 2: Deployment Diversity
# ============================================================================

def analyze_diversity(dataset, all_scores, method_names, k=5):
    """Compute deployment diversity metrics for each method."""
    results = {}
    for mname, scores in zip(method_names, all_scores):
        if scores is None:
            continue

        scores_t = torch.from_numpy(scores).float()
        placement = torch.zeros(dataset.num_models, dataset.num_servers)
        for mid in range(dataset.num_models):
            topk_idx = torch.topk(scores_t[mid], k=k)[1]
            placement[mid, topk_idx] = 1.0

        div = compute_diversity_metrics(placement, dataset.num_servers)
        results[mname] = {k_: round(float(v), 6) for k_, v in div.items()}

    return results


# ============================================================================
# Analysis 3: Per-Category Performance
# ============================================================================

def analyze_per_category(dataset, all_scores, method_names):
    """Evaluate each method separately for each model category."""
    models_df = dataset.models_df
    if 'ModelType' not in models_df.columns:
        model_res = models_df['Modelresource'].values
        categories = []
        for r in model_res:
            if r < 5:
                categories.append('text_small')
            elif r < 15:
                categories.append('text_medium')
            elif r < 40:
                categories.append('text_large')
            elif r < 60:
                categories.append('image')
            else:
                categories.append('video')
        models_df = models_df.copy()
        models_df['ModelType'] = categories

    cat_to_indices = defaultdict(list)
    for idx, cat in enumerate(models_df['ModelType']):
        cat_to_indices[cat].append(idx)

    _, pos_global = dataset.get_evaluation_pairs()
    pos_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in pos_global
    ]

    results = {}
    metrics_calc = RecommendationMetrics(k_list=[5, 10])

    for mname, scores in zip(method_names, all_scores):
        if scores is None:
            continue
        scores_t = torch.from_numpy(scores).float()
        cat_results = {}
        for cat, indices in cat_to_indices.items():
            cat_scores = scores_t[indices]
            cat_pos = [pos_local[i] for i in indices]
            m = metrics_calc.compute_all_metrics(cat_scores, cat_pos)
            cat_results[cat] = {
                'count': len(indices),
                'ndcg@5': round(m.get('ndcg@5', 0.0), 6),
                'precision@5': round(m.get('precision@5', 0.0), 6),
                'ndcg@10': round(m.get('ndcg@10', 0.0), 6),
                'hit_rate@5': round(m.get('hit_rate@5', 0.0), 6),
            }
        results[mname] = cat_results

    return results


# ============================================================================
# Analysis 4: Case Studies
# ============================================================================

def generate_case_studies(dataset, all_scores, method_names, k=5):
    """Select representative models and show Top-K recommendations."""
    models_df = dataset.models_df.copy()
    model_res = models_df['Modelresource'].values

    categories = []
    for r in model_res:
        if r < 5:
            categories.append('text_small')
        elif r < 15:
            categories.append('text_medium')
        elif r < 40:
            categories.append('text_large')
        elif r < 60:
            categories.append('image')
        else:
            categories.append('video')
    models_df['ModelType'] = categories

    cat_to_indices = defaultdict(list)
    for idx, cat in enumerate(categories):
        cat_to_indices[cat].append(idx)

    _, pos_global = dataset.get_evaluation_pairs()
    pos_local = [
        [s - dataset.num_users - dataset.num_models for s in servers]
        for servers in pos_global
    ]

    selected = {}
    for cat, indices in cat_to_indices.items():
        mid_idx = indices[len(indices) // 2]
        selected[cat] = mid_idx

    cases = {}
    for cat, mid in selected.items():
        gt_servers = pos_local[mid]
        model_info = {
            'model_id': mid,
            'category': cat,
            'arena_score': float(models_df.iloc[mid].get('ArenaScore', 0)),
            'model_size': float(models_df.iloc[mid].get('Modelsize', 0)),
            'model_resource': float(models_df.iloc[mid].get('Modelresource', 0)),
            'gt_servers': gt_servers[:k],
        }

        method_recs = {}
        for mname, scores in zip(method_names, all_scores):
            if scores is None:
                continue
            scores_t = torch.from_numpy(scores).float()
            topk_idx = torch.topk(scores_t[mid], k=k)[1].numpy().tolist()
            hits = len(set(topk_idx) & set(gt_servers))
            method_recs[mname] = {
                'topk_servers': topk_idx,
                'hits': hits,
                'precision': round(hits / k, 4),
            }

        model_info['method_recommendations'] = method_recs
        cases[cat] = model_info

    return cases


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='datasetsnew1')
    parser.add_argument('--results_dir', type=str, default='results_v2/datasetsnew1')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_root}...")
    dataset = TopKPlacementDataset(
        split='train', k_positive=10, data_root=args.data_root,
    )
    dataset.prepare()

    out_dir = Path(args.results_dir) / 'inference_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    heuristic_methods = ['Random', 'Popular', 'User-Aware', 'Resource-Match']
    heuristic_scores = [compute_heuristic_scores(dataset, m) for m in heuristic_methods]

    neural_methods = []
    neural_scores = []
    for model_name, display in NEURAL_MODELS.items():
        topk_data, score_matrix = load_model_predictions(args.results_dir, model_name)
        if score_matrix is not None:
            neural_methods.append(display)
            neural_scores.append(score_matrix)
            print(f"  Loaded {display}: score_matrix {score_matrix.shape}")
        elif topk_data is not None:
            nm = dataset.num_models
            ns = dataset.num_servers
            score_matrix = np.zeros((nm, ns))
            topk_indices = topk_data['topk_server_indices']
            topk_vals = topk_data['topk_scores']
            for mid in range(nm):
                for rank, (sid, val) in enumerate(zip(topk_indices[mid], topk_vals[mid])):
                    score_matrix[mid, sid] = val
            neural_methods.append(display)
            neural_scores.append(score_matrix)
            print(f"  Loaded {display}: reconstructed from topk ({len(topk_indices)} models)")
        else:
            print(f"  Skipped {display}: no predictions found")

    all_methods = heuristic_methods + neural_methods
    all_scores = heuristic_scores + neural_scores

    # --- Analysis 1: Constraint Satisfaction ---
    print(f"\n{'='*80}")
    print("ANALYSIS 1: CONSTRAINT SATISFACTION")
    print(f"{'='*80}")
    constraint_results = analyze_constraint_satisfaction(
        dataset, all_scores, all_methods, k=args.k
    )
    print(f"\n{'Method':<20} {'Cap. Sat.':>10} {'Lat. Sat.':>10}")
    print("-" * 42)
    for method, vals in constraint_results.items():
        print(f"{method:<20} {vals['capacity_satisfaction']:>10.4f} "
              f"{vals['latency_satisfaction']:>10.4f}")

    with open(out_dir / 'constraint_satisfaction.json', 'w') as f:
        json.dump(constraint_results, f, indent=2)

    # --- Analysis 2: Diversity ---
    print(f"\n{'='*80}")
    print("ANALYSIS 2: DEPLOYMENT DIVERSITY")
    print(f"{'='*80}")
    diversity_results = analyze_diversity(
        dataset, all_scores, all_methods, k=args.k
    )
    print(f"\n{'Method':<20} {'LoadBal':>8} {'Coverage':>8} {'Gini':>8}")
    print("-" * 48)
    for method, vals in diversity_results.items():
        print(f"{method:<20} {vals.get('load_balance', 0):>8.4f} "
              f"{vals.get('server_coverage', 0):>8.4f} "
              f"{vals.get('gini_coefficient', 0):>8.4f}")

    with open(out_dir / 'diversity_metrics.json', 'w') as f:
        json.dump(diversity_results, f, indent=2)

    # --- Analysis 3: Per-Category ---
    print(f"\n{'='*80}")
    print("ANALYSIS 3: PER-CATEGORY PERFORMANCE")
    print(f"{'='*80}")
    category_results = analyze_per_category(
        dataset, all_scores, all_methods
    )
    for cat in sorted(next(iter(category_results.values())).keys()):
        print(f"\n  Category: {cat}")
        print(f"  {'Method':<20} {'NDCG@5':>8} {'P@5':>8} {'Hit@5':>8}")
        print(f"  {'-'*48}")
        for method in all_methods:
            if method in category_results:
                vals = category_results[method].get(cat, {})
                print(f"  {method:<20} {vals.get('ndcg@5', 0):>8.4f} "
                      f"{vals.get('precision@5', 0):>8.4f} "
                      f"{vals.get('hit_rate@5', 0):>8.4f}")

    with open(out_dir / 'per_category_metrics.json', 'w') as f:
        json.dump(category_results, f, indent=2)

    # --- Analysis 4: Case Studies ---
    print(f"\n{'='*80}")
    print("ANALYSIS 4: CASE STUDIES")
    print(f"{'='*80}")
    case_results = generate_case_studies(
        dataset, all_scores, all_methods, k=args.k
    )
    for cat, case in case_results.items():
        print(f"\n  [{cat}] Model ID={case['model_id']}, "
              f"Resource={case['model_resource']:.1f}, "
              f"ArenaScore={case['arena_score']:.1f}")
        print(f"  GT servers: {case['gt_servers']}")
        for method, rec in case['method_recommendations'].items():
            tag = "*" if rec['hits'] > 0 else " "
            print(f"    {tag} {method:<20} Top-5: {rec['topk_servers']}  "
                  f"Hits={rec['hits']}")

    with open(out_dir / 'case_studies.json', 'w') as f:
        json.dump(case_results, f, indent=2, default=int)

    # --- Save combined summary ---
    summary = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'k': args.k,
        'methods': all_methods,
        'constraint_satisfaction': constraint_results,
        'diversity': diversity_results,
        'per_category': category_results,
        'case_studies': case_results,
    }
    with open(out_dir / 'inference_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=int)

    print(f"\n\nAll results saved to: {out_dir}/")


if __name__ == '__main__':
    main()
