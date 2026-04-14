"""
Neural Network Baselines for Model Placement Task

Implements graph-based and hypergraph-based baselines to provide
comprehensive comparisons:
  - GCN_Placement:  Standard Graph Convolutional Network on clique-expanded graph
  - GAT_Placement:  Graph Attention Network on clique-expanded graph
  - HAN_Placement:  Heterogeneous Attention Network with type-specific transforms
  - HyperGCN_Placement: Alternative hypergraph convolution (HyperGCN-style)

All models output [num_models, num_servers] compatibility scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Utility: Build clique-expanded adjacency from hypergraph incidence matrix
# ============================================================================

def build_clique_adjacency(H, self_loop=True):
    """
    Convert hypergraph incidence matrix H to clique-expanded adjacency.

    A = H @ H^T gives co-occurrence counts.
    Returns symmetrically normalized: D^{-1/2} A D^{-1/2}
    """
    if isinstance(H, np.ndarray):
        A = H @ H.T
        np.fill_diagonal(A, 0)
        if self_loop:
            A = A + np.eye(A.shape[0])
        D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-10))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return A_norm.astype(np.float32)
    else:
        raise ValueError("H must be a numpy array")


# ============================================================================
# GCN Baseline
# ============================================================================

class GCNConv(nn.Module):
    """Standard Graph Convolution layer: X' = sigma(A_norm @ X @ W)"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_ch, out_ch))
        self.bias = nn.Parameter(torch.FloatTensor(out_ch))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support) + self.bias
        return output


class GCN_Placement(nn.Module):
    """GCN for model placement on clique-expanded graph."""

    def __init__(self, in_ch, n_hid, num_users, num_models, num_servers, dropout=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.num_servers = num_servers

        self.gc1 = GCNConv(in_ch, n_hid)
        self.gc2 = GCNConv(n_hid, n_hid)
        self.gc3 = GCNConv(n_hid, n_hid)
        self.dropout = dropout
        self.placement_predictor = nn.Bilinear(n_hid, n_hid, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)

        m_start = self.num_users
        m_end = m_start + self.num_models
        s_start = m_end
        s_end = s_start + self.num_servers

        model_emb = x[m_start:m_end]
        server_emb = x[s_start:s_end]

        nm, ns = model_emb.size(0), server_emb.size(0)
        m_exp = model_emb.unsqueeze(1).expand(-1, ns, -1).reshape(-1, model_emb.size(1))
        s_exp = server_emb.unsqueeze(0).expand(nm, -1, -1).reshape(-1, server_emb.size(1))
        scores = self.placement_predictor(m_exp, s_exp).reshape(nm, ns)
        return scores


# ============================================================================
# GAT Baseline
# ============================================================================

class GATConv(nn.Module):
    """
    Sparse Graph Attention layer (single-head).

    Computes attention only for existing edges (COO format) instead of
    materialising a dense N×N matrix, reducing memory from O(N²) to O(E).
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.W = nn.Linear(in_ch, out_ch, bias=False)
        self.a_l = nn.Linear(out_ch, 1, bias=False)
        self.a_r = nn.Linear(out_ch, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x, edge_index):
        """
        Args:
            x: (N, in_ch) node features
            edge_index: (2, E) int64 COO edge list  [src → dst]
        """
        N = x.size(0)
        h = self.W(x)                               # (N, out_ch)
        src, dst = edge_index                        # each (E,)

        e_l = self.a_l(h).squeeze(-1)               # (N,)  receiver
        e_r = self.a_r(h).squeeze(-1)               # (N,)  sender

        edge_attn = F.leaky_relu(e_l[dst] + e_r[src], 0.2)

        # ---- sparse softmax (numerically stable) ----
        edge_max = torch.full((N,), float('-inf'), device=x.device)
        edge_max.scatter_reduce_(0, dst, edge_attn,
                                 reduce='amax', include_self=True)
        edge_attn = (edge_attn - edge_max[dst]).exp()

        edge_sum = torch.zeros(N, device=x.device)
        edge_sum.scatter_add_(0, dst, edge_attn)
        edge_attn = edge_attn / (edge_sum[dst] + 1e-10)

        # ---- weighted aggregation ----
        messages = h[src] * edge_attn.unsqueeze(-1)  # (E, out_ch)
        out = torch.zeros(N, h.size(1), device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        return out + self.bias


class GAT_Placement(nn.Module):
    """GAT for model placement on clique-expanded graph (sparse attention)."""

    def __init__(self, in_ch, n_hid, num_users, num_models, num_servers,
                 dropout=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.num_servers = num_servers

        self.gat1 = GATConv(in_ch, n_hid)
        self.gat2 = GATConv(n_hid, n_hid)
        self.gat3 = GATConv(n_hid, n_hid)
        self.dropout = dropout
        self.placement_predictor = nn.Bilinear(n_hid, n_hid, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat3(x, edge_index)

        m_start = self.num_users
        m_end = m_start + self.num_models
        s_start = m_end
        s_end = s_start + self.num_servers

        model_emb = x[m_start:m_end]
        server_emb = x[s_start:s_end]

        nm, ns = model_emb.size(0), server_emb.size(0)
        m_exp = model_emb.unsqueeze(1).expand(-1, ns, -1).reshape(-1, model_emb.size(1))
        s_exp = server_emb.unsqueeze(0).expand(nm, -1, -1).reshape(-1, server_emb.size(1))
        scores = self.placement_predictor(m_exp, s_exp).reshape(nm, ns)
        return scores


# ============================================================================
# HAN Baseline (Heterogeneous Attention Network)
# ============================================================================

class HANLayer(nn.Module):
    """
    Type-specific linear transforms + semantic attention aggregation.
    Each node type gets its own linear projection before aggregation.
    """

    def __init__(self, in_ch, out_ch, num_users, num_models, num_servers):
        super().__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.num_servers = num_servers

        self.W_user = nn.Linear(in_ch, out_ch)
        self.W_model = nn.Linear(in_ch, out_ch)
        self.W_server = nn.Linear(in_ch, out_ch)
        self.attn = nn.Linear(out_ch, 1)

    def forward(self, x, adj):
        N = x.size(0)
        nu, nm, ns = self.num_users, self.num_models, self.num_servers

        h = torch.zeros(N, self.W_user.out_features, device=x.device)
        h[:nu] = self.W_user(x[:nu])
        h[nu:nu + nm] = self.W_model(x[nu:nu + nm])
        h[nu + nm:] = self.W_server(x[nu + nm:])

        support = torch.mm(adj, h)

        attn_weights = torch.sigmoid(self.attn(h))
        out = support * attn_weights
        return out


class HAN_Placement(nn.Module):
    """Heterogeneous Attention Network for model placement."""

    def __init__(self, in_ch, n_hid, num_users, num_models, num_servers, dropout=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.num_servers = num_servers

        self.han1 = HANLayer(in_ch, n_hid, num_users, num_models, num_servers)
        self.han2 = HANLayer(n_hid, n_hid, num_users, num_models, num_servers)
        self.han3 = HANLayer(n_hid, n_hid, num_users, num_models, num_servers)
        self.dropout = dropout
        self.placement_predictor = nn.Bilinear(n_hid, n_hid, 1)

    def forward(self, x, adj):
        x = F.relu(self.han1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.han2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.han3(x, adj)

        m_start = self.num_users
        m_end = m_start + self.num_models
        s_start = m_end
        s_end = s_start + self.num_servers

        model_emb = x[m_start:m_end]
        server_emb = x[s_start:s_end]

        nm, ns = model_emb.size(0), server_emb.size(0)
        m_exp = model_emb.unsqueeze(1).expand(-1, ns, -1).reshape(-1, model_emb.size(1))
        s_exp = server_emb.unsqueeze(0).expand(nm, -1, -1).reshape(-1, server_emb.size(1))
        scores = self.placement_predictor(m_exp, s_exp).reshape(nm, ns)
        return scores


# ============================================================================
# HyperGCN Baseline (alternative hypergraph convolution)
# ============================================================================

class HyperGCNConv(nn.Module):
    """
    HyperGCN-style convolution that reduces each hyperedge to a pairwise
    edge between the two most dissimilar nodes (mediator trick), then
    applies standard graph convolution.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_ch, out_ch))
        self.bias = nn.Parameter(torch.FloatTensor(out_ch))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, G):
        support = torch.mm(x, self.weight)
        output = torch.mm(G, support) + self.bias
        return output


class HyperGCN_Placement(nn.Module):
    """HyperGCN for model placement (uses hypergraph Laplacian G)."""

    def __init__(self, in_ch, n_hid, num_users, num_models, num_servers, dropout=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.num_servers = num_servers

        self.hgc1 = HyperGCNConv(in_ch, n_hid)
        self.hgc2 = HyperGCNConv(n_hid, n_hid)
        self.dropout = dropout

        self.placement_predictor = nn.Bilinear(n_hid, n_hid, 1)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, G)

        m_start = self.num_users
        m_end = m_start + self.num_models
        s_start = m_end
        s_end = s_start + self.num_servers

        model_emb = x[m_start:m_end]
        server_emb = x[s_start:s_end]

        nm, ns = model_emb.size(0), server_emb.size(0)
        m_exp = model_emb.unsqueeze(1).expand(-1, ns, -1).reshape(-1, model_emb.size(1))
        s_exp = server_emb.unsqueeze(0).expand(nm, -1, -1).reshape(-1, server_emb.size(1))
        scores = self.placement_predictor(m_exp, s_exp).reshape(nm, ns)
        return scores
