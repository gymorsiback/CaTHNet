import torch
from torch import nn
from models import HGNN_conv
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class TypeAwareNorm(nn.Module):
    """Separate LayerNorm per node type to avoid cross-type mean/var
    contamination that global LayerNorm would cause on heterogeneous graphs."""

    def __init__(self, n_hid, num_users, num_models, num_servers):
        super().__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.ln_user = nn.LayerNorm(n_hid)
        self.ln_model = nn.LayerNorm(n_hid)
        self.ln_server = nn.LayerNorm(n_hid)

    def forward(self, x):
        nu, nm = self.num_users, self.num_models
        out = torch.empty_like(x)
        out[:nu] = self.ln_user(x[:nu])
        out[nu:nu + nm] = self.ln_model(x[nu:nu + nm])
        out[nu + nm:] = self.ln_server(x[nu + nm:])
        return out


class HGNN_ModelPlacement(nn.Module):
    """
    Heterogeneous Hypergraph Neural Network for model placement.

    Architectural advantages over vanilla spectral HGNN / HyperGCN:
      1. Type-specific input projections  — each entity type (user / model /
         server) is mapped to the shared hidden space through its own linear
         layer, preserving heterogeneous semantics.
      2. Residual connections — prevent the notorious over-smoothing problem
         in spectral graph convolutions.
      3. Type-aware gating — a lightweight element-wise gate after each
         convolution layer selectively retains type-specific information
         that would otherwise be washed out by neighbourhood averaging.

    Design note: normalization is intentionally omitted.  With only two
    convolution layers the gradient landscape is already stable, and we
    empirically found that both global LayerNorm and type-specific
    normalization reduce the discriminative variance of heterogeneous
    node embeddings, hurting ranking performance.

    ablation_config keys:
      use_type_projection  – True (default) / False → shared linear
      use_residual         – True (default) / False → no skip connections
      use_gating           – True (default) / False → no type-aware gating
      use_layer_norm       – False (default) / True → add TypeAwareNorm
    """

    DEFAULT_ABLATION = {
        'use_type_projection': True,
        'use_residual': True,
        'use_gating': True,
        'use_layer_norm': False,
    }

    def __init__(self, in_ch, n_hid, num_users, num_models, num_servers,
                 dropout=0.5, ablation_config=None):
        super(HGNN_ModelPlacement, self).__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.num_servers = num_servers
        self.dropout = dropout

        abl = {**self.DEFAULT_ABLATION, **(ablation_config or {})}
        self.abl = abl

        # --- Type-specific input projections ---
        if abl['use_type_projection']:
            self.user_proj = nn.Linear(in_ch, n_hid)
            self.model_proj = nn.Linear(in_ch, n_hid)
            self.server_proj = nn.Linear(in_ch, n_hid)
        else:
            self.shared_proj = nn.Linear(in_ch, n_hid)

        # --- Hypergraph convolution layers (2 layers with residual) ---
        self.hgc1 = HGNN_conv(n_hid, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

        if abl['use_layer_norm']:
            self.ln1 = TypeAwareNorm(n_hid, num_users, num_models, num_servers)
            self.ln2 = TypeAwareNorm(n_hid, num_users, num_models, num_servers)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        # --- Type-aware gating (applied between conv layers) ---
        if abl['use_gating']:
            self.gate_user = nn.Linear(n_hid, n_hid)
            self.gate_model = nn.Linear(n_hid, n_hid)
            self.gate_server = nn.Linear(n_hid, n_hid)

        # --- Placement prediction head ---
        self.placement_predictor = nn.Bilinear(n_hid, n_hid, 1)

    # -----------------------------------------------------------------
    def _type_project(self, x):
        if not self.abl['use_type_projection']:
            return self.shared_proj(x)
        nu, nm = self.num_users, self.num_models
        h = torch.empty(x.size(0), self.user_proj.out_features,
                        device=x.device, dtype=x.dtype)
        h[:nu] = self.user_proj(x[:nu])
        h[nu:nu + nm] = self.model_proj(x[nu:nu + nm])
        h[nu + nm:] = self.server_proj(x[nu + nm:])
        return h

    def _type_gate(self, x):
        if not self.abl['use_gating']:
            return x
        nu, nm = self.num_users, self.num_models
        g = torch.empty_like(x)
        g[:nu] = torch.sigmoid(self.gate_user(x[:nu]))
        g[nu:nu + nm] = torch.sigmoid(self.gate_model(x[nu:nu + nm]))
        g[nu + nm:] = torch.sigmoid(self.gate_server(x[nu + nm:]))
        return x * g

    # -----------------------------------------------------------------
    def forward(self, x, G):
        x = F.relu(self._type_project(x))

        residual = x
        x = self.hgc1(x, G)
        x = self.ln1(x + residual) if self.abl['use_residual'] else self.ln1(x)
        x = F.relu(x)
        x = self._type_gate(x)
        x = F.dropout(x, self.dropout, training=self.training)

        residual = x
        x = self.hgc2(x, G)
        x = self.ln2(x + residual) if self.abl['use_residual'] else self.ln2(x)

        m_s = self.num_users
        m_e = m_s + self.num_models
        s_s = m_e
        s_e = s_s + self.num_servers

        model_emb = x[m_s:m_e]
        server_emb = x[s_s:s_e]

        nm, ns = model_emb.size(0), server_emb.size(0)
        m_exp = model_emb.unsqueeze(1).expand(-1, ns, -1).reshape(-1, model_emb.size(1))
        s_exp = server_emb.unsqueeze(0).expand(nm, -1, -1).reshape(-1, server_emb.size(1))
        scores = self.placement_predictor(m_exp, s_exp).reshape(nm, ns)
        return scores

    def get_embeddings(self, x, G):
        x = F.relu(self._type_project(x))
        residual = x
        x = self.hgc1(x, G)
        x = self.ln1(x + residual) if self.abl['use_residual'] else self.ln1(x)
        x = F.relu(x)
        x = self._type_gate(x)
        residual = x
        x = self.hgc2(x, G)
        x = self.ln2(x + residual) if self.abl['use_residual'] else self.ln2(x)
        return x
