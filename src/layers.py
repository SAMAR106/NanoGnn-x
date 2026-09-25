"""
Custom invariant three-body message-passing layer.

Follows the DimeNet/MEGNet family of ideas: each bond's feature vector is
refined using the angles it forms with every other bond sharing the same
central atom, before that refined bond feature is used to pass a message
from neighbor to center atom. This is what lets the network distinguish,
say, an octahedral TiO6 environment from a tetrahedral one even when the
raw Ti-O distances are similar -- the angles carry the geometric
information a purely distance-based model would miss.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from src.config import CFG
from src.featurizer import BesselRBF, ChebyshevAngleBasis
from src.ops import scatter_mean_safe


class EdgeAngleUpdate(nn.Module):
    """Combines each edge's distance basis with the (aggregated) angle
    basis of every triplet it participates in, into a single edge
    embedding. Distance uses DimeNet's radial Bessel basis; angle uses
    a Chebyshev (Fourier-in-the-angle) basis -- see src/featurizer.py
    for the literature grounding of both."""

    def __init__(self, hidden_dim: int = CFG.hidden_dim):
        super().__init__()
        self.dist_rbf = BesselRBF(cutoff=CFG.r_cut, num_rbf=CFG.num_rbf_dist)
        self.angle_basis = ChebyshevAngleBasis(num_orders=CFG.num_rbf_angle)
        self.dist_proj = nn.Linear(CFG.num_rbf_dist, hidden_dim)
        self.angle_proj = nn.Linear(CFG.num_rbf_angle, hidden_dim)
        self.combine = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        edge_dist: torch.Tensor,
        edge_vec: torch.Tensor,
        triplet_index: torch.Tensor,
        num_edges: int,
    ) -> torch.Tensor:
        dist_feat = self.dist_proj(self.dist_rbf(edge_dist))  # (E, H)

        # No `if triplet_index.numel() > 0` branch on purpose: a Python
        # conditional on a tensor's runtime size gets baked as a fixed
        # constant during ONNX tracing (confirmed via a TracerWarning),
        # which would silently break any future crystal with an isolated
        # atom (zero triplets) if exported from a sample graph that had
        # some.
        e1, e2 = triplet_index[0], triplet_index[1]
        v1, v2 = edge_vec[e1], edge_vec[e2]
        cos_theta = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
        cos_theta = cos_theta.clamp(-1 + 1e-6, 1 - 1e-6)

        # Pad with one dummy row before the Linear layer, then slice it
        # back off. This looks redundant in eager mode (and is a no-op
        # there), but it fixes a real ONNX Runtime bug: its Gemm kernel
        # (what nn.Linear lowers to) rejects a literal zero-row input at
        # *runtime* -- confirmed by exporting and running a genuinely
        # zero-triplet graph (an atom with only one neighbor) and hitting
        # "GemmHelper ... NumDimensions() == 2 || == 1 was false". Since
        # the angle basis and the Linear layer are both strictly row-wise
        # (no cross-row mixing), padding-then-slicing is mathematically
        # identical to not padding at all in the normal case, and turns
        # the zero-triplet case into a harmless one-row-then-dropped case
        # instead of a runtime crash. Verified both ways on real exports.
        padded_cos_theta = torch.cat([cos_theta, cos_theta.new_zeros(1)])
        angle_feat_per_triplet = self.angle_proj(self.angle_basis(padded_cos_theta))[:-1]

        # each triplet is "owned" by its e1 edge; average every triplet's
        # angle contribution back onto that edge
        angle_feat = scatter_mean_safe(angle_feat_per_triplet, e1, dim_size=num_edges)

        return self.combine(torch.cat([dist_feat, angle_feat], dim=-1))


class ThreeBodyMPLayer(MessagePassing):
    """One round of invariant message passing: each node aggregates
    messages from its neighbors, weighted by the (distance + angle)
    edge embedding computed above."""

    def __init__(self, hidden_dim: int = CFG.hidden_dim):
        super().__init__(aggr="add", node_dim=0)
        # NOTE: do not name this attribute `edge_update` -- MessagePassing
        # reserves that name for its own edge_updater() hook mechanism, and
        # assigning over it gets silently shadowed by the inherited method
        # (confirmed: calling it then raises a bogus "takes 1 positional
        # argument" TypeError from the wrong function entirely).
        self.edge_feat_net = EdgeAngleUpdate(hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_vec: torch.Tensor,
        triplet_index: torch.Tensor,
    ) -> torch.Tensor:
        edge_feat = self.edge_feat_net(edge_dist, edge_vec, triplet_index, edge_index.size(1))
        aggregated = self.propagate(edge_index, x=x, edge_feat=edge_feat)
        x_new = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return self.norm(x + x_new)  # residual connection, stabilizes deeper stacks

    def message(self, x_j: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        return self.message_mlp(torch.cat([x_j, edge_feat], dim=-1))
