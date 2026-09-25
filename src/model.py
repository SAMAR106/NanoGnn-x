"""
Full NanoGNN-X assembly:
  atomic number -> Embedding -> N x ThreeBodyMPLayer -> graph mean pool
  -> twin MLP heads (bandgap energy, formation energy)
"""

import torch
import torch.nn as nn

from src.config import CFG
from src.layers import ThreeBodyMPLayer
from src.ops import scatter_mean_safe


class MLPHead(nn.Module):
    """A small regression head; two heads of this shape make up the
    network's multi-task output."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NanoGNNX(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.num_elements, cfg.hidden_dim)
        self.mp_layers = nn.ModuleList(
            [ThreeBodyMPLayer(cfg.hidden_dim) for _ in range(cfg.num_mp_layers)]
        )
        self.bandgap_head = MLPHead(cfg.hidden_dim, cfg.mlp_hidden_dim, cfg.dropout)
        self.formation_energy_head = MLPHead(cfg.hidden_dim, cfg.mlp_hidden_dim, cfg.dropout)

    def forward(self, data) -> torch.Tensor:
        """Convenience entry point for training/inference code that works
        with PyG `Data`/`Batch` objects directly. Determines num_graphs
        from `data` itself, so this path handles arbitrary batch sizes
        (used during training)."""
        batch = getattr(data, "batch", None)
        if batch is None:
            # single, un-batched graph (e.g. at inference time)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
            num_graphs = 1
        else:
            num_graphs = int(batch.max().item()) + 1
        return self.forward_tensors(
            data.x,
            data.edge_index,
            data.edge_dist,
            data.edge_vec,
            data.triplet_index,
            batch,
            num_graphs,
        )

    def forward_tensors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_vec: torch.Tensor,
        triplet_index: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """The actual computation, expressed purely in terms of tensors.

        This split exists because ONNX/TorchScript tracing needs a plain
        tuple of tensor inputs -- it cannot trace through a custom PyG
        `Data` object passed as a single argument (confirmed: tracing
        `forward(self, data)` directly fails, since the exporter can't
        match a `Data` instance against a flat pytree of dynamic axes).
        `api/engine.py` traces this method directly for ONNX export, with
        `num_graphs` fixed at 1 -- the actual serving path
        (`api/server.py`) always sends exactly one crystal per request,
        so the exported graph is validated for that case specifically,
        not for arbitrary dynamic batch sizes. `forward()` above is the
        friendlier, batch-size-agnostic entry point used everywhere else
        (training, explainability).
        """
        h = self.embedding(x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_dist, edge_vec, triplet_index)

        # graph-level mean pool -- see src/ops.py for why this is a
        # hand-rolled scatter-mean rather than PyG's global_mean_pool
        graph_embedding = scatter_mean_safe(h, batch, dim_size=num_graphs)

        bandgap = self.bandgap_head(graph_embedding)
        formation_energy = self.formation_energy_head(graph_embedding)
        return torch.stack([bandgap, formation_energy], dim=-1)  # (num_graphs, 2)
