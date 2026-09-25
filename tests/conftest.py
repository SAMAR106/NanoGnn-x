"""
Shared test fixtures. `make_synthetic_graph` builds a small, fully
connected `CrystalData` graph with real (if arbitrary) 3D coordinates,
so tests can exercise the whole distance/angle/triplet pipeline without
needing pymatgen or an actual .cif file on disk.
"""

from collections import defaultdict

import pytest
import torch

from src.dataset import CrystalData


def make_synthetic_graph(
    n_nodes: int = 5, seed: int = 0, fully_connected: bool = True
) -> CrystalData:
    g = torch.Generator().manual_seed(seed)
    z = torch.randint(1, 30, (n_nodes,), generator=g)
    pos = torch.randn(n_nodes, 3, generator=g)

    src, dst = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and fully_connected:
                src.append(i)
                dst.append(j)
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_vec = torch.zeros((0, 3))
    edge_dist = edge_vec.norm(dim=-1)

    by_center = defaultdict(list)
    for e, i in enumerate(edge_index[0].tolist()):
        by_center[i].append(e)
    e1, e2 = [], []
    for edges in by_center.values():
        for a in range(len(edges)):
            for b in range(len(edges)):
                if a != b:
                    e1.append(edges[a])
                    e2.append(edges[b])
    triplet_index = (
        torch.tensor([e1, e2], dtype=torch.long) if e1 else torch.zeros((2, 0), dtype=torch.long)
    )

    return CrystalData(
        x=z,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        triplet_index=triplet_index,
        num_nodes=n_nodes,
        y=torch.tensor([[1.0, -1.0]]),
    )


@pytest.fixture
def synthetic_graph():
    return make_synthetic_graph(n_nodes=5, seed=1)


@pytest.fixture
def isolated_pair_graph():
    """Two atoms, one bond, zero triplets -- the degenerate edge case
    that broke ONNX export until src/layers.py padded around it."""
    x = torch.tensor([6, 8])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_vec = torch.tensor([[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]], dtype=torch.float32)
    edge_dist = edge_vec.norm(dim=-1)
    triplet_index = torch.zeros((2, 0), dtype=torch.long)
    return CrystalData(
        x=x,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        triplet_index=triplet_index,
        num_nodes=2,
    )
