"""
CIF -> PyTorch Geometric graph conversion.

Each crystal structure becomes a `Data` object:
  x             -- (N,) atomic numbers (embedded inside the model)
  edge_index    -- (2, E) source/target *base* atom indices. Periodic
                   images are collapsed onto their base atom's index; the
                   actual image offset lives in edge_vec, not edge_index.
  edge_vec      -- (E, 3) cartesian bond vectors i -> j (image-aware)
  edge_dist     -- (E,) bond lengths
  triplet_index -- (2, T) pairs of edge indices (e1, e2) that share a
                   center atom, used for the three-body angle term
  y             -- (1, 2) targets [bandgap_eV, formation_energy_eV_per_atom]

Periodic boundary conditions are handled by pymatgen's
`get_neighbor_list`, which already accounts for lattice images -- that's
what lets an atom near a cell edge correctly "see" neighbors across the
boundary.
"""

import os
from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from src.config import CFG

if TYPE_CHECKING:
    # pymatgen is only needed by functions that actually parse CIFs; keep
    # the import lazy (inside those functions) so this module -- and in
    # particular the CrystalData class below -- stays importable in
    # contexts (tests, the API layer) that don't need pymatgen installed.
    from pymatgen.core import Structure


class CrystalData(Data):
    """A `Data` subclass with one override: `triplet_index` indexes into
    *edges*, not nodes.

    PyG's default `Batch` collation auto-increments any attribute whose
    name contains "index" by the running node count, because that's the
    right rule for `edge_index`. It is the *wrong* rule for
    `triplet_index`, which is a pair of edge indices -- left on the
    default, batching a DataLoader with more than one graph per batch
    silently corrupts every triplet after the first graph (verified: a
    graph with node_count != edge_count batched next to another graph
    produces triplet indices that point at the wrong graph's edges,
    with no error raised). Overriding `__inc__` to offset by edge count
    instead of node count fixes this.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "triplet_index":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "triplet_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _build_edges(structure: "Structure", r_cut: float):
    """Builds periodic-boundary-aware edges via pymatgen's neighbor list.

    Verified against a real rock-salt NaCl structure: recovers the
    correct first/second/third coordination shells (2.82 A / 3.99 A /
    4.88 A for a=5.64 A) and octahedral (6-neighbor) first-shell
    coordination.
    """
    import numpy as np

    center_idx, point_idx, offsets, distances = structure.get_neighbor_list(r=r_cut)

    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    cart_coords = torch.tensor(structure.cart_coords, dtype=torch.float32)

    # np.stack (not a raw Python list of arrays) avoids a slow-path
    # warning from torch.tensor() and is meaningfully faster for the
    # neighbor-list sizes real crystals produce.
    edge_index = torch.from_numpy(np.stack([center_idx, point_idx]).astype(np.int64))

    # Offsets are in fractional lattice units; convert to cartesian so we
    # can add them straight onto the neighbor's base cartesian position.
    offsets_cart = torch.tensor(offsets, dtype=torch.float32) @ lattice

    center_cart = cart_coords[edge_index[0]]
    neighbor_cart = cart_coords[edge_index[1]] + offsets_cart

    edge_vec = neighbor_cart - center_cart
    edge_dist = torch.tensor(distances, dtype=torch.float32)

    return edge_index, edge_vec, edge_dist


def _build_triplets(edge_index: torch.Tensor, max_neighbors: int = 32) -> torch.Tensor:
    """Groups edges by their center atom and enumerates every ordered pair
    of distinct edges sharing that center -- these are the (j, i, k)
    three-body angle terms. Truncated per-atom to keep the O(degree^2)
    blow-up bounded for high-coordination atoms."""
    by_center = defaultdict(list)
    for e, i in enumerate(edge_index[0].tolist()):
        by_center[i].append(e)

    e1_list, e2_list = [], []
    for edges in by_center.values():
        edges = edges[:max_neighbors]
        for a in range(len(edges)):
            for b in range(len(edges)):
                if a == b:
                    continue
                e1_list.append(edges[a])
                e2_list.append(edges[b])

    if not e1_list:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([e1_list, e2_list], dtype=torch.long)


def _check_atomic_numbers(z: torch.Tensor, source: str, num_elements: int | None = None) -> None:
    """Raises a clear, actionable error if any atomic number exceeds the
    embedding table's size, instead of letting it surface as an opaque
    IndexError deep inside nn.Embedding at train time."""
    num_elements = num_elements if num_elements is not None else CFG.num_elements
    if z.numel() and int(z.max()) >= num_elements:
        raise ValueError(
            f"{source} contains an element with atomic number {int(z.max())}, "
            f"but CFG.num_elements is only {num_elements}. Raise "
            f"num_elements in src/config.py to at least {int(z.max()) + 1}."
        )


def cif_to_graph(
    cif_path: str, y: torch.Tensor | None = None, r_cut: float | None = None
) -> CrystalData:
    """Loads a single .cif file and converts it into a `CrystalData` graph."""
    import warnings

    from pymatgen.core import Structure  # lazy: see the TYPE_CHECKING note above

    r_cut = r_cut or CFG.r_cut
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*fractional coordinates rounded to ideal values.*",
            category=UserWarning,
        )
        structure = Structure.from_file(cif_path)

    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    _check_atomic_numbers(z, cif_path)

    edge_index, edge_vec, edge_dist = _build_edges(structure, r_cut)
    triplet_index = _build_triplets(edge_index, CFG.max_neighbors)

    data = CrystalData(
        x=z,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_dist=edge_dist,
        triplet_index=triplet_index,
        num_nodes=z.size(0),
    )
    if y is not None:
        data.y = y.view(1, -1)
    return data


class NanoGNNXDataset(Dataset):
    """Expects `raw_dir` to contain *.cif files plus a `targets.csv` with
    columns: filename, bandgap, formation_energy. See
    data/raw_cifs/targets.csv for the expected format.

    Caches each parsed `CrystalData` graph in memory after first access.
    Without this, every training epoch re-parses every CIF from scratch
    (pymatgen's periodic neighbor search included) -- fine at the
    original 101-structure demo scale, but a real, measured cost once
    you scale up: re-parsing dominates epoch time on a
    several-thousand-structure real dataset (e.g. via
    scripts/download_mp20.py), where the actual forward/backward pass is
    comparatively cheap. Safe to cache because a `CrystalData` object,
    once built, is never mutated in place anywhere in this codebase.
    """

    def __init__(
        self,
        raw_dir: str | None = None,
        targets_csv: str | None = None,
        transform=None,
        pre_transform=None,
    ):
        self.raw_dir_ = raw_dir or CFG.raw_cif_dir
        self.targets_csv = targets_csv or os.path.join(self.raw_dir_, "targets.csv")
        self.df = pd.read_csv(self.targets_csv)
        self._cache: dict[int, CrystalData] = {}
        super().__init__(root=self.raw_dir_, transform=transform, pre_transform=pre_transform)

    def len(self):
        return len(self.df)

    def get(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        row = self.df.iloc[idx]
        cif_path = os.path.join(self.raw_dir_, row["filename"])
        y = torch.tensor([row["bandgap"], row["formation_energy"]], dtype=torch.float32)
        data = cif_to_graph(cif_path, y=y)
        self._cache[idx] = data
        return data
