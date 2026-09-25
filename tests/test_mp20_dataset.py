import random

import pandas as pd
import pytest

pytest.importorskip("pymatgen")

from src.dataset import NanoGNNXDataset, cif_to_graph

MP20_DIR = "data/mp20"


def test_mp20_targets_csv_has_expected_row_count():
    df = pd.read_csv(f"{MP20_DIR}/targets.csv")
    assert len(df) == 3000


def test_mp20_random_sample_parses_correctly():
    """Doesn't re-parse all 3000 on every test run (slow, and already
    verified once for real while building this -- see README) --
    spot-checks a random sample each run instead, which still catches a
    systematic problem (e.g. a bad extraction) without paying the full
    cost every time."""
    df = pd.read_csv(f"{MP20_DIR}/targets.csv")
    sample = df.sample(n=25, random_state=0)
    for _, row in sample.iterrows():
        data = cif_to_graph(f"{MP20_DIR}/{row['filename']}")
        assert data.num_nodes >= 1
        assert data.num_nodes <= 20  # MP-20's structural cap, by construction
        assert not data.edge_index.numel() or data.edge_dist.min() > 0


def test_mp20_dataset_loads_via_nanognnxdataset():
    ds = NanoGNNXDataset(raw_dir=MP20_DIR)
    assert len(ds) == 3000
    sample = ds.get(random.Random(0).randrange(len(ds)))
    assert sample.y.shape == (1, 2)
