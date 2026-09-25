import torch
from torch_geometric.data import Batch

from src.config import CFG
from src.model import NanoGNNX


def test_single_graph_forward_shape(synthetic_graph):
    model = NanoGNNX(CFG)
    out = model(synthetic_graph)
    assert out.shape == (1, 2)
    assert not torch.isnan(out).any()


def test_batched_forward_shape(synthetic_graph):
    from tests.conftest import make_synthetic_graph

    g2 = make_synthetic_graph(n_nodes=7, seed=2)
    model = NanoGNNX(CFG)
    batch = Batch.from_data_list([synthetic_graph, g2])
    out = model(batch)
    assert out.shape == (2, 2)


def test_gradients_reach_every_parameter(synthetic_graph):
    from tests.conftest import make_synthetic_graph

    g2 = make_synthetic_graph(n_nodes=7, seed=2)
    model = NanoGNNX(CFG)
    batch = Batch.from_data_list([synthetic_graph, g2])

    out = model(batch)
    target = torch.cat([synthetic_graph.y, g2.y], dim=0)
    loss = torch.nn.functional.huber_loss(out, target)
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.abs().sum() > 0, f"{name} got an all-zero gradient"


def test_isolated_atom_zero_triplets_no_nan(isolated_pair_graph):
    """A crystal where no atom has 2+ neighbors within the cutoff (zero
    triplets) must not produce NaNs -- this used to be a Python-level
    branch that also silently broke under ONNX export; see
    src/layers.py."""
    model = NanoGNNX(CFG)
    out = model(isolated_pair_graph)
    assert not torch.isnan(out).any()


def test_element_overflow_raises_clear_error():
    """The embedding-table guard should raise a clear, actionable error
    for an atomic number the table can't hold, rather than an opaque
    IndexError surfacing deep inside nn.Embedding at train time."""
    import pytest

    from src.dataset import _check_atomic_numbers

    ok = torch.tensor([6, 8, 22])  # carbon, oxygen, titanium -- all fine
    _check_atomic_numbers(ok, source="test.cif", num_elements=119)  # should not raise

    too_big = torch.tensor([6, 8, 500])  # 500 is not a real element, on purpose
    with pytest.raises(ValueError, match="num_elements"):
        _check_atomic_numbers(too_big, source="test.cif", num_elements=119)
