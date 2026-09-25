import torch

from src.ops import scatter_mean_safe


def test_scatter_mean_basic():
    src = torch.tensor([[1.0], [3.0], [10.0]])
    index = torch.tensor([0, 0, 1])
    out = scatter_mean_safe(src, index, dim_size=2)
    assert torch.allclose(out, torch.tensor([[2.0], [10.0]]))


def test_scatter_mean_empty_group_returns_zero_not_nan():
    src = torch.tensor([[1.0], [2.0]])
    index = torch.tensor([0, 0])
    # dim_size=3 -- group 1 and 2 have zero members
    out = scatter_mean_safe(src, index, dim_size=3)
    assert torch.allclose(out[0], torch.tensor([1.5]))
    assert torch.allclose(out[1], torch.tensor([0.0]))
    assert torch.allclose(out[2], torch.tensor([0.0]))
    assert not torch.isnan(out).any()


def test_scatter_mean_fully_empty_input():
    src = torch.zeros((0, 4))
    index = torch.zeros((0,), dtype=torch.long)
    out = scatter_mean_safe(src, index, dim_size=5)
    assert out.shape == (5, 4)
    assert torch.allclose(out, torch.zeros(5, 4))
