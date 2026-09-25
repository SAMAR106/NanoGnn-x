import torch

from src.normalizer import TargetNormalizer


def test_fit_transform_round_trip():
    y = torch.tensor([[0.0, -3.0], [2.0, -1.0], [4.0, 0.0]])
    norm = TargetNormalizer().fit(y)
    transformed = norm.transform(y)
    assert torch.allclose(transformed.mean(dim=0), torch.zeros(2), atol=1e-5)
    recovered = norm.inverse_transform(transformed)
    assert torch.allclose(recovered, y, atol=1e-4)


def test_save_load_round_trip(tmp_path):
    y = torch.tensor([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]])
    norm = TargetNormalizer().fit(y)
    path = str(tmp_path / "normalizer.json")
    norm.save(path)

    loaded = TargetNormalizer.load(path)
    assert torch.allclose(loaded.mean, norm.mean)
    assert torch.allclose(loaded.std, norm.std)

    original = torch.tensor([[2.0, -3.0]])
    assert torch.allclose(loaded.inverse_transform(loaded.transform(original)), original, atol=1e-4)


def test_degenerate_constant_column_does_not_divide_by_zero():
    """If every sample has the exact same value for one target, std
    would be 0 -- guard against a division by zero rather than
    producing inf/nan."""
    y = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    norm = TargetNormalizer().fit(y)
    transformed = norm.transform(y)
    assert not torch.isnan(transformed).any()
    assert not torch.isinf(transformed).any()
