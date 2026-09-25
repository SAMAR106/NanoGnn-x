import torch

from src.config import CFG
from src.model import NanoGNNX


def _random_rotation_matrix(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g)
    q, _ = torch.linalg.qr(a)
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_prediction_is_rotation_invariant(synthetic_graph):
    """The entire architectural claim of an "invariant three-body
    network" rests on rotating the crystal not changing the prediction.
    This was never actually checked until now -- worth verifying
    directly rather than trusting it followed from using distances and
    dot-product angles."""
    model = NanoGNNX(CFG)
    model.eval()

    with torch.no_grad():
        out_original = model(synthetic_graph)

    R = _random_rotation_matrix(seed=42)
    rotated = synthetic_graph.clone()
    rotated.edge_vec = synthetic_graph.edge_vec @ R.T

    with torch.no_grad():
        out_rotated = model(rotated)

    assert torch.allclose(out_original, out_rotated, atol=1e-4)
