import math

import torch

from src.featurizer import BesselRBF, ChebyshevAngleBasis, CosineCutoff


def test_bessel_rbf_vanishes_smoothly_at_cutoff():
    rbf = BesselRBF(cutoff=5.0, num_rbf=8)
    d = torch.tensor([0.5, 2.0, 4.99, 5.0, 5.5])
    out = rbf(d)
    assert not torch.isnan(out).any()
    assert torch.allclose(out[3], torch.zeros(8), atol=1e-4)  # at cutoff
    assert torch.allclose(out[4], torch.zeros(8))  # beyond cutoff


def test_bessel_rbf_no_nan_near_zero_distance():
    rbf = BesselRBF(cutoff=5.0, num_rbf=8)
    out = rbf(torch.tensor([0.0, 1e-9]))
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_cosine_cutoff_shape():
    cutoff = CosineCutoff(cutoff=5.0)
    d = torch.tensor([0.0, 2.5, 5.0, 6.0])
    out = cutoff(d)
    assert out[0].item() == 1.0  # cos(0) = 1 at d=0
    assert out[3].item() == 0.0  # beyond cutoff


def test_chebyshev_matches_true_fourier_series_in_angle():
    """T_k(cos(theta)) must equal cos(k*theta) exactly -- that identity
    is the entire justification for using Chebyshev polynomials as a
    Fourier-in-the-angle basis instead of an arbitrary Gaussian bump."""
    cheb = ChebyshevAngleBasis(num_orders=6)
    for theta_deg in [0, 37, 60, 90, 120, 163, 180]:
        theta = torch.tensor(theta_deg * math.pi / 180)
        cos_theta = torch.cos(theta).unsqueeze(0)
        out = cheb(cos_theta)[0]
        expected = torch.tensor([math.cos(k * theta.item()) for k in range(6)])
        assert torch.allclose(out, expected, atol=1e-4), f"mismatch at {theta_deg} degrees"


def test_chebyshev_output_shape():
    cheb = ChebyshevAngleBasis(num_orders=10)
    cos_theta = torch.linspace(-1, 1, 20)
    out = cheb(cos_theta)
    assert out.shape == (20, 10)
