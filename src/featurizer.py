"""
Invariant featurization utilities.

Distances use the radial Bessel basis from DimeNet (Klicpera, Gross &
Gunnemann, "Directional Message Passing for Molecular Graphs", ICLR
2020): for a cutoff c, the n-th basis function is the zeroth-order
spherical Bessel function j_0(x) = sin(x)/x evaluated at its n-th root
(n*pi), rescaled to the cutoff --

    e_n(d) = sqrt(2/c) * sin(n*pi*d/c) / d,   n = 1..N

These are an orthogonal basis on [0, c] derived directly from the
radial part of the Schrodinger equation for a particle in an infinite
spherical well, which is the physical motivation DimeNet gives for
using them over an arbitrarily-centered Gaussian RBF. They're combined
with a smooth cosine cutoff envelope (the Behler-Parrinello form used
across the interatomic-potential literature, e.g. Behler & Parrinello,
PRL 2007) so the feature -- and its derivative -- goes smoothly to zero
at the cutoff radius, instead of the discontinuity an un-enveloped
Gaussian RBF has right at r_cut.

Angles use a Chebyshev polynomial basis of cos(theta): T_k(cos(theta))
= cos(k * theta), so this is exactly a truncated Fourier series in the
bond angle itself -- the natural orthogonal basis for a periodic
angular variable, and the same cos(k*theta) structure that the m=0
slice of a spherical-harmonics expansion reduces to. Computed directly
from cos(theta) via the standard Chebyshev recurrence (T_0=1, T_1=x,
T_k = 2x*T_{k-1} - T_{k-2}), which avoids ever calling arccos (numerically
safer near cos(theta) = +/-1 than going through the angle itself).
"""

import torch
import torch.nn as nn


class CosineCutoff(nn.Module):
    """Smooth envelope f(d) = 0.5*(cos(pi*d/c)+1) for d < c, else 0.
    Standard smooth-cutoff form (Behler & Parrinello, PRL 2007) used to
    make interatomic features -- and their derivatives -- vanish
    continuously at the cutoff radius, rather than truncating sharply."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        cutoffs = 0.5 * (torch.cos(dist * (torch.pi / self.cutoff)) + 1.0)
        return cutoffs * (dist < self.cutoff).to(dist.dtype)


class BesselRBF(nn.Module):
    """DimeNet's radial Bessel basis (see module docstring), enveloped
    with a `CosineCutoff` so features smoothly vanish at `cutoff`."""

    def __init__(self, cutoff: float = 5.0, num_rbf: int = 64):
        super().__init__()
        self.cutoff = cutoff
        n = torch.arange(1, num_rbf + 1, dtype=torch.float32)
        self.register_buffer("frequencies", n * torch.pi / cutoff)
        self.envelope = CosineCutoff(cutoff)
        self.norm = (2.0 / cutoff) ** 0.5

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: (E,) -> (E, num_rbf). Guard against d=0 (two atoms at
        # the same position shouldn't occur in a valid structure, but a
        # 1e-8 floor keeps this from ever producing a NaN/Inf).
        d = dist.clamp(min=1e-8).unsqueeze(-1)  # (E, 1)
        # mypy can't statically tell a register_buffer'd attribute
        # (always a real Tensor at runtime) apart from a submodule --
        # nn.Module routes both through the same dynamic __getattr__,
        # a known PyTorch/mypy friction point with no clean fix short
        # of heavy stub gymnastics.
        out = self.norm * torch.sin(self.frequencies * d) / d  # type: ignore[operator]
        return out * self.envelope(dist).unsqueeze(-1)


class ChebyshevAngleBasis(nn.Module):
    """Expands cos(theta) into Chebyshev polynomials T_0..T_{K-1}, i.e.
    a K-term Fourier series cos(0*theta), cos(1*theta), ..., cos((K-1)*theta)
    in the bond angle itself (see module docstring)."""

    def __init__(self, num_orders: int = 64):
        super().__init__()
        self.num_orders = num_orders

    def forward(self, cos_theta: torch.Tensor) -> torch.Tensor:
        # cos_theta: (T,) -> (T, num_orders)
        polys = [torch.ones_like(cos_theta), cos_theta]
        for _ in range(2, self.num_orders):
            polys.append(2 * cos_theta * polys[-1] - polys[-2])
        return torch.stack(polys[: self.num_orders], dim=-1)


def compute_angles_from_edge_vec(
    edge_vec: torch.Tensor, triplet_index: torch.Tensor
) -> torch.Tensor:
    """Given per-edge cartesian bond vectors and a (2, T) triplet index of
    (edge_1, edge_2) pairs that share a center atom, returns cos(theta)
    for each triplet.

    We compute angles from the edge vectors directly (not from node
    positions) because in a periodic crystal, two bonds from the same
    center atom can point at two different periodic *images* of a
    neighbor -- the plain node-position difference would be wrong.
    """
    e1, e2 = triplet_index[0], triplet_index[1]
    v1, v2 = edge_vec[e1], edge_vec[e2]
    cos_theta = (v1 * v2).sum(dim=-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
    return cos_theta.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
