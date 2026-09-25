"""
Target normalization.

Bandgap (eV, roughly 0-9 in this dataset) and formation energy
(eV/atom, roughly -4 to 0) live on genuinely different scales -- fitting
both jointly with one Huber `delta` and no normalization biases training
toward whichever target has the larger raw magnitude. Standard fix:
z-score each target independently using *training-set* statistics only,
train the network to predict normalized values, and invert back to
physical units at inference time.

The model itself never needs to know about normalization -- it just
predicts two normalized scalars. This module only handles the fit /
transform / inverse_transform / persistence around that.
"""

import json

import torch


class TargetNormalizer:
    def __init__(self, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.mean = mean
        self.std = std

    def _require_fitted(self) -> None:
        """`mean`/`std` are only set after `.fit()` or `.load()` -- this
        turns "divide by None" (a confusing TypeError deep in a tensor
        op) into a clear, actionable error if `.transform()` or
        `.inverse_transform()` is ever called before either of those."""
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "TargetNormalizer has no fitted statistics yet -- call "
                ".fit(y) or .load(path) before .transform()/.inverse_transform()."
            )

    def fit(self, y: torch.Tensor) -> "TargetNormalizer":
        """y: (N, 2) raw [bandgap, formation_energy] targets from the
        training split only -- never fit on validation/test data, or
        the split stops being a fair measure of generalization."""
        self.mean = y.mean(dim=0)
        self.std = y.std(dim=0).clamp(min=1e-6)  # guard a degenerate all-equal column
        return self

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        assert self.mean is not None and self.std is not None  # narrows for mypy
        return (y - self.mean) / self.std

    def inverse_transform(self, y_normalized: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        assert self.mean is not None and self.std is not None
        return y_normalized * self.std + self.mean

    def save(self, path: str) -> None:
        self._require_fitted()
        assert self.mean is not None and self.std is not None
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    @classmethod
    def load(cls, path: str) -> "TargetNormalizer":
        with open(path) as f:
            stats = json.load(f)
        return cls(mean=torch.tensor(stats["mean"]), std=torch.tensor(stats["std"]))
