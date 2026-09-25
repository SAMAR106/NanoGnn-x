"""
Export-safe scatter-mean primitive.

torch_geometric's own `scatter(..., reduce="mean")` and
`global_mean_pool` hit a real ONNX-export bug found empirically while
building this repo: the legacy JIT tracer's mutation-removal pass
strips out what it thinks is an in-place write into one of the traced
graph's *input* tensors (the exact warning was `Removing mutation from
node aten::scatter_add_ on block input: 'batch'`), which silently
changes the traced graph's semantics -- the exported ONNX model then
produces numerically different output than the original PyTorch model
on the exact same input, with no error raised anywhere.

Everything below is built from plain, non-in-place `scatter_add` (no
trailing underscore, always writing into a freshly allocated zero
tensor) specifically to avoid that failure mode. Verified: after
swapping this in, PyTorch and ONNX Runtime outputs match to 1e-4 on
multiple differently-sized graphs.
"""

import torch


def scatter_mean_safe(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Groups rows of `src` (N, F) by `index` (N,), values in
    [0, dim_size), and averages within each group. A group with zero
    members returns 0 for that row (not NaN)."""
    feat_dim = src.size(-1)
    index_expanded = index.view(-1, 1).expand(-1, feat_dim)

    summed = torch.zeros(dim_size, feat_dim, dtype=src.dtype, device=src.device)
    summed = summed.scatter_add(0, index_expanded, src)

    ones = torch.ones(index.size(0), 1, dtype=src.dtype, device=src.device)
    counts = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    counts = counts.scatter_add(0, index.view(-1, 1), ones)
    counts = counts.clamp(min=1.0)

    return summed / counts
