"""
XAI layer: wraps torch_geometric.explain.Explainer (GNNExplainer)
around a trained NanoGNNX model to highlight which bonds drove a given
bandgap / formation-energy prediction -- e.g. the six Ti-O bonds of an
octahedral TiO6 unit lighting up for a bandgap prediction.

Honest caveat: GNNExplainer natively produces node and edge masks over
the model's *input* tensors. Our node input (`x`) is a discrete atomic
number fed through an nn.Embedding, so a continuous "attribute mask" on
it isn't very interpretable on its own -- masking which *elements*
matter is a strange question when the element is a fixed fact about the
structure. We therefore mask edges only. The resulting edge_mask directly
answers "which bonds mattered", and because every triplet's angle
feature is folded back onto its owning edge, an important edge also
implicates the triplet angles that feed it -- good enough to render a
heatmap over bonds for judges, but if you want per-triplet (not
per-edge) attribution, you'd need a custom Captum-style attribution over
the angle features directly.
"""

import torch
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer


class _ExplainerModelWrapper(nn.Module):
    """Adapts NanoGNNX's `forward(data)` signature to the `forward(x,
    edge_index, **kwargs)` signature GNNExplainer calls.

    This must be a real `nn.Module` holding `model` as a registered
    submodule, not a bound method or closure. GNNExplainer's edge-mask
    mechanism works by calling `torch_geometric.nn.conv.set_masks(model,
    ...)`, which walks `model.modules()` looking for `MessagePassing`
    instances to patch. A bound method has no `.modules()` to walk, so
    the mask is silently never applied to anything -- confirmed by
    running it and finding edge_mask always converges to all-ones,
    regardless of `edges_to_add_loss` weighting.
    """

    def __init__(self, model: nn.Module, target_index: int):
        super().__init__()
        self.model = model
        self.target_index = target_index

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        # GNNExplainer perturbs x / edge_index and calls us with those
        # perturbed tensors; splice them into a clone of the original
        # graph so edge_dist, edge_vec, and triplet_index stay consistent
        # with whatever edge_index it's currently testing.
        #
        # x arrives as float (N, 1) -- GNNExplainer's mask initializer
        # unconditionally does `(N, F) = x.size()` even when
        # node_mask_type=None, so `explain()` below hands it a reshaped
        # float view of the atomic numbers. Since node_mask_type=None
        # means that x is never actually perturbed by a learned mask, the
        # round-trip back to long is exact (verified) for atomic numbers.
        data = kwargs["data"].clone()
        data.x = x.view(-1).long()
        data.edge_index = edge_index
        return self.model(data)[:, self.target_index : self.target_index + 1]


class NanoGNNXExplainer:
    def __init__(self, model: nn.Module, target_head: str = "bandgap", epochs: int = 200):
        """target_head: 'bandgap' (output index 0) or 'formation_energy' (index 1)."""
        self.model = model
        self.target_index = 0 if target_head == "bandgap" else 1
        self._wrapped_model = _ExplainerModelWrapper(model, self.target_index)

        self.explainer = Explainer(
            model=self._wrapped_model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type=None,
            edge_mask_type="object",
            model_config=dict(
                mode="regression",
                task_level="graph",
                return_type="raw",
            ),
        )

    def explain(self, data):
        """Returns an edge_mask (E,) of importance scores in [0, 1] for
        every bond in `data`. Sort descending and take the top-k to get
        e.g. "these are the 6 bonds the model relied on"."""
        # GNNExplainer's mask initializer requires a 2D (N, F) x
        # regardless of node_mask_type -- reshape the 1D atomic-number
        # vector to (N, 1) float here; the wrapper above converts it
        # straight back for the actual model call.
        x_for_explainer = data.x.view(-1, 1).float()
        explanation = self.explainer(x=x_for_explainer, edge_index=data.edge_index, data=data)
        return explanation.edge_mask

    @staticmethod
    def top_bonds(edge_mask: torch.Tensor, edge_index: torch.Tensor, k: int = 6):
        """Convenience helper: returns the top-k (atom_i, atom_j, score)
        triples for rendering a heatmap over the crystal structure."""
        scores, idx = torch.topk(edge_mask, k=min(k, edge_mask.numel()))
        pairs = edge_index[:, idx]
        return [(int(pairs[0, n]), int(pairs[1, n]), float(scores[n])) for n in range(idx.numel())]
