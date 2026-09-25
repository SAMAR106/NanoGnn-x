from src.config import CFG
from src.explainer import NanoGNNXExplainer
from src.model import NanoGNNX


def test_explainer_produces_edge_mask_with_variance(synthetic_graph):
    """A regression guard for two real bugs found while building this:
    (1) passing a bound method instead of an nn.Module as `model=` to
    Explainer silently no-ops the edge mask (GNNExplainer walks
    `model.modules()` to find MessagePassing layers to patch; a bound
    method has none), and (2) GNNExplainer's mask initializer requires
    a 2D `x` even when node_mask_type=None, which our 1D atomic-number
    vector doesn't satisfy without reshaping. Both are fixed in
    src/explainer.py; this test would fail again if either regressed."""
    model = NanoGNNX(CFG)
    model.eval()

    explainer = NanoGNNXExplainer(model, target_head="bandgap", epochs=10)
    edge_mask = explainer.explain(synthetic_graph)

    assert edge_mask.shape[0] == synthetic_graph.edge_index.shape[1]
    # if the mask were never actually applied (bug #1 above), every
    # edge would converge to the same importance and std would be ~0
    assert edge_mask.std().item() > 1e-6


def test_top_bonds_helper(synthetic_graph):
    model = NanoGNNX(CFG)
    model.eval()
    explainer = NanoGNNXExplainer(model, target_head="formation_energy", epochs=5)
    edge_mask = explainer.explain(synthetic_graph)

    top = NanoGNNXExplainer.top_bonds(edge_mask, synthetic_graph.edge_index, k=3)
    assert len(top) == 3
    scores = [score for _, _, score in top]
    assert scores == sorted(scores, reverse=True)
