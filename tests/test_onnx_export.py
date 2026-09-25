import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

onnxruntime = pytest.importorskip("onnxruntime")

from api.engine import build_feed_dict, export_to_onnx
from src.config import CFG
from src.model import NanoGNNX
from tests.conftest import make_synthetic_graph


def _to_feed(batch):
    return build_feed_dict(batch)


@pytest.fixture(scope="module")
def exported_session(tmp_path_factory):
    torch.manual_seed(0)
    model = NanoGNNX(CFG)
    model.eval()
    sample = Batch.from_data_list([make_synthetic_graph(5, seed=1)])
    onnx_path = str(tmp_path_factory.mktemp("onnx") / "model.onnx")
    export_to_onnx(model, sample, onnx_path)
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return model, session


def test_onnx_matches_pytorch_on_several_graph_sizes(exported_session):
    model, session = exported_session
    for n, seed in [(5, 1), (3, 7), (9, 42), (12, 99)]:
        batch = Batch.from_data_list([make_synthetic_graph(n, seed)])
        with torch.no_grad():
            pt_out = model(batch).numpy()
        onnx_out = session.run(None, _to_feed(batch))[0]
        assert np.allclose(pt_out, onnx_out, atol=1e-4), f"mismatch at n_nodes={n}"


def test_onnx_matches_pytorch_on_zero_triplet_graph(exported_session, isolated_pair_graph):
    """Regression test: ONNX Runtime's Gemm kernel used to hard-crash on
    a literal zero-row input (an isolated atom pair has zero triplets),
    both because of a Python-level `if` branch that got baked as a
    constant during tracing, and independently because of a Gemm-kernel
    limitation with zero-sized dimensions. Fixed by removing the branch
    and padding-then-slicing around the Linear layer in
    src/layers.py::EdgeAngleUpdate."""
    model, session = exported_session
    batch = Batch.from_data_list([isolated_pair_graph])
    with torch.no_grad():
        pt_out = model(batch).numpy()
    onnx_out = session.run(None, _to_feed(batch))[0]
    assert not np.isnan(onnx_out).any()
    assert np.allclose(pt_out, onnx_out, atol=1e-4)


def test_export_does_not_mutate_the_live_model():
    """Regression test for a real bug: torch.onnx.export's legacy tracer
    mutated persistent state on PyG MessagePassing layers as a side
    effect of tracing, silently changing the *live* model's predictions
    on every subsequent call even though its weights were untouched.
    Fixed by exporting a deep copy in api/engine.py::export_to_onnx."""
    torch.manual_seed(0)
    model = NanoGNNX(CFG)
    model.eval()
    batch = Batch.from_data_list([make_synthetic_graph(5, seed=1)])

    with torch.no_grad():
        pt_before = model(batch).clone()

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        export_to_onnx(model, batch, f"{d}/model.onnx")

    with torch.no_grad():
        pt_after = model(batch)

    assert torch.allclose(pt_before, pt_after), "exporting mutated the live model's behavior"


def test_dynamic_batch_size(exported_session):
    """The export is traced from a single-graph (batch_size=1) sample,
    but must still work correctly for a genuinely different number of
    graphs per request -- that's the entire point of the
    num_graphs_carrier trick in _ONNXExportWrapper. Exercises batch
    sizes never seen during tracing, including a return to 1."""
    model, session = exported_session
    for n_graphs in [3, 5, 1, 8]:
        graphs = [make_synthetic_graph(4 + i, seed=10 + i) for i in range(n_graphs)]
        batch = Batch.from_data_list(graphs)
        with torch.no_grad():
            pt_out = model(batch).numpy()
        onnx_out = session.run(None, _to_feed(batch))[0]
        assert pt_out.shape == (n_graphs, 2)
        assert onnx_out.shape == (n_graphs, 2)
        assert np.allclose(pt_out, onnx_out, atol=1e-4), f"mismatch at batch_size={n_graphs}"
