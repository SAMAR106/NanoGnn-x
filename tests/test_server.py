import dataclasses
import glob

import pytest
import torch

pytest.importorskip("pymatgen")
pytest.importorskip("fastapi")
onnxruntime = pytest.importorskip("onnxruntime")

from torch_geometric.data import Batch

from api.engine import export_to_onnx
from src.config import CFG
from src.dataset import cif_to_graph
from src.model import NanoGNNX
from src.normalizer import TargetNormalizer

NACL_CIF = "data/raw_cifs/NaCl.cif"


@pytest.fixture
def deployed_checkpoint(tmp_path, monkeypatch):
    """Exports a (randomly initialized, untrained) model + a matching
    identity normalizer into a pytest tmp_path, then monkeypatches the
    module-level `CFG` that api/server.py and api/engine.py actually
    read at call time, so `InferenceEngine`'s startup genuinely loads
    from the temp location.

    This used to write directly to `CFG.onnx_path` -- the *same path*
    `train.py`/`api/engine.py` use for the real production checkpoint
    -- and delete it on teardown with no backup. Running this test
    suite after a real training run silently destroyed the real
    exported ONNX model (confirmed: it vanished between one export and
    the next `pytest` run, with no error anywhere). tmp_path + a
    monkeypatched CFG makes that class of bug structurally impossible:
    this fixture can no longer touch a real checkpoint even by
    accident, whatever `CFG.checkpoint_dir` happens to point at.
    """
    torch.manual_seed(0)
    model = NanoGNNX(CFG)
    model.eval()
    sample = cif_to_graph(NACL_CIF)
    batch = Batch.from_data_list([sample])

    onnx_path = str(tmp_path / "model.onnx")
    normalizer_path = str(tmp_path / "normalizer.json")

    export_to_onnx(model, batch, onnx_path)
    TargetNormalizer(mean=torch.zeros(2), std=torch.ones(2)).save(normalizer_path)

    test_cfg = dataclasses.replace(CFG, onnx_path=onnx_path, normalizer_path=normalizer_path)
    monkeypatch.setattr("api.server.CFG", test_cfg)
    monkeypatch.setattr("api.engine.CFG", test_cfg)

    yield onnx_path


def test_server_end_to_end(deployed_checkpoint):
    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    with client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        with open(NACL_CIF, "rb") as f:
            resp = client.post("/predict", files={"file": ("NaCl.cif", f, "chemical/x-cif")})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "bandgap_eV",
            "formation_energy_eV_per_atom",
            "inference_time_ms",
            "backend",
            "normalized",
        }
        assert body["normalized"] is False  # a normalizer was loaded, so output is in eV
        predictions = [body["bandgap_eV"], body["formation_energy_eV_per_atom"]]
        assert all(v == v for v in predictions)  # no NaN

        rejected = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
        assert rejected.status_code == 400


def test_predict_does_not_leak_temp_files(deployed_checkpoint):
    """Regression test: the /predict endpoint used to write a temp .cif
    file with delete=False and never clean it up -- every request leaked
    one file. Fixed with a try/finally os.unlink in api/server.py."""
    from fastapi.testclient import TestClient

    from api.server import app

    before = set(glob.glob("/tmp/*.cif"))
    client = TestClient(app)
    with client, open(NACL_CIF, "rb") as f:
        client.post("/predict", files={"file": ("NaCl.cif", f, "chemical/x-cif")})
    after = set(glob.glob("/tmp/*.cif"))
    assert after == before, f"predict() leaked temp files: {after - before}"


def test_predict_batch_returns_one_result_per_file(deployed_checkpoint):
    """The whole point of extending ONNX export past batch_size=1: a
    single /predict/batch call runs several structures through one
    batched forward pass and returns one result per input file, in
    the same order they were uploaded."""
    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    with client:
        with open(NACL_CIF, "rb") as f1, open(NACL_CIF, "rb") as f2, open(NACL_CIF, "rb") as f3:
            resp = client.post(
                "/predict/batch",
                files=[
                    ("files", ("a.cif", f1, "chemical/x-cif")),
                    ("files", ("b.cif", f2, "chemical/x-cif")),
                    ("files", ("c.cif", f3, "chemical/x-cif")),
                ],
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert [r["filename"] for r in body["results"]] == ["a.cif", "b.cif", "c.cif"]
        for r in body["results"]:
            assert r["bandgap_eV"] == r["bandgap_eV"]  # no NaN

        # same structure uploaded three times -> identical predictions,
        # since a batched forward pass shouldn't cross-contaminate
        # between graphs in the batch
        bandgaps = [r["bandgap_eV"] for r in body["results"]]
        assert bandgaps[0] == pytest.approx(bandgaps[1], abs=1e-4)
        assert bandgaps[1] == pytest.approx(bandgaps[2], abs=1e-4)


def test_predict_batch_rejects_too_many_files(deployed_checkpoint, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr("api.server.MAX_BATCH_FILES", 2)
    from api.server import app

    client = TestClient(app)
    with client:
        with open(NACL_CIF, "rb") as f1, open(NACL_CIF, "rb") as f2, open(NACL_CIF, "rb") as f3:
            resp = client.post(
                "/predict/batch",
                files=[
                    ("files", ("a.cif", f1, "chemical/x-cif")),
                    ("files", ("b.cif", f2, "chemical/x-cif")),
                    ("files", ("c.cif", f3, "chemical/x-cif")),
                ],
            )
        assert resp.status_code == 400


def test_upload_over_size_limit_is_rejected(deployed_checkpoint, monkeypatch):
    monkeypatch.setattr("api.server.MAX_UPLOAD_BYTES", 100)  # tiny, on purpose
    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    with client:
        oversized = b"#\n" * 1000  # comfortably over 100 bytes, still parses as junk not a real CIF
        resp = client.post("/predict", files={"file": ("big.cif", oversized, "chemical/x-cif")})
        assert resp.status_code == 413


def test_api_key_required_when_configured(deployed_checkpoint, monkeypatch):
    monkeypatch.setattr("api.server.API_KEY", "secret123")
    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    with client:
        with open(NACL_CIF, "rb") as f:
            no_key = client.post("/predict", files={"file": ("NaCl.cif", f, "chemical/x-cif")})
        assert no_key.status_code == 401

        with open(NACL_CIF, "rb") as f:
            wrong_key = client.post(
                "/predict",
                files={"file": ("NaCl.cif", f, "chemical/x-cif")},
                headers={"X-API-Key": "wrong"},
            )
        assert wrong_key.status_code == 401

        with open(NACL_CIF, "rb") as f:
            right_key = client.post(
                "/predict",
                files={"file": ("NaCl.cif", f, "chemical/x-cif")},
                headers={"X-API-Key": "secret123"},
            )
        assert right_key.status_code == 200


def test_api_key_not_required_by_default(deployed_checkpoint):
    """The default (no NANOGNN_API_KEY set) must stay auth-free -- this
    is what keeps local development friction-free; hardening is opt-in,
    not a breaking default."""
    from fastapi.testclient import TestClient

    from api.server import API_KEY, app

    assert API_KEY is None  # sanity: nothing in this test env set it
    client = TestClient(app)
    with client:
        with open(NACL_CIF, "rb") as f:
            resp = client.post("/predict", files={"file": ("NaCl.cif", f, "chemical/x-cif")})
        assert resp.status_code == 200


def test_health_reports_auth_status(deployed_checkpoint):
    from fastapi.testclient import TestClient

    from api.server import app

    client = TestClient(app)
    with client:
        body = client.get("/health").json()
        assert "auth_enabled" in body
        assert body["auth_enabled"] is False


def test_require_cif_filename_helper():
    """Unit test for the filename guard directly, rather than trying to
    reproduce a None filename through a real HTTP request: empirically,
    Starlette's own multipart parsing rejects a file part with no
    filename before it ever reaches an UploadFile-typed handler (a file
    part with no filename is parsed as a plain string field instead,
    and FastAPI's validation 422s it) -- confirmed directly against
    both the simple httpx files= API and a hand-built multipart body
    with no filename in the Content-Disposition header. So this is
    real type-correctness and defense-in-depth, not a fix for a bug a
    client could actually trigger over HTTP -- worth being precise
    about which claim this is."""
    from fastapi import HTTPException

    from api.server import _require_cif_filename

    with pytest.raises(HTTPException) as exc_info:
        _require_cif_filename(None)
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        _require_cif_filename("notes.txt")
    assert exc_info.value.status_code == 400

    assert _require_cif_filename("NaCl.cif") == "NaCl.cif"
