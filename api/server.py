"""
FastAPI serving layer. Upload a .cif file (or several, via
/predict/batch), get bandgap + formation energy predictions back.

Run: uvicorn api.server:app --host 0.0.0.0 --port 8000

Basic hardening (both off by default for easy local dev, on by an
environment variable for anything exposed past localhost):
  NANOGNN_API_KEY            -- if set, requests must send a matching
                                 X-API-Key header
  NANOGNN_MAX_UPLOAD_BYTES   -- per-file upload size cap (default 5 MB
                                 -- real .cif files are text and a few
                                 KB to a few hundred KB; 5 MB is already
                                 generous headroom, not a tight limit)
  NANOGNN_MAX_BATCH_FILES    -- max files per /predict/batch request
                                 (default 64, matching
                                 build_tensorrt_engine's default
                                 max_graphs bound)
"""

import logging
import os
import secrets
import tempfile
import time
import warnings
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torch_geometric.data import Batch

from api.engine import InferenceEngine, build_feed_dict
from src.config import CFG
from src.dataset import cif_to_graph
from src.normalizer import TargetNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("nanognn_x.api")

MAX_UPLOAD_BYTES = int(os.environ.get("NANOGNN_MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
MAX_BATCH_FILES = int(os.environ.get("NANOGNN_MAX_BATCH_FILES", 64))
API_KEY = os.environ.get("NANOGNN_API_KEY")  # None -> auth disabled

engine: InferenceEngine | None = None
normalizer: TargetNormalizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # `@app.on_event("startup")` is deprecated as of the FastAPI version
    # this was tested against; this `lifespan` context manager is the
    # currently-recommended replacement.
    global engine, normalizer
    # prefer_tensorrt=True will silently fall back to ONNX Runtime if no
    # compiled .trt engine is present -- see api/engine.py
    engine = InferenceEngine(prefer_tensorrt=True)
    logger.info("Inference engine loaded (backend=%s)", engine.backend)

    # The model predicts *normalized* targets (see src/normalizer.py and
    # train.py) -- without inverse-transforming here, /predict would
    # silently return z-scored numbers instead of eV, which used to be
    # exactly what this endpoint did before normalization was added.
    if os.path.exists(CFG.normalizer_path):
        normalizer = TargetNormalizer.load(CFG.normalizer_path)
        logger.info("Target normalizer loaded from %s", CFG.normalizer_path)
    else:
        normalizer = None
        logger.warning(
            "No target normalizer found at %s -- /predict will return raw "
            "(likely normalized-scale) model output. Run train.py first.",
            CFG.normalizer_path,
        )

    if API_KEY:
        logger.info("API key auth ENABLED (NANOGNN_API_KEY is set)")
    else:
        logger.warning(
            "API key auth DISABLED (NANOGNN_API_KEY not set) -- fine for "
            "local development, set it before exposing this past localhost."
        )
    yield


app = FastAPI(title="NanoGNN-X Inference API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(x_api_key: str | None = Header(default=None)):
    """A no-op when NANOGNN_API_KEY isn't set, so local dev needs no
    setup. Once set, every request must send a matching X-API-Key
    header. This is deliberately simple (one shared key, no per-user
    accounts or rotation) -- enough to keep a casual public endpoint
    from being freely scraped or abused, not a substitute for a real
    auth system if this ever needs multi-tenant access control."""
    if API_KEY is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "Missing or invalid X-API-Key header")


async def _read_upload_capped(file: UploadFile) -> bytes:
    """Reads an upload in chunks, rejecting it the moment it exceeds
    MAX_UPLOAD_BYTES rather than buffering an arbitrarily large body
    into memory first and checking after the fact -- a client can lie
    about (or omit) Content-Length, so the real limit has to be
    enforced while reading, not just checked against a header."""
    chunks = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _cif_bytes_to_graph(contents: bytes, filename: str):
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        return cif_to_graph(tmp_path)
    finally:
        # The temp file used to leak on every request (never deleted) --
        # clean it up regardless of success or failure.
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _predictions_to_physical_units(raw: np.ndarray) -> np.ndarray:
    if normalizer is None:
        return raw
    return normalizer.inverse_transform(torch.from_numpy(raw)).numpy()


def _get_engine() -> InferenceEngine:
    """Narrows `engine` from `InferenceEngine | None` to `InferenceEngine`
    for both mypy and actual runtime safety. `engine` is only ever None
    before `lifespan`'s startup has run, which shouldn't be reachable
    from a request handler in practice -- but "shouldn't be reachable"
    and "provably can't happen" are different claims, and a clear
    RuntimeError here beats an AttributeError several frames deep in
    whichever line happened to touch `engine` first."""
    if engine is None:
        raise RuntimeError("Inference engine not loaded -- lifespan startup hasn't run yet")
    return engine


def _require_cif_filename(filename: str | None) -> str:
    """A real bug fix, not just a type-checker appeasement: `UploadFile
    .filename` is `str | None` per the multipart spec (a client can
    omit it), and calling `.lower()` on it unguarded used to raise an
    unhandled AttributeError -- a 500 -- instead of the clean 400 this
    is supposed to be for any malformed upload."""
    if filename is None or not filename.lower().endswith(".cif"):
        raise HTTPException(400, f"Please upload a .cif file (got {filename!r})")
    return filename


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...)):
    filename = _require_cif_filename(file.filename)

    start = time.perf_counter()
    try:
        contents = await _read_upload_capped(file)
        graph = _cif_bytes_to_graph(contents, filename)
        batch = Batch.from_data_list([graph])
        prediction = np.asarray(_get_engine().predict(build_feed_dict(batch))).reshape(1, -1)
        prediction = _predictions_to_physical_units(prediction)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed to process upload %r: %s", filename, exc)
        raise HTTPException(422, f"Failed to process structure: {exc}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    bandgap, formation_energy = prediction.reshape(-1)[:2]
    logger.info(
        "predict %r -> bandgap=%.4f formation_energy=%.4f (%.2f ms, backend=%s)",
        filename,
        bandgap,
        formation_energy,
        elapsed_ms,
        _get_engine().backend,
    )
    return {
        "bandgap_eV": float(bandgap),
        "formation_energy_eV_per_atom": float(formation_energy),
        "inference_time_ms": round(elapsed_ms, 3),
        "backend": _get_engine().backend,
        "normalized": normalizer is None,
    }


@app.post("/predict/batch", dependencies=[Depends(require_api_key)])
async def predict_batch(files: list[UploadFile] = File(...)):
    """Runs several crystals through the model in a single forward
    pass, using the same dynamic-batch-size ONNX export the single-file
    /predict endpoint uses (see api/engine.py::export_to_onnx). This is
    where genuine bulk-screening throughput comes from -- one batched
    call instead of N separate round trips through the model."""
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            400,
            f"Batch of {len(files)} files exceeds the {MAX_BATCH_FILES}-file limit "
            f"per request (set NANOGNN_MAX_BATCH_FILES to change it)",
        )
    filenames = [_require_cif_filename(f.filename) for f in files]

    start = time.perf_counter()
    try:
        graphs = []
        for f, filename in zip(files, filenames, strict=True):
            contents = await _read_upload_capped(f)
            graphs.append(_cif_bytes_to_graph(contents, filename))
        batch = Batch.from_data_list(graphs)
        predictions = np.asarray(_get_engine().predict(build_feed_dict(batch))).reshape(
            len(files), -1
        )
        predictions = _predictions_to_physical_units(predictions)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed to process batch of %d files: %s", len(files), exc)
        raise HTTPException(422, f"Failed to process batch: {exc}") from exc
    elapsed_ms = (time.perf_counter() - start) * 1000

    results = [
        {
            "filename": filename,
            "bandgap_eV": float(predictions[i, 0]),
            "formation_energy_eV_per_atom": float(predictions[i, 1]),
        }
        for i, filename in enumerate(filenames)
    ]
    logger.info(
        "predict_batch: %d structures in %.2f ms (%.2f ms/structure, backend=%s)",
        len(files),
        elapsed_ms,
        elapsed_ms / len(files),
        _get_engine().backend,
    )
    return {
        "results": results,
        "count": len(files),
        "inference_time_ms": round(elapsed_ms, 3),
        "backend": _get_engine().backend,
        "normalized": normalizer is None,
    }


@app.post("/explain", dependencies=[Depends(require_api_key)])
async def explain(file: UploadFile = File(...)):
    filename = _require_cif_filename(file.filename)
    try:
        contents = await _read_upload_capped(file)
        graph = _cif_bytes_to_graph(contents, filename)
        
        base_batch = Batch.from_data_list([graph])
        base_pred = np.asarray(_get_engine().predict(build_feed_dict(base_batch))).reshape(1, -1)
        base_pred = _predictions_to_physical_units(base_pred)[0] 
        
        num_atoms = graph.num_nodes
        contributions = []
        
        # Ablation: zero out each atom's Z embedding one by one
        for i in range(num_atoms):
            z_ablated = graph.x.clone()
            z_ablated[i] = 0  # Dummy empty element
            
            ablated_graph = graph.clone()
            ablated_graph.x = z_ablated
            
            batch = Batch.from_data_list([ablated_graph])
            pred = np.asarray(_get_engine().predict(build_feed_dict(batch))).reshape(1, -1)
            pred = _predictions_to_physical_units(pred)[0]
            
            # Absolute difference in bandgap prediction
            diff = np.abs(base_pred[0] - pred[0]) 
            contributions.append(float(diff))
            
        max_c = max(contributions) if contributions and max(contributions) > 0 else 1.0
        normalized_heatmap = [c / max_c for c in contributions]
        
        max_idx = int(np.argmax(contributions)) if contributions else 0
        
        from pymatgen.core import Structure
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
            
        key_element = "Unknown"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                struct = Structure.from_file(tmp_path)
                key_element = struct[max_idx].specie.name
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
        justification = f"The model's prediction is strongly driven by the {key_element} atoms (highlighted in red) in this geometry."
        
        return {
            "heatmap": normalized_heatmap,
            "justification": justification
        }
    except Exception as exc:
        raise HTTPException(422, f"Failed to generate explanation: {exc}") from exc


@app.get("/dataset")
def get_dataset(limit: int = 500):
    import csv
    results = []
    try:
        csv_path = os.path.join(CFG.raw_cif_dir, "targets.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= limit: break
                    results.append(row)
    except Exception as e:
        logger.warning("Could not read targets.csv: %s", e)
    return {"data": results}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": engine.backend if engine else "not loaded",
        "normalizer_loaded": normalizer is not None,
        "auth_enabled": API_KEY is not None,
    }
