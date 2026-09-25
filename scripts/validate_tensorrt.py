"""
Run this ONCE you have real NVIDIA GPU hardware -- it's the actual test
of the TensorRT execution path in api/engine.py, which could not be run
in the environment this repo was built in (no GPU available there).

It builds a TensorRT engine from the repo's already-trained ONNX model,
then runs the SAME inputs through both TensorRT and ONNX Runtime and
checks they agree. ONNX Runtime's output has already been extensively
validated against raw PyTorch elsewhere in this repo (see
tests/test_onnx_export.py) -- so agreement here is real evidence the
TensorRT path is correct, not just "it didn't crash."

Usage:
    pip install tensorrt pycuda   # in addition to requirements.txt
    python scripts/validate_tensorrt.py

Exits non-zero (with a specific message) on the first mismatch, crash,
or shape-out-of-profile-range case, rather than printing a wall of
tracebacks and leaving you to figure out which check actually failed.
"""

import sys

import numpy as np
from torch_geometric.data import Batch

from api.engine import InferenceEngine, build_feed_dict, build_tensorrt_engine
from src.config import CFG
from src.dataset import cif_to_graph


def _to_feed(batch) -> dict:
    return build_feed_dict(batch)


def main():
    print("Step 1/3: building the TensorRT engine from", CFG.onnx_path)
    try:
        engine_path = build_tensorrt_engine(CFG.onnx_path, CFG.tensorrt_engine_path)
    except Exception as exc:
        sys.exit(
            f"Engine build failed: {exc}\n"
            f"Common causes: no NVIDIA GPU visible, a TensorRT version that "
            f"doesn't support an op in the ONNX graph, or checkpoints/"
            f"nanognn_x.onnx not existing yet (run `python train.py` first)."
        )
    print(f"  -> built {engine_path}")

    print("\nStep 2/3: loading both backends")
    trt_engine = InferenceEngine(prefer_tensorrt=True)
    if trt_engine.backend != "tensorrt":
        sys.exit(
            "InferenceEngine silently fell back to ONNX Runtime instead of "
            "loading the TensorRT engine -- something in _load_tensorrt() "
            "raised (pycuda not installed? no GPU?). Check the stack trace "
            "by calling InferenceEngine._load_tensorrt() directly if this "
            "message alone isn't enough to diagnose it."
        )
    onnx_engine = InferenceEngine(prefer_tensorrt=False)
    print("  -> TensorRT and ONNX Runtime both loaded")

    print("\nStep 3/3: comparing predictions on real structures")

    from tests.conftest import make_synthetic_graph  # reuses the repo's own test fixtures

    test_cases = [
        ("NaCl.cif (real, bundled)", cif_to_graph("data/raw_cifs/NaCl.cif")),
        ("synthetic, 5 atoms", make_synthetic_graph(5, seed=1)),
        ("synthetic, 3 atoms", make_synthetic_graph(3, seed=7)),
        ("synthetic, 12 atoms", make_synthetic_graph(12, seed=99)),
    ]

    all_ok = True
    for label, graph in test_cases:
        batch = Batch.from_data_list([graph])
        feed = _to_feed(batch)

        onnx_out = onnx_engine.predict(feed)
        try:
            trt_out = trt_engine.predict(feed)
        except Exception as exc:
            print(f"  [FAIL] {label}: TensorRT raised {exc}")
            all_ok = False
            continue

        match = np.allclose(onnx_out, trt_out, atol=1e-3)
        status = "OK" if match else "MISMATCH"
        print(f"  [{status}] {label}: onnx={onnx_out.flatten()} trt={trt_out.flatten()}")
        if not match:
            all_ok = False

    print()
    if all_ok:
        print(
            "ALL CHECKS PASSED -- TensorRT output matches the validated ONNX "
            "Runtime baseline. The TensorRT path is genuinely confirmed working "
            'on this hardware, not just "didn\'t crash."'
        )
    else:
        sys.exit(
            "One or more checks failed or mismatched -- do not trust the "
            "TensorRT path yet. Report the [FAIL]/[MISMATCH] lines above."
        )


if __name__ == "__main__":
    main()
