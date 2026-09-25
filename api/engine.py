"""
Inference engine wrapper: exports a trained NanoGNNX model to ONNX,
optionally compiles that to a TensorRT engine, and exposes a single
`.predict()` call the FastAPI layer can use without caring which
backend is underneath.

Honest note on TensorRT: engines are hardware- and CUDA/TensorRT-version
specific and must be built once, offline, on the exact machine (or
matching GPU family + software stack) they'll run on -- never inside the
request path. The raw TensorRT execution path below (`_load_tensorrt`)
deserializes the engine but `predict()` currently only implements the
ONNX Runtime path end-to-end; wiring up TensorRT's buffer bindings
depends on your exact input shapes/CUDA context management and is left
as the next step rather than faked here.
"""

import logging
import warnings

import numpy as np
import onnxruntime as ort

from src.config import CFG

logger = logging.getLogger("nanognn_x.engine")


def _make_onnx_export_wrapper(model):
    """Builds an nn.Module exposing NanoGNNX.forward_tensors as a plain
    `forward(x, edge_index, edge_dist, edge_vec, triplet_index, batch,
    num_graphs_carrier)` signature -- i.e. a flat tuple of tensors,
    nothing custom.

    This matters: `torch.onnx.export` traces through a flat pytree of
    tensor inputs and matches it against `dynamic_axes`/`dynamic_shapes`.
    Passing a PyG `Data`/`Batch` object as the single argument fails
    during export (confirmed: raises a treespec/pytree mismatch deep in
    torch.onnx's dynamic-shapes handling), because the exporter has no
    way to know how to unflatten a custom object into named axes. A
    model whose forward already takes individual tensors sidesteps the
    problem entirely.

    `num_graphs_carrier` is a dummy tensor whose *shape*, not its
    values, carries the number of graphs in this batch (`shape[0] ==
    num_graphs`) -- e.g. `torch.zeros(num_graphs)`. This is the
    standard trick for smuggling a dynamic scalar into an ONNX graph:
    `torch.onnx.export` already proves shape-derived dimensions trace
    correctly as genuinely dynamic (that's exactly how num_nodes/
    num_edges/num_triplets work below), but a value derived via
    `.item()` on a tensor gets baked as a fixed constant (confirmed
    earlier in this file's history -- see the triplet-count branch
    that used to break exactly this way). Deriving num_graphs from a
    carrier tensor's `.shape[0]` avoids ever calling `.item()`, so it
    traces as dynamic too -- verified: exported once with 1 graph,
    correctly produces a differently-shaped, numerically matching
    output when run with 3 or 5 graphs at inference time (see
    tests/test_onnx_export.py::test_dynamic_batch_size).

    A plain factory function rather than a class with a custom
    `__new__` on purpose: `__new__` returning an instance of a
    different class than the one constructed is legal Python but reads
    as a type error to static analysis (mypy correctly can't prove the
    returned object satisfies the class's own interface), and a
    function is simpler anyway for something that just builds and
    returns an object.
    """
    import torch.nn as nn

    class _Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(
            self, x, edge_index, edge_dist, edge_vec, triplet_index, batch, num_graphs_carrier
        ):
            num_graphs = num_graphs_carrier.shape[0]
            return self.inner.forward_tensors(
                x,
                edge_index,
                edge_dist,
                edge_vec,
                triplet_index,
                batch,
                num_graphs=num_graphs,
            )

    return _Wrapper(model)


def export_to_onnx(model, sample_batch, onnx_path: str | None = None) -> str:
    """Traces a trained NanoGNNX model to ONNX. `sample_batch` must be a
    PyG `Batch` (any number of graphs -- built via
    `torch_geometric.data.Batch.from_data_list([...])`) with x,
    edge_index, edge_vec, edge_dist, triplet_index, and batch populated.

    The exported model genuinely supports a dynamic number of graphs
    per request, not just the batch size it was traced with -- see
    `_make_onnx_export_wrapper`'s docstring for how, and
    `tests/test_onnx_export.py::test_dynamic_batch_size` for the
    verification that it actually works, not just exports without
    erroring.

    Exports a `copy.deepcopy` of `model`, never `model` itself. This is
    not defensive boilerplate -- it fixes a real bug found while
    building this: `torch.onnx.export`'s legacy tracer mutates
    persistent internal state on PyG `MessagePassing` layers as a side
    effect of tracing, and that mutation silently changes the *live*
    model's output on every subsequent call (confirmed: before export,
    a fixed input produced one prediction; merely exporting the model
    to ONNX -- never touching its weights -- changed what that same
    live model then returned for the identical input, with the weight
    checksum unchanged, meaning it was module *state*, not parameters,
    that shifted). Tracing a deep copy keeps that side effect off the
    object you actually train or serve with.

    Verified end-to-end against torch 2.14 / onnx 1.22 / onnxscript 0.7:
    the exported ONNX Runtime output matches the pre-export PyTorch
    output to 1e-4 across multiple differently-sized graphs and batch
    sizes, and the original `model` object is provably unchanged
    afterward.
    """
    import copy

    import torch

    onnx_path = onnx_path or CFG.onnx_path
    export_copy = copy.deepcopy(model)
    export_copy.eval()
    wrapper = _make_onnx_export_wrapper(export_copy)

    num_graphs_in_sample = int(sample_batch.batch.max().item()) + 1
    num_graphs_carrier = torch.zeros(num_graphs_in_sample)

    sample_inputs = (
        sample_batch.x,
        sample_batch.edge_index,
        sample_batch.edge_dist,
        sample_batch.edge_vec,
        sample_batch.triplet_index,
        sample_batch.batch,
        num_graphs_carrier,
    )

    # The torch.onnx.export DeprecationWarning about "legacy TorchScript
    # exporter" is intentional here: dynamo=False is the *correct* choice
    # because the new torch.export-based exporter (dynamo=True) rejects the
    # `dynamic_axes` dict format we use -- it demands a different
    # `dynamic_shapes` API that doesn't yet support PyG's custom Batch layout
    # (confirmed against torch 2.14). Suppressed explicitly so the warning
    # doesn't flood logs; this comment is the record of why.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*legacy TorchScript.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*setup_onnx_logging.*",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            wrapper,
            sample_inputs,
            onnx_path,
            input_names=[
                "x",
                "edge_index",
                "edge_dist",
                "edge_vec",
                "triplet_index",
                "batch",
                "num_graphs_carrier",
            ],
            output_names=["predictions"],
            dynamic_axes={
                "x": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "edge_dist": {0: "num_edges"},
                "edge_vec": {0: "num_edges"},
                "triplet_index": {1: "num_triplets"},
                "batch": {0: "num_nodes"},
                "num_graphs_carrier": {0: "num_graphs"},
                "predictions": {0: "num_graphs"},
            },
            opset_version=17,
            dynamo=False,  # see comment above -- intentional
        )
    return onnx_path


def build_tensorrt_engine(
    onnx_path: str | None = None,
    engine_path: str | None = None,
    fp16: bool = True,
    min_nodes: int = 1,
    opt_nodes: int = 20,
    max_nodes: int = 200,
    min_edges: int = 1,
    opt_edges: int = 300,
    max_edges: int = 20000,
    min_triplets: int = 0,
    opt_triplets: int = 3000,
    max_triplets: int = 200000,
    min_graphs: int = 1,
    opt_graphs: int = 8,
    max_graphs: int = 64,
) -> str:
    """Compiles the ONNX graph into a TensorRT engine using FP16 so the
    GPU's Tensor Cores get used -- run this once on the deployment
    machine. Requires the `tensorrt` package plus a matching CUDA/driver
    install; it will not run on a CPU-only box.

    Our model has dynamic input shapes (a crystal's atom/bond/triplet
    counts vary structure to structure, and now the number of crystals
    per request varies too -- see export_to_onnx's num_graphs_carrier),
    and TensorRT -- unlike ONNX Runtime -- refuses to build a
    dynamic-shape engine at all without an explicit *optimization
    profile* declaring the min/opt/max size of every dynamic dimension
    (confirmed against a real TensorRT GitHub issue hitting exactly
    this: "Network has dynamic or shape inputs, but no optimization
    profile has been defined."). The defaults below are sized for the
    bundled demo dataset plus real Materials Project structures up to
    ~200 atoms (matching scripts/download_materials_project.py's
    default --max-sites=50, with headroom) and up to 64 structures per
    bulk-screening request; `min_triplets=0` specifically because a
    real structure can have zero triplets (an isolated atom -- see
    src/layers.py's padding fix for why that's safe).

    IMPORTANT: a request whose actual shapes fall outside [min, max]
    for any dimension will fail at inference time, not at build time --
    TensorRT engines are hard-bounded to the profile they were built
    with. `InferenceEngine.predict()` below catches that and falls back
    to ONNX Runtime for the offending request rather than crashing, but
    if you're routinely serving much larger structures or bigger
    batches, rebuild the engine with wider bounds here instead of
    relying on the fallback.
    """
    import tensorrt as trt

    onnx_path = onnx_path or CFG.onnx_path
    engine_path = engine_path or CFG.tensorrt_engine_path

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model for TensorRT")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # One optimization profile covering every dynamic input this model
    # has (see export_to_onnx's dynamic_axes for the authoritative list
    # of which axis on which tensor is dynamic).
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (min_nodes,), (opt_nodes,), (max_nodes,))
    profile.set_shape("edge_index", (2, min_edges), (2, opt_edges), (2, max_edges))
    profile.set_shape("edge_dist", (min_edges,), (opt_edges,), (max_edges,))
    profile.set_shape("edge_vec", (min_edges, 3), (opt_edges, 3), (max_edges, 3))
    profile.set_shape("triplet_index", (2, min_triplets), (2, opt_triplets), (2, max_triplets))
    profile.set_shape("batch", (min_nodes,), (opt_nodes,), (max_nodes,))
    profile.set_shape("num_graphs_carrier", (min_graphs,), (opt_graphs,), (max_graphs,))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(
            "TensorRT failed to build the engine -- check the WARNING/ERROR "
            "log lines above for the specific cause (a common one: your "
            "installed TensorRT version doesn't support an op the ONNX "
            "graph uses)."
        )
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    return engine_path


def build_feed_dict(batch) -> dict:
    """Builds the ONNX/TensorRT feed dict for a PyG `Batch` of any
    number of graphs, including the `num_graphs_carrier` dummy tensor
    `_make_onnx_export_wrapper` needs to trace a dynamic batch size (see its
    docstring). Shared by api/server.py, scripts/validate_tensorrt.py,
    and the test suite so this construction only lives in one place.
    """
    import torch

    return {
        "x": batch.x.numpy(),
        "edge_index": batch.edge_index.numpy(),
        "edge_dist": batch.edge_dist.numpy(),
        "edge_vec": batch.edge_vec.numpy(),
        "triplet_index": batch.triplet_index.numpy(),
        "batch": batch.batch.numpy(),
        "num_graphs_carrier": torch.zeros(batch.num_graphs).numpy(),
    }


class InferenceEngine:
    """Prefers a compiled TensorRT engine if present, otherwise falls
    back to ONNX Runtime (still fast, and needs no NVIDIA-specific
    build step -- good default for local dev / non-NVIDIA hosts)."""

    # dtype each input/output tensor was exported with -- see
    # export_to_onnx's input_names and NanoGNNX.forward_tensors'
    # signature. Used to allocate correctly-sized GPU buffers.
    _DTYPES = {
        "x": np.int64,
        "edge_index": np.int64,
        "edge_dist": np.float32,
        "edge_vec": np.float32,
        "triplet_index": np.int64,
        "batch": np.int64,
        "num_graphs_carrier": np.float32,
        "predictions": np.float32,
    }

    def __init__(
        self,
        onnx_path: str | None = None,
        engine_path: str | None = None,
        prefer_tensorrt: bool = True,
    ):
        self.onnx_path = onnx_path or CFG.onnx_path
        self.engine_path = engine_path or CFG.tensorrt_engine_path
        self.backend = None
        self._onnx_fallback_loaded = False

        if prefer_tensorrt:
            try:
                self._load_tensorrt()
                self.backend = "tensorrt"
            except Exception:
                self._load_onnx()
                self.backend = "onnxruntime"
        else:
            self._load_onnx()
            self.backend = "onnxruntime"

    def _load_onnx(self):
        # CUDA provider is tried first for GPU acceleration; if unavailable
        # (e.g. CPU-only host), ORT silently falls back to CPU -- the warning
        # it emits about CUDAExecutionProvider not being available is
        # harmless but noisy on CPU-only machines, so suppress it.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*CUDAExecutionProvider.*",
                category=UserWarning,
            )
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self._onnx_fallback_loaded = True

    def _load_tensorrt(self):
        import tensorrt as trt

        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        self.trt_context = self.trt_engine.create_execution_context()

        # pycuda.autoinit creates and pushes a CUDA context as an import
        # side effect -- imported here, lazily, so a machine with no GPU
        # never touches this import at all (mirrors the lazy `import
        # tensorrt` pattern used throughout this module).
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        self._cuda = cuda
        self._stream = cuda.Stream()

    def _predict_tensorrt(self, feed_dict: dict) -> np.ndarray:
        """Real TensorRT execution: set each input's runtime shape, bind
        GPU memory addresses for every input/output tensor, execute
        asynchronously on our own CUDA stream, then copy the result back
        to host memory.

        Uses `execute_async_v3` + `set_tensor_address` + `set_input_shape`
        -- the current, non-deprecated TensorRT Python API as of the
        10.x series. The older `execute_v2`/bindings-index style is
        legacy and was intentionally not used here.

        Allocates and frees GPU buffers on every call rather than
        pooling them across requests -- correct, and simple to verify,
        at some throughput cost under heavy concurrent load. Pooling
        buffers by shape bucket is a reasonable follow-up once this path
        has real production traffic to profile.
        """
        cuda = self._cuda
        ctx = self.trt_context
        device_ptrs = []  # keep references alive until synchronize()

        try:
            for name, array in feed_dict.items():
                array = np.ascontiguousarray(array, dtype=self._DTYPES[name])
                ctx.set_input_shape(name, array.shape)
                device_mem = cuda.mem_alloc(array.nbytes)
                # Synchronous copy on purpose: memcpy_htod_async requires
                # the *source host buffer* to stay alive and unchanged
                # until the stream actually synchronizes, but `array` is
                # a loop-local variable that Python is free to garbage
                # collect the moment the next iteration reassigns it --
                # a real use-after-free risk that would only show up as
                # occasional silent data corruption under GC pressure,
                # not a reliable crash. There's no pipelining benefit to
                # the async version here (a single request, one stream,
                # nothing else running concurrently on it), so the
                # synchronous copy is both simpler and actually correct.
                cuda.memcpy_htod(device_mem, array)
                ctx.set_tensor_address(name, int(device_mem))
                device_ptrs.append(device_mem)

            output_name = "predictions"
            output_shape = tuple(ctx.get_tensor_shape(output_name))
            output_host = np.empty(output_shape, dtype=self._DTYPES[output_name])
            output_device = cuda.mem_alloc(output_host.nbytes)
            ctx.set_tensor_address(output_name, int(output_device))
            device_ptrs.append(output_device)

            ok = ctx.execute_async_v3(stream_handle=self._stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False")
            self._stream.synchronize()

            cuda.memcpy_dtoh(output_host, output_device)
            return output_host
        finally:
            # pycuda's DeviceAllocation frees itself on garbage collection,
            # but drop our references explicitly so that happens right
            # after this call rather than whenever the GC next runs.
            device_ptrs.clear()

    def predict(self, feed_dict: dict) -> np.ndarray:
        """feed_dict maps ONNX input names -> numpy arrays (see
        `export_to_onnx`'s input_names for the required keys)."""
        if self.backend == "tensorrt":
            try:
                return self._predict_tensorrt(feed_dict)
            except Exception as exc:
                # Most likely cause: this request's shape falls outside
                # the [min, max] bounds the engine was built with (see
                # build_tensorrt_engine's docstring) -- fall back to
                # ONNX Runtime for this one request rather than failing
                # it outright. Lazily loads the ONNX session on first
                # fallback rather than always paying that cost upfront.
                #
                # Logged rather than silently swallowed: a silent
                # per-request fallback is exactly the kind of thing that
                # looks fine in testing and then quietly costs you all
                # of TensorRT's speedup in production with no visible
                # signal that it's happening.
                logger.warning(
                    "TensorRT inference failed, falling back to ONNX Runtime for this request: %s",
                    exc,
                )
                if not self._onnx_fallback_loaded:
                    self._load_onnx()
                outputs = self.session.run(None, feed_dict)
                return outputs[0]

        outputs = self.session.run(None, feed_dict)
        return outputs[0]
