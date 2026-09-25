"""
Global configuration and hyperparameters for NanoGNN-X.

Centralizing these here means every module (dataset, model, training,
serving) reads from a single source of truth instead of magic numbers
scattered across files.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # --- Graph construction ---
    r_cut: float = 5.0  # Å, neighbor cutoff radius for edges
    max_neighbors: int = 32  # cap neighbors per atom, bounds triplet count (O(n^2))

    # --- Featurization ---
    num_rbf_dist: int = 64  # dims for the distance Bessel radial basis
    num_rbf_angle: int = 64  # Chebyshev orders for the angle basis (0..num_rbf_angle-1)

    # --- Model architecture ---
    num_elements: int = 119  # embedding table size, covers all known elements (Z 1-118)
    hidden_dim: int = 64
    num_mp_layers: int = 4  # message-passing blocks
    mlp_hidden_dim: int = 128
    dropout: float = 0.1

    # --- Training ---
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 300
    lr_patience: int = 5  # ReduceLROnPlateau patience (epochs)
    lr_factor: float = 0.5
    huber_delta: float = 1.0
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    seed: int = 42

    # --- Paths ---
    raw_cif_dir: str = str(ROOT_DIR / "data/raw_cifs")
    processed_dir: str = str(ROOT_DIR / "data/processed")
    checkpoint_dir: str = str(ROOT_DIR / "checkpoints")
    onnx_path: str = str(ROOT_DIR / "checkpoints/nanognn_x.onnx")
    tensorrt_engine_path: str = str(ROOT_DIR / "checkpoints/nanognn_x.trt")
    normalizer_path: str = str(ROOT_DIR / "checkpoints/target_normalizer.json")


CFG = Config()
