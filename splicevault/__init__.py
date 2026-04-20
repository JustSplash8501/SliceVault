"""SpliceVault package."""

from .classify import benchmark, load_model, predict_samples, train_classifier
from .psi import compute_psi_matrix, parse_junction_file
from .signatures import build_signature_matrix, save_signature_matrix

__all__ = [
    "benchmark",
    "build_signature_matrix",
    "compute_psi_matrix",
    "load_model",
    "parse_junction_file",
    "predict_samples",
    "save_signature_matrix",
    "train_classifier",
]

__version__ = "0.1.0"
