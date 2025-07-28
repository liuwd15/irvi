"""Multimodal Transformer-based VAE for gene expression and TCR data."""

__version__ = "0.1.0"

# Define exports
__all__ = [
    "IRVI",
    "IRVAE",
]

# Import main classes
from .model import IRVI
from .module import IRVAE
