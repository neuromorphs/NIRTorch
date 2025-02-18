from importlib.metadata import version as metadata_version, PackageNotFoundError

try:
    __version__ = version = metadata_version("nirtorch")
    del metadata_version
except PackageNotFoundError:
    # package is not installed
    pass

from .from_nir import load  # noqa F401
from .to_nir import extract_nir_graph  # noqa F401
from .nir_interpreter import nir_to_torch
from .torch_tracer import torch_to_nir

__all__ = [
    "extract_torch_graph",
    "extract_nir_graph",
    "load",
    "nir_to_torch",
    "torch_to_nir",
]
