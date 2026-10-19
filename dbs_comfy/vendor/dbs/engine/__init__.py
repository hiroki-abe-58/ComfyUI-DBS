"""Inference protocols only (ComfyUI-DBS shim; the upstream package also exports the training engine)."""
from .inference import separate

__all__ = ["separate"]
