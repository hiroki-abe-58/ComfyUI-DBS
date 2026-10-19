"""ComfyUI-DBS shim for upstream dbs/hub.py: no Hugging Face Hub mixin.

The model classes keep their upstream base list; the pack builds them from a local config.json and loads model.safetensors
itself (dbs_comfy/runtime.py), so nothing here can download.
"""
from __future__ import annotations

HUB_KWARGS: dict = {}


class PyTorchModelHubMixin:
    pass


__all__ = ["HUB_KWARGS", "PyTorchModelHubMixin"]
