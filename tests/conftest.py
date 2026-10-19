"""Test setup. References come from environment variables, never from this repository:

DBS_MODELS_DIR    a models/dbs folder with the files of tools/model_files.sha256 (tools/download_model.py fetches them; 23 MB)
DBS_UPSTREAM_DIR  JusperLee/Dynamic-Band-Splitting checkout at the commit in scripts/vendor_sync.py (the official package)
DBS_COMFYUI_DIR   ComfyUI checkout (v0.38.0 or later)
DBS_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
The tests run the real models on the CPU, on synthetic audio (tests/audio.py), and compare the pack with the official package
(imported from DBS_UPSTREAM_DIR, which needs huggingface_hub and rotary-embedding-torch) in the same process.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("DBS_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("DBS_COMFYUI_DIR")
MODELS = _path("DBS_MODELS_DIR")
UPSTREAM = _path("DBS_UPSTREAM_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (DBS_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (DBS_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def models_dir():
    """The models/dbs folder (holding DBS-TIGER-DnR/ and DBS-BSRNN-DnR/)."""
    return Path(_need(MODELS, "DBS model files (DBS_MODELS_DIR)"))


@pytest.fixture(scope="session")
def upstream_dir():
    return Path(_need(UPSTREAM, "Dynamic-Band-Splitting checkout (DBS_UPSTREAM_DIR)"))


@pytest.fixture(scope="session")
def official(upstream_dir):
    """The official `dbs` package of the upstream checkout (models + inference)."""
    sys.path.insert(0, str(upstream_dir))
    import dbs.engine.inference
    import dbs.models

    return dbs
