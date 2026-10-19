"""Rebuild dbs_comfy/vendor/dbs from an upstream checkout and rewrite vendor/MANIFEST.sha256 (maintainer tool).

    python scripts/vendor_sync.py --upstream <JusperLee/Dynamic-Band-Splitting checkout at UPSTREAM_COMMIT> [--check]

Only the files that inference with the two released DnR v2 models needs are vendored, all of them byte for byte (COPIED). SHIMS
are this pack's own small files replacing upstream ones:
- `dbs/__init__.py`, `dbs/engine/__init__.py`: package markers without the training imports (PyTorch Lightning, OmegaConf);
- `dbs/models/__init__.py`: only DBS-BSRNN and DBS-TIGER (the RoFormer models need rotary-embedding-torch and have no released
  DnR weights);
- `dbs/hub.py`: no Hugging Face Hub mixin; this pack loads config.json and model.safetensors from a local folder itself, so the
  vendored code contains no download path.
tests/test_vendor.py rebuilds the tree from the pinned upstream commit in CI and compares every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "dbs_comfy" / "vendor"
UPSTREAM_COMMIT = "ff526463a58ee0278e7bd05c4b8bce112b743e34"

COPIED = [
    "dbs/models/bsrnn.py", "dbs/models/tiger.py", "dbs/modules/__init__.py", "dbs/modules/band_split.py",
    "dbs/modules/dbs.py", "dbs/modules/stft.py", "dbs/engine/inference.py",
]
SHIMS = ["dbs/__init__.py", "dbs/engine/__init__.py", "dbs/models/__init__.py", "dbs/hub.py"]
LICENSE_FILES = ["LICENSE", "THIRD_PARTY.md"]


def expected_tree(upstream: Path) -> dict[str, bytes]:
    out = {rel: (upstream / rel).read_bytes() for rel in COPIED}
    for name in LICENSE_FILES:
        out[name] = (upstream / name).read_bytes()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="only compare, write nothing")
    args = ap.parse_args(argv)
    tree = expected_tree(args.upstream)
    bad = [rel for rel, data in tree.items() if not (VENDOR / rel).is_file() or (VENDOR / rel).read_bytes() != data]
    if args.check:
        print("\n".join(f"DIFFERS {rel}" for rel in bad) or "vendor tree matches the upstream checkout")
        return 1 if bad else 0
    for rel in bad:
        (VENDOR / rel).parent.mkdir(parents=True, exist_ok=True)
        (VENDOR / rel).write_bytes(tree[rel])
        print("wrote", rel)
    names = sorted(set(tree) | set(SHIMS))
    lines = [f"{hashlib.sha256((VENDOR / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (VENDOR / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"MANIFEST.sha256: {len(lines)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
