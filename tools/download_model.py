"""Download the official DBS models into ComfyUI's models/dbs, and verify them.

    python tools/download_model.py --models-dir <ComfyUI>/models/dbs [--models DBS-TIGER-DnR DBS-BSRNN-DnR] [--verify-only]

Sources (pinned revisions; Apache-2.0, see each model card):
  https://huggingface.co/JusperLee/DBS-TIGER-DnR   4.1 MB
  https://huggingface.co/JusperLee/DBS-BSRNN-DnR   18.9 MB
Result:
  models/dbs/DBS-TIGER-DnR/config.json, model.safetensors, README.md (the model card)
  models/dbs/DBS-BSRNN-DnR/config.json, model.safetensors, README.md
Every file is checked against its SHA-256 in tools/model_files.sha256. Needs `huggingface_hub` (installed with ComfyUI's
requirements). Nothing is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPOS = {
    "DBS-TIGER-DnR": ("JusperLee/DBS-TIGER-DnR", "22da7d4cd3ea63f9365181008bff1c0d2ab6fb2b"),
    "DBS-BSRNN-DnR": ("JusperLee/DBS-BSRNN-DnR", "5af343f57f106fbb95efe8a60f1a3999eeb0aef8"),
}
FILES = ("config.json", "model.safetensors", "README.md")
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{path below models/dbs: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/dbs folder")
    parser.add_argument("--models", nargs="+", choices=list(REPOS), default=list(REPOS), help="models to fetch (default: both)")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    expected = expected_files()
    ok = True
    for model in args.models:
        repo, revision = REPOS[model]
        folder = args.models_dir / model
        for name in FILES:
            target = folder / name
            key = f"{model}/{name}"
            if not args.verify_only and not (target.is_file() and sha256(target) == expected[key]):
                from huggingface_hub import hf_hub_download

                hf_hub_download(repo_id=repo, filename=name, revision=revision, local_dir=str(folder))
            good = target.is_file() and sha256(target) == expected[key]
            ok &= good
            print(f"{'ok      ' if good else ('MISMATCH' if target.is_file() else 'MISSING ')} {key}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
