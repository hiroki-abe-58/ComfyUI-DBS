"""The vendored code is the pinned upstream commit byte for byte (plus the documented shims), and the manifest is current."""

import hashlib
import subprocess

import vendor_sync

from dbs_comfy import runtime


def test_vendor_tree_matches_the_pinned_upstream_commit(upstream_dir):
    head = subprocess.run(["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == vendor_sync.UPSTREAM_COMMIT
    for rel, data in vendor_sync.expected_tree(upstream_dir).items():
        assert (vendor_sync.VENDOR / rel).read_bytes() == data, rel


def test_manifest_lists_every_vendored_file_with_its_hash():
    lines = (vendor_sync.VENDOR / "MANIFEST.sha256").read_text(encoding="ascii").splitlines()
    listed = dict(reversed(line.split("  ", 1)) for line in lines)
    expected = set(vendor_sync.COPIED) | set(vendor_sync.SHIMS) | set(vendor_sync.LICENSE_FILES)
    assert set(listed) == expected
    for rel, digest in listed.items():
        assert hashlib.sha256((vendor_sync.VENDOR / rel).read_bytes()).hexdigest() == digest, rel
    on_disk = {p.relative_to(vendor_sync.VENDOR).as_posix() for p in vendor_sync.VENDOR.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    assert on_disk == expected | {"MANIFEST.sha256", "UPSTREAM.md", "__init__.py"}


def test_shims_import_no_hub_and_no_training_code():
    text = "\n".join((vendor_sync.VENDOR / rel).read_text(encoding="utf-8") for rel in vendor_sync.SHIMS)
    for name in ("huggingface_hub", "lightning", "omegaconf", "roformer", "import_module"):
        assert name not in text
    assert runtime.MODELS["DBS-TIGER-DnR"].cls.__module__.endswith("vendor.dbs.models.tiger")


def test_upstream_md_names_the_pinned_commit_and_license_hashes():
    text = (vendor_sync.VENDOR / "UPSTREAM.md").read_text(encoding="utf-8")
    assert vendor_sync.UPSTREAM_COMMIT in text
    for name in vendor_sync.LICENSE_FILES:
        assert hashlib.sha256((vendor_sync.VENDOR / name).read_bytes()).hexdigest() in text
