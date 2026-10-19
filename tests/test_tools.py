"""The download tool's manifest covers every file the nodes need, its pins match the documentation, and its verification reports
what it finds."""

import hashlib
import importlib.util
from pathlib import Path

from dbs_comfy import runtime

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_model", ROOT / "tools" / "download_model.py")
download_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_model)


def test_manifest_covers_both_models_and_the_files_the_nodes_load():
    files = download_model.expected_files()
    assert set(download_model.REPOS) == set(runtime.MODELS)
    assert set(runtime.FILES) <= set(download_model.FILES)
    assert set(files) == {f"{m}/{f}" for m in download_model.REPOS for f in download_model.FILES}
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in files.values())
    upstream_md = (ROOT / "dbs_comfy" / "vendor" / "UPSTREAM.md").read_text(encoding="utf-8")
    for model, (repo, revision) in download_model.REPOS.items():
        assert repo.endswith(model) and revision in upstream_md and files[f"{model}/model.safetensors"] in upstream_md


def test_the_real_model_folder_verifies(models_dir, capsys):
    assert download_model.main(["--models-dir", str(models_dir), "--verify-only"]) == 0
    assert "all files verified" in capsys.readouterr().out


def test_verify_only_reports_missing_and_mismatching_files(tmp_path, capsys, monkeypatch):
    manifest = tmp_path / "model_files.sha256"
    folder = tmp_path / "models" / "DBS-TIGER-DnR"
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(b"x")
    (folder / "README.md").write_bytes(b"not the card")
    manifest.write_text(f"# test\n{hashlib.sha256(b'x').hexdigest()}  DBS-TIGER-DnR/config.json\n{'0' * 64}  DBS-TIGER-DnR/README.md\n"
                        f"{'1' * 64}  DBS-TIGER-DnR/model.safetensors\n", encoding="ascii")
    monkeypatch.setattr(download_model, "MANIFEST", manifest)
    assert download_model.main(["--models-dir", str(tmp_path / "models"), "--models", "DBS-TIGER-DnR", "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert "ok       DBS-TIGER-DnR/config.json" in out and "MISMATCH DBS-TIGER-DnR/README.md" in out
    assert "MISSING  DBS-TIGER-DnR/model.safetensors" in out
