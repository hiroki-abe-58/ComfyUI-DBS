"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server with this node pack installed, put a
synthetic soundtrack and a short video into its input folder, and run the shipped workflows through POST /prompt with the real
DBS models.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --models-dir <models/dbs> --upstream <Dynamic-Band-Splitting
                                     checkout> [--workdir DIR] [--keep-pack]

The models are small and run on the CPU, so CI runs the real nodes and the real models (tools/download_model.py fetches them).
The check proves that the pack installs into a stock ComfyUI and that through the queue
- Load Audio -> DBS Loader -> DBS Separate -> Save Audio (FLAC) gives the official package's stems for the same file (the README
  usage, to the 16 bits of the FLAC), on the channel mean and per channel, and Merge Audio of music + sfx gives the official
  music + sfx (a soundtrack without dialogue);
- Load Video -> Get Video Components -> DBS Separate -> Merge Audio -> Create Video -> Save Video writes a video with the new
  soundtrack and the dialogue as FLAC;
- the other model gives other audio, a repeated job gives the same audio, and invalid inputs are rejected.
The server is started and stopped by this script only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import audio  # noqa: E402
import queue_client  # noqa: E402

NODE_IDS = {"DBSLoader", "DBSSeparate"}
STEP = 1 / 32768  # one 16-bit step: ComfyUI's FLAC is 16 bits
PROTOCOLS = {"DBS-TIGER-DnR": ("DBSTIGER", {"mode": "chunked", "chunk_seconds": 6.0, "hop_seconds": 3.0}),
             "DBS-BSRNN-DnR": ("DBSBSRNN", {"mode": "chunked", "chunk_seconds": 3.0, "hop_seconds": 1.5})}


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def write_clip(path: Path, seconds: float = 3.0, fps: int = 24) -> None:
    """A small MPEG-4 video with a stereo AAC soundtrack (PyAV, which ComfyUI requires)."""
    import av

    sound = audio.soundtrack(seconds, seed=21)  # [samples, 2]
    with av.open(str(path), "w", format="mp4") as f:
        video = f.add_stream("mpeg4", rate=fps)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        aud = f.add_stream("aac", rate=44100, layout="stereo")
        for i in range(int(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), (i * 7) % 256, dtype=np.uint8), format="rgb24")
            for packet in video.encode(frame):
                f.mux(packet)
        for packet in video.encode():
            f.mux(packet)
        planar = np.ascontiguousarray(sound.T)
        for start in range(0, planar.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(planar[:, start:start + 1024], format="fltp", layout="stereo")
            frame.sample_rate = 44100
            for packet in aud.encode(frame):
                f.mux(packet)
        for packet in aud.encode():
            f.mux(packet)


def main(argv=None) -> int:
    import soundfile as sf
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True, help="a models/dbs folder (holding DBS-TIGER-DnR/, DBS-BSRNN-DnR/)")
    parser.add_argument("--upstream", type=Path, required=True, help="JusperLee/Dynamic-Band-Splitting checkout (the official package)")
    parser.add_argument("--port", type=int, default=8198)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-pack", action="store_true", help="use the pack already installed in custom_nodes (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    models = args.models_dir.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="dbs-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = comfy / "custom_nodes" / "ComfyUI-DBS"
    if pack.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(pack):  # never delete through a link into a working copy
        raise SystemExit(f"{pack} is a link; use a ComfyUI checkout without it")
    if not args.keep_pack:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--keep-pack: {pack} is not installed")
    inputs, output = work / "input", work / "output"
    inputs.mkdir(exist_ok=True)
    sf.write(inputs / "soundtrack.wav", audio.soundtrack(7.0, seed=20), 44100, subtype="FLOAT")  # the names the workflows use
    write_clip(inputs / "clip.mp4")
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"dbs_ci:\n  base_path: {models.parent.as_posix()}\n  dbs: {models.name}\n", encoding="utf-8")

    sys.path.insert(0, str(args.upstream.resolve()))
    import dbs.engine.inference
    import dbs.models

    x, _ = sf.read(inputs / "soundtrack.wav", dtype="float32", always_2d=True)

    def official(name, signal):
        cls, protocol = PROTOCOLS[name]
        model = getattr(dbs.models, cls).from_pretrained(str(models / name)).eval()
        with torch.inference_mode():
            mixture = torch.from_numpy(np.ascontiguousarray(signal)).view(1, 1, -1)
            return dbs.engine.inference.separate(model, mixture, protocol, sample_rate=44100)[0].numpy()  # [3, samples]

    mono = official("DBS-TIGER-DnR", x.mean(1))
    per_channel = np.stack([official("DBS-TIGER-DnR", x[:, c]) for c in range(x.shape[1])], axis=2)  # [3, samples, channels]

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
         "--extra-model-paths-config", str(yaml), "--input-directory", str(inputs), "--output-directory", str(output)],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")

        def run(workflow: str, tag: str, expect_success: bool = True, **overrides) -> dict:
            wf = json.loads((ROOT / "workflows" / "api" / workflow).read_text(encoding="utf-8"))
            for node in wf.values():
                if "filename_prefix" in node["inputs"]:
                    node["inputs"]["filename_prefix"] = node["inputs"]["filename_prefix"].replace("dbs/", f"ci/{tag}_")
            for key, value in overrides.items():
                node, _, field = key.partition(".")
                wf[node]["inputs"][field] = value
            try:
                result = queue_client.run(server, wf, None, 900)
            except SystemExit as rejected:  # POST /prompt refused the workflow (validation error)
                result = {"status": "rejected", "wall_seconds": 0, "error": str(rejected)[:200]}
            error = result.get("error") or {}
            print(json.dumps({"run": tag, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": error.get("exception_message", "")[:200] if isinstance(error, dict) else error}))
            if (result["status"] == "success") != expect_success:
                failures.append(f"{tag}: expected {'success' if expect_success else 'an error'}, got {result['status']} {error}")
            return result

        def saved(stem: str):
            files = sorted((output / "ci").glob(f"{stem}_*.flac"))
            if len(files) != 1:
                failures.append(f"{stem}: expected one saved .flac, found {[f.name for f in files]}")
                return None
            data, rate = sf.read(str(files[0]), dtype="float32", always_2d=True)
            if rate != 44100:
                failures.append(f"{stem}: saved at {rate} Hz, expected 44100")
            return data

        def same(stem: str, got, expected):
            if got is None:
                return
            expected = expected.reshape(expected.shape[0], -1)
            diff = float(np.abs(got - expected).max()) if got.shape == expected.shape else None
            print(json.dumps({"compare": stem, "shape": list(got.shape), "max_diff_in_16bit_steps": None if diff is None else round(diff / STEP, 3)}))
            if diff is None or diff > STEP:
                failures.append(f"{stem}: saved FLAC differs from the official output by more than one 16-bit step ({diff}, {got.shape} vs {expected.shape})")

        result = run("separate_soundtrack.json", "tiger_a")
        tiger_a = {s: saved(f"tiger_a_{s}") for s in ("music", "sfx", "speech")}
        for i, s in enumerate(("music", "sfx", "speech")):
            same(f"tiger_a_{s}", tiger_a[s], mono[i][:, None])
        texts = [json.loads(v[0]) for v in (result.get("text") or {}).values() if v and v[0].lstrip().startswith("{")]
        if not texts or texts[0]["model"] != "DBS-TIGER-DnR" or texts[0]["device"] != "cpu" or texts[0]["windows"] != 2:
            failures.append(f"unexpected report {texts}")
        run("separate_soundtrack.json", "bsrnn", **{"2.model": "DBS-BSRNN-DnR"})
        run("separate_soundtrack.json", "tiger_b")  # after the BSRNN job, so ComfyUI's cache no longer holds the TIGER result
        b, other = saved("tiger_b_speech"), saved("bsrnn_speech")
        if tiger_a["speech"] is not None and b is not None and other is not None:
            if not np.array_equal(tiger_a["speech"], b):
                failures.append("the same job gave different audio")
            if np.array_equal(tiger_a["speech"], other):
                failures.append("DBS-TIGER and DBS-BSRNN gave the same audio")

        run("remove_dialogue.json", "nodialog")
        same("nodialog_music_and_effects", saved("nodialog_music_and_effects"), per_channel[0] + per_channel[1])
        same("nodialog_dialogue", saved("nodialog_dialogue"), per_channel[2])

        run("video_remove_dialogue.json", "video")
        videos = sorted((output / "ci").glob("video_no_dialogue_*.mp4"))
        dialogue = saved("video_dialogue")
        if len(videos) != 1:
            failures.append(f"video: expected one saved .mp4, found {[v.name for v in videos]}")
        else:
            import av

            with av.open(str(videos[0])) as f:
                kinds = sorted(s.type for s in f.streams)
                seconds = float(f.duration / 1_000_000) if f.duration else 0.0
            print(json.dumps({"video": videos[0].name, "streams": kinds, "seconds": round(seconds, 3),
                              "dialogue_samples": None if dialogue is None else dialogue.shape[0]}))
            if kinds != ["audio", "video"] or abs(seconds - 3.0) > 0.2:
                failures.append(f"video: unexpected streams {kinds} or duration {seconds}")
            if dialogue is not None and (dialogue.shape[1] != 2 or abs(dialogue.shape[0] - 3 * 44100) > 2048):
                failures.append(f"video: unexpected dialogue shape {dialogue.shape}")

        run("separate_soundtrack.json", "bad_channels", expect_success=False, **{"3.channels": "stereo"})
        run("separate_soundtrack.json", "bad_model", expect_success=False, **{"2.model": "../DBS-TIGER-DnR"})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
