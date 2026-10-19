"""The node pack as ComfyUI loads it: registration, model discovery, ComfyUI's own validation of the shipped workflows, and node
execution on audio decoded by ComfyUI's own Load Audio, compared with the runtime (which tests/test_parity.py compares with the
official package)."""

import asyncio
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import audio

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"DBSLoader", "DBSSeparate"}


def pack_module(comfy_nodes, name):
    """A module of the pack as ComfyUI imported it (under a path-based package name, so not the `dbs_comfy` the tests import)."""
    return importlib.import_module(comfy_nodes.NODE_CLASS_MAPPINGS["DBSSeparate"].__module__.rsplit(".", 1)[0] + "." + name)


def _write_clip(path):
    """A 2 s 64x64 video with a stereo 44.1 kHz soundtrack, written by ComfyUI's own video code (the name the workflow uses)."""
    from comfy_api.latest import InputImpl, Types

    frames = torch.linspace(0, 1, 48).view(48, 1, 1, 1).expand(48, 64, 64, 3).contiguous()
    sound = {"waveform": torch.from_numpy(audio.soundtrack(2.0, seed=11).T.copy()).unsqueeze(0), "sample_rate": 44100}
    InputImpl.VideoFromComponents(Types.VideoComponents(images=frames, audio=sound, frame_rate=24)).save_to(str(path))


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, models_dir, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    folder_paths.set_output_directory(str(base / "output"))
    folder_paths.set_input_directory(str(base / "input"))
    (base / "input").mkdir()
    sf.write(base / "input" / "soundtrack.wav", audio.soundtrack(4.0, seed=10), 44100, subtype="FLOAT")
    sf.write(base / "input" / "soundtrack_48k.wav", audio.soundtrack(2.5, rate=48000, seed=12), 48000, subtype="FLOAT")
    _write_clip(base / "input" / "clip.mp4")
    folder_paths.add_model_folder_path("dbs", str(models_dir), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    yield nodes
    pack_module(nodes, "runtime").release()


def _load_audio(comfy_nodes, name="soundtrack.wav"):
    return comfy_nodes.NODE_CLASS_MAPPINGS["LoadAudio"].execute(name).args[0]


def _model(comfy_nodes, name="DBS-TIGER-DnR", device="cpu"):
    return comfy_nodes.NODE_CLASS_MAPPINGS["DBSLoader"].execute(name, device).args[0]


def _separate(comfy_nodes, name="DBS-TIGER-DnR", channels="mono", rate="44100", sound=None):
    return comfy_nodes.NODE_CLASS_MAPPINGS["DBSSeparate"].execute(_model(comfy_nodes, name), sound or _load_audio(comfy_nodes),
                                                                  channels, rate).args


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


@pytest.mark.parametrize("name", ["DBS-TIGER-DnR", "DBS-BSRNN-DnR"])
@pytest.mark.parametrize("channels", ["mono", "each channel"])
def test_node_output_equals_the_runtime_on_the_loaded_file(comfy_nodes, models_dir, name, channels):
    from dbs_comfy import runtime

    music, sfx, speech, report = _separate(comfy_nodes, name, channels)
    sound = _load_audio(comfy_nodes)
    expected = runtime.separate_audio(runtime.load(str(models_dir / name), name, "cpu"), name, sound, channels)
    for ours, theirs in zip((music, sfx, speech), expected):
        assert ours["sample_rate"] == 44100 and torch.equal(ours["waveform"], theirs["waveform"])
    parsed = json.loads(report)
    assert parsed["model"] == name and parsed["device"] == "cpu" and parsed["signals_per_item"] == (1 if channels == "mono" else 2)
    import folder_paths

    x, _ = sf.read(Path(folder_paths.get_input_directory()) / "soundtrack.wav", dtype="float32", always_2d=True)
    assert np.array_equal(sound["waveform"][0].numpy(), x.T)  # Load Audio gives the file's samples unchanged


def test_a_48khz_file_is_resampled_in_and_back_out(comfy_nodes):
    import comfy.audio

    sound = _load_audio(comfy_nodes, "soundtrack_48k.wav")
    assert sound["sample_rate"] == 48000
    name = "DBS-BSRNN-DnR"  # one unpadded window for 2.5 s (DBS-TIGER pads to 6 s); the resampling around it is the same
    music, _, speech, report = _separate(comfy_nodes, name, rate="input", sound=sound)
    assert music["sample_rate"] == 48000 and music["waveform"].shape == (1, 1, sound["waveform"].shape[-1])
    at_model_rate = _separate(comfy_nodes, name, sound=sound)[2]
    assert at_model_rate["sample_rate"] == 44100
    assert torch.equal(speech["waveform"], comfy.audio.resample(at_model_rate["waveform"], 44100, 48000))
    assert json.loads(report)["resampled_to_44100"] is True


def test_unknown_or_missing_models_are_reported(comfy_nodes, tmp_path, monkeypatch):
    import folder_paths

    with pytest.raises(ValueError, match="unknown model"):
        _model(comfy_nodes, "../DBS-TIGER-DnR")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path)])
    (tmp_path / "DBS-BSRNN-DnR").mkdir()
    (tmp_path / "DBS-BSRNN-DnR" / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        _model(comfy_nodes, "DBS-BSRNN-DnR")


def test_interrupt_stops_between_windows(comfy_nodes, monkeypatch):
    import comfy.model_management as mm

    before = _separate(comfy_nodes, "DBS-BSRNN-DnR")[2]
    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        if calls["n"] > 2:
            raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        _separate(comfy_nodes, "DBS-BSRNN-DnR")
    assert calls["n"] == 3  # the check before starting, then one per window
    monkeypatch.undo()
    after = _separate(comfy_nodes, "DBS-BSRNN-DnR")[2]
    assert torch.equal(before["waveform"], after["waveform"])


def test_shipped_gui_workflows_use_known_nodes():
    allowed = NODE_IDS | {"LoadAudio", "SaveAudioAdvanced", "PreviewAny", "AudioMerge", "LoadVideo", "GetVideoComponents",
                          "CreateVideo", "SaveVideo", "Note", "MarkdownNote"}
    for path in sorted((ROOT / "workflows" / "gui").glob("*.json")):
        types = {n["type"] for n in json.loads(path.read_text(encoding="utf-8"))["nodes"]}
        assert types <= allowed, path.name
