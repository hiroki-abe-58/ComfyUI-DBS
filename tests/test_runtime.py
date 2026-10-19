"""The pack's runtime with the real models on the CPU: loading, shapes and rates, the per-channel and batch handling, resampling,
progress, and the errors for inputs the models cannot take."""

import json
import math
import shutil

import numpy as np
import pytest
import torch

import audio
from dbs_comfy import runtime

NAMES = list(runtime.MODELS)


def _audio(x, rate=44100):
    """[samples, channels] (or a list of them, one per batch item) -> ComfyUI AUDIO."""
    items = x if isinstance(x, list) else [x]
    return {"waveform": torch.stack([torch.from_numpy(np.ascontiguousarray(i.T)) for i in items]), "sample_rate": rate}


def _model(models_dir, name="DBS-TIGER-DnR"):
    return runtime.load(str(models_dir / name), name, "cpu")


def test_load_is_cached_per_folder_name_and_device(models_dir):
    a = _model(models_dir)
    assert _model(models_dir) is a and not a.training
    assert _model(models_dir, "DBS-BSRNN-DnR") is not a
    with pytest.raises(ValueError, match="unknown model"):
        runtime.load(str(models_dir / "DBS-TIGER-DnR"), "TIGER", "cpu")
    runtime.release()


def test_strict_loading_rejects_weights_of_the_other_model(models_dir, tmp_path):
    shutil.copy(models_dir / "DBS-TIGER-DnR" / "config.json", tmp_path / "config.json")
    shutil.copy(models_dir / "DBS-BSRNN-DnR" / "model.safetensors", tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError):
        runtime.load(str(tmp_path), "DBS-TIGER-DnR", "cpu")


@pytest.mark.parametrize("name", NAMES)
def test_stems_shapes_and_sum(models_dir, name):
    x = audio.soundtrack(3.0, seed=1)
    music, sfx, speech, report = runtime.separate_audio(_model(models_dir, name), name, _audio(x))
    for stem in (music, sfx, speech):
        assert stem["sample_rate"] == 44100 and stem["waveform"].shape == (1, 1, x.shape[0]) and stem["waveform"].dtype == torch.float32
    total = music["waveform"] + sfx["waveform"] + speech["waveform"]
    assert torch.allclose(total[0, 0], torch.from_numpy(x.mean(1)), atol=1e-5)  # the masks of the released models sum to one
    assert report["model"] == name and report["channels"] == "mono" and report["signals_per_item"] == 1
    json.dumps(report)


def test_each_channel_keeps_the_channels(models_dir):
    x = audio.soundtrack(2.5, seed=2)
    music, sfx, speech, report = runtime.separate_audio(_model(models_dir), "DBS-TIGER-DnR", _audio(x), "each channel")
    assert music["waveform"].shape == (1, 2, x.shape[0]) and report["signals_per_item"] == 2
    total = (music["waveform"] + sfx["waveform"] + speech["waveform"])[0]
    assert torch.allclose(total, torch.from_numpy(x.T.copy()), atol=1e-5)


def test_batch_items_are_separated_on_their_own(models_dir):
    a, b = audio.soundtrack(2.0, seed=3), audio.soundtrack(2.0, seed=4)
    model = _model(models_dir)
    both = runtime.separate_audio(model, "DBS-TIGER-DnR", _audio([a, b]))[:3]
    one = runtime.separate_audio(model, "DBS-TIGER-DnR", _audio(a))[:3]
    two = runtime.separate_audio(model, "DBS-TIGER-DnR", _audio(b))[:3]
    for s in range(3):
        assert torch.equal(both[s]["waveform"][0], one[s]["waveform"][0]) and torch.equal(both[s]["waveform"][1], two[s]["waveform"][0])


def test_other_rates_are_resampled_with_comfyuis_resampler(comfy_dir, models_dir):
    import comfy.audio

    x = audio.soundtrack(2.0, rate=48000, seed=5)
    model = _model(models_dir)
    music, _, _, report = runtime.separate_audio(model, "DBS-TIGER-DnR", _audio(x, 48000))
    resampled = comfy.audio.resample(torch.from_numpy(x.mean(1)), 48000, 44100)
    direct = runtime.separate_audio(model, "DBS-TIGER-DnR", {"waveform": resampled.view(1, 1, -1), "sample_rate": 44100})[0]
    assert report["resampled_to_44100"] and music["sample_rate"] == 44100 and torch.equal(music["waveform"], direct["waveform"])
    back, _, _, report = runtime.separate_audio(model, "DBS-TIGER-DnR", _audio(x, 48000), output_rate="input")
    assert back["sample_rate"] == 48000 and back["waveform"].shape[-1] == x.shape[0] and report["output_sample_rate"] == 48000


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seconds", [1.0, 4.5, 7.3])
def test_progress_counts_every_model_call(models_dir, name, seconds):
    x = audio.soundtrack(seconds, seed=6)
    calls = []
    for channels in runtime.CHANNEL_MODES:
        calls.clear()
        report = runtime.separate_audio(_model(models_dir, name), name, _audio(x), channels, progress=calls.append)[3]
        total = runtime.total_windows(name, _audio(x), channels)
        assert calls == list(range(1, total + 1)) and report["windows"] == total
    chunk, hop = runtime.window_samples(name)
    expected = 1 if x.shape[0] <= chunk else math.ceil((x.shape[0] - chunk) / hop) + 1
    assert runtime.windows_for(x.shape[0], chunk, hop) == expected


def test_cpu_runs_are_repeatable(models_dir):
    x = audio.soundtrack(2.0, seed=8)
    first = runtime.separate_audio(_model(models_dir, "DBS-BSRNN-DnR"), "DBS-BSRNN-DnR", _audio(x))[2]["waveform"]
    second = runtime.separate_audio(_model(models_dir, "DBS-BSRNN-DnR"), "DBS-BSRNN-DnR", _audio(x))[2]["waveform"]
    assert torch.equal(first, second)


def test_inputs_the_models_cannot_take(models_dir):
    model = _model(models_dir)
    with pytest.raises(ValueError, match="non-empty"):
        runtime.separate_audio(model, "DBS-TIGER-DnR", {"waveform": torch.zeros(1, 2, 0), "sample_rate": 44100})
    bad = audio.soundtrack(1.0)
    bad[100, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        runtime.separate_audio(model, "DBS-TIGER-DnR", _audio(bad))
    with pytest.raises(ValueError, match="channels"):
        runtime.separate_audio(model, "DBS-TIGER-DnR", _audio(audio.soundtrack(1.0)), "stereo")
