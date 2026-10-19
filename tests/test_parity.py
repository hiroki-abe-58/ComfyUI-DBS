"""The pack against the official package, in the same process on the CPU: the README usage

    model = <cls>.from_pretrained(<folder>).eval()
    separate(model, torch.from_numpy(audio.mean(1)).view(1, 1, -1), {"mode": "chunked", ...}, sample_rate=44100)[0]

must give the pack's stems bit for bit, for both released models, on every path of the chunked overlap-add (several windows,
one unpadded window, one zero-padded window) and for the pack's per-channel mode (the same call on every channel)."""

import numpy as np
import pytest
import torch

import audio
from dbs_comfy import runtime

PROTOCOLS = {"DBS-TIGER-DnR": ("DBSTIGER", {"mode": "chunked", "chunk_seconds": 6.0, "hop_seconds": 3.0}),
             "DBS-BSRNN-DnR": ("DBSBSRNN", {"mode": "chunked", "chunk_seconds": 3.0, "hop_seconds": 1.5})}


@pytest.fixture(scope="module")
def official_models(official, models_dir):
    return {name: getattr(official.models, cls).from_pretrained(str(models_dir / name)).eval() for name, (cls, _) in PROTOCOLS.items()}


def _official(official, model, name, x):
    with torch.inference_mode():
        mixture = torch.from_numpy(np.ascontiguousarray(x)).view(1, 1, -1)
        return official.engine.inference.separate(model, mixture, PROTOCOLS[name][1], sample_rate=44100)[0].numpy()


def _pack(models_dir, name, x, channels):
    model = runtime.load(str(models_dir / name), name, "cpu")
    stems = runtime.separate_audio(model, name, {"waveform": torch.from_numpy(np.ascontiguousarray(x.T)).unsqueeze(0), "sample_rate": 44100},
                                   channels)[:3]
    return np.stack([s["waveform"][0].numpy() for s in stems], axis=1)  # [signals, 3, samples]


@pytest.mark.parametrize("name", list(PROTOCOLS))
@pytest.mark.parametrize("seconds", [9.0, 4.5, 2.0])  # several windows / TIGER one unpadded window / TIGER zero-padded window
def test_mono_equals_the_official_usage(official, official_models, models_dir, name, seconds):
    x = audio.soundtrack(seconds, seed=int(seconds * 10))
    expected = _official(official, official_models[name], name, x.mean(1))
    ours = _pack(models_dir, name, x, "mono")
    assert ours.shape == (1, 3, x.shape[0]) and np.array_equal(ours[0], expected)


@pytest.mark.parametrize("name", list(PROTOCOLS))
def test_each_channel_equals_the_official_usage_per_channel(official, official_models, models_dir, name):
    x = audio.soundtrack(3.5, seed=7)
    ours = _pack(models_dir, name, x, "each channel")
    for c in range(x.shape[1]):
        assert np.array_equal(ours[c], _official(official, official_models[name], name, x[:, c]))


@pytest.mark.parametrize("name", list(PROTOCOLS))
def test_strict_loading_gives_the_official_weights(official_models, models_dir, name):
    ours = runtime.load(str(models_dir / name), name, "cpu").state_dict()
    theirs = official_models[name].state_dict()
    assert ours.keys() == theirs.keys() and all(torch.equal(ours[k], theirs[k]) for k in ours)
