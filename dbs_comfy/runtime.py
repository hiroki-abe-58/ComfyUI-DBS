"""DBS separation without ComfyUI: the released models built from a local folder, and the official chunked overlap-add.

The official usage (README and model cards of JusperLee/Dynamic-Band-Splitting) is

    model = DBSTIGER.from_pretrained(<folder>).eval()
    mixture = torch.from_numpy(audio.mean(1)).view(1, 1, -1)          # 44.1 kHz mono
    music, sfx, speech = separate(model, mixture, {"mode": "chunked", "chunk_seconds": 6.0, "hop_seconds": 3.0},
                                  sample_rate=44100)[0]

`from_pretrained` on a folder builds the class from config.json and loads model.safetensors; `load()` does the same without
huggingface_hub (and with strict key checking). `separate_audio()` runs that call on every batch item, on the channel mean
(official) or on every channel separately, and resamples with ComfyUI's own resampler when the input is not 44.1 kHz.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import torch

from .vendor.dbs.engine.inference import chunked
from .vendor.dbs.models import DBSBSRNN, DBSTIGER

SAMPLE_RATE = 44100
STEMS = ("music", "sfx", "speech")
FILES = ("config.json", "model.safetensors")
CHANNEL_MODES = ("mono", "each channel")
OUTPUT_RATES = ("44100", "input")


@dataclass(frozen=True)
class Spec:
    cls: type
    chunk_seconds: float  # the overlap-add windows of the model card / training recipe
    hop_seconds: float


MODELS = {
    "DBS-TIGER-DnR": Spec(DBSTIGER, 6.0, 3.0),
    "DBS-BSRNN-DnR": Spec(DBSBSRNN, 3.0, 1.5),
}

_cache: dict = {}


def load(folder: str, name: str, device="cpu") -> torch.nn.Module:
    """The model of `folder` (config.json + model.safetensors) on `device`, in eval mode; cached per folder, name and device."""
    from safetensors.torch import load_model

    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {', '.join(MODELS)}")
    key = (os.path.realpath(folder), name, str(device))
    if key not in _cache:
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        model = MODELS[name].cls(**config)
        load_model(model, os.path.join(folder, "model.safetensors"), strict=True)
        if model.sample_rate != SAMPLE_RATE or (getattr(model, "num_sources", None) or model.num_stems) != len(STEMS):
            raise ValueError(f"{folder}: expected a 44.1 kHz model with {len(STEMS)} stems (the DnR v2 release)")
        _cache.clear()  # one model at a time; they are small, but there is no reason to keep old ones
        _cache[key] = model.to(device).eval()
    return _cache[key]


def release() -> None:
    """Drop the cached model."""
    _cache.clear()


def window_samples(name: str) -> tuple[int, int]:
    spec = MODELS[name]
    chunk = int(round(SAMPLE_RATE * spec.chunk_seconds))
    return chunk, max(1, min(int(round(SAMPLE_RATE * spec.hop_seconds)), chunk))


def windows_for(length: int, chunk: int, hop: int) -> int:
    """Model calls of `chunked()` for a signal of `length` samples (the same loop)."""
    if length <= chunk:
        return 1
    count, start = 0, 0
    while start < length:
        count += 1
        if start + chunk >= length:
            break
        start += hop
    return count


def total_windows(name: str, audio: dict, channels: str) -> int:
    """Model calls `separate_audio()` will make (for a progress bar)."""
    waveform, rate = audio["waveform"], int(audio["sample_rate"])
    batch = waveform.shape[0] if waveform.ndim == 3 else 1
    signals = 1 if channels == "mono" else waveform.shape[-2]
    length = waveform.shape[-1] if rate == SAMPLE_RATE else math.ceil(waveform.shape[-1] * SAMPLE_RATE / rate)
    return batch * signals * windows_for(length, *window_samples(name))


def _resample(x: torch.Tensor, orig: int, new: int) -> torch.Tensor:
    if orig == new:
        return x
    import comfy.audio

    return comfy.audio.resample(x, orig, new)


def separate_audio(model: torch.nn.Module, name: str, audio: dict, channels: str = "mono", output_rate: str = "44100",
                   progress=None) -> tuple[dict, dict, dict, dict]:
    """ComfyUI AUDIO -> (music, sfx, speech, report). `progress(done_windows)` is called after every model call."""
    if channels not in CHANNEL_MODES:
        raise ValueError(f"channels must be one of {CHANNEL_MODES}")
    if output_rate not in OUTPUT_RATES:
        raise ValueError(f"output_sample_rate must be one of {OUTPUT_RATES}")
    waveform, rate = audio["waveform"], int(audio["sample_rate"])
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3 or waveform.shape[-1] == 0:
        raise ValueError(f"expected a non-empty audio waveform [batch, channels, samples], got shape {tuple(waveform.shape)}")
    if not torch.isfinite(waveform).all():
        raise ValueError("the audio contains NaN or infinite samples")
    device = next(model.parameters()).device
    chunk, hop = window_samples(name)
    done = 0

    def run(piece):
        nonlocal done
        out = model(piece)
        done += 1
        if progress is not None:
            progress(done)
        return out

    began = time.perf_counter()
    items = []
    with torch.inference_mode():
        for item in waveform.float():
            signals = [item.mean(0)] if channels == "mono" else list(item)
            stems = []
            for x in signals:
                x = _resample(x.cpu(), rate, SAMPLE_RATE)
                stems.append(chunked(run, x.reshape(1, 1, -1).to(device), chunk, hop)[0].float().cpu())  # [3, samples]
            items.append(torch.stack(stems, dim=1))  # [3, signals, samples]
    seconds = time.perf_counter() - began
    out = torch.stack(items, dim=1)  # [3, batch, signals, samples]
    out_rate = rate if output_rate == "input" else SAMPLE_RATE
    results = []
    for s in range(len(STEMS)):
        stem = out[s]
        if out_rate != SAMPLE_RATE:
            stem = _resample(stem, SAMPLE_RATE, out_rate)
        results.append({"waveform": stem.contiguous(), "sample_rate": out_rate})
    audio_seconds = waveform.shape[0] * len(signals) * waveform.shape[-1] / rate
    report = {
        "model": name, "device": str(device), "input_sample_rate": rate, "resampled_to_44100": rate != SAMPLE_RATE,
        "output_sample_rate": out_rate, "channels": channels, "batch": int(waveform.shape[0]),
        "signals_per_item": len(signals), "windows": done, "window_seconds": chunk / SAMPLE_RATE, "hop_seconds": hop / SAMPLE_RATE,
        "audio_seconds": round(audio_seconds, 3), "seconds": round(seconds, 3),
        "times_real_time": round(audio_seconds / seconds, 2) if seconds > 0 else math.inf,
    }
    return results[0], results[1], results[2], report
