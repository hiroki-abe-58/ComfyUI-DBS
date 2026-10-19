"""Inference protocols for signals longer than the training crop.

``mode`` selects how a mixture ``[B, C, T]`` is separated:

* ``full``: one forward pass over the whole signal (EchoSet).
* ``chunked``: windows of ``chunk_seconds`` every ``hop_seconds``, the last
  window zero-padded, averaged with uniform weights (DnR v2).
* ``chunked_tail``: fixed-size windows every ``chunk_samples // num_overlap``
  plus one window aligned to the end, averaged with uniform weights
  (MUSDB18-HQ Mel-RoFormer evaluation).
* ``msst``: the Music-Source-Separation-Training ``demix`` scheme, reflect-padded
  windows with linear fade-in/out weights (MUSDB18-HQ BSRNN / BS-RoFormer).
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def model_output(model: nn.Module, mixture: Tensor) -> Tensor:
    """Separated sources of ``mixture``, dropping any auxiliary outputs."""
    output = model(mixture)
    return output[0] if isinstance(output, (tuple, list)) else output


@torch.no_grad()
def chunked(model: nn.Module, mixture: Tensor, chunk: int, hop: int) -> Tensor:
    length = mixture.shape[-1]
    if length <= chunk:
        if length > chunk // 2:  # the original evaluation ran such signals unpadded
            return model_output(model, mixture)
        # Very short signals are zero-padded to one window (the STFT needs more than n_fft // 2 samples).
        return model_output(model, F.pad(mixture, (0, chunk - length)))[..., :length]
    output = weight = None
    start = 0
    while start < length:
        valid = min(chunk, length - start)
        piece = mixture[..., start:start + valid]
        if valid < chunk:
            piece = F.pad(piece, (0, chunk - valid))
        estimate = model_output(model, piece)[..., :valid]
        if output is None:
            output = mixture.new_zeros(*estimate.shape[:-1], length)
            weight = mixture.new_zeros(length)
        output[..., start:start + valid] += estimate
        weight[start:start + valid] += 1.0
        if start + chunk >= length:
            break
        start += hop
    return output / weight


@torch.no_grad()
def chunked_tail(model: nn.Module, mixture: Tensor, chunk: int, hop: int) -> Tensor:
    length = mixture.shape[-1]
    starts = list(range(0, max(1, length - chunk + 1), hop))
    tail = max(0, length - chunk)
    if starts[-1] != tail:
        starts.append(tail)
    output = None
    weight = mixture.new_zeros(length)
    for start in starts:
        valid = min(chunk, length - start)
        window = mixture.new_zeros(*mixture.shape[:-1], chunk)
        window[..., :valid] = mixture[..., start:start + valid]
        estimate = model_output(model, window)[..., :valid]
        if output is None:
            output = mixture.new_zeros(*estimate.shape[:-1], length)
        output[..., start:start + valid] += estimate
        weight[start:start + valid] += 1.0
    return output / weight


def _fade_window(size: int, fade: int, device) -> Tensor:
    window = torch.ones(size, device=device)
    window[-fade:] = torch.linspace(1, 0, fade, device=device)
    window[:fade] = torch.linspace(0, 1, fade, device=device)
    return window


@torch.no_grad()
def msst(model: nn.Module, mixture: Tensor, chunk: int, num_overlap: int, batch_size: int = 1) -> Tensor:
    """MSST ``demix`` (generic mode) applied to every item of ``mixture [B, C, T]``."""
    return torch.stack([_msst_single(model, item, chunk, num_overlap, batch_size) for item in mixture])


def _msst_single(model: nn.Module, mix: Tensor, chunk: int, num_overlap: int, batch_size: int) -> Tensor:
    fade = chunk // 10
    step = chunk // num_overlap
    border = chunk - step
    length = mix.shape[-1]
    padded = length > 2 * border and border > 0
    if padded:
        mix = F.pad(mix.unsqueeze(0), (border, border), mode="reflect").squeeze(0)
    total = mix.shape[-1]
    base = _fade_window(chunk, fade, mix.device)

    starts = list(range(0, total, step))
    result = counter = None
    for first in range(0, len(starts), batch_size):
        group = starts[first:first + batch_size]
        parts = []
        for start in group:
            part = mix[..., start:start + chunk]
            mode = "reflect" if part.shape[-1] > chunk // 2 else "constant"
            parts.append(F.pad(part.unsqueeze(0), (0, chunk - part.shape[-1]), mode=mode).squeeze(0))
        estimates = model_output(model, torch.stack(parts))
        if result is None:
            result = mix.new_zeros(*estimates.shape[1:-1], total)
            counter = torch.zeros_like(result)
        for j, start in enumerate(group):
            seg_len = min(chunk, total - start)
            window = base.clone()
            if start == 0:
                window[:fade] = 1
            elif start + step >= total:
                window[-fade:] = 1
            result[..., start:start + seg_len] += estimates[j, ..., :seg_len] * window[:seg_len]
            counter[..., start:start + seg_len] += window[:seg_len]
    estimate = torch.nan_to_num(result / counter, nan=0.0)
    return estimate[..., border:-border] if padded else estimate


def separate(model: nn.Module, mixture: Tensor, protocol: Mapping[str, Any] | None = None,
             sample_rate: int | None = None) -> Tensor:
    """Separate ``mixture [B, C, T]`` (or ``[B, T]``) with the given inference protocol."""
    protocol = dict(protocol or {})
    mode = protocol.pop("mode", "full")
    if mode == "full":
        return model_output(model, mixture)
    if mode == "chunked":
        if sample_rate is None:
            raise ValueError("chunked inference needs the sample rate")
        chunk = int(round(sample_rate * float(protocol["chunk_seconds"])))
        hop = max(1, min(int(round(sample_rate * float(protocol.get("hop_seconds", protocol["chunk_seconds"] / 2)))),
                         chunk))
        return chunked(model, mixture, chunk, hop)
    if mode == "chunked_tail":
        chunk = int(protocol["chunk_samples"])
        return chunked_tail(model, mixture, chunk, max(1, chunk // int(protocol.get("num_overlap", 4))))
    if mode == "msst":
        return msst(model, mixture, int(protocol["chunk_samples"]), int(protocol.get("num_overlap", 4)),
                    int(protocol.get("batch_size", 1)))
    raise ValueError(f"unknown inference mode {mode!r}")
