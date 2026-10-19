"""Deterministic synthetic test audio: a "soundtrack" of chords (music), a pitched buzz with moving formants (speech-like) and noise
bursts (effects). The parity tests only need any signal; the shapes are chosen so that every model path is used."""

import numpy as np

RATE = 44100


def soundtrack(seconds: float, rate: int = RATE, channels: int = 2, seed: int = 0) -> np.ndarray:
    """[samples, channels] float32 in about -0.5..0.5."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * rate))
    t = np.arange(n) / rate
    music = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6)) * 0.08
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    buzz = sum(np.sin(k * phase) / k for k in range(1, 25))
    speech = buzz * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t) ** 2) * 0.12
    sfx = np.zeros(n)
    for start in rng.integers(0, max(1, n - rate // 5), size=max(1, int(seconds * 2))):
        burst = rng.standard_normal(rate // 5) * np.exp(-np.arange(rate // 5) / (rate / 30))
        sfx[start:start + burst.size] += burst[: n - start] * 0.2
    mix = music + speech + sfx
    out = np.stack([mix * (1.0 - 0.15 * c) + 0.02 * rng.standard_normal(n) for c in range(channels)], axis=1)
    return out.astype(np.float32)
