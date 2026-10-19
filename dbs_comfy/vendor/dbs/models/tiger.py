"""TIGER backbone and DBS-TIGER.

The separator (``Recurrent``) is the TIGER frequency/frame dual-path network.
DBS-TIGER replaces TIGER's fixed band split with learned boundaries: a router
backbone shared by all regions scores every STFT bin from acoustic features
and learned similarity, hard Top-K keeps a fixed number of boundaries per
region, and each variable-width band is pooled into one token. After
separation, tokens are smoothed, expanded to the STFT grid, refined along
frequency, and decoded into complex masks.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..hub import HUB_KWARGS, PyTorchModelHubMixin
from ..modules.band_split import desired_token_count, validate_region_configs
from ..modules.dbs import flat_segment_ids, segment_softmax_pool, topk_boundaries


# --------------------------------------------------------------------------- TIGER separator


def GlobLN(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(1, channels, eps=1e-8)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2, bias=True, groups=groups)
        self.norm = GlobLN(out_channels)
        self.act = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 groups: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2, bias=bias, groups=groups)
        self.norm = GlobLN(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x))


class DilatedConvNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 dilation: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                              padding=((kernel_size - 1) // 2) * dilation, groups=groups)
        self.norm = GlobLN(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x))


class LayerNormalization4D(nn.Module):
    """Layer norm over the channel axis (and the last axis when it is wider than 1)."""

    def __init__(self, input_dimension: Sequence[int], eps: float = 1e-5):
        super().__init__()
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.dim = (1, 3) if param_size[-1] > 1 else (1,)
        self.gamma = nn.Parameter(torch.ones(*param_size))
        self.beta = nn.Parameter(torch.zeros(*param_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=self.dim, keepdim=True)
        std = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        return (x - mean) / std * self.gamma + self.beta


class ATTConvActNorm(nn.Module):
    """1x1 Conv2d -> PReLU -> LayerNormalization4D, as used by the attention block."""

    def __init__(self, in_chan: int, out_chan: int, n_freqs: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, padding="same")
        self.act = nn.PReLU()
        self.norm = LayerNormalization4D((out_chan, n_freqs))

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.act(self.conv(x)))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, drop: float = 0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size)
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.dwconv(self.fc1(x))))
        return self.drop(self.fc2(x))


class InjectionMultiSum(nn.Module):
    """Gate local features with (upsampled) global features."""

    def __init__(self, inp: int, oup: int, kernel: int = 1):
        super().__init__()
        groups = inp if inp == oup else 1
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l: Tensor, x_g: Tensor) -> Tensor:
        length = x_l.shape[-1]
        gate = F.interpolate(self.act(self.global_act(x_g)), size=length, mode="nearest")
        global_feat = F.interpolate(self.global_embedding(x_g), size=length, mode="nearest")
        return self.local_embedding(x_l) * gate + global_feat


class UConvBlock(nn.Module):
    """Multi-resolution down/up-sampling block with global-local fusion."""

    def __init__(self, out_channels: int = 128, in_channels: int = 512, upsampling_depth: int = 4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([DilatedConvNorm(in_channels, in_channels, 5, stride=1, groups=in_channels)])
        for _ in range(1, upsampling_depth):
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, 5, stride=2, groups=in_channels))
        self.loc_glo_fus = nn.ModuleList([InjectionMultiSum(in_channels, in_channels) for _ in range(upsampling_depth)])
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.globalatt = Mlp(in_channels, in_channels, drop=0.1)
        self.last_layer = nn.ModuleList([InjectionMultiSum(in_channels, in_channels, 5)
                                         for _ in range(upsampling_depth - 1)])

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = [self.spp_dw[0](self.proj_1x1(x))]
        for k in range(1, self.depth):
            out.append(self.spp_dw[k](out[-1]))
        # Kept in float32 as in the original implementation.
        global_f = torch.zeros(out[-1].shape, device=x.device)
        for feature in out:
            global_f = global_f + F.adaptive_avg_pool1d(feature, output_size=out[-1].shape[-1])
        global_f = self.globalatt(global_f)
        fused = [self.loc_glo_fus[i](out[i], global_f) for i in range(self.depth)]
        expanded = None
        for i in range(self.depth - 2, -1, -1):
            expanded = self.last_layer[i](fused[i], fused[i - 1] if i == self.depth - 2 else expanded)
        return self.res_conv(expanded) + residual


class MultiHeadSelfAttention2D(nn.Module):
    """Multi-head self-attention over the last axis of ``[B, C, T, F]`` (transposed when ``dim=4``)."""

    def __init__(self, in_chan: int, n_freqs: int, n_head: int = 4, hid_chan: int = 4, dim: int = 3):
        super().__init__()
        assert in_chan % n_head == 0
        self.n_head = n_head
        self.dim = dim
        self.Queries = nn.ModuleList([ATTConvActNorm(in_chan, hid_chan, n_freqs) for _ in range(n_head)])
        self.Keys = nn.ModuleList([ATTConvActNorm(in_chan, hid_chan, n_freqs) for _ in range(n_head)])
        self.Values = nn.ModuleList([ATTConvActNorm(in_chan, in_chan // n_head, n_freqs) for _ in range(n_head)])
        self.attn_concat_proj = ATTConvActNorm(in_chan, in_chan, n_freqs)

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()
        batch, _, time, freq = x.shape
        residual = x
        q = torch.cat([layer(x) for layer in self.Queries], dim=0).transpose(1, 2).flatten(start_dim=2)
        k = torch.cat([layer(x) for layer in self.Keys], dim=0).transpose(1, 2).flatten(start_dim=2)
        v = torch.cat([layer(x) for layer in self.Values], dim=0).transpose(1, 2)
        v_shape = v.shape
        v = v.flatten(start_dim=2)
        attn = F.softmax(torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5), dim=2)
        v = torch.matmul(attn, v).reshape(v_shape).transpose(1, 2)
        head_dim = v.shape[1]
        x = v.view([self.n_head, batch, head_dim, time, freq]).transpose(0, 1).contiguous()
        x = self.attn_concat_proj(x.view([batch, self.n_head * head_dim, time, freq])) + residual
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()
        return x


class Recurrent(nn.Module):
    """TIGER separator: ``iterations`` passes of frequency-path then frame-path modelling.

    Input and output are ``[B, bands, N, T]``; ``chunk_mask [B, bands]`` zeroes unused token slots.
    """

    def __init__(self, out_channels: int = 128, in_channels: int = 512, upsampling_depth: int = 3,
                 n_head: int = 4, att_hid_chan: int = 4, iterations: int = 4):
        super().__init__()
        self.freq_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head, hid_chan=att_hid_chan, dim=4),
            LayerNormalization4D((out_channels, 1)),
        ])
        self.frame_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head, hid_chan=att_hid_chan, dim=4),
            LayerNormalization4D((out_channels, 1)),
        ])
        self.iterations = iterations
        self.concat_block = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels),
                                          nn.PReLU())

    def forward(self, x: Tensor, chunk_mask: Tensor | None = None) -> Tensor:
        batch, bands, channels, time = x.shape
        if chunk_mask is not None:
            x = x * chunk_mask.to(x.dtype).unsqueeze(-1).unsqueeze(-1)
        x = x.permute(0, 2, 1, 3).contiguous()
        mixture = x.clone()
        mask = None if chunk_mask is None else chunk_mask.to(x.dtype).unsqueeze(1).unsqueeze(-1)
        for i in range(self.iterations):
            inp = x if i == 0 else self.concat_block(mixture + x)
            x = self._freq_time(inp, batch, bands, channels, time, mask)
        return x.permute(0, 2, 1, 3).contiguous()

    def _freq_time(self, x, batch, bands, channels, time, mask):
        if mask is not None:
            x = x * mask
        residual = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()
        freq = self.freq_path[0](x.view(batch * time, channels, bands))
        freq = freq.view(batch, time, channels, bands).permute(0, 2, 1, 3).contiguous()
        freq = self.freq_path[2](self.freq_path[1](freq)).permute(0, 1, 3, 2).contiguous()
        x = freq + residual
        residual = x.clone()
        frame = self.frame_path[0](x.permute(0, 2, 1, 3).contiguous().view(batch * bands, channels, time))
        frame = frame.view(batch, bands, channels, time).permute(0, 2, 1, 3).contiguous()
        x = self.frame_path[2](self.frame_path[1](frame)) + residual
        if mask is not None:
            x = x * mask
        return x


# --------------------------------------------------------------------------- DBS front/back end


def _energy_features(spec_ri: Tensor, mode: str) -> Tensor:
    magnitude = torch.norm(spec_ri, dim=1)
    mean = magnitude.mean(dim=-1)
    features = [torch.log1p(mean)]
    if mode == "mean_peak_std":
        features.append(torch.log1p(magnitude.amax(dim=-1) / mean.clamp(min=1e-6)))
        features.append(torch.log1p(magnitude.std(dim=-1, unbiased=False) / mean.clamp(min=1e-6)))
    return torch.stack(features, dim=-1)


class SharedChunkRouterBackbone(nn.Module):
    """CNN + BiLSTM router features ``[B, F, D]`` shared by all regions."""

    def __init__(self, feature_dim: int, energy_feature_mode: str = "mean"):
        super().__init__()
        self.energy_feature_mode = energy_feature_mode
        self.energy_feature_dim = 3 if energy_feature_mode == "mean_peak_std" else 1
        self.energy_dim = max(1, feature_dim // 4)
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, kernel_size=1),
        )
        self.energy_proj = nn.Sequential(nn.Linear(self.energy_feature_dim, self.energy_dim), nn.ReLU())
        self.lstm = nn.LSTM(input_size=feature_dim + self.energy_dim, hidden_size=feature_dim // 2,
                            num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, spec_ri: Tensor, use_energy: bool) -> Tensor:
        batch, _, freq, _ = spec_ri.shape
        cnn = self.cnn_extractor(spec_ri)
        _, dim, _, time = cnn.shape
        x = cnn.permute(0, 2, 3, 1).reshape(batch * freq, time, dim)
        if use_energy:
            energy = _energy_features(spec_ri, self.energy_feature_mode)
            energy = self.energy_proj(energy.reshape(batch * freq, 1, -1).expand(-1, time, -1))
        else:
            energy = x.new_zeros(batch * freq, time, self.energy_dim)
        _, (hidden, _) = self.lstm(torch.cat([x, energy], dim=-1))
        return torch.cat([hidden[-2], hidden[-1]], dim=-1).view(batch, freq, dim)


class SharedBandSimilarityProjection(nn.Module):
    """Region-conditioned dissimilarity between adjacent bins, gated by energy change."""

    def __init__(self, feature_dim: int, num_bands: int):
        super().__init__()
        self.W_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.band_embedding = nn.Embedding(num_bands, feature_dim)
        self.energy_strength = nn.Parameter(torch.tensor(0.25))

    def forward(self, router_feat: Tensor, band_idx: int, energy: Tensor | None = None) -> Tensor:
        feat = router_feat + self.band_embedding.weight[band_idx].view(1, 1, -1)
        q = F.normalize(self.W_q(feat), dim=-1)
        k = F.normalize(self.W_k(feat), dim=-1)
        score = 0.5 * (1.0 - torch.sum(q[:, 1:] * k[:, :-1], dim=-1))
        if energy is not None:
            delta = torch.abs(energy[:, 1:] - energy[:, :-1])
            if delta.ndim == 3:
                delta = delta.mean(dim=-1)
            score = torch.clamp(score * (1.0 + torch.sigmoid(self.energy_strength) * torch.sigmoid(delta)), max=1.0)
        return F.pad(score, (1, 0), value=1.0)


class PhysicsChunker(nn.Module):
    """Boundary prediction from acoustic cues plus learned similarity, hard Top-K, and band pooling.

    Boundary logit per bin: ``alpha * (1 - temporal correlation) + beta * phase
    discontinuity + gamma * spectral peakness + similarity residual + bias``.
    """

    RATIO_LOSS_MODES = ("density", "fg")
    STE_FULL_EPOCH = 80

    def __init__(self, *, band_width: int, feature_dim: int, target_ratio: float, max_chunks: int,
                 band_id: int, global_start: int, global_enc_dim: int, use_energy: bool = True,
                 energy_feature_mode: str = "mean", ratio_loss_mode: str = "density", ste_start_epoch: int = 30):
        super().__init__()
        if ratio_loss_mode not in self.RATIO_LOSS_MODES:
            raise ValueError(f"ratio_loss_mode must be one of {self.RATIO_LOSS_MODES}")
        self.band_width = int(band_width)
        self.target_ratio = float(target_ratio)
        self.max_chunks = int(max_chunks)
        self.band_id = int(band_id)
        self.use_energy = bool(use_energy)
        self.energy_feature_mode = energy_feature_mode
        self.ratio_loss_mode = ratio_loss_mode
        self.ste_start_epoch = int(ste_start_epoch)
        self.num_tokens = desired_token_count(self.band_width, self.target_ratio, self.max_chunks)

        self.boundary_pool_gain = nn.Parameter(torch.tensor(0.5))
        self.interior_pool_gain = nn.Parameter(torch.tensor(0.15))
        self.encoder_proj = nn.Sequential(nn.GroupNorm(1, 2, eps=1e-8), nn.Conv2d(2, feature_dim, 1))
        hidden = max(1, feature_dim // 2)
        self.chunk_attn = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

        self.alpha = nn.Parameter(torch.tensor(5.0))   # temporal co-modulation
        self.beta = nn.Parameter(torch.tensor(2.0))    # phase discontinuity
        self.gamma = nn.Parameter(torch.tensor(2.0))   # harmonic peakness
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(self.band_width))
        with torch.no_grad():
            # Initial bias by absolute bin: dense low frequencies, sparse high frequencies.
            abs_bins = global_start + torch.arange(self.band_width)
            self.bias.fill_(-3.0)
            self.bias[abs_bins < min(168, global_enc_dim)] = -0.5
            self.bias[abs_bins < min(80, global_enc_dim)] = 0.5
            self.bias[abs_bins < min(40, global_enc_dim)] = 1.5

    @staticmethod
    def physics_features(spec_ri: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mag = torch.norm(spec_ri, dim=1) + 1e-6
        phase = torch.atan2(spec_ri[:, 1], spec_ri[:, 0])
        mag_norm = F.normalize(mag - mag.mean(dim=-1, keepdim=True), dim=-1)
        term_time = 1.0 - (mag_norm[:, :-1] * mag_norm[:, 1:]).sum(dim=-1)
        phase_diff = torch.angle(torch.exp(1j * (phase[:, 1:] - phase[:, :-1])))
        term_phase = phase_diff.std(dim=-1)
        mag_pad = F.pad(mag.mean(dim=-1), (1, 1), mode="replicate")
        term_peak = torch.relu(2 * mag_pad[:, 1:-1] - mag_pad[:, :-2] - mag_pad[:, 2:])
        return term_time, term_phase, term_peak

    def boundary_probs(self, spec_ri: Tensor, router: SharedChunkRouterBackbone,
                       similarity: SharedBandSimilarityProjection) -> Tensor:
        term_time, term_phase, term_peak = self.physics_features(spec_ri)
        router_feat = router(spec_ri, use_energy=self.use_energy)
        energy = _energy_features(spec_ri, self.energy_feature_mode) if self.use_energy else None
        residual = self.residual_scale * similarity(router_feat, self.band_id, energy)
        logits = (self.alpha * F.pad(term_time, (1, 0), value=1.0)
                  + self.beta * F.pad(term_phase, (1, 0), value=1.0)
                  + self.gamma * term_peak + residual + self.bias)
        logits[:, 0] = 10.0
        return torch.sigmoid(logits)

    def pool(self, features: Tensor, probs: Tensor, hard: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Confidence-weighted segment pooling of ``features [B, F, D, T]``."""
        bsz, freq, dim, time = features.shape
        is_boundary = (hard > 0.5).to(torch.long)
        boundary_float = is_boundary.to(features.dtype)
        seg = torch.clamp(torch.cumsum(is_boundary, dim=1) - 1, 0, self.max_chunks - 1).long()
        flat_ids = flat_segment_ids(seg, self.max_chunks)
        num_segments = bsz * self.max_chunks

        p = probs.detach()
        pool_weights = (1.0 + torch.sigmoid(self.boundary_pool_gain) * boundary_float * p
                        + torch.sigmoid(self.interior_pool_gain) * (1.0 - boundary_float) * p)
        scores = self.chunk_attn(features.mean(dim=-1)).squeeze(-1) + torch.log(pool_weights.clamp(min=1e-6))
        tokens = segment_softmax_pool(features, scores, flat_ids, num_segments).view(bsz, self.max_chunks, dim, time)

        counts = torch.zeros(num_segments, device=features.device, dtype=scores.dtype)
        counts.scatter_add_(0, flat_ids, torch.ones_like(scores.reshape(-1)))
        counts = counts.view(bsz, self.max_chunks)
        confidence = torch.where(is_boundary.bool(), p, 1.0 - p)
        conf_sum = torch.zeros(num_segments, device=features.device, dtype=confidence.dtype)
        conf_sum.scatter_add_(0, flat_ids, confidence.reshape(-1))
        conf_mean = conf_sum.view(bsz, self.max_chunks) / counts.clamp(min=1.0)

        # Probability of the first bin of every token (its boundary).
        positions = torch.arange(freq, device=features.device, dtype=torch.long).view(1, freq).expand(bsz, -1)
        boundary_flat = is_boundary.reshape(-1).bool()
        first = torch.full((num_segments,), freq, device=features.device, dtype=torch.long)
        if boundary_flat.any():
            first.scatter_reduce_(0, flat_ids[boundary_flat], positions.reshape(-1)[boundary_flat],
                                  reduce="amin", include_self=True)
        first = first.view(bsz, self.max_chunks)
        valid = first < freq
        boundary_p = probs.gather(1, first.clamp(max=max(freq - 1, 0)))
        token_probs = (0.5 * (boundary_p + conf_mean) * valid.to(boundary_p.dtype)).to(dtype=features.dtype)
        return tokens, token_probs, (counts > 0) & valid

    def ratio_loss(self, probs: Tensor, hard: Tensor, current_epoch: int) -> Tensor:
        """Boundary-density regularizer.

        ``"density"`` pulls the mean boundary probability to the Top-K token
        density. ``"fg"`` is the original TIGER loss on the hard density F and
        the soft density G; during training F receives a straight-through
        gradient ramped in linearly from ``ste_start_epoch`` to epoch 80.
        """
        if self.ratio_loss_mode == "density":
            return (probs.mean(dim=1) - self.num_tokens / float(probs.shape[1])).square().mean()
        start, end = self.ste_start_epoch, self.STE_FULL_EPOCH
        weight = 0.0 if current_epoch < start else 1.0 if current_epoch >= end else (current_epoch - start) / (end - start)
        if self.training and weight > 0.0:
            hard = hard + (probs - probs.detach()) * weight
            hard[:, 0] = 1.0
        f, g = torch.mean(hard), torch.mean(probs)
        n = self.target_ratio if self.target_ratio > 1.0 else 1.0 + 1e-4
        return (n / (n - 1)) * ((n - 1) * f * g + (1 - f) * (1 - g))

    def forward(self, spec_ri: Tensor, router: SharedChunkRouterBackbone,
                similarity: SharedBandSimilarityProjection, current_epoch: int = 0):
        """``spec_ri [B, 2, F, T]`` -> ``(tokens, token_probs, hard, probs, token_mask, ratio_loss)``."""
        probs = self.boundary_probs(spec_ri, router, similarity)
        hard = topk_boundaries(probs, self.num_tokens)
        features = self.encoder_proj(spec_ri).permute(0, 2, 1, 3)
        tokens, token_probs, token_mask = self.pool(features, probs, hard)
        return tokens, token_probs, hard, probs, token_mask, self.ratio_loss(probs, hard, current_epoch)


class FreqDechunker(nn.Module):
    """Token smoothing and expansion, frequency refinement, and the shared complex mask head."""

    def __init__(self, *, feature_dim: int, num_sources: int, band_width: int, refine_kernel_size: int = 3,
                 refine_layers: int = 1, use_mixture_context: bool = False):
        super().__init__()
        if refine_kernel_size < 1 or refine_kernel_size % 2 == 0:
            raise ValueError("refine_kernel_size must be a positive odd integer")
        self.num_sources = int(num_sources)
        self.band_width = int(band_width)
        self.spec_context_proj = (
            nn.Sequential(nn.Conv2d(4, feature_dim, kernel_size=1, bias=False),
                          nn.GroupNorm(1, feature_dim, eps=1e-8), nn.PReLU())
            if use_mixture_context else None
        )
        padding = refine_kernel_size // 2
        self.freq_refine_blocks = nn.ModuleList([
            nn.ModuleDict({
                "dw": nn.Conv2d(feature_dim, feature_dim, kernel_size=(refine_kernel_size, 1),
                                padding=(padding, 0), groups=feature_dim, bias=False),
                "pw": nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=True),
                "norm": nn.GroupNorm(1, feature_dim, eps=1e-8),
                "act": nn.PReLU(),
            })
            for _ in range(int(refine_layers))
        ])
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(feature_dim, feature_dim * 2, 1),
            nn.PReLU(),
            nn.Conv1d(feature_dim * 2, 4 * num_sources, 1, groups=num_sources),
        )

    @staticmethod
    def smooth(tokens: Tensor, token_probs: Tensor, token_mask: Tensor) -> Tensor:
        bsz, count = tokens.shape[:2]
        outs = [tokens[:, 0]]
        for i in range(1, count):
            p = token_probs[:, i].view(bsz, 1, 1)
            value = p * tokens[:, i] + (1 - p) * outs[-1]
            outs.append(torch.where(token_mask[:, i].view(bsz, 1, 1), value, torch.zeros_like(value)))
        return torch.stack(outs, dim=1)

    def forward(self, tokens: Tensor, hard: Tensor, probs: Tensor, token_probs: Tensor,
                token_mask: Tensor, spec_complex: Tensor) -> Tensor:
        """Separated tokens ``[B, N, D, T]`` -> masked spectra ``[B, sources, F, T]``."""
        bsz, count, dim, time = tokens.shape
        freq = self.band_width
        smoothed = self.smooth(tokens, token_probs, token_mask)
        seg = (torch.cumsum(hard, dim=1).long() - 1).clamp(min=0, max=count - 1)
        expanded = torch.gather(smoothed, 1, seg.view(bsz, freq, 1, 1).expand(bsz, freq, dim, time))

        # Confidence path: value 1 in the forward pass, gradient of the boundary probability backward.
        c = torch.where(hard > 0.5, probs, 1.0 - probs).to(dtype=expanded.dtype).view(bsz, freq, 1, 1)
        z = (c + (1.0 - c).detach()) * expanded

        z = z.permute(0, 2, 1, 3).contiguous()
        if self.spec_context_proj is not None:
            context = torch.stack([spec_complex.real, spec_complex.imag, spec_complex.abs(),
                                   c.expand(bsz, freq, 1, time).squeeze(2)], dim=1)
            z = z + self.spec_context_proj(context)
        for block in self.freq_refine_blocks:
            z = block["norm"](z + block["pw"](block["act"](block["dw"](z))))
        z = z.permute(0, 2, 1, 3).contiguous()

        params = self.mask_net(z.view(bsz * freq, dim, time)).view(bsz, freq, self.num_sources, 2, 2, time)
        mask_real = params[..., 0, 0, :] * torch.sigmoid(params[..., 0, 1, :])
        mask_imag = params[..., 1, 0, :] * torch.sigmoid(params[..., 1, 1, :])
        # Mixture consistency: the source masks sum to 1 + 0j.
        mask_real = mask_real - (mask_real.sum(dim=2, keepdim=True) - 1) / self.num_sources
        mask_imag = mask_imag - mask_imag.sum(dim=2, keepdim=True) / self.num_sources
        spec_real, spec_imag = spec_complex.real.unsqueeze(2), spec_complex.imag.unsqueeze(2)
        estimate = torch.complex(spec_real * mask_real - spec_imag * mask_imag,
                                 spec_real * mask_imag + spec_imag * mask_real)
        return estimate.permute(0, 2, 1, 3)


class BiSpectrogramFLABlock(nn.Module):
    """Optional bidirectional gated-linear-attention enhancer of the chunker input.

    Requires ``flash-linear-attention`` (CUDA/Triton). It refines the
    spectrum features used for routing and tokenization; masks are still
    applied to the raw spectrum.
    """

    def __init__(self, in_channels: int = 2, d_model: int = 64):
        super().__init__()
        try:
            from fla.layers import GatedLinearAttention
        except ImportError as error:
            raise ImportError("use_fla=True requires the flash-linear-attention package "
                              "(pip install flash-linear-attention)") from error
        num_heads = d_model // 32 if d_model % 32 == 0 else 1
        self.input_proj = nn.Sequential(nn.Conv2d(in_channels, d_model, kernel_size=1),
                                        nn.GroupNorm(1, d_model), nn.SiLU())
        self.fla_fwd = GatedLinearAttention(hidden_size=d_model, num_heads=num_heads, mode="chunk")
        self.fla_bwd = GatedLinearAttention(hidden_size=d_model, num_heads=num_heads, mode="chunk")
        self.norm = nn.LayerNorm(d_model * 2)
        self.output_proj = nn.Conv2d(d_model * 2, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, freq, time = x.shape
        feats = self.input_proj(x).permute(0, 3, 2, 1).contiguous().view(batch * time, freq, -1)
        pad = math.ceil(freq / 64) * 64 - freq
        if pad > 0:
            feats = F.pad(feats.transpose(1, 2), (0, pad)).transpose(1, 2).contiguous()
        fwd = self.fla_fwd(feats)
        fwd = fwd[0] if isinstance(fwd, tuple) else fwd
        bwd = self.fla_bwd(feats.flip(dims=[1]))
        bwd = (bwd[0] if isinstance(bwd, tuple) else bwd).flip(dims=[1])
        out = self.norm(torch.cat([fwd[:, :freq], bwd[:, :freq]], dim=-1))
        return self.output_proj(out.view(batch, time, freq, -1).permute(0, 3, 2, 1)) + x


class DBSTIGER(nn.Module, PyTorchModelHubMixin, **HUB_KWARGS):
    """TIGER with Dynamic Band Splitting (mono input).

    ``region_configs`` lists the coarse regions as ``start``/``end`` bins with
    ``max_chunks`` (token slots), ``target_ratio`` (bins per token; the hard
    Top-K keeps ``round(width / target_ratio)`` tokens), and ``use_energy``
    (whether the router sees energy descriptors in that region).

    ``forward(mixture)`` returns ``[B, sources, L]``; ``return_aux=True``
    returns ``(estimate, aux)`` with the boundary ratio loss and routes.
    """

    def __init__(
        self,
        out_channels: int = 132,
        in_channels: int = 256,
        num_blocks: int = 8,
        upsampling_depth: int = 5,
        num_sources: int = 3,
        win: int = 2048,
        stride: int = 512,
        sample_rate: int = 44100,
        region_configs: Sequence[Mapping[str, Any]] = (
            dict(start=0, end=40, max_chunks=32, target_ratio=1.25, use_energy=True),
            dict(start=40, end=80, max_chunks=32, target_ratio=1.25, use_energy=True),
            dict(start=80, end=352, max_chunks=32, target_ratio=8.5, use_energy=False),
            dict(start=352, end=1025, max_chunks=32, target_ratio=21.03125, use_energy=False),
        ),
        energy_feature_mode: str = "mean_peak_std",
        dechunker_refine_kernel_size: int = 3,
        dechunker_refine_layers: int = 1,
        use_mixture_context: bool = False,
        ratio_loss_mode: str = "density",
        normalize_ratio_loss: bool = True,
        ste_start_epoch: int = 30,
        use_fla: bool = False,
    ):
        super().__init__()
        if energy_feature_mode not in ("mean", "mean_peak_std"):
            raise ValueError(f"unknown energy_feature_mode {energy_feature_mode!r}")
        self.win = int(win)
        self.stride = int(stride)
        self.sample_rate = int(sample_rate)
        self.enc_dim = self.win // 2 + 1
        self.num_sources = int(num_sources)
        self.normalize_ratio_loss = bool(normalize_ratio_loss)
        self.region_configs = [dict(cfg) for cfg in region_configs]
        validate_region_configs(self.region_configs, self.enc_dim)
        self.band_slices = [(int(cfg["start"]), int(cfg["end"])) for cfg in self.region_configs]

        self.shared_router_backbone = SharedChunkRouterBackbone(out_channels, energy_feature_mode)
        self.shared_similarity_projection = SharedBandSimilarityProjection(out_channels, len(self.region_configs))
        self.separator = Recurrent(out_channels, in_channels, upsampling_depth=upsampling_depth,
                                   iterations=num_blocks)
        self.chunkers = nn.ModuleList()
        self.dechunkers = nn.ModuleList()
        for index, (cfg, (start, end)) in enumerate(zip(self.region_configs, self.band_slices)):
            self.chunkers.append(PhysicsChunker(
                band_width=end - start, feature_dim=out_channels, target_ratio=float(cfg["target_ratio"]),
                max_chunks=int(cfg["max_chunks"]), band_id=index, global_start=start,
                global_enc_dim=self.enc_dim, use_energy=bool(cfg.get("use_energy", True)),
                energy_feature_mode=energy_feature_mode, ratio_loss_mode=ratio_loss_mode,
                ste_start_epoch=ste_start_epoch,
            ))
            self.dechunkers.append(FreqDechunker(
                feature_dim=out_channels, num_sources=self.num_sources, band_width=end - start,
                refine_kernel_size=dechunker_refine_kernel_size, refine_layers=dechunker_refine_layers,
                use_mixture_context=use_mixture_context,
            ))
        # The FLA block is applied on CUDA only, matching the original implementation.
        self.spec_prep = BiSpectrogramFLABlock(in_channels=2, d_model=64) if use_fla else None

    def _window(self, audio: Tensor) -> Tensor:
        return torch.hann_window(self.win).to(audio.device).type(audio.type())

    def forward(self, mixture: Tensor, return_aux: bool = False, current_epoch: int = 0):
        """``current_epoch`` only matters when training with ``ratio_loss_mode="fg"``."""
        if mixture.ndim == 1:
            mixture = mixture.view(1, 1, -1)
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        batch, channels, length = mixture.shape
        audio = mixture.reshape(batch * channels, length)
        spec = torch.stft(audio, n_fft=self.win, hop_length=self.stride, window=self._window(audio),
                          return_complex=True)
        spec_ri = torch.stack([spec.real, spec.imag], 1)
        router_input = spec_ri
        if self.spec_prep is not None and audio.device.type == "cuda":
            router_input = self.spec_prep(spec_ri)

        tokens, masks, states = [], [], []
        ratio_loss = spec_ri.new_tensor(0.0)
        for (start, end), chunker in zip(self.band_slices, self.chunkers):
            x, token_probs, hard, probs, token_mask, region_ratio = chunker(
                router_input[:, :, start:end], self.shared_router_backbone, self.shared_similarity_projection,
                current_epoch)
            tokens.append(x)
            masks.append(token_mask)
            states.append((token_probs, hard, probs, token_mask, x.shape[1]))
            ratio_loss = ratio_loss + region_ratio
        if self.normalize_ratio_loss:
            ratio_loss = ratio_loss / len(self.band_slices)

        separated = self.separator(torch.cat(tokens, dim=1), chunk_mask=torch.cat(masks, dim=1))
        region_specs, offset = [], 0
        for (start, end), dechunker, (token_probs, hard, probs, token_mask, count) in zip(
                self.band_slices, self.dechunkers, states):
            region_specs.append(dechunker(separated[:, offset:offset + count], hard, probs, token_probs,
                                          token_mask, spec[:, start:end]))
            offset += count
        spec_out = torch.cat(region_specs, dim=2)
        estimate = torch.istft(spec_out.reshape(batch * channels * self.num_sources, self.enc_dim, -1),
                               n_fft=self.win, hop_length=self.stride, window=self._window(audio), length=length)
        estimate = estimate.view(batch * channels, self.num_sources, -1)
        if not return_aux:
            return estimate
        aux = dict(
            ratio_loss=ratio_loss,
            boundary_probs=[s[2] for s in states],
            hard_boundaries=[s[1] for s in states],
            token_masks=[s[3] for s in states],
            band_slices=list(self.band_slices),
        )
        return estimate, aux
