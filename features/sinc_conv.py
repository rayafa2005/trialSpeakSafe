"""
features/sinc_conv.py
=====================
Tatvaani ML — Learnable Sinc-Function Filterbank (Path A of TatvaNet)

Each filter is a sinc band-pass function with learnable cutoff frequencies.
The network starts with mel-spaced initialization and adapts during training
to discover which frequency bands best expose deepfake artifacts in Indian
language audio.

This is used as Path A in TatvaNet alongside the 9-channel spectrogram Path B.
The dual-path combination + cross-attention fusion is Tatvaani's novel contribution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """
    Learnable sinc band-pass filterbank applied to raw PCM waveform.

    Args:
        n_filters   : Number of filters (output channels). Default 128.
        kernel_size : Filter length in samples. Must be odd. Default 251.
        sample_rate : Audio sample rate. Default 16000.
        min_low_hz  : Minimum lower cutoff. Default 50 Hz.
        min_band_hz : Minimum bandwidth. Default 50 Hz.

    Input:  (batch, 1, time_samples)
    Output: (batch, n_filters, time_samples)
    """

    def __init__(
        self,
        n_filters:    int   = 128,
        kernel_size:  int   = 251,
        sample_rate:  int   = 16000,
        min_low_hz:   float = 50.0,
        min_band_hz:  float = 50.0,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz  = min_low_hz
        self.min_band_hz = min_band_hz

        # ── Mel-spaced initialization ──────────────────────────────────
        # _hz_to_mel / _mel_to_hz accept both float and Tensor safely
        low_hz  = 30.0
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)

        mel_low    = self._hz_to_mel(low_hz)
        mel_high   = self._hz_to_mel(high_hz)
        mel_points = torch.linspace(
            mel_low.item() if isinstance(mel_low, torch.Tensor) else mel_low,
            mel_high.item() if isinstance(mel_high, torch.Tensor) else mel_high,
            n_filters + 1
        )
        hz_points = self._mel_to_hz(mel_points)

        self.low_hz_  = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).unsqueeze(1))

        # Hamming window — pre-computed, not learned
        self.register_buffer("window_", torch.hamming_window(kernel_size))
        n = torch.arange(0, kernel_size).float() - (kernel_size - 1) / 2.0
        self.register_buffer("n_", n)

    # ── Frequency helpers — accept float or Tensor ────────────────────────────

    @staticmethod
    def _hz_to_mel(hz) -> torch.Tensor:
        """Convert Hz to mel. Accepts Python float or torch.Tensor."""
        if not isinstance(hz, torch.Tensor):
            hz = torch.tensor(hz, dtype=torch.float32)
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel) -> torch.Tensor:
        """Convert mel to Hz. Accepts Python float or torch.Tensor."""
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel, dtype=torch.float32)
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # ── Filter construction ───────────────────────────────────────────────────

    def _build_filters(self) -> torch.Tensor:
        """Build sinc filter bank from current learned cutoffs. (n_filters, 1, kernel_size)"""
        low  = self.min_low_hz  + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2
        )

        f1 = low  / self.sample_rate   # Normalized [0, 0.5]
        f2 = high / self.sample_rate

        n  = self.n_.view(1, -1)

        # Band-pass = high low-pass minus low low-pass
        bp = 2 * f2 * self._sinc(2 * f2 * n) - 2 * f1 * self._sinc(2 * f1 * n)
        bp = bp * self.window_
        bp = bp / (2.0 * bp.abs().sum(dim=-1, keepdim=True) + 1e-8)

        return bp.unsqueeze(1)   # (n_filters, 1, kernel_size)

    @staticmethod
    def _sinc(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable sinc: sin(π·x)/(π·x), equals 1 at x=0."""
        x = torch.where(x == 0, torch.tensor(1e-10, device=x.device, dtype=x.dtype), x)
        return torch.sin(math.pi * x) / (math.pi * x)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, time_samples)
        Returns:
            (batch, n_filters, time_samples)
        """
        filters = self._build_filters()
        if x.dim() == 4: x = x.squeeze(1)  # safety: ensure 3D for conv1d
        out = F.conv1d(x, filters, stride=1,
                       padding=self.kernel_size // 2, bias=None)
        return torch.abs(out)

    def get_filter_frequencies(self) -> dict:
        with torch.no_grad():
            low  = (self.min_low_hz + torch.abs(self.low_hz_)).squeeze().cpu().numpy()
            band = (self.min_band_hz + torch.abs(self.band_hz_)).squeeze().cpu().numpy()
            high = low + band
        return {"low_hz": low, "high_hz": high}


# ─── Unit test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing SincConv...")

    batch = 4
    x     = torch.randn(batch, 1, 2 * 16000)
    model = SincConv(n_filters=128, kernel_size=251, sample_rate=16000)
    out   = model(x)

    assert out.shape == (batch, 128, 2 * 16000), f"Shape wrong: {out.shape}"

    loss = out.mean()
    loss.backward()
    assert model.low_hz_.grad  is not None, "No grad for low_hz_"
    assert model.band_hz_.grad is not None, "No grad for band_hz_"

    freqs = model.get_filter_frequencies()
    print(f"  Output: {out.shape}")
    print(f"  Filter 0:   {freqs['low_hz'][0]:.1f}–{freqs['high_hz'][0]:.1f} Hz")
    print(f"  Filter 127: {freqs['low_hz'][127]:.1f}–{freqs['high_hz'][127]:.1f} Hz")
    print("  Gradient check PASSED.")
    print("SincConv PASSED.")