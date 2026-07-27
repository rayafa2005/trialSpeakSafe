"""
features/delta.py
=================
Tatvaani ML — Delta (Δ) and Delta-Delta (ΔΔ) Temporal Dynamics

This is the core of Tatvaani's architectural novelty.

WHY DELTA STREAMS MATTER FOR DEEPFAKE DETECTION:
  Natural human speech has characteristic temporal dynamics — the way
  spectral features change frame-to-frame follows physical constraints
  of the vocal tract (inertia, articulatory speed limits).

  AI vocoders synthesize audio frame-by-frame. Their transitions are
  often too smooth (GAN over-smoothing) or contain micro-discontinuities
  at synthesis boundaries that human speech never exhibits.

  Delta (Δ)   = rate of change per frame
  Delta-Delta (ΔΔ) = acceleration of change

INPUT / OUTPUT:
  Input:  (batch, 3, 128, T)  — 3-channel [Mel, STFT, CQT]
  Output: (batch, 9, 128, T)  — 9-channel:
    Ch 0-2: originals  [Mel,   STFT,   CQT  ]
    Ch 3-5: Δ streams  [ΔMel,  ΔSTFT,  ΔCQT ]
    Ch 6-8: ΔΔ streams [ΔΔMel, ΔΔSTFT, ΔΔCQT]

IMPLEMENTATION NOTE:
  The delta kernel is applied via F.conv1d along the time axis.
  We avoid F.pad with mode="replicate" on 4D tensors because PyTorch
  only supports replicate padding for 3D/4D with constant-only for dims > 2.
  Instead we manually replicate edge frames before convolving.
  This is mathematically identical and works on all PyTorch versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaExtractor(nn.Module):
    """
    Computes Δ and ΔΔ features for each channel of a spectrogram tensor.

    Delta uses the standard HTK/Kaldi regression formula:
      Δ[t] = Σ_{n=1}^{N} n*(x[t+n] - x[t-n]) / (2 * Σ_{n=1}^{N} n²)

    Args:
        window: Regression half-width N. Default 2 (5-frame window).
    """

    def __init__(self, window: int = 2):
        super().__init__()
        self.window = window

        # Build 1D delta kernel shape: (1, 1, kernel_size)
        # Correct shape for F.conv1d (out_ch, in_ch, kernel_size)
        kernel = self._build_kernel(window)
        self.register_buffer("delta_kernel", kernel)   # not a learned parameter

    @staticmethod
    def _build_kernel(window: int) -> torch.Tensor:
        """
        Build the normalized regression delta kernel.
        For window=2: n = [-2,-1,0,1,2], norm = 2*(1+4) = 10
        Returns: (1, 1, 2*window+1) float32 — shape for F.conv1d
        """
        n    = torch.arange(-window, window + 1).float()
        norm = 2.0 * sum(i ** 2 for i in range(1, window + 1))
        k    = n / norm
        return k.view(1, 1, -1)   # (out=1, in=1, kernel_size)

    def _compute_delta(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute first-order delta along the time axis.

        Args:
            x: (batch, channels, freq, time)
        Returns:
            (batch, channels, freq, time) — same shape
        """
        batch, channels, freq, time = x.shape
        pad = self.window

        # Flatten batch/channel/freq into one dimension for 1D conv
        # Shape: (batch*channels*freq, time)
        x_flat = x.reshape(batch * channels * freq, time)

        # Manual edge replication — avoids the 4D replicate padding PyTorch bug
        # Replicate first `pad` frames on the left, last `pad` frames on the right
        left  = x_flat[:, :pad]       # (N, pad)  — repeat left edge
        right = x_flat[:, -pad:]      # (N, pad)  — repeat right edge
        x_padded = torch.cat([left, x_flat, right], dim=1)  # (N, time+2*pad)

        # Reshape for conv1d: (N, 1, time+2*pad)
        x_padded = x_padded.unsqueeze(1)

        # Apply delta kernel: (1, 1, kernel_size) convolved over time
        delta_flat = F.conv1d(x_padded, self.delta_kernel, padding=0)
        # Shape: (N, 1, time)

        # Restore original shape
        delta = delta_flat.squeeze(1).reshape(batch, channels, freq, time)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        if x.dim() == 3 and x.shape[1] == 1: x = x.squeeze(1)  # accept [B,1,N]
        Args:
            x: (batch, 3, 128, T) — 3-channel spectrogram
        Returns:
            (batch, 9, 128, T) — original + Δ + ΔΔ concatenated

        Channel layout:
            0: Mel      3: Δ-Mel      6: ΔΔ-Mel
            1: STFT     4: Δ-STFT     7: ΔΔ-STFT
            2: CQT      5: Δ-CQT      8: ΔΔ-CQT
        """
        delta  = self._compute_delta(x)       # (batch, 3, 128, T)
        delta2 = self._compute_delta(delta)   # (batch, 3, 128, T)
        return torch.cat([x, delta, delta2], dim=1)  # (batch, 9, 128, T)

    def get_channel_names(self) -> list:
        return ["Mel", "STFT", "CQT",
                "Δ-Mel", "Δ-STFT", "Δ-CQT",
                "ΔΔ-Mel", "ΔΔ-STFT", "ΔΔ-CQT"]


class FeaturePipeline(nn.Module):
    """
    Complete Path B feature pipeline: raw PCM → 9-channel delta tensor.
    Used by the training loop and the server inference endpoint.
    The Android app replicates this in Kotlin DSP code.

    Input:  (batch, samples) — 16kHz mono PCM float32
    Output: (batch, 9, 128, T)
    """

    def __init__(self, sample_rate: int = 16000, delta_window: int = 2):
        super().__init__()
        from features.spectrogram import MultiStreamSpectrogram
        self.spectrogram = MultiStreamSpectrogram(sample_rate=sample_rate)
        self.delta       = DeltaExtractor(window=delta_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        spec = self.spectrogram(x)   # (batch, 3, 128, T)
        return self.delta(spec)      # (batch, 9, 128, T)


# ─── Unit test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing DeltaExtractor...")

    batch, ch, freq, time = 4, 3, 128, 100
    x     = torch.randn(batch, ch, freq, time)
    delta = DeltaExtractor(window=2)
    out   = delta(x)

    assert out.shape == (batch, 9, freq, time), f"Shape wrong: {out.shape}"
    assert torch.allclose(out[:, :3], x),       "Original channels modified"
    assert not torch.allclose(out[:, 3:6], x),  "Delta same as input — bug"

    # Delta of a constant signal must be zero
    const     = torch.ones(1, 1, 10, 50)
    out_const = DeltaExtractor(window=2)._compute_delta(const)
    assert out_const.abs().max() < 1e-5, f"Delta of constant != 0: {out_const.abs().max()}"

    print(f"  Output shape: {out.shape}")
    print(f"  Channels: {delta.get_channel_names()}")
    print("  All assertions PASSED.")

    print("\nTesting FeaturePipeline...")
    samples = 2 * 16000
    x_audio = torch.randn(batch, samples)
    pipe    = FeaturePipeline(sample_rate=16000)
    out_f   = pipe(x_audio)
    assert out_f.shape[1] == 9 and out_f.shape[2] == 128, f"Pipeline shape wrong: {out_f.shape}"
    print(f"  Pipeline output: {out_f.shape}")
    print("  PASSED.")