"""
features/spectrogram.py
=======================
Tatvaani ML — Multi-Stream Spectrogram Extractor (Path B input)

Computes three spectral representations from the same audio buffer:
  1. Mel Spectrogram  — captures tonal texture and formant structure
  2. STFT Magnitude   — captures phase-related artifacts and fine frequency detail
  3. CQT              — captures harmonic structure (geometric frequency spacing)

These three are stacked into a 3-channel 2D tensor (like RGB channels in an image).
Delta streams (Δ, ΔΔ) are added in delta.py to make the final 9-channel tensor.

All three representations are resized to the same (128, T) grid so they can be
stacked as channels for the CNN.

ANDROID DSP NOTE: The constants N_FFT, HOP_LENGTH, N_MELS, SAMPLE_RATE must be
replicated exactly in FeatureExtractor.kt. Any mismatch causes wrong tensor shapes
and garbage inference output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


# ─── Fixed constants ──────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
N_FFT         = 1024
HOP_LENGTH    = 160
N_MELS        = 128
TARGET_HEIGHT = 128
TOP_DB        = 80.0
N_CQT_BINS    = 84
BINS_PER_OCT  = 12
FMIN_CQT      = 32.7


def _normalize_spec(x: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalize spectrogram to [0,1] per sample.
    Works for 3D (B, F, T) or 4D (B, C, F, T) input.
    Always returns the same number of dims as input.
    """
    orig_shape = x.shape
    b = orig_shape[0]
    x_flat = x.reshape(b, -1)
    x_min  = x_flat.min(dim=1)[0]
    x_max  = x_flat.max(dim=1)[0]
    # Expand mins/maxs to broadcast against original shape
    for _ in range(x.dim() - 1):
        x_min = x_min.unsqueeze(-1)
        x_max = x_max.unsqueeze(-1)
    return (x - x_min) / (x_max - x_min + 1e-8)


def _resize_to_target(out: torch.Tensor) -> torch.Tensor:
    """
    Resize frequency axis to TARGET_HEIGHT using bilinear interpolation.
    Input can be 3D (B, F, T) or 4D (B, C, F, T).
    Always returns 4D (B, 1, TARGET_HEIGHT, T).
    """
    # Ensure 4D for F.interpolate
    if out.dim() == 3:
        out = out.unsqueeze(1)   # (B, F, T) → (B, 1, F, T)
    # out is now (B, C, F, T)
    out = F.interpolate(
        out,
        size=(TARGET_HEIGHT, out.shape[-1]),
        mode="bilinear",
        align_corners=False
    )
    return out  # (B, C, TARGET_HEIGHT, T)


class MelSpectrogramExtractor(nn.Module):
    """
    Mel Spectrogram: perceptually-scaled frequency representation.
    Input:  (batch, samples)  or  (batch, 1, samples)
    Output: (batch, 1, 128, T)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=20.0,
            f_max=sample_rate / 2,
            power=2.0,
            normalized=True,
        )
        self.to_db = T.AmplitudeToDB(top_db=TOP_DB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, 1, N) from tatvanet or (B, N) from direct call
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.to_db(self.mel(x))   # (B, 128, T)
        out = _normalize_spec(out)      # (B, 128, T)  in [0,1]
        out = out.unsqueeze(1)          # (B, 1, 128, T)
        # Mel is already 128 bins — no resize needed, but run through
        # _resize_to_target for consistency (handles T alignment)
        return out                      # (B, 1, 128, T)


class STFTExtractor(nn.Module):
    """
    STFT Magnitude: fine-grained frequency detail (513 bins → resized to 128).
    Input:  (batch, samples)  or  (batch, 1, samples)
    Output: (batch, 1, 128, T)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.stft  = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH,
                                   power=1.0, normalized=True)
        self.to_db = T.AmplitudeToDB(top_db=TOP_DB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.to_db(self.stft(x))  # (B, 513, T)  — 3D
        out = _normalize_spec(out)      # (B, 513, T)  — 3D, [0,1]
        out = _resize_to_target(out)    # (B, 1, 128, T) — unsqueezes + interpolates
        return out


class CQTExtractor(nn.Module):
    """
    Pseudo-CQT via triangular filterbank on STFT (84 bins → resized to 128).
    Input:  (batch, samples)  or  (batch, 1, samples)
    Output: (batch, 1, 128, T)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        weights = self._build_filterbank(N_CQT_BINS, BINS_PER_OCT,
                                         FMIN_CQT, N_FFT, sample_rate)
        self.register_buffer("cqt_weights", weights)
        self.stft  = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH,
                                   power=1.0, normalized=True)
        self.to_db = T.AmplitudeToDB(top_db=TOP_DB)

    @staticmethod
    def _build_filterbank(n_bins, bins_per_oct, fmin, n_fft, sr):
        n_stft    = n_fft // 2 + 1
        freqs     = torch.linspace(0, sr / 2, n_stft)
        cqt_freqs = fmin * (2.0 ** (torch.arange(n_bins).float() / bins_per_oct))
        weights   = torch.zeros(n_bins, n_stft)
        for i in range(n_bins):
            fc   = cqt_freqs[i].item()
            bw   = fc / bins_per_oct
            diff = (freqs - fc).abs()
            mask = diff < bw
            if mask.any():
                weights[i, mask] = 1.0 - diff[mask] / bw
        row_sums = weights.sum(dim=1, keepdim=True)
        return (weights / (row_sums + 1e-8)).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        stft = self.stft(x)                                          # (B, 513, T)
        cqt  = torch.einsum("qf,bft->bqt", self.cqt_weights, stft)  # (B, 84, T)
        out  = _normalize_spec(self.to_db(cqt))                      # (B, 84, T)
        out  = _resize_to_target(out)                                 # (B, 1, 128, T)
        return out


class MultiStreamSpectrogram(nn.Module):
    """
    Combines Mel + STFT + CQT → (batch, 3, 128, T).
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.mel  = MelSpectrogramExtractor(sample_rate)
        self.stft = STFTExtractor(sample_rate)
        self.cqt  = CQTExtractor(sample_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        m = self.mel(x)    # (B, 1, 128, T)
        s = self.stft(x)   # (B, 1, 128, T)
        c = self.cqt(x)    # (B, 1, 128, T)
        t = min(m.shape[-1], s.shape[-1], c.shape[-1])
        return torch.cat([m[..., :t], s[..., :t], c[..., :t]], dim=1)  # (B, 3, 128, T)

    def get_output_shape(self, input_samples: int) -> tuple:
        return (3, TARGET_HEIGHT, (input_samples - N_FFT) // HOP_LENGTH + 1)


if __name__ == "__main__":
    print("Testing MultiStreamSpectrogram...")
    batch = 4
    x = torch.randn(batch, 2 * SAMPLE_RATE)
    out = MultiStreamSpectrogram()(x)
    assert out.shape[1] == 3 and out.shape[2] == 128, f"Shape wrong: {out.shape}"
    assert out.min() >= -0.01 and out.max() <= 1.01,  f"Range wrong: {out.min():.3f}–{out.max():.3f}"
    print(f"  Output: {out.shape}  Range: [{out.min():.3f}, {out.max():.3f}]")

    # Also test with (B, 1, N) input — as tatvanet sends it
    x2 = torch.randn(batch, 1, 2 * SAMPLE_RATE)
    out2 = MultiStreamSpectrogram()(x2)
    assert out2.shape == out.shape, f"Shape mismatch with 3D input: {out2.shape}"
    print(f"  3D input test: {out2.shape}  ✅")
    print("PASSED.")