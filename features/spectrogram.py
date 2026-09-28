"""
features/spectrogram.py
=======================
Tatvaani ML — Multi-Stream Spectrogram Extractor (Path B input)

Computes three spectral representations from the same audio buffer:
  1. Mel Spectrogram  — captures tonal texture and formant structure
  2. STFT Magnitude   — captures phase-related artifacts and fine frequency detail
  3. CQT              — captures harmonic structure (geometric frequency spacing)

Uses ConvSTFT (1D convolution DFT basis) for 100% ONNX and TFLite export compatibility
without complex-number symbolic issues.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF

SAMPLE_RATE   = 16000
N_FFT         = 1024
HOP_LENGTH    = 160
N_MELS        = 128
TARGET_HEIGHT = 128
TOP_DB        = 80.0
N_CQT_BINS    = 84
BINS_PER_OCT  = 12
FMIN_CQT      = 32.7


class ConvSTFT(nn.Module):
    """
    Export-safe Short-Time Fourier Transform implemented via Conv1d.
    Avoids PyTorch complex64 ONNX exporter errors.
    """
    def __init__(self, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, win_length: int = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        win_length = win_length or n_fft
        window = torch.hann_window(win_length)
        
        n_stft = n_fft // 2 + 1
        t = torch.arange(n_fft).float()
        k = torch.arange(n_stft).unsqueeze(1).float()
        
        basis_real = torch.cos(2 * math.pi * k * t / n_fft) * window
        basis_imag = -torch.sin(2 * math.pi * k * t / n_fft) * window
        
        # (2 * n_stft, 1, n_fft)
        weight = torch.stack([basis_real, basis_imag], dim=1).reshape(-1, 1, n_fft)
        self.register_buffer("weight", weight.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N) or (B, 1, N)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        pad = self.n_fft // 2
        x = F.pad(x, (pad, pad), mode="reflect")
        out = F.conv1d(x, self.weight, stride=self.hop_length)
        n_stft = self.n_fft // 2 + 1
        out = out.reshape(out.shape[0], n_stft, 2, out.shape[-1])
        mag = torch.sqrt(out[:, :, 0, :] ** 2 + out[:, :, 1, :] ** 2 + 1e-12)
        return mag


def _to_db(x: torch.Tensor, top_db: float = TOP_DB) -> torch.Tensor:
    log_spec = 20.0 * torch.log10(torch.clamp(x, min=1e-5))
    return log_spec


def _normalize_spec(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    b = orig_shape[0]
    x_flat = x.reshape(b, -1)
    x_min = x_flat.min(dim=1)[0]
    x_max = x_flat.max(dim=1)[0]
    for _ in range(x.dim() - 1):
        x_min = x_min.unsqueeze(-1)
        x_max = x_max.unsqueeze(-1)
    return (x - x_min) / (x_max - x_min + 1e-8)


def _resize_to_target(out: torch.Tensor) -> torch.Tensor:
    if out.dim() == 3:
        out = out.unsqueeze(1)
    out = F.interpolate(
        out,
        size=(TARGET_HEIGHT, out.shape[-1]),
        mode="bilinear",
        align_corners=False
    )
    return out


class MelSpectrogramExtractor(nn.Module):
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.stft = ConvSTFT(N_FFT, HOP_LENGTH)
        mel_fb = aF.melscale_fbanks(
            n_freqs=N_FFT // 2 + 1,
            f_min=20.0,
            f_max=sample_rate / 2.0,
            n_mels=N_MELS,
            sample_rate=sample_rate,
            norm="slaney"
        )
        self.register_buffer("mel_fb", mel_fb.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        mag = self.stft(x)  # (B, 513, T)
        mel = torch.matmul(mag.transpose(1, 2), self.mel_fb).transpose(1, 2)  # (B, 128, T)
        out = _to_db(mel)
        out = _normalize_spec(out)
        return out.unsqueeze(1)  # (B, 1, 128, T)


class STFTExtractor(nn.Module):
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.stft = ConvSTFT(N_FFT, HOP_LENGTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        mag = self.stft(x)  # (B, 513, T)
        out = _to_db(mag)
        out = _normalize_spec(out)
        return _resize_to_target(out)  # (B, 1, 128, T)


class CQTExtractor(nn.Module):
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        weights = self._build_filterbank(N_CQT_BINS, BINS_PER_OCT, FMIN_CQT, N_FFT, sample_rate)
        self.register_buffer("cqt_weights", weights)
        self.stft = ConvSTFT(N_FFT, HOP_LENGTH)

    @staticmethod
    def _build_filterbank(n_bins, bins_per_oct, fmin, n_fft, sr):
        n_stft = n_fft // 2 + 1
        freqs = torch.linspace(0, sr / 2, n_stft)
        cqt_freqs = fmin * (2.0 ** (torch.arange(n_bins).float() / bins_per_oct))
        weights = torch.zeros(n_bins, n_stft)
        for i in range(n_bins):
            fc = cqt_freqs[i].item()
            bw = fc / bins_per_oct
            diff = (freqs - fc).abs()
            mask = diff < bw
            if mask.any():
                weights[i, mask] = 1.0 - diff[mask] / bw
        row_sums = weights.sum(dim=1, keepdim=True)
        return (weights / (row_sums + 1e-8)).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        mag = self.stft(x)  # (B, 513, T)
        cqt = torch.einsum("qf,bft->bqt", self.cqt_weights, mag)  # (B, 84, T)
        out = _to_db(cqt)
        out = _normalize_spec(out)
        return _resize_to_target(out)  # (B, 1, 128, T)


class MultiStreamSpectrogram(nn.Module):
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.mel = MelSpectrogramExtractor(sample_rate)
        self.stft = STFTExtractor(sample_rate)
        self.cqt = CQTExtractor(sample_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        m = self.mel(x)
        s = self.stft(x)
        c = self.cqt(x)
        t = min(m.shape[-1], s.shape[-1], c.shape[-1])
        return torch.cat([m[..., :t], s[..., :t], c[..., :t]], dim=1)

    def get_output_shape(self, input_samples: int) -> tuple:
        return (3, TARGET_HEIGHT, (input_samples - N_FFT) // HOP_LENGTH + 1)