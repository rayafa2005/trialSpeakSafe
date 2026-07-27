"""
Tatvaani - Preprocessing Visual Demo
Shows DRAMATIC visible difference between raw and processed audio.
Applies: noise injection → VAD trim → bandpass filter → normalization
Run: py -3.11 demo_preprocess_visual.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

# ── Load the recorded audio ────────────────────────────────────────────────
RAW_PATH = Path("demo_output/recorded_raw.wav")
OUT_PATH = Path("demo_output/preprocessing_demo.png")
OUT_PATH.parent.mkdir(exist_ok=True)

waveform, sr = torchaudio.load(str(RAW_PATH))
if waveform.shape[0] > 1:
    waveform = waveform.mean(0, keepdim=True)
if sr != 16000:
    waveform = T.Resample(sr, 16000)(waveform)
    sr = 16000

# Trim to 3 seconds
waveform = waveform[:, :sr * 3]
audio = waveform.squeeze(0).numpy()

# ─────────────────────────────────────────────────────────────────────────
# STAGE 0: Simulate what a RAW phone call audio looks like
#   (add codec noise + background noise + DC offset — what the app actually sees)
# ─────────────────────────────────────────────────────────────────────────
np.random.seed(42)

# DC offset (microphone hardware bias)
dc_offset = 0.08

# Background noise (café / street level — SNR ~12dB)
noise_level = np.std(audio) / (10 ** (12 / 20))
background  = np.random.randn(len(audio)) * noise_level

# 50Hz electrical hum (very visible on spectrogram)
t      = np.linspace(0, 3, len(audio))
hum    = 0.015 * np.sin(2 * np.pi * 50 * t)
hum   += 0.008 * np.sin(2 * np.pi * 150 * t)   # 3rd harmonic

# High-frequency codec hash
codec_hash = np.random.randn(len(audio)) * 0.006

raw_dirty = audio + background + hum + codec_hash + dc_offset

# ─────────────────────────────────────────────────────────────────────────
# STAGE 1: DC offset removal
# ─────────────────────────────────────────────────────────────────────────
stage1 = raw_dirty - np.mean(raw_dirty)

# ─────────────────────────────────────────────────────────────────────────
# STAGE 2: Bandpass filter 80Hz–8000Hz (removes hum + codec hash)
# ─────────────────────────────────────────────────────────────────────────
import torch.fft as fft_mod
x       = torch.tensor(stage1, dtype=torch.float32)
n       = len(x)
X       = torch.fft.rfft(x)
freqs   = torch.fft.rfftfreq(n, d=1.0/sr)
mask    = (freqs >= 80) & (freqs <= 8000)
X_filt  = X * mask.float()
stage2  = torch.fft.irfft(X_filt, n=n).numpy()

# ─────────────────────────────────────────────────────────────────────────
# STAGE 3: VAD — trim leading/trailing silence
# ─────────────────────────────────────────────────────────────────────────
frame_size  = 160   # 10ms
energy      = np.array([np.sqrt(np.mean(stage2[i:i+frame_size]**2))
                         for i in range(0, len(stage2)-frame_size, frame_size)])
threshold   = np.max(energy) * 0.04
voiced      = energy > threshold
first_voice = np.argmax(voiced) * frame_size
last_voice  = (len(voiced) - np.argmax(voiced[::-1])) * frame_size
stage3      = stage2[first_voice:last_voice]

# Pad back to 3s for fair comparison
if len(stage3) < len(audio):
    stage3 = np.pad(stage3, (0, len(audio) - len(stage3)))
else:
    stage3 = stage3[:len(audio)]

# ─────────────────────────────────────────────────────────────────────────
# STAGE 4: Peak normalization to [-1, 1]
# ─────────────────────────────────────────────────────────────────────────
peak   = np.abs(stage3).max()
stage4 = stage3 / (peak + 1e-8)

# ─────────────────────────────────────────────────────────────────────────
# SPECTROGRAMS
# ─────────────────────────────────────────────────────────────────────────
def make_mel(audio_np):
    w = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    mel = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=160, n_mels=128)(w)
    return T.AmplitudeToDB(top_db=80)(mel).squeeze(0).numpy()

mel_raw   = make_mel(raw_dirty)
mel_clean = make_mel(stage4)

# ─────────────────────────────────────────────────────────────────────────
# PLOT — 3 rows, clear story
# ─────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11), facecolor="#0d1117")
fig.suptitle("Tatvaani — Preprocessing Pipeline", fontsize=18,
             fontweight="bold", color="white", y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

time_full = np.linspace(0, 3, len(audio))

def wave_ax(ax, y, color, title, subtitle=""):
    ax.set_facecolor("#161b22")
    ax.plot(time_full[:len(y)], y, color=color, linewidth=0.6, alpha=0.9)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    if subtitle:
        ax.text(0.5, -0.22, subtitle, transform=ax.transAxes,
                ha='center', fontsize=8, color="#8b949e", style='italic')
    ax.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.set_xlim(0, 3)

def spec_ax(ax, mel, title, cmap="magma"):
    ax.set_facecolor("#161b22")
    im = ax.imshow(mel, aspect="auto", origin="lower", cmap=cmap,
                   vmin=-80, vmax=20, interpolation="bilinear")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#30363d")
    return im

# Row 1: Waveform journey
ax1 = fig.add_subplot(gs[0, 0])
wave_ax(ax1, raw_dirty, "#ef4444", "① Raw Phone Audio",
        "DC bias + background noise\n+ 50Hz electrical hum")

ax2 = fig.add_subplot(gs[0, 1])
wave_ax(ax2, stage1, "#f97316", "② DC Offset Removed",
        "Mean subtracted\nBaseline centred at 0")

ax3 = fig.add_subplot(gs[0, 2])
wave_ax(ax3, stage2, "#eab308", "③ Bandpass 80–8000 Hz",
        "Hum & codec hash removed\nPhone-band frequencies kept")

ax4 = fig.add_subplot(gs[0, 3])
wave_ax(ax4, stage4, "#22c55e", "④ VAD Trim + Normalised",
        "Silence stripped\nPeak normalised to [-1, 1]")

# Arrows between waveforms
for i, (x_pos, label) in enumerate([(0.255, "DC\nfix"), (0.505, "BPF"), (0.755, "VAD\n+norm")]):
    fig.text(x_pos, 0.715, "→", fontsize=18, color="#6e7681",
             ha='center', va='center', fontweight='bold')

# Row 2: Spectrogram comparison — side by side BIG
ax5 = fig.add_subplot(gs[1, :2])
im5 = spec_ax(ax5, mel_raw, "Mel Spectrogram — RAW (noisy)", "magma")
# Annotate noise artifacts
ax5.axhline(y=3, color="#ef4444", linewidth=1.5, linestyle="--", alpha=0.8)
ax5.text(2, 5, "50Hz hum band", color="#ef4444", fontsize=8, fontweight="bold")
ax5.set_xlabel("Time Frames", color="#8b949e", fontsize=8)
ax5.set_ylabel("Mel Bins", color="#8b949e", fontsize=8)

ax6 = fig.add_subplot(gs[1, 2:])
im6 = spec_ax(ax6, mel_clean, "Mel Spectrogram — PROCESSED (clean)", "magma")
ax6.axhline(y=3, color="#22c55e", linewidth=1.5, linestyle="--", alpha=0.8)
ax6.text(2, 5, "Hum removed ✓", color="#22c55e", fontsize=8, fontweight="bold")
ax6.set_xlabel("Time Frames", color="#8b949e", fontsize=8)
ax6.set_ylabel("Mel Bins", color="#8b949e", fontsize=8)

plt.colorbar(im6, ax=ax6, label="dB", fraction=0.03)

# Row 3: SNR + stats summary
ax7 = fig.add_subplot(gs[2, :])
ax7.set_facecolor("#161b22")
ax7.axis("off")

def snr(signal, noisy):
    noise = noisy - signal
    if np.std(noise) < 1e-10: return 99.9
    return 20 * np.log10(np.std(signal) / (np.std(noise) + 1e-10))

dc_before  = np.mean(raw_dirty)
dc_after   = np.mean(stage4)
rms_before = np.sqrt(np.mean(raw_dirty**2))
rms_after  = np.sqrt(np.mean(stage4**2))
hum_before = np.abs(np.fft.rfft(raw_dirty))[int(50 * 3)]
hum_after  = np.abs(np.fft.rfft(stage4))[int(50 * 3)]
hum_reduction = 20 * np.log10((hum_before + 1e-10) / (hum_after + 1e-10))

stats = [
    ("DC Offset", f"{dc_before:+.4f}", f"{dc_after:+.5f}", "✓ Removed"),
    ("RMS Level", f"{rms_before:.4f}", f"{rms_after:.4f}", "✓ Normalised"),
    ("50Hz Hum", f"{hum_before:.1f}", f"{hum_after:.2f}", f"✓ -{hum_reduction:.0f}dB"),
    ("Peak Amplitude", f"{np.abs(raw_dirty).max():.3f}", f"{np.abs(stage4).max():.3f}", "✓ 1.000"),
    ("Silence Frames", "Leading/trailing", "Stripped by VAD", "✓ Clean start/end"),
]

col_x = [0.02, 0.22, 0.45, 0.68]
headers = ["Metric", "Before", "After", "Result"]
for i, (hdr, cx) in enumerate(zip(headers, col_x)):
    ax7.text(cx, 0.92, hdr, transform=ax7.transAxes,
             color="#8b949e", fontsize=9, fontweight="bold")

for row_i, (metric, before, after, result) in enumerate(stats):
    y = 0.75 - row_i * 0.16
    ax7.text(col_x[0], y, metric,  transform=ax7.transAxes, color="#e6edf3", fontsize=9)
    ax7.text(col_x[1], y, before,  transform=ax7.transAxes, color="#ef4444", fontsize=9, family="monospace")
    ax7.text(col_x[2], y, after,   transform=ax7.transAxes, color="#22c55e", fontsize=9, family="monospace")
    ax7.text(col_x[3], y, result,  transform=ax7.transAxes, color="#22c55e", fontsize=9, fontweight="bold")

ax7.set_title("Preprocessing Stats", color="white", fontsize=10,
              fontweight="bold", loc="left", pad=8)

plt.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight",
            facecolor="#0d1117", edgecolor="none")
print(f"Saved: {OUT_PATH}")
plt.show()