"""
Tatvaani — Live Preprocessing & Feature Extraction Demo
Records mic → removes broadband noise (fan/AC/hum) → plays back clean
Shows: Waveform, MFCC, Δ-MFCC, ΔΔ-MFCC, SincConv filterbank

Run: py -3.11 tatvaani_live_demo.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import time

try:
    import sounddevice as sd
except ImportError:
    import subprocess
    subprocess.run(["py", "-3.11", "-m", "pip", "install", "sounddevice"], check=True)
    import sounddevice as sd

SR        = 16000
NOISE_DUR = 1.5
SPEECH_DUR = 4
OUT_DIR   = Path("demo_output")
OUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  TATVAANI — Live Audio Demo")
print("=" * 60)

# ── Step 1: Capture noise profile ─────────────────────────────────────────
print(f"\n🔇 Stay SILENT for {NOISE_DUR}s — capturing noise profile (fan/AC)...")
time.sleep(0.8)
noise_np = sd.rec(int(NOISE_DUR * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()
noise_np = noise_np.flatten()
print("  ✅ Noise profile captured\n")

# ── Step 2: Record speech ──────────────────────────────────────────────────
print(f"🎙  SPEAK NOW — recording {SPEECH_DUR}s...")
time.sleep(0.3)
raw_np = sd.rec(int(SPEECH_DUR * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()
raw_np = raw_np.flatten()
print("  ✅ Recording done\n")

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: DC removal
s1 = raw_np - np.mean(raw_np)

# Stage 2: Spectral subtraction — removes broadband fan/AC noise
# This is the key technique: estimate noise power spectrum from the
# silent recording, then subtract it from every frame of the speech.
def spectral_subtract(signal, noise_profile, frame_size=512, hop=256,
                      over_subtract=2.0, floor=0.002):
    # Build noise power spectrum from silent frames
    n_noise = (len(noise_profile) - frame_size) // hop
    noise_pwr = np.zeros(frame_size // 2 + 1)
    for i in range(n_noise):
        frame = noise_profile[i*hop:i*hop+frame_size] * np.hanning(frame_size)
        noise_pwr += np.abs(np.fft.rfft(frame)) ** 2
    noise_pwr /= max(n_noise, 1)

    output = np.zeros(len(signal))
    win    = np.hanning(frame_size)
    n_frames = (len(signal) - frame_size) // hop

    for i in range(n_frames):
        start   = i * hop
        frame   = signal[start:start+frame_size] * win
        spec    = np.fft.rfft(frame)
        pwr     = np.abs(spec) ** 2
        phase   = np.angle(spec)
        # Subtract noise, keep a floor so we don't get musical noise artifacts
        clean_pwr  = np.maximum(pwr - over_subtract * noise_pwr, floor * pwr)
        clean_spec = np.sqrt(clean_pwr) * np.exp(1j * phase)
        output[start:start+frame_size] += np.fft.irfft(clean_spec) * win

    return output

print("  🔧 Spectral subtraction (removes fan noise)...")
s2 = spectral_subtract(s1, noise_np)

# Stage 3: Bandpass 80–8000 Hz
x     = torch.tensor(s2, dtype=torch.float32)
X     = torch.fft.rfft(x)
freqs = torch.fft.rfftfreq(len(x), d=1.0/SR)
s3    = torch.fft.irfft(X * ((freqs >= 80) & (freqs <= 8000)).float(), n=len(x)).numpy()

# Stage 4: VAD trim
fe     = 160
energy = np.array([np.mean(s3[i:i+fe]**2) for i in range(0, len(s3)-fe, fe)])
voiced = np.where(energy > np.max(energy) * 0.04)[0]
s4     = s3[voiced[0]*fe:(voiced[-1]+1)*fe] if len(voiced) else s3
target = SR * SPEECH_DUR
s4     = np.pad(s4,(0,max(0,target-len(s4))))[:target]

# Stage 5: Peak normalise
clean = (s4 / (np.abs(s4).max() + 1e-8)).astype(np.float32)

print("  ✅ Preprocessing complete\n")

torchaudio.save(str(OUT_DIR/"raw.wav"),   torch.tensor(raw_np).unsqueeze(0), SR)
torchaudio.save(str(OUT_DIR/"clean.wav"), torch.tensor(clean).unsqueeze(0), SR)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
waveform = torch.tensor(clean).unsqueeze(0)

mfcc_tf   = T.MFCC(sample_rate=SR, n_mfcc=13,
                    melkwargs={"n_fft":1024,"hop_length":160,"n_mels":128})
mfcc_t    = mfcc_tf(waveform)
mfcc_np   = mfcc_t.squeeze(0).numpy()
delta_tf  = T.ComputeDeltas()
delta_np  = delta_tf(mfcc_t).squeeze(0).numpy()
delta2_np = delta_tf(delta_tf(mfcc_t)).squeeze(0).numpy()
mel_np    = T.AmplitudeToDB(top_db=80)(
    T.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=160, n_mels=128)(waveform)
).squeeze(0).numpy()

sinc_ok = False
try:
    from features.sinc_conv import SincConv
    sinc = SincConv(n_filters=128, kernel_size=251, sample_rate=SR).eval()
    with torch.no_grad():
        out      = sinc(waveform.unsqueeze(0))
        sinc_np  = (10 * torch.log10(
            F.avg_pool1d(out.squeeze(0).pow(2), 160, 160) + 1e-8
        )).numpy()
    sinc_ok = True
except Exception as e:
    print(f"  SincConv skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
BG,PANEL,BORDER = "#0d1117","#161b22","#30363d"
WHITE,GRAY      = "#e6edf3","#8b949e"
RED,GREEN,BLUE  = "#ef4444","#22c55e","#58a6ff"
AMBER,PURPLE    = "#f59e0b","#a78bfa"

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle("Tatvaani — Live Preprocessing & Feature Extraction Pipeline",
             fontsize=17, fontweight="bold", color=WHITE, y=0.99)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.32)
t   = np.linspace(0, SPEECH_DUR, len(raw_np))

def style(ax, title, xlabel="", ylabel="", tc=WHITE):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=tc, fontsize=10, fontweight="bold", pad=5)
    ax.tick_params(colors=GRAY, labelsize=7)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY, fontsize=8)
    for sp in ax.spines.values(): sp.set_color(BORDER)

# Row 1: waveforms
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(t, raw_np, color=RED, lw=0.5, alpha=0.85)
ax1.axhline(np.mean(raw_np), color=AMBER, lw=1.2, ls="--",
            label=f"DC offset = {np.mean(raw_np):+.4f}")
ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=AMBER, framealpha=0.7)
style(ax1,"① RAW — fan noise + DC bias","Time (s)","Amplitude",RED)
ax1.set_xlim(0,SPEECH_DUR)

ax2 = fig.add_subplot(gs[0, 2:])
ax2.plot(np.linspace(0,SPEECH_DUR,len(clean)), clean, color=GREEN, lw=0.5)
ax2.axhline(0, color=GRAY, lw=0.7, ls="--", alpha=0.4)
ax2.set_ylim(-1.1,1.1)
style(ax2,"② CLEAN — spectral subtraction · BPF · VAD · normalised","Time (s)","Amplitude",GREEN)
ax2.set_xlim(0,SPEECH_DUR)

fig.text(0.505,0.756,"→",fontsize=22,color=GREEN,ha='center',va='center',fontweight='bold')
fig.text(0.505,0.732,"noise removed",fontsize=7,color=GRAY,ha='center',va='center')

# Row 2: features
for ax, data, cm, title, cbar_lbl in [
    (fig.add_subplot(gs[1,0]), mfcc_np,   "plasma",   "MFCC (13 coefficients)", "Energy"),
    (fig.add_subplot(gs[1,1]), delta_np,  "coolwarm", "Δ-MFCC (rate of change)","Δ"),
    (fig.add_subplot(gs[1,2]), delta2_np, "coolwarm", "ΔΔ-MFCC (acceleration)", "ΔΔ"),
    (fig.add_subplot(gs[1,3]), mel_np,    "magma",    "Mel Spectrogram (Path B)","dB"),
]:
    vmin,vmax = (-80,20) if cbar_lbl=="dB" else (None,None)
    im = ax.imshow(data,aspect="auto",origin="lower",cmap=cm,
                   vmin=vmin,vmax=vmax,interpolation="bilinear")
    style(ax,title,"Frames","")
    plt.colorbar(im,ax=ax,fraction=0.05,label=cbar_lbl).ax.tick_params(labelsize=6,colors=GRAY)

# MFCC annotations
ax_m = fig.axes[2]   # first feature ax
for y,lbl in [(0,"C0 energy"),(3,"timbre"),(8,"texture")]:
    ax_m.text(mfcc_np.shape[1]*0.97, y+0.3, lbl, color=WHITE, fontsize=5.5,
              ha='right', bbox=dict(boxstyle='round,pad=0.2',fc=PANEL,alpha=0.8))

# Row 3: SincConv + summary
ax7 = fig.add_subplot(gs[2,:3])
if sinc_ok:
    im5 = ax7.imshow(sinc_np,aspect="auto",origin="lower",cmap="inferno",
                     interpolation="bilinear",extent=[0,SPEECH_DUR,0,128])
    style(ax7,"SincConv Filterbank — Path A (128 learnable band-pass filters on raw waveform)",
          "Time (s)","Filter # (low → high freq)")
    plt.colorbar(im5,ax=ax7,fraction=0.015,pad=0.01,
                 label="Energy dB").ax.tick_params(labelsize=6,colors=GRAY)
    for fi,lbl in [(8,"~150Hz"),(32,"~800Hz"),(64,"~2kHz"),(96,"~4kHz"),(116,"~7kHz")]:
        ax7.text(SPEECH_DUR*0.99,fi,lbl,color=AMBER,fontsize=7,ha='right',va='center',
                 bbox=dict(boxstyle='round,pad=0.15',fc="#0d1117",alpha=0.7))
else:
    ax7.set_facecolor(PANEL)
    ax7.text(0.5,0.5,"SincConv — run from tatvaani_ml/ directory to see filterbank",
             transform=ax7.transAxes,ha='center',va='center',color=GRAY,fontsize=11)
    style(ax7,"SincConv Filterbank — Path A")

ax8 = fig.add_subplot(gs[2,3])
ax8.set_facecolor(PANEL); ax8.axis("off")
style(ax8,"What TatvaNet Sees")
for i,(txt,color,bold) in enumerate([
    ("PATH A — SincConv",   BLUE,   True),
    (" 128 band-pass filters",WHITE,False),
    (" Learnable cutoffs",   WHITE, False),
    (" → 256-dim embedding", AMBER, False),
    ("",WHITE,False),
    ("PATH B — Spectrogram", PURPLE,True),
    (" Mel + STFT + CQT",    WHITE, False),
    (" + Δ + ΔΔ = 9ch",      WHITE, False),
    (" → 256-dim embedding", AMBER, False),
    ("",WHITE,False),
    ("FUSION + GRU",         GREEN, True),
    (" Cross-attention",     WHITE, False),
    (" → SAFE / CAUTION",    GREEN, False),
    ("   / DANGER 🔴",       RED,   False),
]):
    ax8.text(0.05,0.97-i*0.067,txt,transform=ax8.transAxes,color=color,fontsize=8.5,
             fontweight="bold" if bold else "normal",
             family="monospace" if txt.startswith(" ") else "sans-serif")

plt.savefig(str(OUT_DIR/"tatvaani_pipeline_demo.png"),
            dpi=150,bbox_inches="tight",facecolor=BG)
print("  ✅ Saved: demo_output/tatvaani_pipeline_demo.png\n")

# ── Playback ───────────────────────────────────────────────────────────────
print("🔊 RAW audio (fan noise audible)...")
sd.play(raw_np, SR); sd.wait(); time.sleep(0.5)

print("🔊 CLEAN audio (fan noise removed)...")
sd.play(clean, SR); sd.wait()

print("\n" + "="*60 + "\n  Demo complete!\n" + "="*60)
plt.show()