import os
import sys
import glob
import torch
import torchaudio
import torchaudio.transforms as T

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from model.tatvanet import TatvaNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TatvaNet(sample_rate=16000, n_classes=2).to(device)
ckpt = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

print("=" * 105)
print(f"{'FILE NAME':<45} | {'EXPECTED':<12} | {'p_fake':<8} | {'VERDICT':<16} | {'STATUS'}")
print("=" * 105)

audio_files = []
for ext in ["*.wav", "*.flac", "*.mp3", "*.mp4", "*.m4a"]:
    audio_files.extend(glob.glob("test_audio(razz)/**/" + ext, recursive=True))

for fpath in sorted(audio_files):
    fname = os.path.basename(fpath)
    if "checkpoint" in fname:
        continue
    try:
        wav, sr = torchaudio.load(fpath)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = T.Resample(sr, 16000)(wav)
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak
        if wav.shape[1] < 32000:
            wav = torch.nn.functional.pad(wav, (0, 32000 - wav.shape[1]))
        else:
            wav = wav[:, :32000]

        with torch.no_grad():
            p_fake = torch.softmax(model(wav.to(device))["logits"], dim=-1)[0, 1].item()

        is_real = "(real)" in fname.lower() or "whatsapp" in fname.lower()
        expected = "Real Human" if is_real else "Synthetic AI"

        if p_fake > 0.65:
            verdict, matched = "🔴 DANGER (Fake)", not is_real
        elif p_fake >= 0.35:
            verdict, matched = "🟡 CAUTION", True
        else:
            verdict, matched = "🟢 SAFE (Real)", is_real

        status = "✅ MATCH" if matched else "❌ MISMATCH"
        print(f"{fname[:43]:<45} | {expected:<12} | {p_fake:<8.4f} | {verdict:<16} | {status}")
    except Exception as e:
        print(f"{fname[:43]:<45} | ERROR: {e}")

print("=" * 105)
