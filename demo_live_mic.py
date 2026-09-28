"""
demo_live_mic.py
================
Live Microphone Deepfake Audio Detector Demo (Tatvaani / TatvaNet)

Records 2.0s of live audio directly from your laptop microphone,
runs inference through TatvaNet (or ONNX), and displays real-time stats:
  - Latency (ms)
  - Fake Probability (0.0 to 1.0)
  - Color-Coded Verdict (SAFE, CAUTION, DANGER)
  - Audio Level Meter

Usage:
  python demo_live_mic.py
  python demo_live_mic.py --checkpoint checkpoints/best_model.pt
  python demo_live_mic.py --onnx export/tatvanet.onnx
"""

import os
import sys
import time
import argparse

# Ensure Windows PowerShell prints UTF-8 and emojis cleanly without encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch

try:
    import sounddevice as sd
except ImportError:
    print("⚠️ 'sounddevice' not installed. Installing it now...")
    os.system(f'"{sys.executable}" -m pip install -q sounddevice')
    import sounddevice as sd

try:
    import colorama
    colorama.init()
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
except Exception:
    RED = GREEN = YELLOW = CYAN = BOLD = RESET = ""

from model.tatvanet import TatvaNet

SAMPLE_RATE = 16000
DURATION_SEC = 2.0
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_SEC) # 32,000 samples


def record_microphone(duration=DURATION_SEC, sr=SAMPLE_RATE):
    """Record mono audio from system default microphone."""
    print(f"\n{CYAN}🎤 Recording {duration:.1f}s from microphone... SPEAK NOW!{RESET}", flush=True)
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait() # Wait until recording is finished
    audio = audio.flatten()
    return audio


def run_demo():
    parser = argparse.ArgumentParser(description="Live Microphone Deepfake Demo")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--onnx", default="export/tatvanet.onnx", help="Path to ONNX model")
    parser.add_argument("--threshold_safe", type=float, default=0.35)
    parser.add_argument("--threshold_danger", type=float, default=0.65)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print(f"{BOLD}🎙️ TATVAANI - REAL-TIME VOICE DEEPFAKE DETECTOR DEMO{RESET}")
    print("=" * 65)
    print(f"Hardware Device: {device}")
    print(f"Sample Rate:     {SAMPLE_RATE} Hz")
    print(f"Audio Window:    {DURATION_SEC} seconds ({TARGET_SAMPLES} samples)")
    print(f"Thresholds:      Safe < {args.threshold_safe:.2f} | Danger > {args.threshold_danger:.2f}")

    # Load Model
    if os.path.exists(args.checkpoint):
        print(f"Loading PyTorch checkpoint: {args.checkpoint}...")
        model = TatvaNet(sample_rate=SAMPLE_RATE, n_classes=2).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        use_onnx = False
    elif os.path.exists(args.onnx):
        import onnxruntime as ort
        print(f"Loading ONNX model: {args.onnx}...")
        session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        use_onnx = True
    else:
        print(f"{RED}❌ Error: Neither {args.checkpoint} nor {args.onnx} was found.{RESET}")
        return

    print(f"{GREEN}✅ Model loaded and ready for live demo!{RESET}\n")

    while True:
        try:
            input(f"{BOLD}👉 Press [ENTER] to record 2 seconds (or Ctrl+C to quit)...{RESET}")
            
            # 1. Record
            t_rec_start = time.time()
            raw_audio = record_microphone(duration=DURATION_SEC, sr=SAMPLE_RATE)
            
            # Peak normalize
            peak = np.max(np.abs(raw_audio))
            rms = np.sqrt(np.mean(raw_audio ** 2))
            
            if peak > 1e-5:
                norm_audio = raw_audio / peak
            else:
                norm_audio = raw_audio

            # 2. Run Inference
            t_inf_start = time.time()
            if not use_onnx:
                tensor_in = torch.from_numpy(norm_audio).unsqueeze(0).to(device) # (1, 32000)
                with torch.no_grad():
                    out = model(tensor_in)
                    logits = out["logits"]
                    probs = torch.softmax(logits, dim=-1)[0]
                    p_safe = probs[0].item()
                    p_fake = probs[1].item()
            else:
                inp = norm_audio.reshape(1, TARGET_SAMPLES).astype(np.float32)
                ort_outs = session.run(None, {input_name: inp})
                logits = ort_outs[0][0]
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)
                p_safe = float(probs[0])
                p_fake = float(probs[1])

            latency_ms = (time.time() - t_inf_start) * 1000.0

            # 3. Determine Verdict
            if p_fake > args.threshold_danger:
                verdict_str = f"{RED}{BOLD}🚨 DANGER — AI DEEPFAKE / CLONED VOICE DETECTED!{RESET}"
                conf_pct = p_fake * 100.0
                bar = "█" * int(p_fake * 30) + "░" * (30 - int(p_fake * 30))
            elif p_fake < args.threshold_safe:
                verdict_str = f"{GREEN}{BOLD}🛡️ SAFE — AUTHENTIC HUMAN VOICE{RESET}"
                conf_pct = p_safe * 100.0
                bar = "█" * int(p_safe * 30) + "░" * (30 - int(p_safe * 30))
            else:
                verdict_str = f"{YELLOW}{BOLD}⚠️ CAUTION — BORDERLINE / NOISY VOICE{RESET}"
                conf_pct = (1.0 - abs(p_fake - 0.5) * 2.0) * 100.0
                bar = "█" * 15 + "░" * 15

            # 4. Print Dashboard
            print("\n" + "─" * 65)
            print(f"📊 {BOLD}LIVE INFERENCE RESULTS:{RESET}")
            print(f"   Peak Level:       {peak:6.3f} | RMS Energy: {rms:6.4f}")
            print(f"   Inference Speed:  {CYAN}{latency_ms:5.1f} ms{RESET} (Android Target: < 50ms)")
            print(f"   Fake Probability: {BOLD}{p_fake * 100:5.1f}%{RESET}")
            print(f"   Verdict:          {verdict_str}")
            print(f"   Confidence Meter: [{bar}] {conf_pct:5.1f}%")
            print("─" * 65 + "\n")

        except KeyboardInterrupt:
            print(f"\n{CYAN}Exiting demo. Good luck with your presentation! 🎉{RESET}")
            break
        except Exception as e:
            print(f"{RED}Error during live run: {e}{RESET}")


if __name__ == "__main__":
    run_demo()
