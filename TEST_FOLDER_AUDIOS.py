"""
=============================================================================
TEST_FOLDER_AUDIOS.py
=============================================================================
Dedicated benchmark script to evaluate audio files in `test_audio(razz)/fake`
and `tr/` against BOTH:
  1. On-Device / Server ONNX model (`tatvaani_android/app/src/main/assets/tatvanet.onnx`)
  2. PyTorch Checkpoint (`checkpoints/best_model.pt`)

Usage:
  python TEST_FOLDER_AUDIOS.py
  python TEST_FOLDER_AUDIOS.py --folder "test_audio(razz)/fake"
  python TEST_FOLDER_AUDIOS.py --folder "tr"
  python TEST_FOLDER_AUDIOS.py --onnx /kaggle/working/tatvanet.onnx --checkpoint /kaggle/working/best_model.pt
=============================================================================
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import onnxruntime as ort

from model.tatvanet import TatvaNet
from training.train import extract_active_2s_window

SAMPLE_RATE = 16000
TARGET_SAMPLES = 32000  # 2.0 seconds @ 16kHz
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".mp4", ".m4a", ".ogg", ".opus"}


def load_raw_audio_16k(fpath: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Loads any audio file (WAV, FLAC, MP3, MP4, M4A, OGG, OPUS) into a mono 16kHz float32 numpy array.
    """
    ext = fpath.suffix.lower()
    if ext not in {".mp4", ".m4a", ".opus"}:
        try:
            wav, sr = torchaudio.load(str(fpath))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wav = T.Resample(sr, target_sr)(wav)
            return wav.squeeze(0).numpy().astype(np.float32)
        except Exception:
            pass

    # Fast ffmpeg pipe fallback for container formats (.mp4, .m4a, .opus, etc.)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-v", "error", "-i", str(fpath),
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(target_sr), "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if proc.returncode == 0 and len(proc.stdout) > 0:
            return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    # Final fallback: librosa
    import librosa
    data, _ = librosa.load(str(fpath), sr=target_sr, mono=True)
    return data.astype(np.float32)


def prepare_model_input(
    raw_audio: np.ndarray,
    target_samples: int = TARGET_SAMPLES,
    active_window: bool = True
) -> np.ndarray:
    """
    Prepares a (1, 32000) float32 peak-normalized audio tensor.
    When active_window=True (default), selects the highest-energy 2.0s voiced window
    so leading dead air in file recordings does not skew evaluation.
    """
    if active_window:
        t_wav = torch.from_numpy(raw_audio).unsqueeze(0)
        t_win = extract_active_2s_window(t_wav, target_n=target_samples, random_crop=False)
        return t_win.numpy().astype(np.float32)

    if len(raw_audio) < target_samples:
        audio = np.pad(raw_audio, (0, target_samples - len(raw_audio)))
    else:
        audio = raw_audio[:target_samples].copy()

    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        audio = audio / peak
    return audio.reshape(1, target_samples).astype(np.float32)


def format_verdict(p_fake: float) -> tuple[str, float]:
    if p_fake < 0.35:
        return "SAFE (Real)", 1.0 - p_fake
    elif p_fake > 0.65:
        return "DANGER (Fake)", p_fake
    else:
        return "CAUTION", 1.0 - abs(p_fake - 0.5) * 2.0


def infer_ground_truth_is_real(fpath: Path) -> bool:
    """
    Determines ground truth label (True = Real Human, False = Synthetic AI):
      - All files inside `tr/` or `real/` are Real Human.
      - Files with `(real)` or `whatsapp` in their filename are Real Human.
      - Otherwise Synthetic AI.
    """
    parent_lower = fpath.parent.name.lower()
    fname_lower = fpath.name.lower()
    if parent_lower in {"tr", "real"}:
        return True
    if "(real)" in fname_lower or "whatsapp" in fname_lower:
        return True
    return False


def evaluate_folder(
    folder: Path,
    ort_session: ort.InferenceSession,
    pt_model: TatvaNet,
    device: torch.device,
    ckpt_info: str,
    onnx_path: str,
    active_window: bool = True,
    print_rows: bool = True
) -> dict:
    files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS and "checkpoint" not in f.name.lower()
    ])
    if not files:
        print(f"[WARNING] No audio files found in {folder}")
        return {}

    if print_rows:
        print("=" * 124)
        print(f"TESTING {len(files)} AUDIO FILES IN: {folder}")
        print(f"  ONNX Model:        {onnx_path}")
        print(f"  PyTorch Model:     {ckpt_info}")
        print("=" * 124)
        print(f"{'FILE NAME':<44} | {'EXPECTED':<12} | {'ONNX p_fake':<11} | {'PT p_fake':<9} | {'VERDICT (CONF)':<22} | {'MATCH'}")
        print("-" * 124)

    real_correct, real_total = 0, 0
    fake_correct, fake_total = 0, 0
    caution_count = 0

    for fpath in files:
        fname = fpath.name
        try:
            raw_audio = load_raw_audio_16k(fpath)
            audio_tensor = prepare_model_input(raw_audio, active_window=active_window)

            # 1. ONNX Inference
            onnx_logits = ort_session.run(None, {"audio": audio_tensor})[0][0]
            exp_onnx = np.exp(onnx_logits - np.max(onnx_logits))
            p_fake_onnx = float(exp_onnx[1] / np.sum(exp_onnx))

            # 2. PyTorch Inference
            with torch.no_grad():
                t_in = torch.from_numpy(audio_tensor).to(device)
                pt_logits = pt_model(t_in)["logits"][0].cpu().numpy()
                exp_pt = np.exp(pt_logits - np.max(pt_logits))
                p_fake_pt = float(exp_pt[1] / np.sum(exp_pt))

            is_real = infer_ground_truth_is_real(fpath)
            expected = "Real Human" if is_real else "Synthetic AI"
            verdict_str, conf = format_verdict(p_fake_onnx)

            if 0.35 <= p_fake_onnx <= 0.65:
                caution_count += 1

            # Binary pass/fail at 0.50 decision boundary + strict 3-tier check
            if is_real:
                real_total += 1
                matched = p_fake_onnx < 0.35
                if matched:
                    real_correct += 1
            else:
                fake_total += 1
                matched = p_fake_onnx > 0.65
                if matched:
                    fake_correct += 1

            if print_rows:
                status = "PASS" if matched else "FAIL"
                verdict_col = f"{verdict_str} ({conf*100:4.1f}%)"
                print(
                    f"{fname[:42]:<44} | {expected:<12} | {p_fake_onnx:<11.4f} | "
                    f"{p_fake_pt:<9.4f} | {verdict_col:<22} | {status}"
                )
        except Exception as e:
            if print_rows:
                print(f"{fname[:42]:<44} | ERROR: {e}")

    n_correct = real_correct + fake_correct
    n_total = real_total + fake_total
    if print_rows:
        print("=" * 124)
        print(
            f"SUMMARY ({folder}): {n_correct}/{n_total} STRICT PASS ({100.0 * n_correct / max(n_total, 1):.1f}%) | "
            f"Real Human SAFE: {real_correct}/{real_total} | Synthetic AI DANGER: {fake_correct}/{fake_total} | "
            f"Caution: {caution_count}"
        )
        print("=" * 124)

    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "real_correct": real_correct,
        "real_total": real_total,
        "fake_correct": fake_correct,
        "fake_total": fake_total,
        "caution": caution_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Test TatvaNet on test_audio(razz) and tr/ folders")
    parser.add_argument("--folder", default=None, help="Specific folder to test (default: tests test_audio(razz)/fake and tr/)")
    parser.add_argument("--onnx", default="tatvaani_android/app/src/main/assets/tatvanet.onnx")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--raw_start", action="store_true", help="Slice first 2.0s [:32000] instead of active voiced 2.0s window")
    args = parser.parse_args()

    ort_session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt_model = TatvaNet(sample_rate=SAMPLE_RATE, n_classes=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pt_model.load_state_dict(ckpt["model_state"], strict=False)
    pt_model.eval()
    ckpt_info = f"{args.checkpoint} (Epoch {ckpt.get('epoch')}, Val EER {100*ckpt.get('val_eer', 0):.2f}%)"

    active_window = not args.raw_start

    if args.folder is not None:
        evaluate_folder(Path(args.folder), ort_session, pt_model, device, ckpt_info, args.onnx, active_window=active_window)
    else:
        razz_folder = Path("test_audio(razz)/fake")
        if razz_folder.exists():
            evaluate_folder(razz_folder, ort_session, pt_model, device, ckpt_info, args.onnx, active_window=active_window)

        tr_folder = Path("tr")
        if tr_folder.exists():
            print()
            evaluate_folder(tr_folder, ort_session, pt_model, device, ckpt_info, args.onnx, active_window=active_window)


if __name__ == "__main__":
    main()
