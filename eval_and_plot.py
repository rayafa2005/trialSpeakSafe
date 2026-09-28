import os
import sys
import glob
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from model.tatvanet import TatvaNet

def load_audio_robust(fpath, target_sr=16000, target_samples=32000):
    """Loads audio with torchaudio, fallback to librosa if container format."""
    try:
        wav, sr = torchaudio.load(fpath)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = T.Resample(sr, target_sr)(wav)
    except Exception:
        import librosa
        data, sr = librosa.load(fpath, sr=target_sr, mono=True)
        wav = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak

    if wav.shape[1] < target_samples:
        wav = torch.nn.functional.pad(wav, (0, target_samples - wav.shape[1]))
    else:
        wav = wav[:, :target_samples]
    return wav

def run_evaluation(model_path="checkpoints/best_model.pt", test_dir="test_audio(razz)", output_dir="evaluation_plots"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {model_path} on {device}")
    model = TatvaNet(sample_rate=16000, n_classes=2).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    audio_files = []
    for ext in ["*.wav", "*.flac", "*.mp3", "*.mp4", "*.m4a", "*.ogg"]:
        audio_files.extend(glob.glob(os.path.join(test_dir, "**", ext), recursive=True))

    y_true = []
    y_scores = []
    file_results = []

    print("\n" + "=" * 105)
    print(f"{'FILE NAME':<45} | {'EXPECTED':<12} | {'p_fake':<8} | {'VERDICT':<16} | {'STATUS'}")
    print("=" * 105)

    for fpath in sorted(audio_files):
        fname = os.path.basename(fpath)
        if "checkpoint" in fname:
            continue
        try:
            wav = load_audio_robust(fpath).to(device)
            with torch.no_grad():
                logits = model(wav)["logits"]
                p_fake = torch.softmax(logits, dim=-1)[0, 1].item()

            is_real = "(real)" in fname.lower() or "whatsapp" in fname.lower()
            label = 0 if is_real else 1
            expected = "Real Human" if is_real else "Synthetic AI"

            if p_fake > 0.65:
                verdict, matched = "🔴 DANGER (Fake)", (label == 1)
            elif p_fake >= 0.35:
                verdict, matched = "🟡 CAUTION", True
            else:
                verdict, matched = "🟢 SAFE (Real)", (label == 0)

            status = "✅ MATCH" if matched else "❌ MISMATCH"
            print(f"{fname[:43]:<45} | {expected:<12} | {p_fake:<8.4f} | {verdict:<16} | {status}")

            y_true.append(label)
            y_scores.append(p_fake)
            file_results.append((fname, expected, p_fake, verdict, status))
        except Exception as e:
            print(f"{fname[:43]:<45} | ERROR: {e}")

    print("=" * 105)

    if len(y_true) < 2 or len(set(y_true)) < 2:
        print("\nNote: Need at least 1 sample from each class (Real & Fake) to compute ROC/PR curves.")
        return

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    # 1. Classification Metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_thresh = thresholds[eer_idx]

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    print("\n--- Summary Performance Metrics ---")
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"PR-AUC:        {pr_auc:.4f}")
    print(f"EER:           {eer * 100:.2f}% (Operating Threshold: {eer_thresh:.4f})")
    print("\n" + classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))

    # 2. Plotting Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # Plot 1: ROC Curve
    axs[0, 0].plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    axs[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2)
    axs[0, 0].scatter(fpr[eer_idx], tpr[eer_idx], color="red", s=60, zorder=5, label=f"EER = {eer*100:.2f}%")
    axs[0, 0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axs[0, 0].set_xlabel("False Positive Rate (FPR)")
    axs[0, 0].set_ylabel("True Positive Rate (TPR)")
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].grid(alpha=0.3)

    # Plot 2: Precision-Recall Curve
    axs[0, 1].plot(recall, precision, color="#008080", lw=2.5, label=f"PR Curve (AP = {pr_auc:.3f})")
    axs[0, 1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axs[0, 1].set_xlabel("Recall (Sensitivity)")
    axs[0, 1].set_ylabel("Precision")
    axs[0, 1].legend(loc="lower left")
    axs[0, 1].grid(alpha=0.3)

    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real (0)", "Fake (1)"])
    disp.plot(ax=axs[1, 0], cmap="Blues", colorbar=False)
    axs[1, 0].set_title("Confusion Matrix (Threshold = 0.50)", fontsize=13, fontweight="bold")

    # Plot 4: Score Distribution Histogram
    real_scores = y_scores[y_true == 0]
    fake_scores = y_scores[y_true == 1]
    if len(real_scores) > 0:
        axs[1, 1].hist(real_scores, bins=10, alpha=0.6, color="#2ca02c", label="Genuine Human", edgecolor="black")
    if len(fake_scores) > 0:
        axs[1, 1].hist(fake_scores, bins=10, alpha=0.6, color="#d62728", label="Synthetic Fake", edgecolor="black")
    axs[1, 1].axvline(0.35, color="orange", linestyle="--", label="Safe/Caution Thresh (0.35)")
    axs[1, 1].axvline(0.65, color="red", linestyle="--", label="Caution/Danger Thresh (0.65)")
    axs[1, 1].set_title("Probability Distribution (p_fake)", fontsize=13, fontweight="bold")
    axs[1, 1].set_xlabel("Predicted Probability of Synthetic Audio")
    axs[1, 1].set_ylabel("Sample Count")
    axs[1, 1].legend(loc="upper center")
    axs[1, 1].grid(alpha=0.3)

    plot_file = os.path.join(output_dir, "model_performance_evaluation.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\n✅ Evaluation plots generated and saved to: {plot_file}")

if __name__ == "__main__":
    run_evaluation()
