import os
import sys
import time
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from model.tatvanet import TatvaNet

# --- Dataset Loader ---
class ManifestEvalDataset(Dataset):
    def __init__(self, df, target_samples=32000, target_sr=16000):
        self.df = df
        self.target_samples = target_samples
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fpath = row["path"]
        label = int(row.get("bin_label", row.get("label", 0)))
        if label == 2:
            label = 1

        try:
            wav, sr = torchaudio.load(fpath)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.target_sr:
                wav = T.Resample(sr, self.target_sr)(wav)
            peak = wav.abs().max()
            if peak > 1e-6:
                wav = wav / peak
            if wav.shape[1] < self.target_samples:
                wav = torch.nn.functional.pad(wav, (0, self.target_samples - wav.shape[1]))
            else:
                wav = wav[:, :self.target_samples]
            return wav.squeeze(0), label
        except Exception:
            return torch.zeros(self.target_samples, dtype=torch.float32), label


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_thresh = thresholds[eer_idx]
    return eer, eer_thresh, fpr, tpr


def compute_det_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    return fpr, fnr


def generate_all_technical_graphs(
    manifest_path="data/val_manifest_4k.csv",
    checkpoint_path="checkpoints/best_model.pt",
    output_dir="technical_graphs",
    batch_size=128
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"🔬 GENERATING TECHNICAL BENCHMARK GRAPHS ON {device}")
    print("=" * 75)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
    if not os.path.exists(manifest_path):
        manifest_path = "data/val_manifest.csv"

    print(f"Loading Model:    {checkpoint_path}")
    print(f"Loading Dataset:  {manifest_path}")

    df = pd.read_csv(manifest_path)
    # Limit to 4000 samples for swift evaluation
    if len(df) > 4000:
        df = df.sample(n=4000, random_state=42).reset_index(drop=True)

    dataset = ManifestEvalDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = TatvaNet(sample_rate=16000, n_classes=2).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    y_true_list = []
    y_scores_list = []

    print(f"Running inference across {len(df)} audio samples...")
    t0 = time.time()

    with torch.no_grad():
        for batch_wavs, batch_labels in loader:
            batch_wavs = batch_wavs.to(device)
            out = model(batch_wavs)
            probs = torch.softmax(out["logits"], dim=-1)
            p_fake = probs[:, 1].cpu().numpy()
            y_scores_list.extend(p_fake)
            y_true_list.extend(batch_labels.numpy())

    eval_time = time.time() - t0
    avg_latency_ms = (eval_time / len(df)) * 1000.0

    y_true = np.array(y_true_list)
    y_scores = np.array(y_scores_list)
    y_pred = (y_scores >= 0.5).astype(int)

    # Core Metrics
    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    pr_auc = average_precision_score(y_true, y_scores)
    eer, eer_thresh, fpr, tpr = compute_eer(y_true, y_scores)
    det_fpr, det_fnr = compute_det_curve(y_true, y_scores)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 50)
    print("📊 TECHNICAL METRICS REPORT SUMMARY")
    print("=" * 50)
    print(f"• Dataset Size:          {len(df):,} audio samples")
    print(f"• Total Evaluation Time: {eval_time:.2f} s ({avg_latency_ms:.2f} ms/sample)")
    print(f"• Accuracy:              {acc * 100:.2f}%")
    print(f"• Equal Error Rate (EER):{eer * 100:.2f}% (Threshold = {eer_thresh:.4f})")
    print(f"• Area Under ROC (AUC):  {roc_auc:.4f}")
    print(f"• Precision-Recall AUC:  {pr_auc:.4f}")
    print(f"• Precision (Fake):      {prec * 100:.2f}%")
    print(f"• Recall / Sensitivity:  {rec * 100:.2f}%")
    print(f"• Specificity (Human):   {spec * 100:.2f}%")
    print(f"• F1-Score:              {f1 * 100:.2f}%")
    print("=" * 50)

    # Style Configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # ─────────────────────────────────────────────────────────────
    # FIGURE 1: 6-PANEL COMPREHENSIVE TECHNICAL DASHBOARD
    # ─────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    # 1. ROC Curve
    axs[0, 0].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    axs[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.2)
    axs[0, 0].scatter(eer, 1 - eer, color="#d62728", s=70, zorder=5, label=f"EER = {eer*100:.2f}%")
    axs[0, 0].set_title("(a) Receiver Operating Characteristic (ROC)", fontweight="bold")
    axs[0, 0].set_xlabel("False Positive Rate (FPR)")
    axs[0, 0].set_ylabel("True Positive Rate (TPR)")
    axs[0, 0].legend(loc="lower right", frameon=True)
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    axs[0, 1].plot(recall_vals, precision_vals, color="#2ca02c", lw=2.5, label=f"PR Curve (AP = {pr_auc:.4f})")
    axs[0, 1].set_title("(b) Precision-Recall Curve (PR-AUC)", fontweight="bold")
    axs[0, 1].set_xlabel("Recall (Sensitivity)")
    axs[0, 1].set_ylabel("Precision")
    axs[0, 1].legend(loc="lower left", frameon=True)
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Detection Error Trade-Off (DET)
    axs[0, 2].plot(det_fpr * 100, det_fnr * 100, color="#9467bd", lw=2.5, label="TatvaNet DET")
    axs[0, 2].scatter(eer * 100, eer * 100, color="#d62728", s=70, zorder=5, label=f"EER Operating Point ({eer*100:.2f}%)")
    axs[0, 2].set_title("(c) Detection Error Trade-Off (DET)", fontweight="bold")
    axs[0, 2].set_xlabel("False Alarm Probability (%)")
    axs[0, 2].set_ylabel("Miss Probability (%)")
    axs[0, 2].legend(loc="upper right", frameon=True)
    axs[0, 2].grid(True, alpha=0.3)

    # 4. Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real (Human)", "Fake (AI)"])
    disp.plot(ax=axs[1, 0], cmap="Blues", colorbar=False, values_format="d")
    axs[1, 0].set_title(f"(d) Confusion Matrix (N = {len(df):,})", fontweight="bold")

    # 5. Score Distribution Histogram
    real_sc = y_scores[y_true == 0]
    fake_sc = y_scores[y_true == 1]
    axs[1, 1].hist(real_sc, bins=25, alpha=0.65, color="#2ca02c", label="Real Speech (0)", edgecolor="black", density=True)
    axs[1, 1].hist(fake_sc, bins=25, alpha=0.65, color="#d62728", label="Deepfake AI (1)", edgecolor="black", density=True)
    axs[1, 1].axvline(0.35, color="#ff7f0e", linestyle="--", lw=1.8, label="Safe Threshold (0.35)")
    axs[1, 1].axvline(0.65, color="#d62728", linestyle="--", lw=1.8, label="Danger Threshold (0.65)")
    axs[1, 1].set_title("(e) Predicted Probability Distribution ($p_{fake}$)", fontweight="bold")
    axs[1, 1].set_xlabel("Predicted Probability of Deepfake")
    axs[1, 1].set_ylabel("Density")
    axs[1, 1].legend(loc="upper center", frameon=True, fontsize=8)
    axs[1, 1].grid(True, alpha=0.3)

    # 6. Metrics Summary Bar Chart
    metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score"]
    metric_vals = [acc * 100, prec * 100, rec * 100, spec * 100, f1 * 100]
    bars = axs[1, 2].bar(metric_names, metric_vals, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"], edgecolor="black", alpha=0.85)
    axs[1, 2].set_ylim(0, 110)
    axs[1, 2].set_title("(f) Core Performance Metrics (%)", fontweight="bold")
    axs[1, 2].set_ylabel("Score (%)")
    for bar in bars:
        yval = bar.get_height()
        axs[1, 2].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
    axs[1, 2].grid(True, alpha=0.3, axis='y')

    dashboard_path = os.path.join(output_dir, "tatvanet_technical_dashboard.png")
    fig.suptitle("TatvaNet Deepfake Voice Detection — Comprehensive Technical Evaluation", fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Main Technical Dashboard saved to: {dashboard_path}")

    # ─────────────────────────────────────────────────────────────
    # FIGURE 2: STANDALONE HIGH-RES ROC & PR CURVES
    # ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(fpr, tpr, color="#1f77b4", lw=3, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax1.scatter(eer, 1 - eer, color="red", s=90, zorder=5, label=f"EER = {eer*100:.2f}%")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(recall_vals, precision_vals, color="#2ca02c", lw=3, label=f"PR Curve (AP = {pr_auc:.4f})")
    ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.legend(loc="lower left", fontsize=11)
    ax2.grid(True, alpha=0.3)

    roc_pr_path = os.path.join(output_dir, "roc_pr_curves_publication.png")
    plt.savefig(roc_pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Standalone ROC & PR Curves saved to: {roc_pr_path}")

    print("\n🎉 All technical evaluation graphs successfully generated!")

if __name__ == "__main__":
    generate_all_technical_graphs()
