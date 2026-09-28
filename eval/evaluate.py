"""
eval/evaluate.py
================
Tatvaani ML — Blind Test Set Evaluation

Run this ONLY on the held-out test set — never on training or validation data.
This produces the numbers you present in the demo and in your research report.

METRICS COMPUTED:
  - EER (Equal Error Rate)       — industry standard for anti-spoofing
  - Accuracy per class           — Safe / Caution / Danger
  - Danger Precision             — when we say Danger, how often are we right
  - Safe Recall                  — how often do we correctly clear genuine voices
  - Confusion matrix             — full 3x3 breakdown
  - Per-dataset breakdown        — EER on ASVspoof vs IndicSynth vs In-the-Wild
  - Latency measurement          — on-device inference time per clip

TARGET SCORES (from master document):
  EER on ASVspoof 2019 LA    < 5%
  EER on IndicSynth test     < 15%
  Danger Precision           > 90%
  Safe Recall                > 95%
  On-device latency          < 50ms

USAGE:
  py -3.11 eval/evaluate.py --checkpoint checkpoints/best_model.pt
  py -3.11 eval/evaluate.py --checkpoint checkpoints/best_model.pt --manifest data/test_manifest.csv
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix, classification_report

from model.tatvanet import TatvaNet
from training.train import TatvaaniDataset, compute_eer


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate TatvaNet on test set")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--manifest",    default="data/test_manifest.csv")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--clip_duration", type=float, default=2.0)
    parser.add_argument("--output",      default="eval/eval_results.json",
                        help="Where to save results JSON")
    parser.add_argument("--device",      default=None,
                        help="cuda or cpu. Auto-detected if not set.")
    return parser.parse_args()


@torch.no_grad()
def run_evaluation(model, loader, device) -> dict:
    """
    Full evaluation pass over a DataLoader.
    Collects predictions, probabilities, latencies per sample.

    Returns dict with arrays:
      labels, verdicts, probs (N,3), scores (N,) danger prob, latencies (N,)
    """
    model.eval()

    all_labels   = []
    all_verdicts = []
    all_probs    = []
    all_latencies= []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device)

        # Measure inference latency per batch
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        with autocast(enabled=(device.type == "cuda")):
            output = model(waveforms)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        batch_ms = (t_end - t_start) * 1000.0 / waveforms.shape[0]  # ms per sample

        all_labels   .extend(labels.numpy())
        all_verdicts .extend(output["verdict"].cpu().numpy())
        all_probs    .extend(output["probs"].cpu().numpy())
        all_latencies.extend([batch_ms] * waveforms.shape[0])

    return {
        "labels":    np.array(all_labels),
        "verdicts":  np.array(all_verdicts),
        "probs":     np.array(all_probs),
        "scores":    np.array(all_probs)[:, 2],   # Danger probability
        "latencies": np.array(all_latencies),
    }


def compute_all_metrics(results: dict, dataset_name: str = "all") -> dict:
    """
    Compute all evaluation metrics from raw predictions.

    Args:
        results     : Dict from run_evaluation()
        dataset_name: Label for logging
    Returns:
        Dict of all metric values
    """
    labels   = results["labels"]
    verdicts = results["verdicts"]
    scores   = results["scores"]

    # Core metrics
    accuracy = (verdicts == labels).mean()
    eer      = compute_eer(labels, scores)

    # Confusion matrix
    cm = confusion_matrix(labels, verdicts, labels=[0, 1, 2])

    # Danger Precision: of all clips predicted as Danger, how many are truly synthetic?
    danger_predicted = cm[:, 2].sum()
    danger_precision = cm[2, 2] / danger_predicted if danger_predicted > 0 else 0.0

    # Safe Recall: of all truly Safe clips, how many do we correctly identify?
    safe_total  = cm[0].sum()
    safe_recall = cm[0, 0] / safe_total if safe_total > 0 else 0.0

    # Danger Recall: of all truly Danger clips, how many do we catch?
    danger_total  = cm[2].sum()
    danger_recall = cm[2, 2] / danger_total if danger_total > 0 else 0.0

    # Latency
    lat_mean = results["latencies"].mean()
    lat_p95  = np.percentile(results["latencies"], 95)

    metrics = {
        "dataset":          dataset_name,
        "n_samples":        int(len(labels)),
        "accuracy":         float(accuracy),
        "eer":              float(eer),
        "eer_pct":          float(eer * 100),
        "danger_precision": float(danger_precision),
        "safe_recall":      float(safe_recall),
        "danger_recall":    float(danger_recall),
        "latency_mean_ms":  float(lat_mean),
        "latency_p95_ms":   float(lat_p95),
        "confusion_matrix": cm.tolist(),
        "class_report":     classification_report(
                                labels, verdicts,
                                target_names=["Safe", "Caution", "Danger"],
                                output_dict=True
                            )
    }

    # Check against target thresholds
    metrics["targets_met"] = {
        "danger_precision_gt_90": danger_precision > 0.90,
        "safe_recall_gt_95":      safe_recall > 0.95,
        "latency_lt_50ms":        lat_mean < 50.0,
    }

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics to console."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS — {metrics['dataset'].upper()}")
    print("="*60)
    print(f"  Samples:           {metrics['n_samples']:,}")
    print(f"  Accuracy:          {100*metrics['accuracy']:.2f}%")
    print(f"  EER:               {metrics['eer_pct']:.2f}%  (target: <5% ASVspoof, <15% IndicSynth)")
    print(f"  Danger Precision:  {100*metrics['danger_precision']:.2f}%  (target: >90%)")
    print(f"  Safe Recall:       {100*metrics['safe_recall']:.2f}%  (target: >95%)")
    print(f"  Danger Recall:     {100*metrics['danger_recall']:.2f}%")
    print(f"  Latency (mean):    {metrics['latency_mean_ms']:.1f}ms  (target: <50ms)")
    print(f"  Latency (p95):     {metrics['latency_p95_ms']:.1f}ms")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    cm = np.array(metrics["confusion_matrix"])
    print(f"              Safe  Caution  Danger")
    for i, row_label in enumerate(["Safe   ", "Caution", "Danger "]):
        print(f"    {row_label}   {cm[i, 0]:5d}  {cm[i, 1]:7d}  {cm[i, 2]:6d}")

    print(f"\n  Targets met:")
    for k, v in metrics["targets_met"].items():
        status = "✓ PASS" if v else "✗ FAIL"
        print(f"    {status}  {k}")
    print("="*60)


def evaluate_by_dataset(
    model:      TatvaNet,
    manifest:   str,
    device:     torch.device,
    sample_rate: int,
    clip_duration: float,
    batch_size: int
) -> list:
    """
    Run evaluation broken down by source dataset.
    This shows performance on ASVspoof vs IndicSynth vs WaveFake etc. separately.
    Critical for understanding where the model struggles.
    """
    # Group test manifest by dataset field
    dataset_rows = defaultdict(list)
    with open(manifest, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset_rows[row["dataset"]].append(row)

    all_results = []

    for ds_name, rows in dataset_rows.items():
        if len(rows) < 10:
            print(f"  [SKIP] {ds_name}: only {len(rows)} samples — too few to evaluate")
            continue

        # Write a temporary manifest for this dataset subset
        tmp_path = f"eval/tmp_{ds_name}.csv"
        os.makedirs("eval", exist_ok=True)
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # Create dataset and loader for this subset
        ds     = TatvaaniDataset(tmp_path, sample_rate, clip_duration, is_train=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        if len(ds) == 0:
            os.remove(tmp_path)
            continue

        results = run_evaluation(model, loader, device)
        metrics = compute_all_metrics(results, dataset_name=ds_name)
        print_metrics(metrics)
        all_results.append(metrics)

        # Clean up temp file
        os.remove(tmp_path)

    return all_results


def main():
    args   = get_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Checkpoint: {args.checkpoint}")
    print(f"[Eval] Test manifest: {args.manifest}")

    # Load model
    model = TatvaNet(sample_rate=args.sample_rate).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    print(f"[Eval] Model loaded. Best EER from training: "
          f"{100*ckpt.get('best_eer', 0):.2f}%")

    # Full test set evaluation
    test_dataset = TatvaaniDataset(
        args.manifest, args.sample_rate, args.clip_duration, is_train=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"[Eval] Running on {len(test_dataset):,} test samples...")
    results = run_evaluation(model, test_loader, device)
    metrics = compute_all_metrics(results, dataset_name="full_test_set")
    print_metrics(metrics)

    # Per-dataset breakdown
    print("\n[Eval] Per-dataset breakdown...")
    per_dataset = evaluate_by_dataset(
        model, args.manifest, device,
        args.sample_rate, args.clip_duration, args.batch_size
    )

    # Save all results
    output = {
        "overall":     metrics,
        "per_dataset": per_dataset,
        "checkpoint":  args.checkpoint,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[Eval] Results saved to {args.output}")
    print("[Eval] Done. Use these numbers in your demo and report.")


if __name__ == "__main__":
    main()