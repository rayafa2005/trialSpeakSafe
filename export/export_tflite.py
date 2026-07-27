"""
export/export_tflite.py
=======================
Tatvaani ML — Export TatvaNet to TFLite INT8

PIPELINE:
  PyTorch checkpoint (.pt)
    → ONNX (.onnx)        — intermediate portable format
    → TFLite FP32 (.tflite)   — baseline Android model
    → TFLite INT8 (.tflite)   — quantized, ~4x faster, <2MB

INT8 QUANTIZATION:
  Post-training quantization with a calibration dataset.
  Calibration = run 100-1000 representative samples through the model
  so the quantizer knows the real activation ranges.
  Without calibration, INT8 accuracy drops significantly.
  With proper calibration, accuracy drop is typically < 1% EER.

TARGET:
  Model file size: < 2MB
  Inference latency: < 50ms on mid-range Android (ARM Cortex-A55)

USAGE:
  py -3.11 export/export_tflite.py --checkpoint checkpoints/best_model.pt
  py -3.11 export/export_tflite.py --checkpoint checkpoints/best_model.pt --validate

OUTPUTS:
  export/tatvanet.onnx          — ONNX model (also usable for ONNX Runtime Mobile)
  export/tatvanet_fp32.tflite   — Full precision TFLite
  export/tatvanet_int8.tflite   — Quantized TFLite (deploy this to Android)
"""

import os
import sys
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from model.tatvanet import TatvaNet
from features.spectrogram import SAMPLE_RATE


def get_args():
    parser = argparse.ArgumentParser(description="Export TatvaNet to TFLite")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--output_dir",   default="export")
    parser.add_argument("--sample_rate",  type=int, default=SAMPLE_RATE)
    parser.add_argument("--clip_duration",type=float, default=2.0,
                        help="Input clip duration in seconds. Must match training.")
    parser.add_argument("--calib_manifest", default="data/train_manifest.csv",
                        help="Manifest for INT8 calibration samples")
    parser.add_argument("--n_calib",      type=int, default=200,
                        help="Number of calibration samples for INT8 quantization")
    parser.add_argument("--validate",     action="store_true",
                        help="Run FP32 vs INT8 accuracy comparison after export")
    return parser.parse_args()


# ─── Step 1: Load model ────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> TatvaNet:
    """Load TatvaNet from checkpoint and set to eval mode."""
    model = TatvaNet(sample_rate=SAMPLE_RATE).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Export] Model loaded from {checkpoint_path}")
    print(f"[Export] Parameters: {model.count_parameters():,}")
    return model


# ─── Step 2: Export to ONNX ───────────────────────────────────────────────────

def export_onnx(
    model:         TatvaNet,
    output_path:   str,
    clip_samples:  int,
    device:        torch.device
) -> str:
    """
    Export TatvaNet to ONNX format.

    ONNX is an intermediate format that bridges PyTorch and TFLite.
    We export with dynamic batch size so the model works for both
    batch inference (server) and single-sample inference (Android).

    Returns: path to saved .onnx file
    """
    import torch.onnx

    print(f"\n[Export] Exporting to ONNX...")

    # Representative input: (1, clip_samples) — single sample, mono audio
    dummy_input = torch.randn(1, clip_samples).to(device)

    # Disable feature pipeline lazy init by running a warmup pass
    with torch.no_grad():
        _ = model(dummy_input)

    # Export with dynamic axes so batch size is flexible
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,          # Opset 12 has good TFLite converter support
        do_constant_folding=True,  # Fold constant ops for smaller model
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={
            "audio":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        verbose=False,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[Export] ONNX saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


# ─── Step 3: ONNX → TFLite ────────────────────────────────────────────────────

def onnx_to_tflite_fp32(onnx_path: str, output_path: str) -> str:
    """
    Convert ONNX model to TFLite FP32.
    Uses onnx2tf which has better op coverage than the official converter for audio models.

    Returns: path to saved .tflite file
    """
    print(f"\n[Export] Converting ONNX → TFLite FP32...")
    try:
        import onnx2tf
    except ImportError:
        print("[Export] onnx2tf not installed. Run: py -3.11 -m pip install onnx2tf")
        return None

    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=str(Path(output_path).parent),
            output_tfv5_pb=False,
            non_verbose=True,
        )
        # onnx2tf saves as <model_name>_float32.tflite
        onnx_stem   = Path(onnx_path).stem
        generated   = Path(output_path).parent / f"{onnx_stem}_float32.tflite"
        if generated.exists():
            generated.rename(output_path)

        size_mb = os.path.getsize(output_path) / 1e6 if Path(output_path).exists() else 0
        print(f"[Export] TFLite FP32 saved: {output_path} ({size_mb:.2f} MB)")
        return output_path

    except Exception as e:
        print(f"[Export] onnx2tf conversion failed: {e}")
        print("[Export] Trying direct TFLite conversion via tensorflow...")
        return _tflite_via_tensorflow(onnx_path, output_path)


def _tflite_via_tensorflow(onnx_path: str, output_path: str) -> str:
    """
    Fallback TFLite conversion using TensorFlow's SavedModel path.
    Used if onnx2tf is unavailable.
    """
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare

        print("[Export] Using tensorflow + onnx-tf backend...")
        onnx_model = onnx.load(onnx_path)
        tf_rep     = prepare(onnx_model)
        saved_path = str(Path(output_path).parent / "tatvanet_saved_model")
        tf_rep.export_graph(saved_path)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(output_path) / 1e6
        print(f"[Export] TFLite FP32 saved: {output_path} ({size_mb:.2f} MB)")
        return output_path

    except Exception as e:
        print(f"[Export] TFLite conversion failed entirely: {e}")
        print("[Export] Install: py -3.11 -m pip install onnx2tf tensorflow onnx-tf")
        return None


def quantize_to_int8(
    fp32_tflite_path: str,
    output_path:      str,
    calib_data:       np.ndarray
) -> str:
    """
    Post-training INT8 quantization of a TFLite FP32 model.

    Requires TensorFlow. Uses representative dataset calibration
    to determine real activation ranges — critical for accuracy.

    Args:
        fp32_tflite_path : Path to FP32 .tflite model
        output_path      : Where to save INT8 .tflite
        calib_data       : (N, clip_samples) numpy array of calibration audio

    Returns: path to INT8 .tflite
    """
    print(f"\n[Export] Quantizing to INT8...")
    try:
        import tensorflow as tf
    except ImportError:
        print("[Export] TensorFlow not installed. Run: py -3.11 -m pip install tensorflow")
        return None

    # Representative dataset generator — yields calibration samples one at a time
    def representative_dataset():
        for i in range(len(calib_data)):
            sample = calib_data[i:i+1].astype(np.float32)   # (1, clip_samples)
            yield [sample]

    # Load FP32 model and set up INT8 quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(
        str(Path(fp32_tflite_path).parent / "tatvanet_saved_model")
    )

    # INT8 quantization settings
    converter.optimizations          = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Fallback for unsupported INT8 ops
    ]
    converter.inference_input_type  = tf.float32  # Keep float input for Android
    converter.inference_output_type = tf.float32  # Keep float output for Android

    tflite_int8 = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_int8)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[Export] INT8 model saved: {output_path} ({size_mb:.2f} MB)")

    if size_mb > 2.0:
        print(f"[Export] WARNING: Model is {size_mb:.2f}MB — target is <2MB")
        print("[Export] Consider reducing n_filters in SincConv or CNN channels")
    else:
        print(f"[Export] Size check PASSED: {size_mb:.2f}MB < 2MB target")

    return output_path


# ─── Step 4: Load calibration data ────────────────────────────────────────────

def load_calibration_data(
    manifest_path: str,
    n_samples:     int,
    clip_samples:  int,
    sample_rate:   int
) -> np.ndarray:
    """
    Load a random subset of training samples for INT8 calibration.
    Calibration data must be representative of the full distribution —
    include safe, caution, and danger examples.
    """
    import torchaudio
    import random

    print(f"\n[Export] Loading {n_samples} calibration samples...")

    # Load manifest
    rows = []
    if Path(manifest_path).exists():
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)

    if not rows:
        print("[Export] No calibration data found — generating synthetic data")
        return np.random.randn(n_samples, clip_samples).astype(np.float32) * 0.1

    # Sample evenly across classes for representative calibration
    by_class = {0: [], 1: [], 2: []}
    for row in rows:
        lbl = int(row["label"])
        if lbl in by_class:
            by_class[lbl].append(row["path"])

    # Take n_samples//3 from each class
    per_class = n_samples // 3
    selected  = []
    for lbl in [0, 1, 2]:
        paths = by_class[lbl]
        random.shuffle(paths)
        selected.extend(paths[:per_class])
    random.shuffle(selected)

    # Load audio
    calib = []
    for path in selected:
        if not os.path.exists(path):
            continue
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = wav.squeeze(0)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            # Fix length
            if wav.shape[0] < clip_samples:
                wav = torch.nn.functional.pad(wav, (0, clip_samples - wav.shape[0]))
            else:
                wav = wav[:clip_samples]
            # Normalize
            peak = wav.abs().max()
            if peak > 1e-6:
                wav = wav / peak
            calib.append(wav.numpy())
        except Exception:
            continue

    if not calib:
        print("[Export] Could not load any calibration files — using synthetic")
        return np.random.randn(n_samples, clip_samples).astype(np.float32) * 0.1

    calib_arr = np.stack(calib, axis=0).astype(np.float32)
    print(f"[Export] Calibration data shape: {calib_arr.shape}")
    return calib_arr


# ─── Step 5: Validate INT8 accuracy ───────────────────────────────────────────

def validate_quantization(
    fp32_path:   str,
    int8_path:   str,
    calib_data:  np.ndarray
):
    """
    Run 50 samples through both FP32 and INT8 models and compare outputs.
    Checks that INT8 quantization did not significantly degrade accuracy.
    Target: output difference < 5% on average.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("[Export] TensorFlow not installed — skipping validation")
        return

    print(f"\n[Export] Validating INT8 vs FP32 accuracy...")

    def run_tflite(model_path, inputs):
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        in_idx  = interp.get_input_details()[0]["index"]
        out_idx = interp.get_output_details()[0]["index"]
        outputs = []
        for inp in inputs:
            interp.set_tensor(in_idx, inp[np.newaxis])
            interp.invoke()
            outputs.append(interp.get_tensor(out_idx)[0])
        return np.stack(outputs)

    n_test = min(50, len(calib_data))
    test_data = calib_data[:n_test]

    if not Path(fp32_path).exists() or not Path(int8_path).exists():
        print("[Export] Model files not found — skipping validation")
        return

    fp32_out = run_tflite(fp32_path, test_data)   # (N, 3) logits
    int8_out = run_tflite(int8_path, test_data)   # (N, 3) logits

    # Compare predicted classes
    fp32_preds = fp32_out.argmax(axis=1)
    int8_preds = int8_out.argmax(axis=1)
    agreement  = (fp32_preds == int8_preds).mean()

    print(f"  FP32 vs INT8 prediction agreement: {100*agreement:.1f}%")
    print(f"  Mean logit difference: {np.abs(fp32_out - int8_out).mean():.4f}")

    if agreement >= 0.95:
        print(f"  [OK] Quantization validated — >95% agreement with FP32")
    else:
        print(f"  [WARNING] Agreement below 95% — check calibration data quality")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args        = get_args()
    device      = torch.device("cpu")   # Export always on CPU for TFLite compatibility
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_samples = int(args.clip_duration * args.sample_rate)

    print("="*60)
    print("TATVANET EXPORT PIPELINE")
    print("="*60)
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Clip samples:  {clip_samples} ({args.clip_duration}s @ {args.sample_rate}Hz)")
    print(f"Output dir:    {output_dir}")

    # Step 1: Load model
    model = load_model(args.checkpoint, device)

    # Step 2: Export ONNX
    onnx_path = str(output_dir / "tatvanet.onnx")
    onnx_path = export_onnx(model, onnx_path, clip_samples, device)

    if onnx_path is None:
        print("[Export] ONNX export failed. Check model for non-exportable ops.")
        return

    # Step 3: ONNX → TFLite FP32
    fp32_path = str(output_dir / "tatvanet_fp32.tflite")
    fp32_path = onnx_to_tflite_fp32(onnx_path, fp32_path)

    if fp32_path is None:
        print("[Export] TFLite FP32 export failed.")
        print("[Export] ONNX model is still usable with ONNX Runtime Mobile on Android.")
        print("[Export] Install onnx2tf: py -3.11 -m pip install onnx2tf")
        return

    # Step 4: Load calibration data
    calib_data = load_calibration_data(
        args.calib_manifest, args.n_calib, clip_samples, args.sample_rate
    )

    # Step 5: Quantize to INT8
    int8_path = str(output_dir / "tatvanet_int8.tflite")
    int8_path = quantize_to_int8(fp32_path, int8_path, calib_data)

    # Step 6: Validate quantization (optional)
    if args.validate and int8_path:
        validate_quantization(fp32_path, int8_path, calib_data)

    # Summary
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    if Path(onnx_path).exists():
        print(f"  ONNX:       {onnx_path} ({os.path.getsize(onnx_path)/1e6:.2f} MB)")
    if fp32_path and Path(fp32_path).exists():
        print(f"  TFLite FP32:{fp32_path} ({os.path.getsize(fp32_path)/1e6:.2f} MB)")
    if int8_path and Path(int8_path).exists():
        print(f"  TFLite INT8:{int8_path} ({os.path.getsize(int8_path)/1e6:.2f} MB) ← deploy this")
    print(f"\nNext step: Copy tatvanet_int8.tflite to tatvaani_android/app/src/main/assets/")
    print("="*60)


if __name__ == "__main__":
    main()