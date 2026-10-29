# ADR 0001: Standardize on 2.0-Second (32,000-Sample) Audio Window Across Training, Server, and Android

## Status
Accepted

## Context
Previously, experimental scripts mixed a 5.0-second (`80,000`-sample) target with a 2.0-second (`32,000`-sample) mobile capture window, causing dimension mismatch errors at inference time (`INVALID_ARGUMENT: Got: 80000 Expected: 32000`).

## Decision
Standardize the entire SpeakSafe / TatvaNet pipeline on a **2.0-second (`32,000` samples @ 16 kHz)** audio window:
1. **`training/train.py`**: Train `TatvaNet` on `target_samples = 32000` (2.0s @ 16 kHz).
2. **`export/export_tflite.py`**: Export `tatvanet.onnx` with input `audio: [batch_size, 32000]`.
3. **`AudioCaptureService.kt` & `FeatureExtractor.kt`**: Capture 2.0 seconds (`32,000` samples) of raw mono 16 kHz PCM and peak-normalize into a `FloatArray(32000)`.
4. **`TatvaaniViewModel.kt`, `OnnxEngine.kt`, `ServerClient.kt`, and `server/app.py`**: Feed the exact same `(1, 32000)` raw waveform into both the primary on-device ONNX engine and the secondary FastAPI server.

## Consequences
- `checkpoints/best_model.pt` (trained on Kaggle for 2.0s windows, achieving `0.52%` validation EER at Epoch 16) and `tatvaani_android/app/src/main/assets/tatvanet.onnx` (`8.54 MB`) share the exact same `(1, 32000)` input contract and model weights.
- On-device and server pipelines execute on bit-for-bit identical 32,000-sample waveforms without shape mismatches.
