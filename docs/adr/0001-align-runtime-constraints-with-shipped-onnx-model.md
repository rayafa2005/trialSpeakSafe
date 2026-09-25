# ADR 0001: Align Runtime Constraints with Shipped ONNX Model

## Status
Accepted

## Context
The Android client and Python server previously threw runtime errors (`Analysis failed — please try again` on Android, HTTP 500 on Server) when attempting to run inference against `tatvanet.onnx`.

### Root Cause
While experimental training scripts (`train.py`, `checkpoints/best_model.pt`) and certain client constants had been updated to target a 5.0-second window (80,000 samples, 501 frames), the physical asset file `tatvaani_android/app/src/main/assets/tatvanet.onnx` currently bundled in the app is the 2.0-second baseline model expecting:
- `audio`: `[batch_size, 32000]`
- `spec_features`: `[batch_size, 9, 128, 201]`

Sending 80,000 samples and 501 frames caused ONNX Runtime to fail immediately with an `INVALID_ARGUMENT: Got: 80000 Expected: 32000` dimension mismatch error.

## Decision
Apply Option 1: Re-align runtime constraints in both the Android application and the FastAPI server to match the 2.0-second baseline (`32,000` samples @ 16 kHz, `201` feature frames):
1. **AudioCaptureService.kt**: Set `CAPTURE_DURATION = 2` seconds.
2. **FeatureExtractor.kt**: Set `TARGET_SAMPLES = 32000` and `TARGET_FRAMES = 201`.
3. **OnnxEngine.kt**: Pass `audio` tensor of shape `(1, 32000)` and 9-channel spectrogram features of shape `(1, 9, 128, 201)` (extracted via `FeatureExtractor` or zero-padded fallback).
4. **server/app.py**: Set `TARGET_SAMPLES = 32000`, `TARGET_FRAMES = 201`, and compute 201-frame spectrogram features.

## Consequences
### Positive
- Android on-device inference executes immediately without crash.
- Server `/predict` endpoint returns HTTP 200 with valid verdicts and confidence scores.
- End-to-end functionality restored without requiring re-export or re-quantization.

### Negative / Future Work
- `checkpoints/best_model.pt` (which was trained on 80,000 samples) is not currently active in mobile assets until an ONNX/TFLite export pipeline is executed and validated.
