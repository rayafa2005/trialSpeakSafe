# Domain Context: Tatvaani Audio Deepfake Detection

## System Overview
Tatvaani is an AI-powered voice deepfake detection system consisting of a dual-path deep learning model (**TatvaNet**), an on-device Android application, and a FastAPI benchmarking inference server.

---

## Glossary & Domain Entities

### Audio Input Specifications
- **Sample Rate (`SAMPLE_RATE`)**: 16,000 Hz mono PCM.
- **Audio Clip Duration**:
  - **Runtime Baseline (Current Shipped ONNX)**: 2.0 seconds (`32,000` samples).
  - **Retrained Model Target**: 5.0 seconds (`80,000` samples).
- **Hop Length (`HOP_LENGTH`)**: 160 samples (10 ms at 16 kHz).
- **FFT Size (`N_FFT`)**: 512 for baseline DSP; 1024 for extended pipeline.
- **Target Frames (`TARGET_FRAMES`)**:
  - For 2.0s: $1 + 32,000 / 160 = 201$ frames.
  - For 5.0s: $1 + 80,000 / 160 = 501$ frames.

### Model Architecture (TatvaNet)
- **Path A (Temporal Waveform)**: SincConv learnable band-pass filters + 1D Depthwise Separable CNN.
- **Path B (Spectral Representation)**: 9-channel spectrogram feature representation (3 base mel bands + 3 first-order deltas $\Delta$ + 3 second-order deltas $\Delta\Delta$) processed with 2D Depthwise Separable CNN and Squeeze-and-Excitation (SE) channel attention.
- **Cross-Attention Fusion**: Inter-modal attention weighting combining Path A and Path B embeddings (dimension 256).
- **Temporal Modeling**: Gated Recurrent Unit (GRU, hidden dimension 128).
- **Classification Head**: Dense linear layer producing class logits.

### Inference Engines & Execution Paths
- **`OnnxEngine`**: Executes `tatvanet.onnx` on-device via the ONNX Runtime Android library.
- **`TFLiteEngine`**: Executes quantized `tatvanet_int8.tflite` via CPU interpreter (fallbacks to ONNX if TFLite model is 0 bytes or uninitialized).
- **`ServerClient`**: Sends base64-encoded PCM to FastAPI `/predict` endpoint over HTTP.

### Classification & Verdict Scoring
- **3-Tier Calibrated Verdict**:
  - **Safe**: $p_{\text{fake}} < 0.35$ (Genuine human speech). Confidence: $1.0 - p_{\text{fake}}$.
  - **Caution**: $0.35 \le p_{\text{fake}} \le 0.65$ (Borderline / uncertain audio). Confidence: $1.0 - 2 \cdot |p_{\text{fake}} - 0.5|$.
  - **Danger**: $p_{\text{fake}} > 0.65$ (Synthetically generated / manipulated speech). Confidence: $p_{\text{fake}}$.
