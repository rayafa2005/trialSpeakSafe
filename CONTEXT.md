# SpeakSafe (Tatvaani)

Edge-native real-time audio deepfake detection system that classifies voice clips into a calibrated three-tier trust verdict on mobile devices while benchmarking against a cloud server.

## Language

### Core Domain & Model

**SpeakSafe**:
The end-to-end mobile and cloud audio deepfake detection product (packaged as `com.tatvaani.app`).
_Avoid_: Tatvaani app (when referring to the published paper system name)

**TatvaNet**:
The dual-path neural network that fuses raw waveform filtering (Path A) and multi-spectral dynamics with first/second-order deltas (Path B) via bi-directional cross-attention and a GRU.
_Avoid_: Dual-path CNN, baseline model

**Path A (Temporal Branch)**:
The time-domain branch of TatvaNet that extracts phase and waveform artifacts directly from raw audio using learnable band-pass `SincConv` filters and 1D separable convolutions.
_Avoid_: Raw branch, waveform CNN

**Path B (Multi-Spectral Branch)**:
The frequency-domain branch of TatvaNet that computes Mel, STFT, and CQT spectrograms in-graph via `ConvSTFT`, stacks their $\Delta$ and $\Delta\Delta$ derivatives into 9 channels, and applies 2D separable convolutions with Squeeze-and-Excitation attention.
_Avoid_: Spectrogram branch, Librosa pipeline

**In-Graph Feature Extraction**:
Computing spectral representations (`ConvSTFT`, Mel, CQT, and $\Delta/\Delta\Delta$) as native neural network layers inside the exported model graph so the model accepts a single raw audio tensor (`audio`) without external DSP preprocessing.
_Avoid_: Dual-input model, external `spec_features` input

### Pipelines & Execution

**Dual-Execution Pipeline**:
The parallel runtime architecture that dispatches captured audio simultaneously to the **Primary Edge Pipeline** (local on-device inference for privacy and speed) and the **Secondary Cloud Benchmarking Pipeline** (FastAPI server for roundtrip latency and quantization parity comparison).
_Avoid_: Fallback server mode (not to be confused with the Dual-Path architecture inside TatvaNet)

**Audio Window**:
A fixed 2.0-second mono 16 kHz audio segment (`32,000` samples) captured strictly in volatile memory and peak-normalized before inference.
_Avoid_: 5-second clip, 80k window, recorded file

### Classification & Verdicts

**Calibrated 3-Tier Verdict**:
The mapping of the model's binary deepfake probability ($p_{\text{fake}}$) into three user-facing trust states: **Safe** ($p_{\text{fake}} < 0.35$, genuine human), **Caution** ($0.35 \le p_{\text{fake}} \le 0.65$, uncertain/noisy), and **Danger** ($p_{\text{fake}} > 0.65$, synthetic/AI-generated).
_Avoid_: 3-class softmax, binary label

**Speaker-Disjoint Split**:
Dataset partitioning across training, validation, and test manifests where no speaker identity appears in more than one split, preventing speaker-memorization leakage.
_Avoid_: Random split
