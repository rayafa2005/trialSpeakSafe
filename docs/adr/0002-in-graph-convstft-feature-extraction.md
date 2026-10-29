# ADR 0002: Embed Spectral Feature Extraction In-Graph via Real-Valued ConvSTFT

## Status
Accepted

## Context
TatvaNet is a dual-path architecture combining raw waveform filtering (Path A: `SincConv`) and 9-channel multi-spectral dynamics with $\Delta/\Delta\Delta$ derivatives (Path B: Mel, STFT, and CQT). Originally, Path B relied on external DSP preprocessing (`librosa` on the Python server and custom FFT/Mel loops in Kotlin `FeatureExtractor.kt`) passed via a second `spec_features` input tensor, while `torchaudio`'s complex-valued `torch.stft` blocked clean ONNX Opset 17 export.

Maintaining separate external DSP implementations in Python and Kotlin risked train/serve skew and input bias between the mobile device and server.

## Decision
Replace complex-valued STFT operations in `features/spectrogram.py` with `ConvSTFT` (a real-valued 1D convolution DFT basis) and embed the entire `FeaturePipeline` (`MultiStreamSpectrogram` + `DeltaExtractor`) directly inside `TatvaNet.forward()`.

## Consequences
- **Single Raw Audio Input (`audio: [batch_size, 32000]`)**: Both `OnnxEngine` on Android and `server/app.py` pass raw peak-normalized audio directly to the model with zero external Kotlin or Python spectral filtering.
- **Dual-Path Preserved Internally**: Inside the model graph, the raw audio branches into Path A (`SincConv` + 1D SepCNN) and Path B (`ConvSTFT` Mel/STFT/CQT + 9-channel $\Delta/\Delta\Delta$ + 2D SepCNN + SE attention) before bi-directional Cross-Attention fusion.
- **Zero Train/Serve or Edge/Cloud Skew**: On-device ONNX and server ONNX/PyTorch execute identical mathematical operations on the same raw audio buffer.
