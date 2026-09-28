# SpeakSafe & TatvaNet: Complete Technical Implementation Specification
**Project Name:** SpeakSafe  
**Model Name:** TatvaNet  
**Target Goal:** Single Comprehensive Source of Truth for Academic & Engineering Implementation Paper  
**Status:** 100% Aligned with Full-System Dual-Pipeline Architecture, In-Graph Model Graph, and 1:1 Balanced Indic Training  
**Date:** September 2026  

---

## Executive Summary
**SpeakSafe** is an edge-native, real-time audio deepfake detection system designed to detect AI-generated, voice-cloned, and converted speech directly on mobile devices without sending private audio to the cloud. 

The system implements a **Dual-Pipeline Execution Architecture**:
1. **Primary On-Device Edge Pipeline**: Runs 100% locally on Android devices via ONNX Runtime Mobile and INT8 TFLite, ensuring zero cloud dependency, low latency (~45 ms on mobile ARM CPU), and complete user data privacy (audio is processed in volatile memory and never written to disk).
2. **Secondary Cloud Benchmarking Pipeline**: An asynchronous REST infrastructure running FastAPI that performs parallel cloud inference for latency profiling, performance verification, and cross-platform benchmarking.

At the core is **TatvaNet** (939,378 parameters), a lightweight dual-path neural network that combines raw time-domain waveforms (Path A: learnable `SincConv` + 1D Depthwise Separable CNN) with spectral acceleration dynamics (Path B: in-graph `ConvSTFT` + 9-channel $\Delta / \Delta\Delta$ spectrograms with Squeeze-and-Excitation attention). The model achieves a **Validation Equal Error Rate (EER) of 0.52% – 0.55%** on balanced multi-corpus benchmarks and an INT8 quantized footprint of **< 2.0 MB** (8.54 MB FP32 ONNX).

---

## 1. Whole-Project System Architecture (Macro Overview)

The SpeakSafe ecosystem consists of three major interconnected subsystems:
- **Subsystem A: MLOps, Ingestion & Model Export Pipeline** (Offline)
- **Subsystem B: Android Edge-Native Mobile Application** (Client Runtime)
- **Subsystem C: FastAPI Cloud Benchmarking Server** (Verification & Latency Profiling)

```
====================================================================================================
                            SPEAKSAFE: WHOLE-PROJECT SYSTEM ARCHITECTURE
====================================================================================================

 ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                      SUBSYSTEM A: OFFLINE MLOps & EXPORT PIPELINE                              │
 │                                                                                                │
 │   Raw Audio Datasets              Balanced Ingestion              PyTorch Training             │
 │   ┌──────────────────────┐        ┌──────────────────────┐        ┌────────────────────────┐   │
 │   │ • ASVspoof 2019 LA   │        │ 1:1 Ratio Balancing  │        │ • AdamW + Focal Loss   │   │
 │   │ • LibriSpeech        │───────►│ 20,000 Total Samples │───────►│ • AudioAugmenter (DSP) │   │
 │   │ • MLAAD / AI4Bharat  │        │ (10k Real : 10k Fake)│        │ • Checkpoint:          │   │
 │   │   (hi, kn, te, ks)   │        │ 70/15/15 Disjoint    │        │   best_model.pt (EER)  │   │
 │   └──────────────────────┘        └──────────────────────┘        └───────────┬────────────┘   │
 │                                                                               │                │
 │                                                                               ▼                │
 │                          In-Graph Model Export & Optimization                                  │
 │                          ┌─────────────────────────────────────────────────┐                   │
 │                          │ • Real-valued ConvSTFT + Delta in graph         │                   │
 │                          │ • ONNX Opset 17 Export: export/tatvanet.onnx    │                   │
 │                          │ • Post-Training INT8: export/tatvanet_int8.tflite│                  │
 │                          └────────────────────────┬────────────────────────┘                   │
 └───────────────────────────────────────────────────┼────────────────────────────────────────────┘
                                                     │ Deployed Model Assets
                                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                  SUBSYSTEM B: ANDROID EDGE-NATIVE CLIENT RUNTIME (com.tatvaani.app)            │
 │                                                                                                │
 │   User Interface (Jetpack Compose / Material 3)                                                │
 │   ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
 │   │  MainActivity  ◄──►  TatvaaniViewModel (StateFlow: Idle | Recording | Processing | UI) │   │
 │   └───────────────────────────────────┬────────────────────────────────────────────────────┘   │
 │                                       │ Triggers 2.0s Capture                                  │
 │                                       ▼                                                        │
 │   Audio Capture Layer                                                                          │
 │   ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
 │   │  AudioCaptureService (Android Foreground Service, FOREGROUND_SERVICE_MICROPHONE)       │   │
 │   │  • Multi-Source HAL Fallback: VOICE_RECOGNITION ➔ VOICE_COMMUNICATION ➔ MIC            │   │
 │   │  • In-Memory PCM Buffer: 16 kHz Mono Float32 (32,000 samples)                          │   │
 │   │  • Zero-Disk Persistence: Audio never saved to storage (100% in volatile memory)       │   │
 │   └───────────────────────────────────┬────────────────────────────────────────────────────┘   │
 │                                       │ Dispatches PCM Buffer in Parallel                      │
 │                     ┌─────────────────┴─────────────────┐                                      │
 │                     │                                   │                                      │
 │                     ▼ [PRIMARY PIPELINE]                ▼ [SECONDARY BENCHMARK PIPELINE]       │
 │   ┌───────────────────────────────────┐   ┌────────────────────────────────────────────────┐   │
 │   │   On-Device Inference Engine      │   │   ServerClient (OkHttp REST Client)            │   │
 │   │   • OnnxEngine (ORT Mobile CPU)   │   │   • Base64 PCM Payload Encoding                │   │
 │   │   • TFLiteEngine (INT8 4-Thread)  │   │   • HTTP POST /predict                         │   │
 │   │   • In-Graph TatvaNet Execution   │   │   • Timeout: 5.0s (Failsafe)                   │   │
 │   │   • Latency: ~45 ms (Local)       │   │   • Latency: ~280 ms (Cloud Roundtrip)         │   │
 │   └─────────────────┬─────────────────┘   └────────────────────────┬───────────────────────┘   │
 │                     │                                              │                           │
 │                     └─────────────────┬────────────────────────────┘                           │
 │                                       ▼                                                        │
 │   Decision & Display Arbiter                                                                   │
 │   ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
 │   │  VerdictResult (3-Tier Output: Safe / Caution / Danger)                                │   │
 │   │  Dual Latency Benchmark Card: On-Device (45 ms) vs. Cloud Server (280 ms)              │   │
 │   └────────────────────────────────────────────────────────────────────────────────────────┘   │
 └────────────────────────────────────────────────────────────────────┬───────────────────────────┘
                                                                      │
                                         HTTP REST Request (PCM B64)  │  HTTP Response (JSON)
                                                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                      SUBSYSTEM C: FASTAPI CLOUD BENCHMARKING SERVER                            │
 │                                                                                                │
 │   FastAPI Backend (server/app.py)                                                              │
 │   ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
 │   │  POST /predict  ──► Base64 Decode ──► TatvaNet Inference ──► Calibrated Verdict (JSON) │   │
 │   │  GET  /latency  ──► Rolling Latency Profiler (p95, Mean, Min, Max stats)               │   │
 │   │  GET  /health   ──► Server & Model Status Healthcheck                                  │   │
 │   └────────────────────────────────────────────────────────────────────────────────────────┘   │
 └────────────────────────────────────────────────────────────────────────────────────────────────┘
====================================================================================================
```

---

## 2. The Dual-Execution Pipeline: Why It Matters

A core contribution of the SpeakSafe system architecture is its **Dual-Execution Pipeline**, providing both operational autonomy and experimental verifiability:

```
                               DUAL-PIPELINE INFERENCE FLOW
                               
                           Raw Audio Buffer (16 kHz, 32000 samples)
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      │                                             │
                      ▼                                             ▼
         [PRIMARY: EDGE-NATIVE]                          [SECONDARY: CLOUD BENCHMARK]
         • Zero network dependency                       • Asynchronous REST API call
         • 100% offline functionality                    • Base64 PCM streaming
         • Audio stays in device RAM                     • Validates edge quantization parity
         • Latency: ~45 ms (ARM CPU)                     • Latency: ~280 ms (Network + Compute)
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             │
                                             ▼
                                 [DECISION & UI ARBITER]
                                 • Real-time 3-Tier Verdict
                                 • Side-by-Side Latency Card
```

### 2.1 Primary Pipeline: On-Device Edge Inference
- **Zero Privacy Risk**: User speech captured during calls or voice notes is buffered strictly in volatile application RAM and processed locally. Audio bytes are never written to mobile flash storage and never transmitted over the internet.
- **Ultra-Low Latency**: By running quantized INT8 TatvaNet weights on 4 CPU threads via ONNX Runtime Mobile / TFLite, inference completes in **~45 ms**, enabling instant feedback.
- **Autonomous Reliability**: Fully operational in airplane mode or low-connectivity environments.

### 2.2 Secondary Pipeline: Cloud Benchmarking & Parity Profiling
- **Real-Time Latency Comparison**: The mobile client sends an asynchronous HTTP POST request to the FastAPI server (`/predict`) in parallel with edge execution. The UI displays the speed difference between on-device compute (~45 ms) and cloud network roundtrips (~280 ms).
- **Quantization Parity Validation**: Serves as a live testbed to verify that INT8 post-training quantization on mobile does not drift from full FP32 PyTorch server outputs.
- **Fail-Safe Decoupling**: If the server times out or loses connection, the mobile app seamlessly relies on the on-device result without blocking the user interface.

---

## 3. End-to-End Application Life-Cycle & State Machine

```
   ┌────────────┐     User taps "Analyze"      ┌─────────────────┐
   │            ├─────────────────────────────►│                 │
   │    IDLE    │                              │   RECORDING     │
   │            │◄─────────────────────────────┤   (2.0 Seconds) │
   └────────────┘        Capture Error         └────────┬────────┘
                                                        │
                                                        ▼ Buffer Full (32,000 samples)
                                               ┌─────────────────┐
                                               │   PROCESSING    │
                                               └────────┬────────┘
                                                        │
                                       Parallel Dispatch (Edge + Server)
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  VERDICT RESULT │
                                               │  🟢 Safe        │
                                               │  🟡 Caution     │
                                               │  🔴 Danger      │
                                               └────────┬────────┘
                                                        │
                                                 "Analyze Again"
                                                        │
                                                        ▼
                                               (Returns to IDLE)
```

1. **IDLE State**: Application displays instruction guidelines and permission indicators.
2. **RECORDING State**: 
   - `AudioCaptureService` transitions to foreground mode, displaying a system notification (`tatvaani_capture`).
   - Captures 16 kHz mono 16-bit PCM for 2.0 seconds into a direct memory buffer.
   - UI displays a pulsating reactive microphone animation.
3. **PROCESSING State**:
   - Audio is peak-normalized to $[-1.0, 1.0]$.
   - Buffer is passed simultaneously to `OnnxEngine` / `TFLiteEngine` (local thread) and `ServerClient` (coroutine IO thread).
4. **VERDICT RESULT State**:
   - UI renders the 3-tier verdict card (`Safe` in green, `Caution` in orange, `Danger` in red).
   - If `Danger` is detected, haptic vibration feedback is triggered.
   - The **Latency Benchmark Card** renders both On-Device and Cloud roundtrip timings.

---

## 4. TatvaNet Deep Learning Model Architecture

TatvaNet has **939,378 trainable parameters** (~0.94M). All feature transformations (`ConvSTFT`, Mel/CQT filterbanks, and $\Delta / \Delta\Delta$ extractors) run inside the model graph.

```
                    ╔══════════════════════════════════════════════╗
                    ║           RAW AUDIO INPUT (16 kHz)           ║
                    ║           Shape: (Batch, 1, 32000)           ║
                    ║            Duration: 2.0 Seconds             ║
                    ╚══════════════════════╦═══════════════════════╝
                                           ║
                     ┌─────────────────────╩─────────────────────┐
                     ▼                                           ▼
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │    PATH A: TIME-DOMAIN      │             │   PATH B: MULTI-SPECTRAL    │
      │  ┌───────────────────────┐  │             │  ┌───────────────────────┐  │
      │  │ Learnable SincConv    │  │             │  │ In-Graph ConvSTFT     │  │
      │  │ 128 Filters (L=251)   │  │             │  │ Mel + STFT + CQT      │  │
      │  └───────────┬───────────┘  │             │  └───────────┬───────────┘  │
      │              ▼              │             │              ▼              │
      │  ┌───────────────────────┐  │             │  ┌───────────────────────┐  │
      │  │ 1D Depthwise Sep CNN  │  │             │  │ Delta Extractor       │  │
      │  │ 4 Blocks (Stride 2/4) │  │             │  │ 9-Channel Δ / ΔΔ      │  │
      │  └───────────┬───────────┘  │             │  └───────────┬───────────┘  │
      │              ▼              │             │              ▼              │
      │  ┌───────────────────────┐  │             │  ┌───────────────────────┐  │
      │  │ 1D Adaptive Pooling   │  │             │  │ 2D Depthwise Sep CNN  │  │
      │  │ + Dense Projection    │  │             │  │ + SE Channel Attention│  │
      │  └───────────┬───────────┘  │             │  └───────────┬───────────┘  │
      │              ▼              │             │              ▼              │
      │      Path A Embedding       │             │      Path B Embedding       │
      │        (Batch, 256)         │             │        (Batch, 256)         │
      └──────────────┬──────────────┘             └──────────────┬──────────────┘
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │        BI-DIRECTIONAL CROSS-ATTENTION       │
                    │    • Path A queries Path B (A ➔ B Head)     │
                    │    • Path B queries Path A (B ➔ A Head)     │
                    │    • LayerNorm + Dense Fusion Projection    │
                    └──────────────────────┬──────────────────────┘
                                           │ (Batch, 256)
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │         TEMPORAL SEQUENCE MODELING          │
                    │        Gated Recurrent Unit (GRU)           │
                    │           Hidden Dimension: 128             │
                    └──────────────────────┬──────────────────────┘
                                           │ (Batch, 128)
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │             CLASSIFICATION HEAD             │
                    │         Dropout(0.3) ➔ Dense(64 ➔ 2)        │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │         CALIBRATED 3-TIER VERDICT           │
                    │   🟢 SAFE  |  🟡 CAUTION  |  🔴 DANGER      │
                    └─────────────────────────────────────────────┘
```

### Parameter Breakdown

| Module | Component | Layer Details | Parameters |
| :--- | :--- | :--- | :--- |
| **Path A** | Time-Domain Waveform | Learnable `SincConv` (128 filters, $L=251$) + 4 1D SepConv blocks + Dense(32 $\to$ 256) | **54,656** |
| **Path B** | Multi-Spectral Dynamics | In-Graph `ConvSTFT`, Mel/CQT + 4 2D SepConv blocks with SE Attention + Dense(64 $\to$ 256)| **69,424** |
| **Fusion** | Cross-Attention | Bi-directional Multihead Attention (4 heads, dim 256) + LayerNorm + Dense(512 $\to$ 256) | **658,688** |
| **Sequence**| Temporal Modeling | Gated Recurrent Unit (GRU, $d_{\text{hidden}}=128$) | **148,224** |
| **Classifier**| Output Head | Linear(128 $\to$ 64) $\to$ ReLU $\to$ Linear(64 $\to$ 2) | **8,386** |
| **TOTAL** | **Full TatvaNet Model** | **Complete Dual-Path Network** | **939,378** (~0.94M) |

---

## 5. Dataset Composition & Indic Language Strategy

```
                           TRAINING DATASET DISTRIBUTION
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ Master Ingested Corpus: 239,696 Audio Segments (16 kHz Mono Float32)                    │
 │  • ASVspoof 2019 LA: 116,490 samples (TTS & Voice Conversion algorithms A01–A19)        │
 │  • LibriSpeech (train-clean-100): 99,635 samples (Clean bonafide human speech)          │
 │  • MLAAD / AI4Bharat Indic: 23,571 samples (Multilingual synthetic & regional speech)   │
 └────────────────────────────────────────────┬────────────────────────────────────────────┘
                                              │
                                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ Production 1:1 Balanced Training Dataset (train_manifest_20k.csv)                       │
 │  • Total Samples: 20,000 clips                                                          │
 │  • Ratio: 1:1 (Exactly 10,000 Real : 10,000 Fake)                                       │
 │  • Validation Set: 4,000 clips (2,000 Real : 2,000 Fake in val_manifest_4k.csv)        │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Core Indic Languages from AI4Bharat
To capture distinct linguistic intonations, the training pipeline specifically incorporates audio across **4 key Indic languages**:

| Language | Family / Group | Language Code | Deepfake Detection Significance |
| :--- | :--- | :--- | :--- |
| **Hindi** | Indo-Aryan (Northern/Central) | `hi` | High-resource, wide tonal range and expressive pitch contours |
| **Kannada** | Dravidian (Southern) | `kn` | Distinct vowel duration dynamics and retroflex consonants |
| **Telugu** | Dravidian (Southern) | `te` | Rapid articulatory vowel transitions and harmonic cadence |
| **Kashmiri** | Indo-Aryan / Dardic (Northern) | `ks` | Unique central vowels and complex palatalized consonants |

### 5.2 1:1 Balanced Training Ratio
Enforcing a strict **10,000 Real : 10,000 Fake** ratio eliminated majority-class bias and accelerated training convergence to **~15–20 minutes** on cloud GPUs. Partitions adhere to a strict **speaker-disjoint** protocol (zero speaker identity overlap between train and test sets).

---

## 6. Mathematical Formulations (Implemented Core)

- **`SincConv` Band-Pass Filter (Path A)**:
  $$g[n, f_1, f_2] = 2f_2 \text{sinc}(2\pi f_2 n) - 2f_1 \text{sinc}(2\pi f_1 n), \quad h[n] = \frac{g[n] \cdot w[n]}{2 \sum |g[n] \cdot w[n]| + 10^{-8}}$$
- **Temporal Velocity ($\Delta$) & Acceleration ($\Delta\Delta$) (Path B)**:
  $$\Delta[t] = \frac{\sum_{n=1}^{N} n \cdot (x[t+n] - x[t-n])}{2 \sum_{n=1}^{N} n^2} \quad (N=2)$$
- **Balanced Multi-Class Focal Loss**:
  $$\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^2 \log(p_t)$$
- **Continuous 3-Tier Verdict Calibration**:
  $$\text{Verdict}(p_{\text{fake}}) = \begin{cases} 
  \textbf{Safe} \text{ (Genuine Voice)} & p_{\text{fake}} < 0.35, \quad \text{Confidence} = 1.0 - p_{\text{fake}} \\
  \textbf{Caution} \text{ (Borderline/Noisy)} & 0.35 \le p_{\text{fake}} \le 0.65, \quad \text{Confidence} = 1.0 - 2|p_{\text{fake}} - 0.5| \\
  \textbf{Danger} \text{ (AI Deepfake)} & p_{\text{fake}} > 0.65, \quad \text{Confidence} = p_{\text{fake}}
  \end{cases}$$

---

## 7. Experimental Results & Verification Metrics

| Metric | Measured Ground Truth | Target Specification | Status |
| :--- | :--- | :--- | :--- |
| **Validation Equal Error Rate (EER)** | **`0.52% – 0.55%`** ($0.005199$) | $< 5.00\%$ | **PASS** |
| **Area Under ROC Curve (AUC)** | **`0.9998`** | $> 0.9900$ | **PASS** |
| **Danger Precision (Fake Detection)** | **`99.4%`** | $> 90.0\%$ | **PASS** |
| **Safe Recall (Real Voice Retention)** | **`99.5%`** | $> 95.0\%$ | **PASS** |
| **Total Model Parameters** | **`939,378`** (~0.94M) | $< 2,000,000$ | **PASS** |
| **On-Device Inference Latency (CPU)** | **`~45.0 ms`** | $< 50.0\text{ ms}$ | **PASS** |
| **Cloud Server Roundtrip Latency** | **`~280.0 ms`** | Benchmark Baseline | **PASS** |
| **Quantized INT8 Model Size** | **`< 2.0 MB`** | $< 2.0\text{ MB}$ | **PASS** |
| **FP32 ONNX Model Size** | **`8.54 MB`** | Baseline Portable | **PASS** |

---

## 8. Complete Tech Stack Reference

```
Layer                   Technology / Library                Purpose
-----------------------------------------------------------------------------------------------------------------
Model Definition        PyTorch 2.x, TorchAudio             TatvaNet dual-path architecture, SincConv, Cross-Attention
Signal Processing       NumPy, SciPy, Librosa, SoundFile    Audio IO, ConvSTFT, Mel/STFT/CQT feature extraction
Training & Optimization AdamW, CosineAnnealingLR, AMP       Loss convergence, mixed precision, Focal Loss
Augmentations           Pure PyTorch DSP Tensor Ops         In-memory telephony bandpass, noise, reverb, mu-law
Model Export            ONNX 1.17, onnx2tf, TensorFlow Lite  Export, ConvSTFT tracing, Post-Training INT8 Quantization
Edge Mobile App         Android 14 (API 34), Kotlin 2.0     Native mobile client, background service, memory capture
Mobile UI               Jetpack Compose, Material 3         Reactive state UI, 3-tier verdict display, latency card
Edge Inference Engines  ONNX Runtime Mobile (1.24.3), TFLite On-device ARM CPU execution with 4 threads
Benchmarking Server     FastAPI, Uvicorn, Pydantic, OkHttp  REST benchmarking server, latency profiling
-----------------------------------------------------------------------------------------------------------------
```
