# Peer Review & Structural Fix Guide: SpeakSafe Implementation Paper

**Paper Title in Draft:** *A Survey on Deepfake Audio Detection Techniques: Challenges, Approaches, and Prospects for Mobile Deployment*  
**Review Target:** IEEE / Conference Implementation Paper for **SpeakSafe & TatvaNet**  
**Audited Against:** Codebase Ground Truth, Model Architecture, Dataset Manifests, and IEEE Publication Standards  
**Date:** September 2026  

---

## 1. Executive Verdict & Assessment

### The Good (Technical Accuracy: 10/10)
- **100% Codebase Accuracy**: The core numbers, parameter counts (939,378 params), model architecture details (SincConv, in-graph ConvSTFT, $\Delta/\Delta\Delta$, Cross-Attention, GRU), dataset counts (239,696 master; 20,000 1:1 balanced), Indic languages (Hindi, Kannada, Telugu, Kashmiri), and empirical results (EER 0.52%–0.55%, ~45ms on-device latency, <2.0MB size) are **completely accurate and faithfully match the repository**.
- **Accurate Problem Framing**: Properly frames the real-world gap between offline benchmark models and edge mobile deployment constraints.

### Critical Issues That Must Be Fixed Before Submission (Urgent)

> [!CAUTION]
> **Issue 1: Misleading Title ("Survey" vs. Implementation Paper)**  
> The paper presents an original novel architecture (**TatvaNet**), an edge Android implementation (**SpeakSafe**), and empirical training results. However, the title starts with *"A Survey on..."*. A survey paper reviews existing literature without presenting a new model. Reviewers will reject or miscategorize the paper immediately if the title says "Survey".

> [!CAUTION]
> **Issue 2: IEEE Template Placeholder Headings Retained in Body Text**  
> The draft body text literally contains the raw instructional placeholder headings from the standard IEEE conference template (`IEEEtran` guidelines):
> - `II. EASE OF USE`
> - `A. Selecting a Template`, `B. Maintaining the Integrity of the Specifications`
> - `III. PREPARE YOUR PAPER BEFORE STYLING`
> - `A. Abbreviations and Acronyms`, `B. Units`, `C. Equations`, `D. Some Common Mistakes`
> - `IV. USING THE TEMPLATE`, `1) For papers with more than six authors`, `2) For papers with less than six authors`, `a) Selection`, `b) Change number of columns`, `c) Deletion`, `B. Identify the Headings`, `C. Figures and Tables`
> 
> **These template guidelines must be completely replaced with proper academic research sections.**

> [!WARNING]
> **Issue 3: Missing Visual System and Architecture Diagrams**  
> The text describes complex dual-path fusion and dual edge-cloud pipelines, but contains zero figures. Reviewers expect at least two visual figures:
> 1. Whole-Project Dual-Pipeline System Architecture (Android App + MLOps + FastAPI Server).
> 2. TatvaNet In-Graph Model Architecture Flow.

---

## 2. Specific Line-by-Line Fixes

### Fix 1: Update Paper Title
- **Current Title (Incorrect):**  
  `A Survey on Deepfake Audio Detection Techniques: Challenges, Approaches, and Prospects for Mobile Deployment`
- **Recommended Titles (Choose One):**
  - **Option A (Recommended):** `SpeakSafe: Real-Time Edge-Native Deepfake Audio Detection for Mobile Devices Using TatvaNet`
  - **Option B:** `SpeakSafe: A Dual-Path Cross-Attention Deep Learning Architecture for On-Device Audio Deepfake Detection in Multilingual Contexts`
  - **Option C:** `TatvaNet: An Efficient Dual-Path Neural Network for Real-Time On-Device Voice Deepfake Detection`

---

### Fix 2: Section-by-Section Academic Restructuring

Replace the IEEE template placeholder sections with this standardized 8-section layout:

```
Current Draft (Template Text)                 Correct Academic Implementation Paper Sections
───────────────────────────────────────────── ────────────────────────────────────────────────────────
Title: "A Survey on..."                     ➔ Title: "SpeakSafe: Real-Time Edge-Native Deepfake Audio Detection..."
Abstract & Keywords                         ➔ Abstract & Keywords (Retain existing text - it is great)
I. INTRODUCTION                             ➔ I. INTRODUCTION & MOTIVATION
  A. Contributions                          ➔   A. Primary Contributions
II. EASE OF USE                             ➔ II. SYSTEM ARCHITECTURE & DUAL-PIPELINE DESIGN
  A. Selecting a Template                   ➔   A. Edge-Native Client & Cloud Benchmarking Subsystems
  B. Maintaining Integrity...               ➔   B. Design Constraints & Edge Target Specifications
III. PREPARE YOUR PAPER BEFORE STYLING      ➔ III. TATVANET MODEL ARCHITECTURE & SIGNAL PROCESSING
  A. Abbreviations                          ➔   A. Path A: Time-Domain SincConv Waveform Branch
  B. Units                                  ➔   B. Path B: Multi-Spectral Dynamics (ConvSTFT & Delta/Delta-Delta)
  C. Equations                              ➔   C. Bi-Directional Cross-Attention Fusion
  D. Some Common Mistakes                   ➔   D. Temporal Modeling (GRU) & Calibrated 3-Tier Verdict
IV. USING THE TEMPLATE                      ➔ IV. DATASET COMPOSITION, INDIC LANGUAGES & TRAINING
  A. Authors and Affiliations               ➔   A. Multi-Corpus Master Ingestion & 1:1 Balanced Sampling
  B. Identify Headings                      ➔   B. Core Indic Language Representation (AI4Bharat)
  C. Figures and Tables                     ➔   C. Loss Formulation & In-Memory DSP Telephony Augmentation
[No existing section]                       ➔ V. EXPERIMENTAL RESULTS & DISCUSSION
[No existing section]                       ➔   A. Benchmark Metrics (EER, AUC, Precision, Recall)
[No existing section]                       ➔   B. On-Device Edge Latency vs. Cloud Server Profiling
[No existing section]                       ➔ VI. PRACTICAL CONSTRAINTS & LIMITATIONS
[No existing section]                       ➔ VII. CONCLUSION & FUTURE WORK
ACKNOWLEDGMENT                              ➔ ACKNOWLEDGMENT
REFERENCES                                  ➔ REFERENCES
```

---

## 3. Ready-to-Copy Section Replacement Content for Your Team

Your teammates can directly copy the text below into their LaTeX (`.tex`) or MS Word template to replace the broken sections.

```latex
% ==============================================================================
% SECTION II: SYSTEM ARCHITECTURE & DUAL-PIPELINE DESIGN
% ==============================================================================
\section{System Architecture \& Dual-Pipeline Design}
SpeakSafe is engineered as an edge-native, zero-cloud-dependent system designed to run during active voice calls without user friction or privacy compromise. The system architecture is partitioned into three distinct subsystems:

\subsection{Subsystem Breakdown}
\begin{itemize}
    \item \textbf{Subsystem A (Offline MLOps Pipeline):} Handles multi-corpus ingestion, 1:1 real-to-fake class balancing, speaker-disjoint dataset partitioning, and automated in-graph PyTorch-to-ONNX/TFLite export.
    \item \textbf{Subsystem B (Android Edge Client Runtime):} Packaged as \texttt{com.tatvaani.app}, built with Jetpack Compose and Material 3. A dedicated \texttt{AudioCaptureService} records 2.0 seconds of 16 kHz mono PCM in volatile device memory with zero flash disk persistence, executing local inference via ONNX Runtime Mobile and INT8 TFLite.
    \item \textbf{Subsystem C (FastAPI Cloud Benchmarking Backend):} An asynchronous REST server providing \texttt{/predict} and \texttt{/latency} endpoints, utilized strictly for side-by-side latency profiling and quantization parity validation.
\end{itemize}

\subsection{Dual-Execution Inference Pipeline}
SpeakSafe operates two parallel inference paths upon capturing audio:
\begin{enumerate}
    \item \textbf{Primary Edge Pipeline:} Executes locally on the mobile CPU in $\sim 45\text{ ms}$. Operates offline, incurs zero cloud computing costs, and eliminates the risk of voice data interception.
    \item \textbf{Secondary Benchmarking Pipeline:} Streams base64-encoded PCM over HTTP to the FastAPI server, returning cloud latency ($\sim 280\text{ ms}$). The UI renders both metrics simultaneously in a dual-benchmark card, proving that edge inference is over $6\times$ faster than cloud transmission.
\end{enumerate}

% ==============================================================================
% SECTION III: TATVANET MODEL ARCHITECTURE & SIGNAL PROCESSING
% ==============================================================================
\section{TatvaNet Model Architecture}
TatvaNet is a 939,378-parameter dual-path neural network designed to capture complementary physical speech cues:

\subsection{Path A: Time-Domain Waveform Branch}
Path A processes raw PCM samples $x \in \mathbb{R}^{1 \times 32000}$ using a learnable \texttt{SincConv} filterbank containing 128 band-pass filters ($L=251$):
\begin{equation}
g[n, f_1, f_2] = 2f_2 \text{sinc}(2\pi f_2 n) - 2f_1 \text{sinc}(2\pi f_1 n), \quad h[n] = g[n] \cdot w[n]
\end{equation}
where $w[n]$ is a Hamming window and $f_1, f_2$ are learnable cutoffs initialized along the Mel scale. The output is processed by four 1D depthwise-separable convolution blocks, global pooling, and a dense projection to yield a 256-dimensional embedding $E_A \in \mathbb{R}^{256}$.

\subsection{Path B: In-Graph Multi-Spectral \& Delta Dynamics}
To eliminate mobile DSP drift and bypass ONNX complex-number limitations, spectral extraction is traced directly inside the model graph via a custom real-valued \texttt{ConvSTFT}:
\begin{itemize}
    \item \textbf{Base Streams (0--2):} Mel Spectrogram, Linear STFT Magnitude, and Constant-Q Transform (CQT).
    \item \textbf{Delta ($\Delta$) \& Acceleration ($\Delta\Delta$) Streams (3--8):} 1st- and 2nd-order temporal derivatives computed via a 5-frame regression window ($N=2$):
    \begin{equation}
    \Delta[t] = \frac{\sum_{n=1}^{N} n \cdot (x[t+n] - x[t-n])}{2 \sum_{n=1}^{N} n^2}
    \end{equation}
\end{itemize}
The resulting $9 \times 128 \times 201$ tensor is processed by 2D separable convolutions equipped with Squeeze-and-Excitation (SE) channel attention ($r=8$), producing embedding $E_B \in \mathbb{R}^{256}$.

\subsection{Bi-Directional Cross-Attention Fusion}
Embeddings $E_A$ and $E_B$ are fused via bi-directional multi-head cross-attention ($h=4, d_k=64$), allowing the temporal branch to query spectral features ($A \to B$) and vice versa ($B \to A$):
\begin{equation}
E_{\text{fused}} = \text{ReLU}\left(W_f \cdot [\text{MHA}(E_A, E_B) \,\|\, \text{MHA}(E_B, E_A)]\right) \in \mathbb{R}^{256}
\end{equation}

\subsection{Temporal Modeling \& Continuous 3-Tier Calibration}
$E_{\text{fused}}$ is passed through a 128-unit Gated Recurrent Unit (GRU) and a linear classification head. The predicted fake probability $p_{\text{fake}}$ is mapped to a calibrated 3-tier verdict:
\begin{equation}
\text{Verdict}(p_{\text{fake}}) = \begin{cases} 
\textbf{Safe} \text{ (Genuine)}, & p_{\text{fake}} < 0.35, \quad \text{Conf} = 1.0 - p_{\text{fake}} \\
\textbf{Caution} \text{ (Borderline)}, & 0.35 \le p_{\text{fake}} \le 0.65, \quad \text{Conf} = 1.0 - 2|p_{\text{fake}} - 0.5| \\
\textbf{Danger} \text{ (AI Fake)}, & p_{\text{fake}} > 0.65, \quad \text{Conf} = p_{\text{fake}}
\end{cases}
\end{equation}

% ==============================================================================
% SECTION IV: DATASET COMPOSITION, INDIC LANGUAGES & TRAINING
% ==============================================================================
\section{Dataset Composition \& Training Methodology}

\subsection{Multi-Corpus Ingestion & 1:1 Balanced Sampling}
The master corpus comprises 239,696 audio segments (16 kHz mono) aggregated from ASVspoof 2019 LA (116,490), LibriSpeech (99,635), and MLAAD/AI4Bharat (23,571). To eliminate majority-class bias, a standardized 1:1 balanced training dataset of 20,000 clips (10,000 Real : 10,000 Fake) and a 4,000-clip validation set (2,000 Real : 2,000 Fake) are used. Partitions enforce strict speaker-disjoint isolation.

\subsection{Core Indic Language Representation}
The dataset specifically incorporates 4 Indic languages from AI4Bharat representing diverse phonetic structures:
\begin{itemize}
    \item \textbf{Hindi (\texttt{hi}):} Indo-Aryan family; wide pitch dynamic range and tonal variations.
    \item \textbf{Kannada (\texttt{kn}):} Dravidian family; retroflex consonant transitions and vowel duration dynamics.
    \item \textbf{Telugu (\texttt{te}):} Dravidian family; rapid vowel-ending cadences.
    \item \textbf{Kashmiri (\texttt{ks}):} Indo-Aryan/Dardic group; palatalized consonants and central vowels.
\end{itemize}

\subsection{Loss Function & In-Memory Audio Augmentation}
Training optimizes Weighted Focal Loss ($\gamma = 2.0$):
\begin{equation}
\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^2 \log(p_t)
\end{equation}
On-the-fly PyTorch tensor augmentations simulate real-world mobile degradations: gain jitter ($0.7\times$--$1.3\times$), Gaussian noise (SNR 15--35 dB), telephony bandpass filtering (PSTN/GSM 300--3400 Hz, VoIP 100--4000 Hz), 8-bit $\mu$-law quantization companding, and synthetic room impulse response reverberation.
```

---

## 4. Figures to Add to the Final Paper

Your team should include these two architecture figures:

### Figure 1: SpeakSafe Full-System Dual-Pipeline Architecture
```
  [User Taps "Analyze" (2.0s PCM)] 
                 │
                 ▼
      [AudioCaptureService]
      (Foreground Service, In-Memory)
                 │
        ┌────────┴────────┐
        ▼                 ▼
 [Primary Pipeline]   [Secondary Pipeline]
 OnnxEngine / TFLite   ServerClient (OkHttp)
 (Local ARM CPU)       (FastAPI POST /predict)
   ~45 ms Latency        ~280 ms Latency
        │                 │
        └────────┬────────┘
                 ▼
      [Dual Benchmark Card UI]
    🟢 Safe | 🟡 Caution | 🔴 Danger
```

### Figure 2: TatvaNet In-Graph Model Architecture
```
         Raw PCM Input (1, 32000)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    [Path A: SincConv]    [Path B: ConvSTFT + Δ/ΔΔ]
    (1D SepConv + Pool)   (2D SepConv + SE Attention)
         │                     │
         └──────────┬──────────┘
                    ▼
       [Cross-Attention Fusion]
                    │
           [GRU (128 units)]
                    │
          [Linear (64 ➔ 2)]
                    │
       [Calibrated 3-Tier Verdict]
```

---

## 5. Summary Checklist for Your Teammate

| Checkpoint | Action Required | Status |
| :--- | :--- | :--- |
| **Title** | Change from *"A Survey on..."* to *"SpeakSafe: Real-Time Edge-Native Deepfake Audio Detection..."* | ⚠️ Must Fix |
| **Section Headings** | Remove all template text (`EASE OF USE`, `PREPARE YOUR PAPER...`, `USING THE TEMPLATE`, `For papers with less than six authors`) | ⚠️ Must Fix |
| **Equations & Numbers** | Keep existing parameter counts ($939,378$), EER ($0.52\%-0.55\%$), 1:1 ratio ($10\text{k}:10\text{k}$), Indic languages (Hindi, Kannada, Telugu, Kashmiri) | ✅ 100% Accurate |
| **Figures** | Insert System Architecture diagram and TatvaNet Model Flow diagram | ⚠️ Add Figures |
| **Tables I, II, III** | Retain existing tables (Parameter Breakdown, Target vs Measured, Tech Stack) | ✅ Well Structured |
