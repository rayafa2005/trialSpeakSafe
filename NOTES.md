# Workspace Notes & Domain Terminology (`NOTES.md`)

## Canonical Terminology
- **SpeakSafe / Tatvaani (`TatvaNet`)**: Dual-path audio deepfake detection architecture combining `PathA_SincCNN` (raw waveform learnable Sinc filterbank) and `PathB_SpecCNN` (in-graph `ConvSTFT` 9-channel Mel + STFT + CQT + $\Delta/\Delta\Delta$ spectrogram CNN) via `CrossAttentionFusion` and a 128-unit `GRU`.
- **Single-Input Zero-Bias Contract**: Both the Android app (`OnnxEngine.kt`) and the FastAPI server (`server/app.py`) feed the exact same peak-normalized `float32[1, 32000]` (`2.0s` @ `16 kHz`) waveform tensor directly into `tatvanet.onnx` without any external Kotlin or Python spectral/VAD filtering.
- **Custom Real Mobile Dataset (`tr/`)**: 46 authentic real-human voice recordings (`40 .m4a`, `5 .mp3`, `1 .opus`, `8.52 MB` total, `419.3s` duration) across 16 Indian speakers (`Ankit Pandey`, `Anmol Naik.S`, `Geetanjali`, `Hrishikesh Y`, `JEEVAN K`, `Manish Roy`, `Md Dilshad`, `Pranav C.M.`, `Pranav Shastri`, `Rajjan Singh`, `Rupali Soni`, `Shreya Srivastava`, `Siddhant Srivastav`, `Sufiyaan Meccai`, `Tejaswani Singh`, `yukti neerukonda`).
- **Held-Out Gatekeeper Benchmark (`test_audio(razz)/fake`)**: 9 evaluation clips containing 4 real human recordings (`2` WhatsApp `.mp4` clips + `2` `LA_E_*(real).flac` clips) and 5 synthetic AI clips (`2` ElevenLabs `.mp3` previews + `3` `LA_E_*.flac` spoofs), evaluated via `TEST_FOLDER_AUDIOS.py`.
- **Kaggle Training Repo (`trialSpeakSafe`)**: Remote GitHub repository (`https://github.com/rayafa2005/trialSpeakSafe.git`) cloned by `kaggle_train_from_scratch.ipynb` on Kaggle T4 GPUs.
- **Kaggle PyTorch 2.5+ / Python 3.13 Compatibility**:
  - Kaggle kernels run PyTorch 2.5+/2.6+ on Python 3.13, where `torch.onnx.export` imports `onnxscript` internally.
  - Required export environment packages: `onnx`, `onnxruntime`, `onnxscript`, `soundfile`, `librosa`.
  - Cell 6 in `kaggle_train_from_scratch.ipynb` includes self-healing dependency installation and directory navigation (`%cd /kaggle/working/trialSpeakSafe`) so that exported weights can be generated even if an interactive session restarts after training without re-running training.
  - `export/export_tflite.py` embeds defensive auto-installation of `onnxscript`.
- **Baseline-Gated Saving**: `training/train.py` rule ensuring `checkpoints/best_model.pt` is only overwritten if a newly trained or fine-tuned epoch achieves a strictly lower validation EER than the existing `best_model.pt` evaluated on the same active validation set.
