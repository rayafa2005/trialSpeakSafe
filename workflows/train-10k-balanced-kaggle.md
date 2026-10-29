# Workflow: Train & Export TatvaNet From Scratch on Kaggle (Zero-Leakage + `tr/` Real Dataset)

**Trigger:** New or updated training corpora (including custom mobile recordings in `tr/`, `ai4bharat/indicvoices_r`, `ASVspoof 2019 LA`, `LibriSpeech`, and modern neural TTS datasets `skypro1111/elevenlabs_dataset`, `gereon/voxguard-synthetic-speech`, `garystafford/deepfake-audio-detection`) pushed to `https://github.com/rayafa2005/trialSpeakSafe.git`.

**Checkpoint (Pushed Right):** After Kaggle completes training (`Cell 4`), exports `/kaggle/working/tatvanet.onnx` (`Cell 6`), and runs the held-out gatekeeper benchmark (`Cell 7` via `TEST_FOLDER_AUDIOS.py` on `test_audio(razz)/fake` and `tr/`). The freshly trained model remains on Kaggle until the human reviews the benchmark brief and clicks the 1-click download link.

**Definition of done:**
1. `checkpoints/best_model.pt` trained from scratch with in-graph `ConvSTFT` + `top_db = 80.0` clamping (`strict=True` state dict with 0 missing/unexpected keys) and baseline-gated `best_eer` saving.
2. Zero speaker overlap (`Train Speakers ∩ Val Speakers ∩ Test Speakers = ∅`) and zero file overlap across `train`, `val`, and `test`, with shuffling performed strictly within each split separately.
3. Single-input ONNX exported at **2.0 s** (`32,000` samples @ 16 kHz) to `/kaggle/working/tatvanet.onnx` (`audio: [batch_size, 32000]` $\rightarrow$ `logits: [batch_size, 2]`) with PyTorch numerical parity $< 10^{-3}$.
4. Gatekeeper benchmark (`TEST_FOLDER_AUDIOS.py`) confirms real phone-mic/WhatsApp audio (`test_audio(razz)` real clips & `tr/`) is classified as `SAFE (Real)` and synthetic TTS/vocoder audio is classified as `DANGER (Fake)` before replacing local `tatvaani_android/app/src/main/assets/tatvanet.onnx`.

---

## Steps (Agent-Implementable via `kaggle_train_from_scratch.ipynb`)

1. **Cell 1 — Sync Repo & Verify In-Graph `ConvSTFT`**:
   - Clone `https://github.com/rayafa2005/trialSpeakSafe.git` on Kaggle GPU (`T4` / `P100`).
   - Verify `_to_db(top_db=80.0)` clamps dynamic range to `<= 80.0 dB` and confirm all 46 files in `tr/` and 9 files in `test_audio(razz)/fake` are present.
2. **Cell 2 — Process Custom `tr/` Dataset + Ingest External Real/Fake Corpora**:
   - Group the 46 real mobile recordings in `tr/` (`40 .m4a`, `5 .mp3`, `1 .opus`) by their **16 true speaker names** (`<title> - <Speaker>.<ext>`).
   - Split the 16 `tr/` speakers strictly into **13 Train speakers** and **3 Val speakers**, decode via `ffmpeg` to `16 kHz` mono float32, strip silence, and slice `2.0s` (`32,000`-sample) voiced chunks into `data/raw/tr_chunks/train` (`0.5s` stride) and `data/raw/tr_chunks/val` (`2.0s` stride).
   - Download & decode `ai4bharat/indicvoices_r` (~3,000 real clips with true parquet `speaker_id`) and verified modern synthetic TTS clips (`skypro1111/elevenlabs_dataset`, `gereon/voxguard-synthetic-speech`, `garystafford/deepfake-audio-detection`) to `16 kHz` mono WAVs.
3. **Cell 3 — Build Zero-Leakage, Speaker-Disjoint Manifests**:
   - Parse all ASVspoof CM protocol `.txt` files across `/kaggle/input` without early `break`.
   - Split each domain strictly by `speaker_id` into `train` (`80%`), `val` (`10%`), and `test` (`10%`).
   - Oversample `tr/` train chunks `3x` inside `train` only; assert `0` speaker overlap and `0` file overlap across splits; balance 1:1 and shuffle strictly inside each split (`data/train_manifest_kaggle.csv`, `data/val_manifest_kaggle.csv`, `data/test_manifest_kaggle.csv`).
4. **Cell 4 — Train From Scratch (`Epoch 0 -> 30`)**:
   - Run `python training/train.py --train_manifest data/train_manifest_kaggle.csv --val_manifest data/val_manifest_kaggle.csv --epochs 30 --batch_size 64 --lr 0.0005 --num_workers 4 --checkpoint_dir checkpoints --reset_best`.
5. **Cell 5 (Optional) — Baseline-Gated Fine-Tuning**:
   - Run `python training/train.py ... --finetune checkpoints/best_model.pt --epochs 8 --lr 0.0001`. `train.py` first evaluates the existing `best_model.pt` on `val_manifest_kaggle.csv` and only overwrites `best_model.pt` if a fine-tuned epoch achieves a strictly lower Val EER.
6. **Cell 6 — Export `/kaggle/working/tatvanet.onnx` & Check Parity**:
   - Export `checkpoints/best_model.pt` to `/kaggle/working/tatvanet.onnx` (`clip_samples=32000`) and verify PyTorch vs. ONNX Runtime parity.
7. **Cell 7 — Gatekeeper Benchmark & Download Brief**:
   - Run `python TEST_FOLDER_AUDIOS.py --onnx /kaggle/working/tatvanet.onnx --checkpoint /kaggle/working/best_model.pt` and present 1-click download links.
