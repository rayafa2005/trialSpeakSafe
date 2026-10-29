# Kaggle Training Notebook Audit, Root-Cause Diagnosis & Retraining Plan

**Primary Sources Audited:**
- [raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1) (Kaggle execution notebook, 13 cells serialized on line 1)
- [training/train.py](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L1-L467) (TatvaNet training loop, dataset loader, checkpointing, and CLI)
- [training/augment.py](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/augment.py#L1-L138) (`AudioAugmenter` in-memory PyTorch waveform augmentation)
- [features/spectrogram.py](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L1-L185) (`ConvSTFT`, `MelSpectrogramExtractor`, `STFTExtractor`, `CQTExtractor`, `MultiStreamSpectrogram`)
- [features/delta.py](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/delta.py#L1-L181) (`DeltaExtractor` and `FeaturePipeline`)
- [features/sinc_conv.py](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/sinc_conv.py#L1-L167) (`SincConv` learnable filterbank)
- [model/tatvanet.py](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/model/tatvanet.py#L1-L646) (`TatvaNet`, `PathA_SincCNN`, `PathB_SpecCNN`, `CrossAttentionFusion`)
- [process_dataset.py](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/process_dataset.py#L1-L730) (Offline dataset ingestion & VAD pipeline bypassed on Kaggle)
- [checkpoints/best_model.pt](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/checkpoints/best_model.pt) (Saved PyTorch checkpoint artifact, inspected at commits `ee4b6a2` and `de39db60`)

---

## 1. Chronological Breakdown of `raziyaspeaksafetraining.ipynb` on Kaggle

### 1.1 Kaggle Environment & Attached Dataset Metadata
From the top-level notebook metadata in [raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1):
- **Runtime Environment**: Python `3.12.13`, Kaggle accelerator `nvidiaTeslaT4`, `isInternetEnabled: true`.
- **Mounted `/kaggle/input` Data Sources**:
  1. `sourceId: 492337` (`sourceType: "datasetVersion"`): **ASVspoof 2019 Logical Access (LA)** corpus containing `.flac` audio clips and CM protocol files (`ASVspoof2019.LA.cm.*.txt`) with speaker IDs (`LA_0079`) and file IDs (`LA_T_1138215`).
  2. `sourceId: 3842332` (`sourceType: "datasetVersion"`): Secondary Kaggle audio corpus containing real (`libri`/`clean`/`human`/`real`/`bonafide`) and synthetic (`elevenlabs`/`wavefake`/`fake`/`deepfake`/`synthetic`/`cloned`) directories matched by path heuristics in Cell 3.

### 1.2 Two-Session Execution Timeline (Reconstructed from Cell `iopub` Timestamps & Checkpoint State)
Inspecting the `metadata.execution` timestamps of each cell in [raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1) alongside the serialized dictionary inside [checkpoints/best_model.pt](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/checkpoints/best_model.pt) reveals that the notebook was edited and executed across **two sessions**:

| Notebook Cell Index | Cell Header Comment | Executed Timestamp (`iopub.status.busy` → `idle`) | Execution Session & Role |
| :--- | :--- | :--- | :--- |
| **Cell 1** | `# Cell 1: Clean setup` | `2026-09-26T02:03:15Z` → `02:03:25Z` | Re-run in Session 2 (originally run in Session 1 to clone repo at commit `ee4b6a2`). |
| **Cell 2** | `# Cell 2: Fetch Indian Voices directly from Hugging Face Parquets (Zero NumPy errors)` | `2026-09-26T02:03:31Z` → `02:12:20Z` | Re-run in Session 2; downloads 2,300 clips from `ai4bharat/indicvoices_r`. |
| **Cell 3** | `# Cell 3: Instant Fast Manifest Builder (Fixed Speaker ID Mapping)` | `2026-09-26T02:12:33Z` → `02:22:58Z` | Re-run in Session 2; builds initial `train_manifest_kaggle.csv`, `val_manifest_kaggle.csv`, `test_manifest_kaggle.csv`. |
| **Cell 4** | `# Step 1: Download Modern Synthetic AI Voices (ElevenLabs & VoxGuard)` | `2026-09-26T02:24:41Z` → `02:24:56Z` | Inserted after Cell 5A failure in Session 1; re-run in Session 2. Downloads 933 modern fake clips. |
| **Cell 5** | `# Step 2: Combine AI4Bharat + ASVspoof + Modern ElevenLabs` | `2026-09-26T02:26:41Z` → `02:26:41Z` | Inserted after Cell 5A failure in Session 1; re-run in Session 2. Appends 933 fake + 933 duplicate Indic real clips to train/val manifests. |
| **Cell 6** | `# Cell 4: Train with Symmetric Room Noise & Codec Invariance` | *(No timestamp in final save — executed in Session 1 prior to `2026-09-25T18:29:20Z`)* | **Stage 1 Training (`--epochs 20 --lr 0.0003`)**. Produced the saved `checkpoints/best_model.pt` at 0-indexed `epoch: 16` (`val_eer: 0.005200`). |
| **Cell 7** | `# Cell 5A: Evaluate on your custom test_audio(razz) files` | `2026-09-25T18:29:20Z` → `18:29:20Z` | Session 1 evaluation of Stage 1 `best_model.pt` on `test_audio(razz)`. |
| **Cell 8** | `# Step 3: Fine-tune for 6 Epochs on the enhanced dataset` | `2026-09-25T18:46:23Z` → `20:04:52Z` | Session 1 **Stage 2 Fine-Tuning (`--resume checkpoints/best_model.pt --epochs 20 --lr 0.0001`)**. Ran for 1h 18m 29s (Epochs 18–20) but **never overwrote `best_model.pt`**. |
| **Cell 9** | `# Evaluate Custom Test Audios` | `2026-09-25T20:13:36Z` → `20:13:37Z` | Session 1 post-Step-3 3-tier evaluation on `test_audio(razz)`. |
| **Cell 10** | `# Decisive 2-Tier Breakdown` | `2026-09-25T20:12:32Z` → `20:12:32Z` | Session 1 post-Step-3 binary (`p_fake >= 0.50`) evaluation on `test_audio(razz)`. |
| **Cell 11** | `# Create 1-click download link` | `2026-09-26T02:00:02Z` → `02:00:02Z` | Session 2 download helper copying `best_model.pt` to `/kaggle/working/best_model.pt`. |
| **Cell 12** | `# Check Checkpoint Details & Run Custom Test` | `2026-09-26T06:13:18Z` → `06:13:24Z` | Session 2 inspection cell; failed at line 7 with `ModuleNotFoundError: No module named 'model'` because the kernel had restarted in `/kaggle/working`. |

---

### 1.3 Cell-by-Cell Technical Audit (Cells 1 through 12)

#### Cell 1 (`# Cell 1: Clean setup`, cell lines 1–5)
- Executes `!pip install -q huggingface_hub pyarrow onnx onnxruntime`, removes any existing `trialSpeakSafe` folder, clones `https://github.com/rayafa2005/trialSpeakSafe.git`, and `%cd trialSpeakSafe` ([raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1)).
- At the time of execution (`2026-09-25` / `2026-09-26`), `HEAD` of `trialSpeakSafe` was commit `ee4b6a2` (`feat: align SpeakSafe to 2.0s baseline, add balanced dataset manifests, and configure single-input pipeline`), where `features/spectrogram.py` still used `torchaudio.transforms.MelSpectrogram`, `torchaudio.transforms.Spectrogram`, and `torchaudio.transforms.AmplitudeToDB(top_db=80.0)`.

#### Cell 2 (`# Cell 2: Fetch Indian Voices directly from Hugging Face Parquets`, cell lines 1–73)
- Authenticates with Hugging Face and lists all files in dataset `ai4bharat/indicvoices_r` ([raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1), Cell 2 lines 10–22).
- Iterates over 4 target languages (`LANGUAGES = {"hindi": 1000, "kannada": 500, "telugu": 500, "kashmiri": 300}`, total `2,300` clips, lines 14–19), downloads matching `.parquet` shards via `hf_hub_download`, and extracts `row.get("audio")["bytes"]` (lines 51–57).
- **Defects in Cell 2**:
  1. **Raw Byte Dump Without Decoding or Resampling (lines 60–62)**: Writes `raw_bytes` directly to `data/raw/indic_voices/indic_{lang}_{saved:04d}.wav` (`with open(out_path, "wb") as f: f.write(raw_bytes)`), bypassing [`process_dataset.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/process_dataset.py#L137-L206) (`vad_trim` and `split_long_clip`).
  2. **Discarded Speaker Metadata**: Ignores the native `speaker_id` column in the `ai4bharat/indicvoices_r` parquet table, saving only sequential filenames `indic_{lang}_{saved:04d}.wav`.

#### Cell 3 (`# Cell 3: Instant Fast Manifest Builder (Fixed Speaker ID Mapping)`, cell lines 1–131)
- Builds the Stage 1 manifests (`data/train_manifest_kaggle.csv`, `data/val_manifest_kaggle.csv`, `data/test_manifest_kaggle.csv`).
- Contains **three severe data-pipeline bugs**:
  1. **Premature `break` in Protocol Parsing Loop (Cell 3, lines 15–37)**:
     ```python
     for kdir in kaggle_folders:
         for root, _, files in os.walk(kdir):
             for f in files:
                 if f.endswith(".txt") and any(k in f.lower() for k in ["cm", "protocol", "trn", "trl"]):
                     ...
             if len(protocol_map) > 0:
                 break
     ```
     As soon as `os.walk(kdir)` encounters the first directory with any matching `.txt` file that populates `protocol_map`, `if len(protocol_map) > 0: break` (lines 36–37) terminates the entire `os.walk(kdir)` traversal for that Kaggle dataset. If protocol files are in separate subdirectories or if a non-ASVspoof `.txt` file matches first, remaining protocol files are never read. Furthermore, pooling `trn` (train) and `trl` (dev/eval) protocols together mixes ASVspoof 2019 LA's official `train`, `dev`, and `eval` partitions (which have disjoint spoofing algorithms `A01–A06` vs. `A07–A19`) into a single pool.
  2. **Per-File Fake Speaker IDs Breaking Speaker-Disjoint Splitting (Cell 3, lines 48–52 & 75–83)**:
     - For `indic_voices` (lines 48–52), speaker IDs are fabricated by dividing the sorted file index by 15: `spk_id = f"indic_{f.split('_')[1]}_{idx // 15}"` (line 51). A single speaker with >15 consecutive utterances in a parquet shard is split across multiple synthetic `spk_id` buckets and leaks across `train`, `val`, and `test`.
     - For `/kaggle/input` files not found in `protocol_map` (lines 75–83, covering `sourceId: 3842332` — LibriSpeech, WaveFake, ElevenLabs, etc.), Cell 3 assigns **a unique speaker ID per file**:
       ```python
       fake_records.append({"path": full_path, "label": 1, "speaker_id": f"fake_{fname_no_ext}"})
       ...
       real_records.append({"path": full_path, "label": 0, "speaker_id": f"real_{fname_no_ext}"})
       ```
       When `split_by_speaker` (lines 87–104) groups by `r["speaker_id"]`, every non-ASVspoof audio clip forms a singleton speaker group of size 1. Shuffling `speakers` (line 92) therefore performs a **random per-file split** rather than a speaker-disjoint split, placing recordings from the exact same LibriSpeech readers and synthetic TTS voices into both `train_manifest_kaggle.csv` and `val_manifest_kaggle.csv`.
  3. **Post-Split Truncation via `balance_pair` (Cell 3, lines 109–119)**:
     Because `split_by_speaker` splits by speaker count (`70% / 15% / 15%` of unique `speaker_id` keys, lines 94–95) and non-ASVspoof files have 1 file per `speaker_id` while ASVspoof speakers have ~100–500 files per `speaker_id`, the resulting `real_train` and `fake_train` lists have skewed dataset proportions before `balance_pair` truncates the larger class to `min(len(reals), len(fakes))`.

#### Cell 4 (`# Step 1: Download Modern Synthetic AI Voices (ElevenLabs & VoxGuard)`, cell lines 1–72)
- Attempts to download up to `TARGET = 1500` synthetic clips from three Hugging Face datasets ([raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1), Cell 4 lines 19–23):
  1. `garystafford/deepfake-audio-detection`
  2. `gereon/voxguard-synthetic-speech`
  3. `skypro1111/elevenlabs_dataset`
- **Defects in Cell 4**:
  - Line 31 contains a boolean tautology: `parquet_files = [f for f in files if f.endswith(".parquet") and ("train" in f or "data" in f or "fake" in f or True)]`, which matches every `.parquet` file regardless of split name, and slices `parquet_files[:2]` (line 33).
  - Lines 48–51 filter out real clips only if `str(val).lower() in ["0", "real", "bonafide"]`. If a dataset encodes `0 = fake, 1 = real` or uses boolean `False` for fake (`str(False).lower() == "false"`), or has no `label`/`is_fake` column, real human clips are saved as `modern_fake_{saved:04d}.wav`!
  - Line 61–63 writes raw parquet bytes directly to `.wav` files (`modern_fake_{saved:04d}.wav`) without decoding container headers (many Hugging Face audio parquets store MP3 or FLAC bytes inside `audio["bytes"]` or use 22.05/24/44.1/48 kHz sample rates).
  - Total clips saved across the candidate repos: `933` clips (as documented in Cell 5, line 13: `# 1. New Modern ElevenLabs/TTS Fake Audio (933 clips)`).

#### Cell 5 (`# Step 2: Combine AI4Bharat + ASVspoof + Modern ElevenLabs`, cell lines 1–41)
- Loads `df_train_prev` and `df_val_prev` from `data/train_manifest_kaggle.csv` and `data/val_manifest_kaggle.csv` (lines 10–11).
- Shuffles `modern_fakes = glob.glob("data/raw/modern_fakes/*.wav")` (933 clips) and assigns synthetic speaker IDs `f"modern_fake_{i//15}"` (lines 14–16).
- **Direct Train/Validation Data Leakage Bug (Cell 5, lines 18–35)**:
  ```python
  indic_files = glob.glob("data/raw/indic_voices/*.wav")
  random.shuffle(indic_files)
  new_real_records = [{"path": os.path.abspath(f), "label": 0, "speaker_id": f"indic_pair_{i//15}"} for i, f in enumerate(indic_files[:len(new_fake_records)])]
  ```
  All 2,300 files in `data/raw/indic_voices/*.wav` had **already been partitioned and written** into `df_train_prev`, `df_val_prev`, and `df_test` in Cell 3 (lines 46–52)! Cell 5 reshuffles `indic_files`, takes 933 of them, splits 85% (`793` clips) into `train_add_real` and 15% (`140` clips) into `val_add_real`, and concatenates them onto `df_train_prev` and `df_val_prev`. Consequently:
  - `933` AI4Bharat files are **duplicated** in the manifests.
  - AI4Bharat files that Cell 3 placed in `val_manifest_kaggle.csv` (and `test_manifest_kaggle.csv`) were randomly selected into `train_add_real` and appended to `train_manifest_kaggle.csv`, causing **exact file overlap between train and validation sets**.

#### Cell 6 (`# Cell 4: Train with Symmetric Room Noise & Codec Invariance`, cell lines 1–9)
- Runs Stage 1 training via [`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L445-L467):
  ```bash
  !python training/train.py \
      --train_manifest data/train_manifest_kaggle.csv \
      --val_manifest data/val_manifest_kaggle.csv \
      --epochs 20 \
      --batch_size 64 \
      --lr 0.0003 \
      --num_workers 4 \
      --checkpoint_dir checkpoints
  ```
- Inspecting [checkpoints/best_model.pt](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/checkpoints/best_model.pt) confirms that this Stage 1 run saved `best_model.pt` at **0-indexed `epoch = 16` (Epoch 17/20)** with:
  - `val_eer = 0.005199597450519953` (**`0.52%`**)
  - `best_eer = 0.005535055350553492` (**`0.55%`**)
  - `no_improve = 1`
  - **Why `ckpt["best_eer"]` (`0.55%`) differs from `ckpt["val_eer"]` (`0.52%`)**: In [`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L410-L427), the `ckpt` dictionary is constructed at lines 410–420 **before** `if val_eer < best_eer:` updates `best_eer = val_eer` and `no_improve = 0` at lines 423–426! Thus, when Epoch 17 (`epoch = 16`) beat the previous best EER (`0.5535%`) with a new best `val_eer = 0.5200%`, `torch.save(ckpt, checkpoint_dir / "best_model.pt")` at line 426 wrote the stale pre-update `best_eer` (`0.005535`) and stale `no_improve` (`1`) into `best_model.pt`!

#### Cell 7 (`# Cell 5A: Evaluate on your custom test_audio(razz) files`, cell lines 1–75)
- Loads `checkpoints/best_model.pt` into `TatvaNet(sample_rate=16000, n_classes=2)` and evaluates `test_audio(razz)/**/*` ([raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1), Cell 7 lines 12–74).
- Uses `torchaudio.load(fpath)` (line 33) with no `ffmpeg` or `librosa` fallback, causing the two `.mp4` WhatsApp recordings (`WhatsApp Audio 2026-05-29 at 17.17.39.mp4` and `(1).mp4`) to throw `Error opening ... Format not recognised`, and misclassifies the ElevenLabs voice previews (`voice_preview_anika`, `voice_preview_monika sogam`) and real ASVspoof eval clips (`LA_E_4581379(real).flac`, `LA_E_5849185(real).flac`).

#### Cell 8 (`# Step 3: Fine-tune for 6 Epochs on the enhanced dataset`, cell lines 1–10)
- Attempts to fine-tune the Stage 1 model for 6 epochs on the Step 2 manifest (which added `modern_fakes`) by executing ([raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1), Cell 8 lines 2–10):
  ```bash
  !python training/train.py \
      --train_manifest data/train_manifest_kaggle.csv \
      --val_manifest data/val_manifest_kaggle.csv \
      --resume checkpoints/best_model.pt \
      --epochs 20 \
      --batch_size 64 \
      --lr 0.0001 \
      --num_workers 4 \
      --checkpoint_dir checkpoints
  ```
- **Four Compounding Bugs in Step 3 (`training/train.py#L285-L335` & `#L423-L427`)**:
  1. **Epoch Range Bug (`3` epochs run instead of `6`)**: Because `checkpoints/best_model.pt` was saved at 0-indexed `ckpt["epoch"] = 16`, [`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L317) set `start_epoch = ckpt["epoch"] + 1 = 17`. With `--epochs 20`, the loop `for epoch in range(start_epoch, config["epochs"]):` ([`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L330)) only executed `epoch = 17, 18, 19` (**3 epochs total**, printed as `Epoch 18/20`, `19/20`, `20/20`).
  2. **CLI `--lr 0.0001` Overwritten by `optimizer.load_state_dict` & `scheduler.load_state_dict`**: In [`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L285-L294), `optimizer` and `CosineAnnealingLR` are initialized with `config["lr"] = 0.0001`, but lines 314–316 immediately overwrite both with the checkpoint's serialized states (`optimizer.load_state_dict(ckpt["optimizer_state"])` and `scheduler.load_state_dict(ckpt["scheduler_state"])`).
  3. **Exhausted `CosineAnnealingLR` Schedule**: In `best_model.pt`, `scheduler_state` had `{'T_max': 17, 'eta_min': 1e-05, 'base_lrs': [0.0003], 'last_epoch': 14, '_step_count': 15, '_last_lr': [3.17185e-05]}`. Thus, during Epochs 18–20, the learning rate was not `1e-4` (`0.0001`), but decayed along the tail of the Stage 1 cosine curve from `3.17e-5` → `2.22e-5` → `1.55e-5` → `1.00e-5` (`eta_min`), preventing the weights from adapting to the new ElevenLabs distribution.
  4. **`best_model.pt` Was NEVER Updated by Step 3**: At [`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L318), `best_eer = ckpt.get("best_eer", float("inf"))` loaded `0.005535` from Stage 1 (computed on the easier pre-Step-2 validation set). In Step 3, validation was evaluated on `df_val_new` (which added 140 uncurated `modern_fakes` + 140 `indic_voices`). Because `val_eer` on this harder validation set remained above `0.005535` across epochs 17–19, `if val_eer < best_eer:` ([`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L423)) evaluated to `False` in all 3 epochs. **Proof**: Inspecting [checkpoints/best_model.pt](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/checkpoints/best_model.pt) shows `ckpt["epoch"] == 16` and `ckpt["config"]["lr"] == 0.0003`. Step 3 only wrote to `last_checkpoint.pt`, while `best_model.pt` remained the untouched Stage 1 Epoch 16 checkpoint!

#### Cells 9, 10, 11, and 12 (Post-Training Evaluation & Checkpoint Export)
- **Cell 9 (`# Evaluate Custom Test Audios`, cell lines 1–50)** and **Cell 10 (`# Decisive 2-Tier Breakdown`, cell lines 1–15)** re-evaluated `checkpoints/best_model.pt` on `test_audio(razz)` — unknowingly testing the exact same unmodified Epoch 16 checkpoint from Cell 6.
- **Cell 11 (`# Create 1-click download link`, cell lines 1–24)** copied `/kaggle/working/trialSpeakSafe/checkpoints/best_model.pt` to `/kaggle/working/best_model.pt` for browser download.
- **Cell 12 (`# Check Checkpoint Details & Run Custom Test`, cell lines 1–77)** ran after a kernel restart (`execution_count: 4` at `2026-09-26T06:13:18Z`) when the working directory was `/kaggle/working`, crashing at line 7 with `ModuleNotFoundError: No module named 'model'`.

---

### 1.4 Post-Training `ConvSTFT` Replacement in `features/spectrogram.py` (Commit `de39db60`)
When `best_model.pt` was trained on Kaggle (commit `ee4b6a2`), `features/spectrogram.py` computed Path B's 3 base spectral streams using `torchaudio.transforms`:
- **Training-time (`commit ee4b6a2`) `features/spectrogram.py` state dict buffers in `de39db60:checkpoints/best_model.pt`**:
  - `feature_pipeline.spectrogram.mel.mel.spectrogram.window` (`T.MelSpectrogram(..., power=2.0, normalized=True)` + `T.AmplitudeToDB(top_db=80.0)`)
  - `feature_pipeline.spectrogram.mel.mel.mel_scale.fb`
  - `feature_pipeline.spectrogram.stft.stft.window` (`T.Spectrogram(..., power=1.0, normalized=True)` + `T.AmplitudeToDB(top_db=80.0)`)
  - `feature_pipeline.spectrogram.cqt.cqt_weights`
  - `feature_pipeline.spectrogram.cqt.stft.window` (`T.Spectrogram(..., power=1.0, normalized=True)` + `T.AmplitudeToDB(top_db=80.0)`)

After downloading `best_model.pt` from Kaggle, commit `de39db60` (`feat: in-graph ConvSTFT single-input pipeline...`) replaced `torchaudio.transforms` in [`features/spectrogram.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L32-L166) with `ConvSTFT` for ONNX Opset 17 export compatibility, loaded `best_model.pt` with `strict=False` ([`eval_custom.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/eval_custom.py#L16), [`demo_live_mic.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/demo_live_mic.py#L92)), and later re-saved `checkpoints/best_model.pt` on disk with the new `ConvSTFT` buffers **without retraining the model weights**.

This post-training replacement introduced three mathematical discrepancies in [`features/spectrogram.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L55-L132):
1. **Unimplemented `top_db` Dynamic-Range Clamping (`features/spectrogram.py#L68-L70`)**:
   ```python
   def _to_db(x: torch.Tensor, top_db: float = TOP_DB) -> torch.Tensor:
       log_spec = 20.0 * torch.log10(torch.clamp(x, min=1e-5))
       return log_spec
   ```
   The `top_db: float = TOP_DB` (`80.0`) parameter is **unused**! During training on Kaggle, `T.AmplitudeToDB(top_db=80.0)` clamped every spectrogram at `max_db - 80.0` before `_normalize_spec(x)` ([`features/spectrogram.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L73-L82)) scaled `[x_min, x_max]` to `[0, 1]`. Without `top_db` clamping, any silent time-frequency bin hits the floor `20 * log10(1e-5) = -100.0 dB` (a `~115–118 dB` span instead of `80 dB`), shifting the normalized `[0, 1]` mean of every Mel spectrogram upwards by `+0.14` to `+0.21`!
2. **Magnitude (`power=1.0`) vs. Power (`power=2.0`) Mel Filterbank Projection (`features/spectrogram.py#L100-L116`)**:
   In [`MelSpectrogramExtractor.forward`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L111-L118), `mag = self.stft(x)` computes the linear magnitude $\sqrt{\text{Re}^2 + \text{Im}^2 + 10^{-12}}$ ([`features/spectrogram.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L64)) and projects `torch.matmul(mag.transpose(1, 2), self.mel_fb)` (`norm="slaney"`, lines 101–108), whereas training used `T.MelSpectrogram(..., power=2.0, norm=None, normalized=True)`.
3. **Window Frame Normalization**:
   Training used `normalized=True` in `T.Spectrogram` and `T.MelSpectrogram` (dividing by $\sqrt{\sum w[n]^2}$), whereas [`ConvSTFT`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L48-L65) does not normalize by window energy.

---

## 2. Why the Model Achieved `0.52%` Val EER on Kaggle Yet Inverted Predictions on `test_audio(razz)/fake` & Live Phone Recordings

### 2.1 Empirical Diagnostic & Path A vs. Path B Ablation on `test_audio(razz)/fake`
Running the shipped model ([`checkpoints/best_model.pt`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/checkpoints/best_model.pt) / [`tatvanet.onnx`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/tatvaani_android/app/src/main/assets/tatvanet.onnx)) on all 9 files in `test_audio(razz)/fake` yields **1/9 correct (`11.1%` accuracy)** — systematically predicting **SAFE (Real)** on modern ElevenLabs AI voices and **DANGER (Fake)** on real WhatsApp/human recordings:

| Audio File in `test_audio(razz)/fake` | Ground Truth | Clip Dur. | 1st-2s Silence Ratio | Norm Mel Mean | Full Model `p_fake` | Path A Only (`SincConv`) `p_fake` | Path B Only (`SpecCNN`) `p_fake` | After VAD (`top_db=25`) `p_fake` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LA_E_2834763.flac` | Synthetic AI | `1.4s` | `0.46` | `0.393` | `0.4224` (CAUTION) | `0.609` | `0.267` | `0.6473` |
| `LA_E_4581379(real).flac` | **Real Human** | `2.1s` | `0.58` | `0.445` | **`0.8490` (FAKE ❌)** | **`0.250` (REAL ✅)** | **`0.978` (FAKE ❌)** | **`0.2621` (REAL ✅)** |
| `LA_E_5849185(real).flac` | **Real Human** | `4.4s` | **`0.80`** | `0.311` | **`0.7931` (FAKE ❌)** | `0.825` | `0.538` | `0.7868` |
| `LA_E_6828287.flac` | Synthetic AI | `1.6s` | `0.70` | `0.392` | `0.9797` (FAKE ✅) | `0.808` | `0.983` | `0.4371` |
| `LA_E_8877452.flac` | **Synthetic AI** | `4.0s` | `0.55` | `0.502` | **`0.0312` (REAL ❌)** | `0.076` | `0.313` | `0.0312` |
| `voice_preview_anika ... .mp3` (ElevenLabs) | **Synthetic AI** | `5.5s` | `0.22` | `0.538` | **`0.0432` (REAL ❌)** | `0.196` | `0.235` | `0.0247` |
| `voice_preview_monika sogam ... .mp3` (ElevenLabs) | **Synthetic AI** | `7.0s` | `0.26` | `0.552` | **`0.0676` (REAL ❌)** | `0.055` | `0.342` | `0.0522` |
| `WhatsApp Audio 2026-05-29 ... (1).mp4` | **Real Human** | `15.7s` | `0.15` | **`0.675`** | **`0.9508` (FAKE ❌)** | **`0.097` (REAL ✅)** | **`0.992` (FAKE ❌)** | `0.9508` |
| `WhatsApp Audio 2026-05-29 ... .mp4` | **Real Human** | `16.6s` | `0.09` | **`0.684`** | **`0.9564` (FAKE ❌)** | **`0.034` (REAL ✅)** | **`0.995` (FAKE ❌)** | `0.9564` |

Similarly, on [`demo_output/`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/demo_output) live laptop microphone recordings:
- `recorded_raw.wav` (unfiltered live mic recording with room/fan noise): **`p_fake = 0.9586` (DANGER / Fake)**

---

### 2.2 Five Primary-Source Root Causes of the Prediction Inversion

#### Root Cause 1: Step 3 Fine-Tuning Never Updated `best_model.pt` — Zero Modern TTS / ElevenLabs Learning
As proven in Section 1.3 (Cell 8) and verified from [checkpoints/best_model.pt](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/checkpoints/best_model.pt) (`epoch: 16`, `lr: 0.0003`), the Step 3 fine-tuning run on `garystafford/deepfake-audio-detection`, `gereon/voxguard-synthetic-speech`, and `skypro1111/elevenlabs_dataset` **never triggered `if val_eer < best_eer:`** ([`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L423-L426)).
- Consequently, the shipped model was trained strictly on **Stage 1** (`ASVspoof 2019 LA` classical 2017–2019 vocoders/TTS + `LibriSpeech` + `AI4Bharat`).
- Modern diffusion/flow-matching/neural-codec TTS voices (`voice_preview_anika`, `voice_preview_monika sogam`, and ASVspoof eval unseen algorithm `LA_E_8877452.flac`) lack the coarse buzz, phase discontinuities, and narrowband vocoder artifacts of ASVspoof 2019 `train` (`A01–A06`). Because they are studio-clean, full-bandwidth signals with smooth formant trajectories, the Stage 1 model classifies them with `93%–97%` confidence as `SAFE (Real)` (`p_fake = 0.0312–0.0676`).

#### Root Cause 2: Path B (`MultiStreamSpectrogram` + `_normalize_spec`) Confuses Ambient Microphone / WhatsApp Noise Floors with Spoofing Artifacts
Look at the Path A vs. Path B ablation in the table above for the two real `WhatsApp Audio` clips and `LA_E_4581379(real).flac`:
- **Path A (`PathA_SincCNN`, [`model/tatvanet.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/model/tatvanet.py#L161-L220))** correctly classifies both real WhatsApp recordings as **Real Human** (`p_A = 0.097` and `p_A = 0.034`) and `LA_E_4581379(real).flac` as **Real Human** (`p_A = 0.250`)!
- **Path B (`PathB_SpecCNN`, [`model/tatvanet.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/model/tatvanet.py#L224-L291))** overrides Path A inside `CrossAttentionFusion` ([`model/tatvanet.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/model/tatvanet.py#L296-L370)) by firing **`p_B = 0.992`**, **`0.995`**, and **`0.978` (DANGER / Fake)**!
- **Why Path B misfires on real mobile/WhatsApp audio**:
  1. In [`features/spectrogram.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L68-L82), `_to_db` followed by per-sample min-max normalization `_normalize_spec` maps `[x_min, x_max]` of every 2.0-second spectrogram to `[0.0, 1.0]`.
  2. In clean studio recordings (`LibriSpeech`, `ElevenLabs`), inter-word silence drops to the `-100 dB` numerical floor (`1e-5`), so `x_min = -100 dB` and the normalized spectrogram has a low background mean (`0.33–0.55` unclipped, `0.23–0.39` with `top_db=80`) and high frame-to-frame $\Delta$-Mel standard deviation (`dmel_std = 0.0365–0.0413`).
  3. In real WhatsApp recordings (`WhatsApp Audio ... .mp4`) and live phone/laptop microphone captures (`recorded_raw.wav`), continuous room tone, AC/fan noise, microphone pre-amp self-noise, and AAC/Opus comfort noise fill the inter-word gaps and compress dynamic range. This raises the normalized spectrogram mean to **`0.675–0.684`** and flattens temporal $\Delta/\Delta\Delta$ contrasts (`dmel_std = 0.0207–0.0226`), which Path B learned to associate with synthetic vocoder noise/over-smoothing!

#### Root Cause 3: Un-Trimmed Leading Silence & Blind First-2.0s Truncation (`training/train.py#L133-L141` & Notebook Evaluation Cells)
- Originally, [`process_dataset.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/process_dataset.py#L137-L148) applied `vad_trim(samples, sr, top_db=30.0)` (`librosa.effects.trim`) to strip leading and trailing silence from every dataset file before training.
- On Kaggle, Cell 3 in [raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1) bypassed `process_dataset.py` and fed raw files directly into [`TatvaaniDataset.__getitem__`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L108-L146).
- In [`TatvaaniDataset.__getitem__`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L133-L141) (and in [`eval_custom.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/eval_custom.py#L40-L43) / [`TEST_FOLDER_AUDIOS.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/TEST_FOLDER_AUDIOS.py#L83-L86)), validation and inference deterministically slice the **first 32,000 samples (2.0 seconds)**: `waveform = waveform[:, :self.target_n]`.
- In `LA_E_5849185(real).flac` (`4.4s` total), **80% of the first 2.0s is silence + low-frequency rumble (`<300 Hz` = 69.6% of energy)**. In `LA_E_4581379(real).flac`, **58% of the first 2.0s is silence**. Simply applying `librosa.effects.trim(raw, top_db=25)` to `LA_E_4581379(real).flac` before slicing 2.0s flips the model's prediction from **`p_fake = 0.8488` (DANGER ❌)** to **`p_fake = 0.2621` (SAFE ✅)**!

#### Root Cause 4: Unrealistic & Artifact-Inducing Augmentations in `training/augment.py`
Inspecting [`AudioAugmenter`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/augment.py#L25-L127) reveals why its synthetic degradations failed to generalize to real mobile microphones or speaker-to-mic replay:
1. **Circular Wrap-Around Splice Discontinuities (`training/augment.py#L75-L77`)**:
   ```python
   if random.random() < self.p_roll:
       shift = random.randint(-self.sr // 4, self.sr // 4)
       waveform = torch.roll(waveform, shifts=shift, dims=-1)
   ```
   `torch.roll` circularly wraps the end of the 2.0s waveform onto the beginning, creating an instantaneous phase/amplitude step discontinuity inside the clip on 30% of training samples.
2. **White Gaussian Noise Only (`training/augment.py#L56-L60`)**:
   `torch.randn_like(waveform) * noise_rms` injects flat white noise (`SNR 15–35 dB`), which has equal energy up to `8000 Hz`. Real microphone/room noise (fans, AC, traffic, voice babble, 50/60 Hz hum, pre-amp $1/f$ noise) is heavily concentrated below `500 Hz` (`LF<300 Hz` is `21%–72%` on real recordings).
3. **Brick-Wall FFT Zeroing (`training/augment.py#L86-L99`)**:
   `_fast_bandpass` multiplies `torch.fft.rfft` by a hard boolean mask `(freqs >= low_hz) & (freqs <= high_hz)`. A hard rectangular cutoff in frequency causes sinc-ringing (Gibbs phenomenon) in the time domain and sets out-of-band FFT bins to exact mathematical zero (`1e-5` floor after `_to_db`), which never occurs in real hardware or Opus/AAC audio codecs.
4. **No Speaker-to-Mic Acoustic Replay Simulation**:
   When a user plays a fake ElevenLabs voice over a laptop or phone speaker into the mobile microphone, the signal undergoes loudspeaker high-pass roll-off (`<150–250 Hz`), non-linear harmonic distortion, early + late room reflections, and microphone AGC compression. `_fast_reverb` ([`training/augment.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/augment.py#L116-L126)) convolves with a smooth exponential curve $e^{-\alpha t}$ (`rir = torch.exp(-decay * t)`), which acts as a simple low-pass filter rather than a sparse multi-reflection room impulse response (RIR).

#### Root Cause 5: Inflated Kaggle Validation Accuracy (`0.52%` EER) Due to Data Leakage & Post-Training `ConvSTFT` Drift
The `0.52%` Val EER on Kaggle did not reflect real-world generalization because:
1. Non-ASVspoof files in Cell 3 ([raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1), Cell 3 lines 79, 82) were assigned per-file `speaker_id` values (`fake_{fname_no_ext}`, `real_{fname_no_ext}`), leaking the same speakers across train and val.
2. AI4Bharat `indic_voices` were bucketed by `idx // 15` instead of true speaker ID (Cell 3 line 51) and duplicated across train/val in Cell 5 (lines 18–35).
3. ASVspoof `train`, `dev`, and `eval` protocols were pooled together in Cell 3 (lines 18–33) before splitting, so spoofing attack algorithms `A07–A19` were no longer held out as unseen attacks.
4. Post-training replacement of `torchaudio.transforms` with `ConvSTFT` in [`features/spectrogram.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L32-L166) without `top_db` clipping altered the Path B feature distribution (`SpecL1Diff` $\approx 0.02–0.03$) after `best_model.pt` had already been frozen.

---

## 3. Concrete, Primary-Source-Backed Fixes for the Next Kaggle Retraining Notebook

### Fix 1: Restore `top_db` Dynamic-Range Clamping & Window Normalization in `features/spectrogram.py` Before Training
Train TatvaNet **directly** with the in-graph `ConvSTFT` pipeline so there is zero post-training feature swap (`strict=True`), and fix [`_to_db`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L68-L70) so that `top_db = 80.0` (or `top_db = 65.0` to reject low-level room noise floors) is actually applied before [`_normalize_spec`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/features/spectrogram.py#L73-L82):

```python
# In features/spectrogram.py (replacing lines 68-70):
def _to_db(x: torch.Tensor, top_db: float = TOP_DB) -> torch.Tensor:
    log_spec = 20.0 * torch.log10(torch.clamp(x, min=1e-5))
    max_val = log_spec.amax(dim=(-2, -1), keepdim=True)
    log_spec = torch.clamp(log_spec, min=max_val - top_db)
    return log_spec
```
**Why this works**: Clamping the spectrogram floor at `max_val - top_db` prevents studio recordings with `-100 dB` digital silence from having a `118 dB` normalization denominator while live phone/WhatsApp recordings have a `65 dB` denominator. Additionally, add `nn.InstanceNorm2d(9, affine=True)` at the input of [`PathB_SpecCNN.stem`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/model/tatvanet.py#L240-L245) so Path B cannot use global channel mean (`mel_mean`) as a shortcut for real vs. fake.

---

### Fix 2: Fix `training/train.py` Checkpoint Saving Order & `--resume` vs. `--finetune` Logic
In [`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L306-L321) and [`#L410-L431`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L410-L431):
1. **Update `best_eer` and `no_improve` BEFORE constructing `ckpt`** (fixing lines 410–430):
   ```python
   is_best = val_eer < best_eer
   if is_best:
       best_eer   = val_eer
       no_improve = 0
   else:
       no_improve += 1

   ckpt = {
       "epoch":           epoch,
       "model_state":     model.state_dict(),
       "optimizer_state": optimizer.state_dict(),
       "scaler_state":    scaler.state_dict(),
       "scheduler_state": scheduler.state_dict(),
       "best_eer":        best_eer,
       "no_improve":      no_improve,
       "val_eer":         val_eer,
       "config":          config,
   }
   torch.save(ckpt, checkpoint_dir / "last_checkpoint.pt")
   if is_best:
       torch.save(ckpt, checkpoint_dir / "best_model.pt")
   ```
2. **Add `--finetune` flag to `training/train.py` (fixing lines 306–321)**:
   When fine-tuning on a new dataset (`--finetune checkpoints/best_model.pt`), load **only** `ckpt["model_state"]`, keep `start_epoch = 0`, keep `best_eer = float("inf")`, and use the freshly initialized `AdamW(lr=config["lr"])` and `CosineAnnealingLR(T_max=config["epochs"] - config["warmup_epochs"])` instead of overwriting them with the old checkpoint's exhausted scheduler and old validation EER threshold.

---

### Fix 3: Single-Pass, Deduplicated, Speaker-Disjoint Manifest Builder in `raziyaspeaksafetraining.ipynb`
Replace Cells 2, 3, 4, and 5 in [raziyaspeaksafetraining.ipynb](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/raziyaspeaksafetraining.ipynb#L1) with a **single unified dataset preparation & manifest builder** before training a single 30-epoch run (eliminating the broken two-stage resume workflow):
1. **Remove `if len(protocol_map) > 0: break`** in Cell 3 (lines 36–37) so every `.txt` protocol file in `/kaggle/input` (`ASVspoof2019.LA.cm.train.trn.txt`, `dev.trl.txt`, `eval.trl.txt`) is parsed completely.
2. **Decode, Resample to 16 kHz Mono, and VAD-Trim All Hugging Face Parquet Audio**:
   In Cells 2 and 4, decode `raw_bytes` via `sf.read(io.BytesIO(raw_bytes), dtype="float32")` (with `torchaudio.load`/`librosa.load` fallback), average stereo to mono, resample to `16,000 Hz`, strip leading/trailing silence (`top_db=28.0`), discard clips shorter than `1.0s`, and write valid `16 kHz` float32 WAV files.
3. **Preserve True Speaker / Voice IDs Across All Corpora**:
   - `ai4bharat/indicvoices_r`: Extract `row.get("speaker_id")` (or `row.get("ULCA_speaker_id")`) from the parquet row: `spk_id = f"indic_{lang}_{row_spk_id}"`.
   - `ASVspoof 2019 LA`: Use `parts[0]` (`LA_0079`) from the CM protocol.
   - `LibriSpeech`: Extract reader ID from parent directory structure (`<reader_id>/<chapter_id>/<reader_id>-<chapter_id>-<utt_id>.flac` → `spk_id = f"libri_{reader_id}"`, matching [`process_dataset.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/process_dataset.py#L552)).
   - `WaveFake` / `MLAAD` / `ElevenLabs` / `VoxGuard`: Group by voice/generator ID (`row.get("voice_id")`, `row.get("speaker")`, `row.get("model")`, or parent folder name) rather than `f"fake_{fname_no_ext}"`.
4. **Enforce True Modern Fake & Real Mic Diversity (1:1 Balanced)**:
   - Include **at least 3,000–5,000 modern neural TTS/VC clips** across `skypro1111/elevenlabs_dataset`, `gereon/voxguard-synthetic-speech`, `garystafford/deepfake-audio-detection` (verifying label polarity per dataset before ingesting!), `MLAAD`, and `WaveFake`, alongside `ASVspoof 2019 LA` spoof clips.
   - Balance 1:1 with `AI4Bharat IndicVoices` (Hindi, Kannada, Telugu, Kashmiri, Tamil, Malayalam, Bengali, Marathi), `ASVspoof 2019 LA` bonafide, and `LibriSpeech`, with **zero file duplication** between train, val, and test.

---

### Fix 4: Active-Speech Energy Windowing & DC-Offset Removal in `training/train.py` and Inference
In [`TatvaaniDataset.__getitem__`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L126-L146) and in inference ([`TEST_FOLDER_AUDIOS.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/TEST_FOLDER_AUDIOS.py#L77-L91), [`server/app.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/server/app.py), and Android `FeatureExtractor.kt`):
1. **Subtract DC Offset**: Always center the waveform (`waveform = waveform - waveform.mean(dim=-1, keepdim=True)`) prior to peak normalization.
2. **Strip Leading/Trailing Silence & Select Max-Energy 2.0s Window**:
   Instead of `waveform = waveform[:, :self.target_n]` ([`training/train.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/train.py#L141)), which feeds 58%–80% leading silence on `LA_E_4581379(real).flac` and `LA_E_5849185(real).flac`:
   - Trim leading/trailing frames whose short-time RMS energy (10 ms frame = 160 samples) is below `5%` of peak frame RMS (`top_db ≈ 26 dB`).
   - For clips still longer than `32,000` samples (`2.0s`):
     - During **training** (`augment=True`): randomly sample a `32,000`-sample window from the active voiced region.
     - During **validation / inference** (`augment=False`): slide a `32,000`-sample window (stride `8,000` samples = `0.5s`) across the voiced signal and select the highest-energy window (or average logits across top-$K$ voiced windows for multi-window file/call scoring).
   - Peak-normalize (`waveform = waveform / waveform.abs().max().clamp(min=1e-6)`) **after** extracting the `32,000`-sample window so that an isolated transient click outside the 2.0s window cannot attenuate the active speech inside the window.

---

### Fix 5: Upgrade `AudioAugmenter` in `training/augment.py` for Live Phone Mic, WhatsApp Codecs & Speaker-to-Mic Replay
Replace the simplified augmentations in [`training/augment.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/augment.py#L43-L127) with physically realistic, symmetrically applied channel degradations:
1. **Remove Circular `torch.roll` (`training/augment.py#L75-L77`)**:
   Replace `torch.roll` with zero-padded or reflection-padded time shifting so boundary splice clicks are never introduced.
2. **Add Colored ($1/f$ Pink/Brown) Ambient Noise + 50/60 Hz Hum (`p = 0.50`, Symmetric Across Real & Fake)**:
   In [`AudioAugmenter.__call__`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/augment.py#L56-L60), shape white noise in the RFFT domain by $1 / \max(f, 20\text{ Hz})^{\beta}$ ($\beta \in [0.5, 1.5]$) plus optional `50 Hz`/`60 Hz` mains hum harmonics (`SNR 10–30 dB`, matching [`demo_preprocess_visual.py`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/demo_preprocess_visual.py#L41-L49)). Applying colored low-frequency room noise equally to `label=0` (Real) and `label=1` (Fake) forces `PathB_SpecCNN` to ignore the ambient noise floor that caused `WhatsApp Audio` (`p_fake = 0.95`) and `recorded_raw.wav` (`p_fake = 0.96`) to be misclassified as Fake.
3. **Replace Brick-Wall FFT Masking with Smooth Roll-Off Bandpass (`training/augment.py#L86-L99`)**:
   Instead of multiplying `fft` by a binary `(freqs >= low_hz) & (freqs <= high_hz)` mask (which zeros out bins to `0.0`), use a smooth raised-cosine or Butterworth-style magnitude response $\frac{1}{\sqrt{1 + (f / f_{\text{high}})^{2N}}} \cdot \frac{1}{\sqrt{1 + (f_{\text{low}} / f)^{2N}}}$ ($N \in [2, 4]$) with a realistic stopband floor (`-40 dB` to `-60 dB`), matching mobile hardware anti-aliasing and WhatsApp/Opus filters.
4. **Add Speaker-to-Mic Acoustic Replay & Non-Linear Transducer Distortion (`p = 0.30`)**:
   To ensure ElevenLabs/TTS played from a laptop or phone speaker into the mobile microphone is caught as `DANGER (Fake)`:
   - Apply gentle non-linear loudspeaker saturation: `waveform = torch.tanh(drive * waveform) / math.tanh(drive)` with `drive` $\in [1.2, 2.5]$.
   - Convolve with a realistic **multi-tap Room Impulse Response (RIR)**: direct path impulse at $t=0$, $5\text{–}12$ random early reflections within $5\text{–}40\text{ ms}$, plus exponentially decaying diffuse reverberation tail ($T_{60} \in [0.08\text{ s}, 0.35\text{ s}]$) modulated by Gaussian velvet noise (replacing the deterministic $e^{-\alpha t}$ low-pass filter in [`_fast_reverb`](file:///c:/Users/RAZIYA%20KHAN/Desktop/Tatvaani_Reema%20-%20Clean%20-%20Copy/training/augment.py#L116-L126)).
   - Apply small-speaker bass cut (`high-pass` at `150–300 Hz`) followed by simulated mobile microphone AGC / dynamic range compression.
