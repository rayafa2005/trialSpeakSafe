# Workflow: Train TatvaNet 10k real + 10k fake (Kaggle)

**Trigger:** New `data/manifest.csv` after processing the three raw folders under `data/raw/` (LA / ASVspoof, LibriSpeech, MLAAD-tiny).

**Definition of done:** `checkpoints/best_model.pt` with val EER logged; ONNX exported at **2.0 s** (32k samples) and copied to `tatvaani_android/app/src/main/assets/tatvanet.onnx`.

## Checkpoint (human)

- Confirm shipped app uses **2 s** audio (ADR 0001). Training must use `target_samples: 32000` in `training/train.py` before cloud run—not the current 80k default.

## Steps (agent-implementable)

1. `process_dataset.py` for each raw dataset → `data/processed/` + `data/manifest.csv`.
2. Filter manifest rows to `dataset in {asvspoof, librispeech, mlaad}` only.
3. Speaker-disjoint split: `split_dataset.py` (optionally on filtered manifest).
4. Subsample **exactly** 10_000 binary-real and 10_000 binary-fake **train** rows (stratified by dataset), 15% val from held-out speakers—not only `balance_split` ratio cap.
5. Kaggle: New Notebook → GPU T4/P100 → clone repo → pip install `-r requirements` → run `training/train.py`.
6. `export/export_tflite.py --checkpoint checkpoints/best_model.pt --clip_duration 2.0` → copy ONNX to Android assets.

**Brief at checkpoint:** Val EER %, train real/fake counts (must be 10000/10000), path to `best_model.pt`, and whether ONNX input shape is `(1, 32000)`.
