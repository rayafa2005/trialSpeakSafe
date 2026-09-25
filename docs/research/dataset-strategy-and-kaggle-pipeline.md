# Research & Technical Specification: Indic Dataset Diversity & Kaggle Training

## 1. Multi-Language & Accent Diversity Verification

To ensure that the 4th dataset provides genuine representation of Indian languages and regional accents without silent gaps:

### Language Distribution (AI4Bharat IndicVoices / Kathbath)
The dataset contains explicit language partitions across 22 Indic languages:
- **North / Central**: Hindi (`hi`), Punjabi (`pa`), Marathi (`mr`), Gujarati (`gu`), Rajasthani (`raj`), Bhojpuri (`bho`)
- **South**: Tamil (`ta`), Telugu (`te`), Kannada (`kn`), Malayalam (`ml`)
- **East / North-East**: Bengali (`bn`), Odia (`or`), Assamese (`as`), Manipuri (`mni`)
- **Indian English**: Indian-accented English (`en-IN`)

### Automated Manifest Validation
Before feeding any manifest into PyTorch `DataLoader`, run a validation sweep:
```python
def validate_dataset_manifest(df):
    assert df['label'].isin([0, 1, 2]).all(), "Invalid labels detected"
    assert (df['duration'] >= 0.5).all(), "Audio clips shorter than 0.5s found"
    print("Language Breakdown:\n", df['language'].value_counts())
    print("Real vs Fake Balance:\n", df['label'].value_counts())
```

---

## 2. Dataset Sizing & Sweet-Spot Composition

| Dataset Scope | Sample Count | Real / Fake Breakdown | Estimated Training Time (Kaggle T4 / P100) |
| :--- | :--- | :--- | :--- |
| **Fast Prototype (20k)** | 20,000 | 10,000 Real : 10,000 Fake | **~15–20 minutes** (30 epochs) |
| **Optimal Production (50k)** | 50,000 | 25,000 Real : 25,000 Fake | **~45–60 minutes** (30 epochs) |
| **Full Massive (160k+)** | 166,426 | 83,000 Real : 83,000 Fake | **~3–3.5 hours** (30 epochs) |

### Recommended 50,000 Balanced Composition:
1. **Real Audio (25,000 samples)**:
   - `9,000` samples: IndicVoices / Kathbath (Indian regional languages + Indian English)
   - `8,000` samples: LibriSpeech (Western clean read English)
   - `8,000` samples: ASVspoof 2019 Bonafide (Telephony / conversational real speech)
2. **Fake Audio (25,000 samples)**:
   - `12,500` samples: ASVspoof 2019 Spoofed (Algorithms A01–A19: vocoders, VC, classical neural synthesis)
   - `12,500` samples: MLAAD / IndicSynth (Modern neural synthesis: VITS, FastSpeech2, XTTS, ElevenLabs)

---

## 3. Kaggle Cloud Architecture Advantages

- **Linux `fork` multiprocessing**: Unlike Windows `spawn`, Kaggle workers (`num_workers=4`) share zero-copy memory pointers, eliminating RAM thrashing.
- **30GB RAM & 73GB Disk**: Eliminates out-of-memory crashes completely.
- **Zero-Wait Dataset Mounting**: Direct access via `/kaggle/input/` with zero download bandwidth usage.
