# Training platform & supplemental datasets (Tatvaani)

Primary sources only. Written for 20k balanced clip training (10k real / 10k fake).

## Recommendation: **Kaggle Notebooks** (over Colab Free)

| Factor | Kaggle | Google Colab (free) |
|--------|--------|---------------------|
| GPU quota | [30 h/week GPU on Notebooks](https://www.kaggle.com/docs/efficient-gpu-usage) | [Variable; sessions often disconnect](https://research.google.com/colaboratory/faq.html) |
| Session length | Up to ~9 h interactive (per Kaggle docs on notebook limits) | Free tier ~hours, not guaranteed |
| Dataset size | Attach large datasets to the notebook / use Kaggle Datasets | Drive mount + manual upload |
| Reproducibility | Versioned dataset + git clone in notebook | Same possible but quota more brittle |

**Pick Kaggle** if you are on the free tier: more predictable weekly GPU budget for multi-epoch TatvaNet training. Use **Colab Pro** only if you already pay for it and need longer single sessions.

Balancing references:

- Downsampling majority class: standard sklearn / imbalanced-learn practice ([User guide — imbalanced datasets](https://scikit-learn.org/stable/modules/imbalance.html)).
- Class weights in loss: [PyTorch `CrossEntropyLoss` weight parameter](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). This repo combines both via `split_dataset.py --balance` and `FocalLoss` alpha in `training/train.py`.

## Extra public datasets (if you need more **real** English or **Indian** synthetic)

| Dataset | Adds | Access |
|---------|------|--------|
| [ASVspoof 2019 LA](https://www.asvspoof.org/index2019.html) | Standard CM benchmark; bonafide + many spoof attacks | Already in `data/raw/LA` |
| [WaveFake](https://github.com/RUB-SysSec/WaveFake) | Many vocoders; strong **fake** diversity | Download to `data/raw/WaveFake`; `process_dataset.py --dataset wavefake` |
| [IndicSynth](https://huggingface.co/datasets) (project-specific; see `process_dataset.py`) | **Indian-language synthetic** speech | `data/raw/IndicSynth` |
| [ASVspoof 2021 DF/LA](https://www.asvspoof.org/index2021.html) | Newer attacks, optional upgrade path | Registration on asvspoof.org |
| [LibriSpeech](https://www.openslr.org/12) | Clean **real** English | Already used in current manifest |

For strict 10k/10k from **only** LA + LibriSpeech + MLAAD-tiny: you already have enough **fake** in ASVspoof and enough **real** in LibriSpeech; MLAAD-tiny adds extra fake variety.
