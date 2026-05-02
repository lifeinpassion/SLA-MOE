# SLA-MoE: Self-Learning-Aware Mixture of Experts for EEG Artifact Removal

Reference implementation accompanying the manuscript
**"Self-Learning-Aware Mixture-of-Experts (SLA-MoE) with ICA-Based Initialization for Enhanced EEG Artifact Removal"** (under revision for IEEE JBHI).

This code is the JBHI-revision version. It supersedes the original
`run_all_experiments.py` with proper SNR-based contamination, subject-grouped
cross-validation, full-seed baselines, paired statistical tests, an ablation
runner, and a downstream motor-imagery BCI evaluation on BCI Competition IV-2a.

---

## Repository layout

```
MoE-main/
├── run_all_experiments.py              # main runner (SNR sweep + LOSO + stats)
├── src/
│   ├── experiments/
│   │   ├── experiment_configs.py       # full + quick configs, baseline configs
│   │   └── ablation.py                 # ablation runner (E, top-k, pretraining, ...)
│   ├── models/
│   │   ├── baselines.py                # LMS, RLS, Kalman, EEGDnoiseNet, ResNet, ...
│   │   └── ica_moe_original.py         # SLA-MoE applier
│   ├── utils/
│   │   ├── data_utils.py               # SNR-based contamination, subject-K-fold
│   │   ├── metrics.py                  # RMSE, SNR, correlation, spectral distortion
│   │   ├── stats.py                    # paired Wilcoxon, Holm-Bonferroni, LaTeX tables
│   │   └── seed_utils.py
│   ├── visualization/                  # figure scripts
│   └── downstream/
│       └── bci_iv_2a.py                # MI-BCI clinical-task pipeline
└── results/                            # auto-populated
```

---

## Hardware recommendations

| Hardware                                | What you can do                                                          |
|-----------------------------------------|--------------------------------------------------------------------------|
| MacBook Air M1, 8 GB                    | `--mode quick` smoke runs only (~90 minutes for 2 seeds × 2 folds × SimpleCNN baseline). Not for paper figures. |
| Single consumer GPU (RTX 3060+)         | Full SNR sweep + 5-fold LOSO + 5 seeds, ~overnight per artifact scenario. |
| Colab T4 (free) / Kaggle P100 (free)    | Full run for one scenario per session; chunk across sessions.            |
| Colab Pro L4 / A100, university cluster | Full sweep + ablation + downstream BCI in a single day.                  |

The code auto-selects the best available device: `cuda` → `mps` (Apple Silicon)
→ `cpu`. Apple MPS is sufficient for the quick configuration only.

---

## Install

```bash
git clone https://github.com/<your-username>/SLA-MoE.git
cd SLA-MoE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the BCI-IV-2a downstream task, you additionally need:
```bash
pip install moabb mne braindecode scikit-learn
```

---

## Data: EEGdenoiseNet

Download the three `.npy` files from the [EEGdenoiseNet repo](https://github.com/ncclabsustech/EEGdenoiseNet)
and place them in `MoE-main/data/`:

```
data/
├── EEG_all_epochs.npy       # (4514, 512)
├── EOG_all_epochs.npy       # (3400, 512)
└── EMG_all_epochs.npy       # (5598, 512)
```

If you also have the per-segment **subject-id** mapping (one int per row of
`EEG_all_epochs.npy`), save it as `data/subject_ids.npy` and the runner will
use proper subject-grouped K-fold. Without it, the runner falls back to
chunked grouping (documented as a limitation in the paper).

---

## Quick start

### Quick sanity check (works on M1 / 8 GB)

```bash
python run_all_experiments.py --mode quick
```

Runs at one SNR (-3 dB), 2 seeds, 2 folds, with SimpleCNN + LMS baselines.
**~90 minutes on an M1 Air**, mostly LSTM training in the SLA-MoE pre-training
phase. Verifies the pipeline end-to-end before you launch the full sweep on Colab.

### Full paper sweep (run on Colab / Kaggle / GPU box)

```bash
python run_all_experiments.py --mode full --experiment all
```

This runs:
- SNR sweep over {-7, -5, -3, -1, 0, +1, +2} dB
- 5 seeds for SLA-MoE *and* every baseline
- 5-fold subject-grouped cross-validation
- Three artifact scenarios (EEG+EOG, EEG+EMG, EEG+EOG+EMG)
- Paired Wilcoxon + Holm-Bonferroni vs SLA-MoE
- LaTeX tables ready for the manuscript

Outputs land in `results/<experiment>/`:
- `raw_results.json`
- `stats_at_-3.0dB.json`
- `table_at_-3.0dB.tex`

### Ablation study

```bash
python -m src.experiments.ablation --data-path data
```

### Downstream BCI-IV-2a

```bash
python -m src.downstream.bci_iv_2a --subjects 1 2 3 4 5 6 7 8 9
```

Reports 4-class accuracy and Cohen's κ for: clean-upper-bound, no-denoising,
SLA-MoE-denoised. Without MOABB installed, runs on synthetic data so you can
verify the pipeline.

---

## Running on Colab from GitHub (no GPU required locally)

1. Push this repo to GitHub.
2. Open `notebooks/colab_full_sweep.ipynb` in Colab via the GitHub badge.
3. The notebook clones the repo, installs deps, downloads EEGdenoiseNet, and
   launches `run_all_experiments.py --mode full` on the Colab T4/A100.
4. Results live in `/content/SLA-MoE/results/` and can be downloaded as a zip
   at the end of the notebook.

This is the recommended workflow for users on M1/Air-class hardware.

---

## Reproducing the paper tables

```bash
# Table II (overall comparison @ SNR = -3 dB)
python run_all_experiments.py --mode full --experiment eeg_eog_emg
cat results/eeg_eog_emg/table_at_-3.0dB.tex

# Ablation table
python -m src.experiments.ablation
cat results/ablation/ablation_table.tex

# Downstream BCI table
python -m src.downstream.bci_iv_2a
cat results/downstream_bci_iv_2a/summary.json
```

---

## Code availability

All code, configurations, and per-seed JSON results are released under the MIT
license. The repository will be assigned a Zenodo DOI on manuscript acceptance.

---

## Citation

```bibtex
@article{liang2026slamoe,
  title   = {Self-Learning-Aware Mixture-of-Experts (SLA-MoE) with ICA-Based
             Initialization for Enhanced EEG Artifact Removal},
  author  = {Liang, Ping and De Ocampo, Anton Louise},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```
