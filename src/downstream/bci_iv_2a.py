"""
Downstream clinical-task pipeline: motor-imagery BCI on BCI Competition IV-2a.

This is the experiment that directly addresses the JBHI editor's
"biomedical/health-informatics contribution" concern. The story is:

    raw EEG  ->  SLA-MoE denoising  ->  EEGNet classifier  ->  4-class accuracy

We compare three pipelines:
    (i)   classifier on uncontaminated raw EEG          [upper bound]
    (ii)  classifier on artifact-contaminated EEG      [no denoising]
    (iii) classifier on contaminated -> SLA-MoE        [our method]
    (iv)  classifier on contaminated -> baseline DL    [each baseline]

Acceptance metric: Cohen's kappa and 4-class accuracy averaged over the 9
BCI-IV-2a subjects, reported with standard error and paired Wilcoxon.

DATA ACCESS:
  BCI-IV-2a is publicly hosted by the BNCI Horizon project. The simplest
  way to load it is via MOABB:

      pip install moabb mne braindecode

      from moabb.datasets import BNCI2014_001
      from moabb.paradigms import LeftRightImagery
      paradigm = LeftRightImagery()
      X, labels, meta = paradigm.get_data(BNCI2014_001(), subjects=[1])

  We keep MOABB imports local so the rest of the repo doesn't require
  the heavy dependency tree just to run artifact-removal experiments.

NOTE: This file is a *scaffold*. It runs end-to-end on synthetic data, but
the real BCI-IV-2a runs require MOABB and ~30 minutes per subject on a
single GPU. Run it on Colab or Kaggle, not on your M1.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.data_utils import contaminate
from src.utils.seed_utils import set_all_seeds


# =============================================================================
# Config
# =============================================================================

@dataclass
class BCIConfig:
    subjects: List[int]
    n_classes: int = 4               # left, right, foot, tongue
    sfreq: float = 250.0
    window_sec: float = 4.0          # 4-second MI windows per BCI-IV-2a
    target_snr_db: float = -3.0      # contamination level applied to raw EEG
    classifier_epochs: int = 200
    batch_size: int = 64
    seeds: Tuple[int, ...] = (40, 41, 42, 43, 44)
    use_moabb: bool = True


DEFAULT_BCI_CONFIG = BCIConfig(subjects=list(range(1, 10)))   # 9 subjects


# =============================================================================
# Data loading (MOABB; falls back to synthetic if not installed)
# =============================================================================

def load_bci_iv_2a(subject_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (X, y, sfreq) where:
        X : (n_trials, n_channels, n_samples)
        y : (n_trials,) integer labels in [0, 3]
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError:
        logging.warning("MOABB not installed; returning synthetic BCI-IV-2a-shaped data.")
        return _synthetic_bci_data(subject_id)

    paradigm = MotorImagery(n_classes=4)
    dataset = BNCI2014_001()
    X, labels, _ = paradigm.get_data(dataset, subjects=[subject_id])
    # MOABB labels are strings; map to ints
    label_map = {l: i for i, l in enumerate(sorted(set(labels)))}
    y = np.array([label_map[l] for l in labels], dtype=int)
    return X.astype(np.float32), y, paradigm.resample or 250.0


def _synthetic_bci_data(subject_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.RandomState(subject_id)
    n_trials = 288
    n_channels = 22
    n_samples = 1000   # 4 sec @ 250 Hz
    X = rng.randn(n_trials, n_channels, n_samples).astype(np.float32) * 1e-5
    y = rng.randint(0, 4, size=n_trials)
    return X, y, 250.0


# =============================================================================
# Artifact contamination using EEGdenoiseNet artifacts
# =============================================================================

def _resample_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[-1] == target_len:
        return arr
    from scipy.signal import resample
    return resample(arr, target_len, axis=-1)


def contaminate_bci_with_artifacts(X: np.ndarray,
                                   eog_segments: np.ndarray,
                                   emg_segments: np.ndarray,
                                   target_snr_db: float,
                                   seed: int) -> np.ndarray:
    """
    Add EOG + EMG artifacts (sampled from EEGdenoiseNet) to BCI-IV-2a trials.
    Each (trial, channel) pair gets a random EOG and a random EMG segment,
    rescaled to match the trial length, then added at the requested input SNR.
    """
    rng = np.random.RandomState(seed)
    n_trials, n_channels, n_samples = X.shape

    # Resample each artifact bank to the trial length
    eog_rs = _resample_to_length(eog_segments, n_samples)
    emg_rs = _resample_to_length(emg_segments, n_samples)

    X_noisy = X.copy()
    for t in range(n_trials):
        for c in range(n_channels):
            eog_idx = rng.randint(eog_rs.shape[0])
            emg_idx = rng.randint(emg_rs.shape[0])
            artifact = (eog_rs[eog_idx] + emg_rs[emg_idx]).astype(np.float32)

            clean = X[t, c]
            # SNR-based scaling
            p_sig = float(np.mean(clean ** 2))
            p_art = float(np.mean(artifact ** 2)) + 1e-12
            lam = float(np.sqrt(p_sig / (p_art * 10 ** (target_snr_db / 10.0))))
            X_noisy[t, c] = clean + lam * artifact
    return X_noisy


# =============================================================================
# Apply SLA-MoE / baseline denoising channel-by-channel
# =============================================================================

def denoise_per_channel(X_noisy: np.ndarray,
                        denoise_fn,
                        seed: int) -> np.ndarray:
    """
    Apply a denoising callable to each (trial, channel) signal.
    `denoise_fn(signal_2d)` should accept and return shape (n_trials_subset, n_samples).
    """
    n_trials, n_channels, n_samples = X_noisy.shape
    out = np.empty_like(X_noisy)
    for c in range(n_channels):
        signal_2d = X_noisy[:, c, :]
        denoised = denoise_fn(signal_2d, seed=seed)
        out[:, c, :] = denoised
    return out


# =============================================================================
# Lightweight EEGNet classifier
# =============================================================================

def build_eegnet(n_channels: int, n_samples: int, n_classes: int = 4):
    import torch.nn as nn
    return nn.Sequential(
        nn.Unflatten(1, (1, n_channels)),
        nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
        nn.BatchNorm2d(16),
        nn.ELU(),
        nn.AvgPool2d((1, 4)),
        nn.Dropout(0.5),
        nn.Conv2d(16, 16, kernel_size=(1, 16), padding=(0, 8), bias=False),
        nn.BatchNorm2d(16),
        nn.ELU(),
        nn.AvgPool2d((1, 8)),
        nn.Dropout(0.5),
        nn.Flatten(),
        nn.LazyLinear(n_classes),
    )


def train_classifier(X: np.ndarray, y: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     seed: int,
                     epochs: int = 200,
                     batch_size: int = 64,
                     device=None) -> Tuple[float, float]:
    """Train EEGNet, return (val_accuracy, val_kappa)."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import cohen_kappa_score

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")

    set_all_seeds(seed)
    n_channels, n_samples = X.shape[1], X.shape[2]
    model = build_eegnet(n_channels, n_samples).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)
    X_v = torch.from_numpy(X_val).float().to(device)
    y_v = torch.from_numpy(y_val).long().to(device)

    n = X_t.shape[0]
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        idx = rng.permutation(n)
        model.train()
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            opt.zero_grad()
            logits = model(X_t[b])
            loss = loss_fn(logits, y_t[b])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(X_v).argmax(dim=1).cpu().numpy()
    y_v_np = y_v.cpu().numpy()
    acc = float((pred == y_v_np).mean())
    kappa = float(cohen_kappa_score(y_v_np, pred))
    return acc, kappa


# =============================================================================
# End-to-end pipeline
# =============================================================================

def run_subject(subject_id: int,
                eog_segments: np.ndarray,
                emg_segments: np.ndarray,
                cfg: BCIConfig,
                denoisers: Dict[str, callable]) -> Dict[str, Dict[str, float]]:
    """
    Returns: {pipeline_name: {accuracy, kappa}} averaged over seeds.
    Pipelines:
      'raw_clean', 'noisy_no_denoise', 'noisy_<denoiser_name>'
    """
    X, y, _ = load_bci_iv_2a(subject_id)
    n = X.shape[0]
    perm = np.random.RandomState(0).permutation(n)
    train_idx, val_idx = perm[: int(0.8 * n)], perm[int(0.8 * n):]

    pipelines: Dict[str, Dict[str, List[float]]] = {}

    def evaluate(name: str, X_train: np.ndarray, X_val: np.ndarray):
        accs, kaps = [], []
        for seed in cfg.seeds:
            acc, kap = train_classifier(X_train, y[train_idx], X_val, y[val_idx],
                                        seed=seed, epochs=cfg.classifier_epochs,
                                        batch_size=cfg.batch_size)
            accs.append(acc); kaps.append(kap)
        pipelines[name] = {"accuracy": accs, "kappa": kaps}

    # (i) clean upper bound
    evaluate("raw_clean", X[train_idx], X[val_idx])

    # contaminated
    X_noisy = contaminate_bci_with_artifacts(X, eog_segments, emg_segments,
                                             target_snr_db=cfg.target_snr_db,
                                             seed=42)

    # (ii) no denoise
    evaluate("noisy_no_denoise", X_noisy[train_idx], X_noisy[val_idx])

    # (iii)/(iv) denoise + classify, for each denoiser
    for d_name, fn in denoisers.items():
        denoised = denoise_per_channel(X_noisy, fn, seed=42)
        evaluate(f"noisy_{d_name}", denoised[train_idx], denoised[val_idx])

    return pipelines


def aggregate(per_subject: Dict[int, Dict[str, Dict[str, List[float]]]]) -> Dict[str, Dict[str, float]]:
    """Pool results across subjects: report mean ± std of per-subject means."""
    pipelines = next(iter(per_subject.values())).keys()
    out = {}
    for p in pipelines:
        accs = []
        kaps = []
        for s, data in per_subject.items():
            accs.append(np.mean(data[p]["accuracy"]))
            kaps.append(np.mean(data[p]["kappa"]))
        out[p] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std":  float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "kappa_mean":    float(np.mean(kaps)),
            "kappa_std":     float(np.std(kaps, ddof=1)) if len(kaps) > 1 else 0.0,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data",
                   help="Directory with EEGdenoiseNet artifact .npy files (used for contamination).")
    p.add_argument("--output-dir", default="results/downstream_bci_iv_2a")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--snr", type=float, default=-3.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    eog = np.load(Path(args.data_path) / "EOG_all_epochs.npy") if (Path(args.data_path) / "EOG_all_epochs.npy").exists() else None
    emg = np.load(Path(args.data_path) / "EMG_all_epochs.npy") if (Path(args.data_path) / "EMG_all_epochs.npy").exists() else None
    if eog is None or emg is None:
        logging.warning("EEGdenoiseNet artifacts not found; using synthetic artifacts.")
        eog = np.random.randn(2000, 1000).astype(np.float32)
        emg = np.random.randn(2000, 1000).astype(np.float32)

    cfg = DEFAULT_BCI_CONFIG
    cfg.subjects = args.subjects
    cfg.target_snr_db = args.snr

    # Define denoisers as channel-callables that accept (n_trials_subset, n_samples).
    from src.models.ica_moe_original import apply_rnn_moe_filter_ica

    def sla_moe_denoiser(signal_2d: np.ndarray, seed: int) -> np.ndarray:
        # Single-channel input shape: (n_trials, n_samples).
        # The applier denormalizes via `out * stds + means` where stds/means
        # are broadcast across the trial dimension, so they must have shape
        # (n_samples,) -- not (n_trials,). The previous version had this
        # transposed and produced a broadcasting error.
        zeros = np.zeros_like(signal_2d)
        n_samples = signal_2d.shape[1]
        stds = np.ones(n_samples)
        means = np.zeros(n_samples)
        return apply_rnn_moe_filter_ica(
            signal_2d, signal_2d, zeros, zeros, stds, means, seed=seed)

    denoisers = {"SLA-MoE": sla_moe_denoiser}

    per_subject = {}
    for s in cfg.subjects:
        logging.info(f"Subject {s}/{cfg.subjects[-1]}...")
        per_subject[s] = run_subject(s, eog, emg, cfg, denoisers)

    summary = aggregate(per_subject)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_subject.json").write_text(json.dumps(per_subject, indent=2,
        default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("Done. Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
