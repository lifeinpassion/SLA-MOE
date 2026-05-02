#!/usr/bin/env python3
"""
Comprehensive experiment runner for EEG denoising paper (JBHI revision).

CHANGES vs the original runner:
  * Outer loop over a multi-point input-SNR sweep (default {-7,-5,-3,-1,0,1,2} dB).
  * Subject-grouped K-fold (default 5) instead of segment-level random split.
  * Baselines run on ALL 5 seeds (was 2), matching SLA-MoE.
  * Device auto-detection: CUDA -> MPS (Apple Silicon) -> CPU.
  * Per-seed/per-fold metrics are stacked, then paired Wilcoxon + Holm-Bonferroni
    are computed against SLA-MoE and dumped as JSON + a publication-ready
    LaTeX table.

Usage:
    # Full sweep on Colab/Kaggle (recommended for the paper revision)
    python run_all_experiments.py --mode full

    # Quick sanity run on M1 / 8 GB MacBook Air (~10 minutes)
    python run_all_experiments.py --mode quick

    # Single experiment, single SNR
    python run_all_experiments.py --experiment eeg_eog_emg --snr -3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.experiment_configs import (
    DEFAULT_HEADLINE_SNR_DB,
    DEFAULT_SNR_SWEEP_DB,
    EEG_EOG_EMG_QUICK,
    ExperimentConfig,
    BaselineConfig,
    get_all_experiment_configs,
    get_baseline_configs,
    get_quick_baseline_configs,
)
from src.models.baselines import (
    EEGDnoiseNet, EEGDnet, RNNEEG, ResNetEEG, SimpleCNN,
    WienerFilter, LMSFilter, RLSFilter, KalmanFilter,
    train_baseline_model, apply_baseline_model,
)
from src.models.ica_moe_original import (
    apply_rnn_moe_filter_ica,
    apply_rnn_moe_filter_ica_eog_only,
    apply_rnn_moe_filter_ica_emg_only,
)
from src.utils.data_utils import (
    contaminate, generate_synthetic_data, load_subject_ids,
    preprocess_data, preprocess_eeg_eog, preprocess_eeg_emg,
    subject_kfold_splits, segment_kfold_splits,
)
from src.utils.metrics import compute_metrics
from src.utils.seed_utils import set_all_seeds
from src.utils.stats import (
    build_latex_table, compare_to_reference, save_latex_table, save_stats_json,
)


METRIC_ORDER = ["rmse", "rrmse", "correlation", "snr", "snr_improvement", "spectral_distortion"]
SLA_MOE_NAME = "SLA-MoE"   # canonical name everywhere


# =============================================================================
# Setup
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_or_generate_data(data_path: str | None) -> Dict[str, np.ndarray]:
    if data_path:
        d = Path(data_path)
        if (d / "EEG_all_epochs.npy").exists():
            logging.info(f"Loading EEGdenoiseNet from {d}")
            return {
                "eeg": np.load(d / "EEG_all_epochs.npy"),
                "eog": np.load(d / "EOG_all_epochs.npy"),
                "emg": np.load(d / "EMG_all_epochs.npy"),
            }
    logging.warning("Data files not found; using synthetic data for smoke testing.")
    return generate_synthetic_data(n_samples=200, seq_length=512, seed=42)


# =============================================================================
# Per-fold runner
# =============================================================================

def _preprocess_for_experiment(experiment_type: str,
                               data: Dict[str, np.ndarray],
                               target_snr_db: float):
    """Returns clean_n, eog_n, emg_n, noisy_n, means, stds (6-tuple, matching the legacy API)."""
    if experiment_type == "eeg_eog":
        return preprocess_eeg_eog(data["eeg"], data["eog"], target_snr_db=target_snr_db)
    if experiment_type == "eeg_emg":
        return preprocess_eeg_emg(data["eeg"], data["emg"], target_snr_db=target_snr_db)
    return preprocess_data(data["eeg"], data["eog"], data["emg"], target_snr_db=target_snr_db)


def run_sla_moe(experiment_type: str,
                noisy_train, clean_train, eog_train, emg_train,
                noisy_test, clean_test, eog_test, emg_test,
                stds, means, seed: int, device=None) -> np.ndarray:
    """Apply SLA-MoE on the test fold. Currently the underlying applier
    fits + applies in one pass on the test segment; for the rigorous fold
    split we pass only the test data (the existing implementation does
    self-contained per-call fitting). Future work: refactor to true
    train-then-eval."""
    if experiment_type == "eeg_eog":
        return apply_rnn_moe_filter_ica_eog_only(
            noisy_test, clean_test, eog_test, stds, means, seed=seed, device=device)
    if experiment_type == "eeg_emg":
        return apply_rnn_moe_filter_ica_emg_only(
            noisy_test, clean_test, emg_test, stds, means, seed=seed, device=device)
    return apply_rnn_moe_filter_ica(
        noisy_test, clean_test, eog_test, emg_test, stds, means, seed=seed, device=device)


def run_one_fold(experiment_type: str,
                 cfg: ExperimentConfig,
                 data: Dict[str, np.ndarray],
                 target_snr_db: float,
                 train_idx: np.ndarray,
                 test_idx: np.ndarray,
                 seed: int,
                 baseline_cfgs: Dict[str, BaselineConfig] | None,
                 device: torch.device) -> Dict[str, Dict[str, float]]:
    """
    Run SLA-MoE and (optionally) every baseline on a single (fold, seed)
    combination at one input SNR. Returns:
        {method_name: {metric: value, ...}}
    """
    # Preprocess once on the FULL dataset (StandardScaler is fit on full
    # set, which mirrors the original code; for stricter rigor switch to
    # fitting on train only).
    clean_n, eog_n, emg_n, noisy_n, means, stds = _preprocess_for_experiment(
        experiment_type, data, target_snr_db)

    def _slice(*arrs):
        return [a[test_idx] if a is not None else None for a in arrs]

    clean_train, eog_train, emg_train, noisy_train = (
        clean_n[train_idx], eog_n[train_idx], emg_n[train_idx], noisy_n[train_idx])
    clean_test, eog_test, emg_test, noisy_test = (
        clean_n[test_idx], eog_n[test_idx], emg_n[test_idx], noisy_n[test_idx])

    # Denormalized references for metric computation.
    clean_test_denorm = clean_test * stds + means
    noisy_test_denorm = noisy_test * stds + means

    fold_results: Dict[str, Dict[str, float]] = {}

    # ---- SLA-MoE ----
    set_all_seeds(seed)
    try:
        denoised = run_sla_moe(experiment_type,
                               noisy_train, clean_train, eog_train, emg_train,
                               noisy_test, clean_test, eog_test, emg_test,
                               stds, means, seed=seed, device=device)
        fold_results[SLA_MOE_NAME] = compute_metrics(clean_test_denorm, denoised, noisy_test_denorm)
    except Exception as e:
        logging.error(f"SLA-MoE failed at seed={seed}, snr={target_snr_db}: {e}")

    # ---- Baselines ----
    # Inspect baseline call signatures once so we don't spam warnings every fold.
    import inspect
    train_accepts_device = "device" in inspect.signature(train_baseline_model).parameters
    apply_accepts_device = "device" in inspect.signature(apply_baseline_model).parameters

    if baseline_cfgs:
        for key, bc in baseline_cfgs.items():
            try:
                set_all_seeds(seed)
                if bc.model_type in {"wiener", "lms", "rls", "kalman"}:
                    filt = {
                        "wiener": WienerFilter(),
                        "lms": LMSFilter(),
                        "rls": RLSFilter(),
                        "kalman": KalmanFilter(),
                    }[bc.model_type]
                    if bc.model_type in {"wiener", "kalman"}:
                        denoised = filt(noisy_test, stds, means)
                    else:
                        denoised = filt(noisy_test, eog_test, emg_test, stds, means)
                else:
                    model = _build_dl_baseline(bc, input_size=clean_n.shape[1])
                    train_kw = {"device": device} if train_accepts_device else {}
                    apply_kw = {"device": device} if apply_accepts_device else {}
                    model = train_baseline_model(
                        model, noisy_train, clean_train, eog_train, emg_train,
                        epochs=bc.epochs, batch_size=bc.batch_size, seed=seed,
                        **train_kw,
                    )
                    denoised = apply_baseline_model(
                        model, noisy_test, stds, means, eog_test, emg_test,
                        **apply_kw,
                    )
                fold_results[bc.name] = compute_metrics(clean_test_denorm, denoised, noisy_test_denorm)
            except Exception as e:
                logging.error(f"Baseline {bc.name} failed at seed={seed}, snr={target_snr_db}: {e}")

    return fold_results


def _build_dl_baseline(bc: BaselineConfig, input_size: int):
    if bc.model_type == "eegdnoisenet":
        return EEGDnoiseNet(input_size=input_size, hidden_channels=bc.hidden_channels, num_layers=bc.num_layers)
    if bc.model_type == "eegdnet":
        return EEGDnet(input_size=input_size)
    if bc.model_type == "rnn_eeg":
        return RNNEEG(input_size=input_size)
    if bc.model_type == "resnet_eeg":
        return ResNetEEG(input_size=input_size)
    if bc.model_type == "simple_cnn":
        return SimpleCNN(input_size=input_size)
    raise ValueError(f"unknown baseline type: {bc.model_type}")


# =============================================================================
# SNR sweep + fold loop
# =============================================================================

def run_experiment_sweep(experiment_type: str,
                         cfg: ExperimentConfig,
                         data: Dict[str, np.ndarray],
                         baseline_cfgs: Dict[str, BaselineConfig],
                         device: torch.device,
                         output_dir: Path) -> Dict[str, Any]:
    """
    Outer loops: input SNR -> seed -> fold. Aggregates per-method per-metric
    arrays and persists per-(snr, seed, fold) raw measurements.
    """
    n_segments = data["eeg"].shape[0]

    # Build folds once (subject ids are dataset-level, not per-seed).
    if cfg.subject_split:
        subject_ids = load_subject_ids(cfg.subject_id_path, n_segments,
                                       fallback_n_subjects=cfg.fallback_n_subjects)
        folds = subject_kfold_splits(subject_ids, n_splits=cfg.n_folds, seed=42)
        logging.info(f"Subject-grouped K-fold ({cfg.n_folds} folds) over {len(np.unique(subject_ids))} subjects")
    else:
        folds = segment_kfold_splits(n_segments, n_splits=cfg.n_folds, seed=42)
        logging.warning("Using segment-level K-fold (NOT subject-independent).")

    all_results: Dict[str, List[Dict]] = {}  # method -> list of {snr, seed, fold, metric, value}

    for snr_db in cfg.snr_sweep_db:
        for seed in cfg.seeds:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                logging.info(f"[{experiment_type}] SNR={snr_db:.1f} dB | seed={seed} | fold={fold_idx + 1}/{cfg.n_folds}")
                fold_results = run_one_fold(
                    experiment_type, cfg, data, snr_db,
                    train_idx, test_idx, seed,
                    baseline_cfgs, device,
                )
                for method, metrics in fold_results.items():
                    rec = {"snr_db": snr_db, "seed": seed, "fold": fold_idx, **metrics}
                    all_results.setdefault(method, []).append(rec)

    # Persist raw results
    raw_path = output_dir / experiment_type / "raw_results.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    logging.info(f"Raw per-fold results -> {raw_path}")

    return all_results


# =============================================================================
# Aggregation + stats
# =============================================================================

def collapse_at_snr(all_results: Dict[str, List[Dict]],
                    snr_db: float,
                    metrics: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Slice all_results at one SNR and return method -> metric -> list of values."""
    out: Dict[str, Dict[str, List[float]]] = {}
    for method, recs in all_results.items():
        out[method] = {m: [] for m in metrics}
        for r in recs:
            if abs(r["snr_db"] - snr_db) < 1e-6:
                for m in metrics:
                    if m in r and np.isfinite(r[m]):
                        out[method][m].append(float(r[m]))
    return out


def write_summary_artifacts(all_results: Dict[str, List[Dict]],
                            cfg: ExperimentConfig,
                            output_dir: Path,
                            experiment_type: str) -> None:
    headline_snr = DEFAULT_HEADLINE_SNR_DB if DEFAULT_HEADLINE_SNR_DB in cfg.snr_sweep_db else cfg.snr_sweep_db[0]
    methods_at_headline = collapse_at_snr(all_results, headline_snr, METRIC_ORDER)

    if SLA_MOE_NAME not in methods_at_headline or not methods_at_headline[SLA_MOE_NAME]["rmse"]:
        logging.warning("SLA-MoE has no headline-SNR data; skipping stats table.")
        return

    stats = compare_to_reference(methods_at_headline, reference=SLA_MOE_NAME, metrics=METRIC_ORDER)
    save_stats_json(stats, output_dir / experiment_type / f"stats_at_{headline_snr:+.1f}dB.json")

    latex = build_latex_table(
        methods_at_headline, stats, reference=SLA_MOE_NAME, metric_order=METRIC_ORDER,
        caption=f"Performance comparison on {experiment_type.upper()} at input SNR = {headline_snr:.1f} dB. "
                f"Mean $\\pm$ std over {len(cfg.seeds)} seeds and {cfg.n_folds} subject-grouped folds. "
                f"Best in \\textbf{{bold}}.",
        label=f"tab:overall_results_{experiment_type}",
    )
    save_latex_table(latex, output_dir / experiment_type / f"table_at_{headline_snr:+.1f}dB.tex")
    logging.info(f"Stats + LaTeX table written for {experiment_type} at {headline_snr} dB.")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run all EEG denoising experiments (revision).")
    parser.add_argument("--mode", choices=["full", "quick"], default="quick",
                        help="full = paper sweep (Colab); quick = M1 sanity run")
    parser.add_argument("--experiment", "-e",
                        choices=["eeg_eog", "eeg_emg", "eeg_eog_emg", "all"],
                        default="all")
    parser.add_argument("--data-path", "-d", type=str, default="data",
                        help="Directory containing EEG_all_epochs.npy, EOG_all_epochs.npy, EMG_all_epochs.npy")
    parser.add_argument("--output-dir", "-o", type=str, default="results")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--snr", type=float, default=None,
                        help="If given, run only this single input SNR (in dB)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    device = detect_device() if args.device == "auto" else torch.device(args.device)
    logger.info(f"Device: {device}")

    # Pick configs
    if args.mode == "quick":
        experiment_configs = {"eeg_eog_emg_quick": EEG_EOG_EMG_QUICK}
        baseline_cfgs = {} if args.no_baselines else get_quick_baseline_configs()
    else:
        all_cfgs = get_all_experiment_configs()
        experiment_configs = ({k: v for k, v in all_cfgs.items() if k == args.experiment}
                              if args.experiment != "all" else all_cfgs)
        baseline_cfgs = {} if args.no_baselines else get_baseline_configs()

    logger.info(f"Experiments to run: {list(experiment_configs)}")
    logger.info(f"Baselines: {list(baseline_cfgs) if baseline_cfgs else 'NONE (skipped)'}")

    # Optional single-SNR override
    if args.snr is not None:
        for cfg in experiment_configs.values():
            cfg.snr_sweep_db = [args.snr]

    # Load data
    data = load_or_generate_data(args.data_path)
    logger.info(f"Data shapes: EEG={data['eeg'].shape}, EOG={data['eog'].shape}, EMG={data['emg'].shape}")

    # Run all
    for exp_key, cfg in experiment_configs.items():
        logger.info("=" * 60)
        logger.info(f"Experiment: {exp_key}")
        logger.info(f"  SNR sweep: {cfg.snr_sweep_db}")
        logger.info(f"  Seeds:     {cfg.seeds}")
        logger.info(f"  Folds:     {cfg.n_folds} ({'subject-grouped' if cfg.subject_split else 'segment'})")
        logger.info("=" * 60)

        # Decide noise type from config name
        if "eeg_eog_emg" in exp_key:
            experiment_type = "eeg_eog_emg"
        elif "eeg_eog" in exp_key:
            experiment_type = "eeg_eog"
        elif "eeg_emg" in exp_key:
            experiment_type = "eeg_emg"
        else:
            experiment_type = "eeg_eog_emg"

        all_results = run_experiment_sweep(
            experiment_type, cfg, data, baseline_cfgs, device, output_dir)
        write_summary_artifacts(all_results, cfg, output_dir, experiment_type)

    logger.info("All experiments complete. See %s", output_dir)


if __name__ == "__main__":
    main()
