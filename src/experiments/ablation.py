"""
Ablation study runner for SLA-MoE.

Variants reported in the paper revision:
  A1) Full SLA-MoE                       (reference)
  A2) No ICA pre-training                (skip pretrain; gate on raw features only)
  A3) Pretraining without independence loss (lambda_independence = 0)
  A4) No ICA features at gating          (gate sees only raw signal)
  A5) Varying number of experts          E in {2, 4, 6, 8}
  A6) Varying top-k routing              k in {1, 2, 3}

Note on coverage: the underlying applier (`apply_rnn_moe_filter_ica`) is a
self-contained per-call routine. To make the ablations meaningful without a
deep refactor of that file, this runner accepts a dict of "ablation knobs"
that are forwarded as monkey-patches via environment variables read inside
the applier (we patch the applier's defaults via attributes set on the
function). If you have already refactored the applier to take a config
object, you can switch to that path by setting USE_REFACTORED_APPLIER=True.

This runner does NOT do the full SNR sweep by default; it runs at the
headline SNR only (default -3 dB) to keep cost tractable on a single GPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.experiment_configs import (
    DEFAULT_HEADLINE_SNR_DB, EEG_EOG_EMG_CONFIG, ExperimentConfig,
)
from src.utils.data_utils import (
    load_subject_ids, preprocess_data, subject_kfold_splits,
)
from src.utils.metrics import compute_metrics
from src.utils.seed_utils import set_all_seeds
from src.utils.stats import (
    build_latex_table, compare_to_reference, save_latex_table, save_stats_json,
)


# Knob keys consumed by the SLA-MoE applier (see ica_moe_original.py).
# These names should match attributes on the applier function or the model
# config object you wire up after refactoring.
ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "Full SLA-MoE":                {},                                       # reference
    "No ICA pretraining":          {"pretrain_epochs": 0},
    "No independence loss":        {"lambda_independence": 0.0},
    "No ICA features at gate":     {"use_ica_features_in_gate": False},
    "E=2":                         {"num_experts": 2},
    "E=6":                         {"num_experts": 6},
    "E=8":                         {"num_experts": 8},
    "top_k=1":                     {"top_k": 1},
    "top_k=3":                     {"top_k": 3},
}


METRIC_ORDER = ["rmse", "rrmse", "correlation", "snr", "snr_improvement", "spectral_distortion"]


def _apply_with_overrides(experiment_type: str,
                          noisy, clean, eog, emg, stds, means,
                          seed: int, overrides: Dict[str, Any]):
    """
    Apply SLA-MoE with knob overrides. We call the existing applier function
    after temporarily patching module-level attributes that the applier
    reads. After the call we restore the originals.

    NOTE: this is a pragmatic shim until the applier is refactored to take a
    config dataclass directly. If you already refactored, replace the body of
    this function with: `model = SLAMoE(**overrides); ... ; return model.predict(...)`.
    """
    from src.models import ica_moe_original as M

    saved = {}
    for k, v in overrides.items():
        if hasattr(M, k):
            saved[k] = getattr(M, k)
            setattr(M, k, v)
    try:
        if experiment_type == "eeg_eog":
            return M.apply_rnn_moe_filter_ica_eog_only(
                noisy, clean, eog, stds, means, seed=seed)
        if experiment_type == "eeg_emg":
            return M.apply_rnn_moe_filter_ica_emg_only(
                noisy, clean, emg, stds, means, seed=seed)
        return M.apply_rnn_moe_filter_ica(
            noisy, clean, eog, emg, stds, means, seed=seed)
    finally:
        for k, v in saved.items():
            setattr(M, k, v)


def run_ablation(data: Dict[str, np.ndarray],
                 cfg: ExperimentConfig,
                 output_dir: Path,
                 experiment_type: str = "eeg_eog_emg",
                 snr_db: float = DEFAULT_HEADLINE_SNR_DB) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subject-grouped folds shared across variants.
    n = data["eeg"].shape[0]
    subject_ids = load_subject_ids(cfg.subject_id_path, n, cfg.fallback_n_subjects)
    folds = subject_kfold_splits(subject_ids, n_splits=cfg.n_folds, seed=42)

    # Preprocess once at the headline SNR.
    clean_n, eog_n, emg_n, noisy_n, means, stds = preprocess_data(
        data["eeg"], data["eog"], data["emg"], target_snr_db=snr_db)

    results: Dict[str, List[Dict[str, float]]] = {v: [] for v in ABLATION_VARIANTS}

    for variant, overrides in ABLATION_VARIANTS.items():
        logging.info("=" * 60)
        logging.info(f"Variant: {variant}  overrides={overrides}")
        logging.info("=" * 60)
        for seed in cfg.seeds:
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                set_all_seeds(seed)
                clean_test = clean_n[test_idx]
                noisy_test = noisy_n[test_idx]
                eog_test = eog_n[test_idx]
                emg_test = emg_n[test_idx]
                clean_test_dn = clean_test * stds + means
                noisy_test_dn = noisy_test * stds + means
                try:
                    denoised = _apply_with_overrides(
                        experiment_type, noisy_test, clean_test, eog_test, emg_test,
                        stds, means, seed=seed, overrides=overrides,
                    )
                    metrics = compute_metrics(clean_test_dn, denoised, noisy_test_dn)
                    metrics.update({"seed": seed, "fold": fold_idx, "snr_db": snr_db})
                    results[variant].append(metrics)
                    logging.info(f"  seed={seed} fold={fold_idx} RMSE={metrics['rmse']:.3f} "
                                 f"SNR={metrics['snr']:.2f} dB")
                except Exception as e:
                    logging.error(f"  variant={variant} seed={seed} fold={fold_idx}: {e}")

    # Persist raw + summary
    raw_path = output_dir / "ablation_raw.json"
    raw_path.write_text(json.dumps(results, indent=2,
                                   default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))

    # Build per-variant per-metric arrays for stats vs Full SLA-MoE.
    per_variant: Dict[str, Dict[str, List[float]]] = {}
    for v, runs in results.items():
        per_variant[v] = {m: [r[m] for r in runs if m in r and np.isfinite(r[m])]
                          for m in METRIC_ORDER}

    if "Full SLA-MoE" in per_variant and per_variant["Full SLA-MoE"]["rmse"]:
        stats = compare_to_reference(per_variant, reference="Full SLA-MoE", metrics=METRIC_ORDER)
        save_stats_json(stats, output_dir / "ablation_stats.json")

        latex = build_latex_table(
            per_variant, stats, reference="Full SLA-MoE", metric_order=METRIC_ORDER,
            caption=f"Ablation study at input SNR = {snr_db:.1f} dB on the EEG+EOG+EMG task. "
                    "Mean $\\pm$ std over seeds and subject-grouped folds. "
                    "$\\dagger$ = Holm-significant difference vs full SLA-MoE.",
            label="tab:ablation",
        )
        save_latex_table(latex, output_dir / "ablation_table.tex")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data")
    p.add_argument("--output-dir", default="results/ablation")
    p.add_argument("--snr", type=float, default=DEFAULT_HEADLINE_SNR_DB)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from run_all_experiments import load_or_generate_data
    data = load_or_generate_data(args.data_path)
    cfg = deepcopy(EEG_EOG_EMG_CONFIG)
    run_ablation(data, cfg, Path(args.output_dir), snr_db=args.snr)


if __name__ == "__main__":
    main()
