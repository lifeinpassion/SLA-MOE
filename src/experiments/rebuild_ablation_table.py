"""
Recompute the ablation stats + LaTeX table from ablation_raw.json without
re-running training. Used when ablation.py finished training but crashed
on the post-processing stats step.

Usage:
    python -m src.experiments.rebuild_ablation_table \
        --raw results_ablation/ablation_raw.json \
        --output-dir results_ablation \
        --snr -3.0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.stats import (
    build_latex_table, compare_to_reference, save_latex_table, save_stats_json,
)

METRIC_ORDER = ["rmse", "rrmse", "correlation", "snr", "snr_improvement", "spectral_distortion"]
REFERENCE = "Full SLA-MoE"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="Path to ablation_raw.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--snr", type=float, default=-3.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.raw) as f:
        results: Dict[str, List[Dict]] = json.load(f)

    # Build per-variant key->record maps once.
    by_variant_key = {
        v: {(r.get("seed"), r.get("fold")): r for r in runs}
        for v, runs in results.items()
    }

    # Step 1: find the (seed, fold) intersection across all variants.
    all_keys = None
    for v, by_key in by_variant_key.items():
        keys = set(by_key.keys())
        all_keys = keys if all_keys is None else (all_keys & keys)
    if all_keys is None:
        all_keys = set()
    logging.info(f"(seed, fold) intersection across {len(results)} variants: "
                 f"{len(all_keys)} keys.")
    for v, by_key in by_variant_key.items():
        logging.info(f"  {v}: raw n={len(by_key)}")

    # Step 2: PER METRIC, restrict to keys where EVERY variant has a finite
    # value. This guarantees aligned arrays for every paired Wilcoxon. Keys
    # with NaN/Inf metrics in any variant are dropped for that metric only.
    per_variant: Dict[str, Dict[str, List[float]]] = {v: {} for v in by_variant_key}
    for m in METRIC_ORDER:
        good_keys = sorted(
            k for k in all_keys
            if all(
                m in by_variant_key[v].get(k, {})
                and np.isfinite(by_variant_key[v][k][m])
                for v in by_variant_key
            )
        )
        if not good_keys:
            logging.warning(f"  metric {m}: no aligned keys with all-finite values; skipping.")
            for v in by_variant_key:
                per_variant[v][m] = []
            continue
        n_drop = len(all_keys) - len(good_keys)
        if n_drop:
            logging.info(f"  metric {m}: keeping {len(good_keys)}/{len(all_keys)} keys "
                         f"({n_drop} dropped due to non-finite values in some variant).")
        for v, by_key in by_variant_key.items():
            per_variant[v][m] = [by_key[k][m] for k in good_keys]

    if REFERENCE not in per_variant or not per_variant[REFERENCE]["rmse"]:
        raise SystemExit(f"No reference data for '{REFERENCE}'.")

    stats = compare_to_reference(per_variant, reference=REFERENCE, metrics=METRIC_ORDER)
    out = Path(args.output_dir)
    save_stats_json(stats, out / "ablation_stats.json")

    latex = build_latex_table(
        per_variant, stats, reference=REFERENCE, metric_order=METRIC_ORDER,
        caption=f"Ablation study at input SNR = {args.snr:.1f} dB on the EEG+EOG+EMG task. "
                f"Mean $\\pm$ std over {len(sorted_keys)} (seed, fold) combinations. "
                "$\\dagger$ = Holm-significant difference vs full SLA-MoE.",
        label="tab:ablation",
    )
    save_latex_table(latex, out / "ablation_table.tex")
    logging.info(f"Wrote {out / 'ablation_table.tex'} and {out / 'ablation_stats.json'}.")


if __name__ == "__main__":
    main()
