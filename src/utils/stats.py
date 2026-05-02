"""
Statistical comparison utilities for EEG-denoising experiments.

Adds the missing pieces that JBHI's reviewer flagged:
  * paired Wilcoxon signed-rank tests across seeds (and folds)
  * Holm-Bonferroni multiple-comparison correction
  * effect sizes (rank-biserial r, Cohen's d_z)
  * LaTeX-ready table builder mirroring Table II of the paper

Usage:
    from src.utils.stats import compare_to_reference, build_latex_table

    # methods is a dict: name -> dict of metric -> array of per-seed values
    stats = compare_to_reference(methods, reference="SLA-MoE")
    latex = build_latex_table(methods, stats, metric_order=[...])

The stats output is also written as JSON next to the per-method results so
the paper's revision can cite p-values directly.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


# Lower-is-better metrics; higher-is-better otherwise.
LOWER_IS_BETTER = {"rmse", "rrmse", "spectral_distortion"}


# =============================================================================
# Effect sizes
# =============================================================================

def cohen_dz(diff: np.ndarray) -> float:
    """Within-subject Cohen's d_z = mean(diff)/std(diff)."""
    diff = np.asarray(diff, dtype=float)
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd < 1e-12:
        return float("inf") if abs(diff.mean()) > 0 else 0.0
    return float(diff.mean() / sd)


def rank_biserial(diff: np.ndarray) -> float:
    """
    Rank-biserial correlation as the standard Wilcoxon effect size.
    r = (sum_pos_ranks - sum_neg_ranks) / sum_all_ranks  in [-1, 1].
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    abs_ranks = np.argsort(np.argsort(np.abs(diff))) + 1
    pos = abs_ranks[diff > 0].sum()
    neg = abs_ranks[diff < 0].sum()
    total = abs_ranks.sum()
    return float((pos - neg) / total)


# =============================================================================
# Paired Wilcoxon comparison
# =============================================================================

def paired_wilcoxon(reference: Sequence[float],
                    other: Sequence[float],
                    metric: str) -> Dict[str, float]:
    """
    Paired Wilcoxon signed-rank test of `other` vs `reference` for one metric.

    Differences are computed as `reference - other` and signed so that a
    POSITIVE difference always means *reference is better*. Direction is
    inferred from LOWER_IS_BETTER.

    Returns a dict:
        diff_mean, diff_median, p, statistic,
        cohen_dz, rank_biserial, n
    """
    ref = np.asarray(reference, dtype=float)
    oth = np.asarray(other, dtype=float)
    if ref.shape != oth.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {oth.shape}")

    if metric in LOWER_IS_BETTER:
        # ref better iff ref < oth, i.e. (oth - ref) > 0
        diff = oth - ref
    else:
        diff = ref - oth

    # Drop NaNs/Infs
    mask = np.isfinite(diff)
    diff = diff[mask]
    n = diff.size
    if n < 2 or np.all(diff == 0):
        return {"n": n, "p": float("nan"), "statistic": float("nan"),
                "diff_mean": float(diff.mean()) if n else float("nan"),
                "diff_median": float(np.median(diff)) if n else float("nan"),
                "cohen_dz": float("nan"), "rank_biserial": float("nan")}

    try:
        stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = float("nan"), float("nan")

    return {
        "n": int(n),
        "p": float(p),
        "statistic": float(stat),
        "diff_mean": float(diff.mean()),
        "diff_median": float(np.median(diff)),
        "cohen_dz": cohen_dz(diff),
        "rank_biserial": rank_biserial(diff),
    }


# =============================================================================
# Multiple-comparison correction
# =============================================================================

def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Holm-Bonferroni correction. Returns (adjusted_p, reject_at_alpha).
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        adj_p = min(p[idx] * factor, 1.0)
        running_max = max(running_max, adj_p)
        adj[idx] = running_max
    reject = adj < alpha
    return adj.tolist(), reject.tolist()


# =============================================================================
# Top-level: compare every method to a reference
# =============================================================================

def compare_to_reference(methods: Mapping[str, Mapping[str, Sequence[float]]],
                         reference: str,
                         metrics: Optional[Sequence[str]] = None,
                         alpha: float = 0.05) -> Dict[str, Dict[str, Dict]]:
    """
    For each non-reference method, run paired Wilcoxon vs the reference for
    each metric, and apply Holm-Bonferroni across methods (within each metric).

    Args:
        methods: {method_name: {metric: per-seed/fold values}}
        reference: name of the reference method (your proposed SLA-MoE)
        metrics: subset of metrics to test (default: keys of reference)
        alpha: family-wise error rate

    Returns:
        {method: {metric: {p, p_holm, reject_holm, ...}}}
    """
    if reference not in methods:
        raise KeyError(f"reference '{reference}' not in methods")
    ref_data = methods[reference]
    if metrics is None:
        metrics = list(ref_data.keys())

    others = [m for m in methods if m != reference]
    out: Dict[str, Dict[str, Dict]] = {m: {} for m in others}

    for metric in metrics:
        if metric not in ref_data:
            continue
        per_method = {}
        for m in others:
            if metric not in methods[m]:
                continue
            per_method[m] = paired_wilcoxon(ref_data[metric], methods[m][metric], metric)

        # Holm-Bonferroni across methods, within this metric.
        names = list(per_method.keys())
        ps = [per_method[n]["p"] for n in names]
        finite_mask = [math.isfinite(p) for p in ps]
        if any(finite_mask):
            adj, reject = holm_bonferroni([p if math.isfinite(p) else 1.0 for p in ps], alpha=alpha)
            for n, a, r, fm in zip(names, adj, reject, finite_mask):
                per_method[n]["p_holm"] = float(a) if fm else float("nan")
                per_method[n]["reject_holm"] = bool(r) if fm else False
        else:
            for n in names:
                per_method[n]["p_holm"] = float("nan")
                per_method[n]["reject_holm"] = False

        for n, d in per_method.items():
            out[n][metric] = d

    return out


# =============================================================================
# LaTeX builder
# =============================================================================

def _format_cell(values: Sequence[float], best: bool, sig_marker: str = "") -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "--"
    mean = arr.mean()
    std = arr.std(ddof=1) if arr.size > 1 else 0.0
    body = f"{mean:.3f} $\\pm$ {std:.3f}{sig_marker}"
    return f"\\textbf{{{body}}}" if best else body


def build_latex_table(methods: Mapping[str, Mapping[str, Sequence[float]]],
                      stats: Mapping[str, Mapping[str, Mapping]],
                      reference: str,
                      metric_order: Sequence[str],
                      caption: str = "Performance comparison.",
                      label: str = "tab:overall_results") -> str:
    """
    Build a LaTeX `tabular` mirroring Table II in the manuscript:
      method | metric_1 | metric_2 | ...
    Bold = winner per column. A trailing \\dagger marks Holm-significant
    differences vs the reference.
    """
    # Determine winners per metric
    winners = {}
    for metric in metric_order:
        candidates = []
        for m in methods:
            arr = np.asarray(methods[m].get(metric, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                candidates.append((m, arr.mean()))
        if not candidates:
            winners[metric] = None
            continue
        if metric in LOWER_IS_BETTER:
            winners[metric] = min(candidates, key=lambda x: x[1])[0]
        else:
            winners[metric] = max(candidates, key=lambda x: x[1])[0]

    cols = "l" + "c" * len(metric_order)
    lines = [
        "\\begin{table*}[htbp]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
        "Method & " + " & ".join(_metric_header(m) for m in metric_order) + " \\\\",
        "\\midrule",
    ]
    for m in methods:
        cells = [m]
        for metric in metric_order:
            sig = ""
            if m != reference:
                d = stats.get(m, {}).get(metric, {})
                if d.get("reject_holm", False):
                    sig = "$^{\\dagger}$"
            cells.append(_format_cell(methods[m].get(metric, []), winners[metric] == m, sig))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\bottomrule", "\\end{tabular}",
        f"\\caption*{{$\\dagger$: Holm-Bonferroni-significant difference vs {reference} (paired Wilcoxon, $\\alpha=0.05$).}}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def _metric_header(metric: str) -> str:
    pretty = {
        "rmse": "RMSE $\\downarrow$",
        "rrmse": "rRMSE $\\downarrow$",
        "correlation": "Corr $\\uparrow$",
        "snr": "SNR (dB) $\\uparrow$",
        "snr_improvement": "$\\Delta$SNR (dB) $\\uparrow$",
        "spectral_distortion": "Spectral Dist. $\\downarrow$",
    }
    return pretty.get(metric, metric)


# =============================================================================
# Convenience persistence
# =============================================================================

def save_stats_json(stats: Mapping, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)


def save_latex_table(latex: str, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
