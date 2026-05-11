"""
Regenerate Figures 2, 3, and 4 from the paper, reading the JSON outputs of
the AutoDL/Colab runs.

Outputs three PNG files into the directory specified by --output-dir
(default: ../paper/figures relative to this file's parent):
  - fig2_method_comparison.png  (replaces the old qualitative-waveform Figure 2)
  - fig3_ablation.png           (replaces the old residual-error Figure 3)
  - fig4_downstream_bci.png     (replaces the old training-dynamics Figure 4)

Usage:
    python -m src.visualization.regenerate_paper_figures \
        --headline-json results/extracted/results_headline/eeg_eog_emg/raw_results.json \
        --ablation-json results/extracted/results_ablation/ablation_raw.json \
        --bci-json      results/extracted/results_downstream_bci/summary.json \
        --output-dir    paper/figures
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Visual style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REF_COLOR = "#1f77b4"      # SLA-MoE / Full / "good" reference
DL_COLOR = "#ff7f0e"
TRAD_COLOR = "#7f7f7f"
ABL_COLOR = "#2ca02c"
ABL_BAD = "#d62728"


# =============================================================================
# Figure 2 -- Method comparison on the headline EEG+EOG+EMG task at -3 dB
# =============================================================================

def _agg(records: List[dict], metric: str) -> tuple:
    vals = np.array([r[metric] for r in records if metric in r and np.isfinite(r[metric])])
    return vals.mean(), vals.std(ddof=1) if vals.size > 1 else 0.0


def figure2_method_comparison(headline: Dict[str, List[dict]], out: Path):
    methods = ["SLA-MoE", "EEGDnet", "RNN_EEG", "EEGDnoiseNet", "ResNet_EEG",
               "SimpleCNN", "RLSFilter", "LMSFilter", "KalmanFilter"]
    methods = [m for m in methods if m in headline]
    metrics = [
        ("snr_improvement", r"$\Delta$SNR (dB)", "higher is better", False),
        ("rmse",            "RMSE",              "lower is better",  True),
        ("correlation",     "Correlation",       "higher is better", False),
        ("spectral_distortion", "Spectral Distortion", "lower is better", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), constrained_layout=True)
    for ax, (key, label, direction, lower_better) in zip(axes, metrics):
        means, stds = zip(*[_agg(headline[m], key) for m in methods])
        means, stds = np.array(means), np.array(stds)
        colors = []
        for m in methods:
            if m == "SLA-MoE":
                colors.append(REF_COLOR)
            elif m in {"LMSFilter", "RLSFilter", "KalmanFilter"}:
                colors.append(TRAD_COLOR)
            else:
                colors.append(DL_COLOR)
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
               linewidth=0.5, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("Filter", "").replace("_", " ")
                            for m in methods], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"{label}  ({direction})", fontsize=10)
        # mark winner
        winner_i = int(np.argmin(means)) if lower_better else int(np.argmax(means))
        ax.bar(x[winner_i], means[winner_i], color=colors[winner_i],
               edgecolor="black", linewidth=2, zorder=3)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Headline comparison on EEG+EOG+EMG at input SNR = -3 dB "
                 "(mean $\\pm$ std over 5 seeds $\\times$ 5 subject-grouped folds; "
                 "thick outline = best per metric)", y=1.06, fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# =============================================================================
# Figure 3 -- Ablation
# =============================================================================

def figure3_ablation(ablation: Dict[str, List[dict]], out: Path):
    order = [
        "Full SLA-MoE", "No ICA pretraining", "No independence loss",
        "No ICA features at gate", "E=2", "E=6", "E=8", "top_k=1", "top_k=3",
    ]
    order = [v for v in order if v in ablation]
    full_mean, _ = _agg(ablation["Full SLA-MoE"], "snr")

    means_snr, stds_snr = [], []
    means_rmse, stds_rmse = [], []
    for v in order:
        m, s = _agg(ablation[v], "snr"); means_snr.append(m); stds_snr.append(s)
        m, s = _agg(ablation[v], "rmse"); means_rmse.append(m); stds_rmse.append(s)
    means_snr = np.array(means_snr); stds_snr = np.array(stds_snr)
    means_rmse = np.array(means_rmse); stds_rmse = np.array(stds_rmse)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.0), constrained_layout=True)

    # SNR bars: red if hurts vs Full, green if helps, grey if = Full
    colors_snr = []
    for v, m in zip(order, means_snr):
        if v == "Full SLA-MoE":
            colors_snr.append(REF_COLOR)
        elif m < full_mean - 0.3:
            colors_snr.append(ABL_BAD)
        elif m > full_mean + 0.3:
            colors_snr.append(ABL_COLOR)
        else:
            colors_snr.append(TRAD_COLOR)
    x = np.arange(len(order))
    ax1.bar(x, means_snr, yerr=stds_snr, color=colors_snr, edgecolor="black",
            linewidth=0.5, capsize=3)
    ax1.axhline(full_mean, color=REF_COLOR, linestyle="--", linewidth=1, alpha=0.7,
                label=f"Full SLA-MoE = {full_mean:.2f} dB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Output SNR (dB)")
    ax1.set_title("Ablation: SNR vs. Full SLA-MoE", fontsize=10)
    ax1.set_ylim(bottom=min(means_snr) - 1.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=9, loc="lower right")

    # RMSE bars (lower is better -- flip color logic)
    full_mean_rmse, _ = _agg(ablation["Full SLA-MoE"], "rmse")
    colors_rmse = []
    for v, m in zip(order, means_rmse):
        if v == "Full SLA-MoE":
            colors_rmse.append(REF_COLOR)
        elif m > full_mean_rmse + 2:
            colors_rmse.append(ABL_BAD)
        elif m < full_mean_rmse - 2:
            colors_rmse.append(ABL_COLOR)
        else:
            colors_rmse.append(TRAD_COLOR)
    ax2.bar(x, means_rmse, yerr=stds_rmse, color=colors_rmse, edgecolor="black",
            linewidth=0.5, capsize=3)
    ax2.axhline(full_mean_rmse, color=REF_COLOR, linestyle="--", linewidth=1, alpha=0.7,
                label=f"Full SLA-MoE = {full_mean_rmse:.1f}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("RMSE")
    ax2.set_title("Ablation: RMSE vs. Full SLA-MoE", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle("Ablation study at input SNR = -3 dB on EEG+EOG+EMG  "
                 "(red = significantly worse than Full, green = better, grey = no significant change)",
                 y=1.03, fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# =============================================================================
# Figure 4 -- Downstream BCI
# =============================================================================

def figure4_downstream_bci(bci: Dict[str, dict], out: Path):
    pipelines = ["raw_clean", "noisy_no_denoise", "noisy_SLA-MoE"]
    pretty = ["Clean EEG\n(upper bound)", "Contaminated\n(no denoising)", "Contaminated\n+ SLA-MoE"]
    acc_mean = [bci[p]["accuracy_mean"] * 100 for p in pipelines]
    acc_std  = [bci[p]["accuracy_std"]  * 100 for p in pipelines]
    kap_mean = [bci[p]["kappa_mean"] for p in pipelines]
    kap_std  = [bci[p]["kappa_std"]  for p in pipelines]
    colors = ["#2ca02c", "#7f7f7f", REF_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    x = np.arange(len(pipelines))
    ax1.bar(x, acc_mean, yerr=acc_std, color=colors, edgecolor="black",
            linewidth=0.5, capsize=4)
    ax1.axhline(25, linestyle=":", color="black", alpha=0.5, label="chance (25%)")
    ax1.set_xticks(x); ax1.set_xticklabels(pretty, fontsize=9)
    ax1.set_ylabel("4-class Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Motor-imagery classification accuracy", fontsize=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=9, loc="upper right")
    for xi, m in zip(x, acc_mean):
        ax1.text(xi, m + 2, f"{m:.1f}%", ha="center", fontsize=9)

    ax2.bar(x, kap_mean, yerr=kap_std, color=colors, edgecolor="black",
            linewidth=0.5, capsize=4)
    ax2.axhline(0.0, linestyle=":", color="black", alpha=0.5, label="chance ($\\kappa$=0)")
    ax2.set_xticks(x); ax2.set_xticklabels(pretty, fontsize=9)
    ax2.set_ylabel("Cohen's $\\kappa$")
    ax2.set_ylim(-0.1, 1.0)
    ax2.set_title("Inter-rater agreement", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9, loc="upper right")
    for xi, m in zip(x, kap_mean):
        ax2.text(xi, m + 0.03, f"{m:.2f}", ha="center", fontsize=9)

    fig.suptitle("Zero-shot transfer to BCI Competition IV-2a (9 subjects, 5 seeds; "
                 "input SNR = -3 dB on contaminated pipelines)",
                 y=1.04, fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--headline-json", required=True)
    p.add_argument("--ablation-json", required=True)
    p.add_argument("--bci-json", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    headline = json.loads(Path(args.headline_json).read_text())
    ablation = json.loads(Path(args.ablation_json).read_text())
    bci      = json.loads(Path(args.bci_json).read_text())

    figure2_method_comparison(headline, out_dir / "fig2_method_comparison.png")
    figure3_ablation(ablation,         out_dir / "fig3_ablation.png")
    figure4_downstream_bci(bci,        out_dir / "fig4_downstream_bci.png")


if __name__ == "__main__":
    main()
