"""
Results visualization for EEG denoising experiments.

This module provides publication-quality figures for:
- Metrics comparison across methods
- Training convergence
- Expert utilization analysis
- Summary figures for papers
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300
})


def plot_metrics_comparison(results: Dict, metric: str = 'rmse',
                            title: str = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create bar plot comparing metrics across methods.

    Args:
        results: Dictionary with results from experiment runner
        metric: Metric to plot ('rmse', 'snr', 'correlation', etc.)
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['ICA-MoE']
    means = []
    stds = []

    # Get ICA-MoE results
    ica_moe_agg = results.get('ica_moe', {}).get('aggregated', {})
    if metric in ica_moe_agg:
        means.append(ica_moe_agg[metric]['mean'])
        stds.append(ica_moe_agg[metric]['std'])
    else:
        means.append(np.nan)
        stds.append(np.nan)

    # Get baseline results
    for name, baseline_results in results.get('baselines', {}).items():
        methods.append(name)
        agg = baseline_results.get('aggregated', {})
        if metric in agg:
            means.append(agg[metric]['mean'])
            stds.append(agg[metric]['std'])
        else:
            means.append(np.nan)
            stds.append(np.nan)

    # Create bar plot
    x = np.arange(len(methods))
    colors = ['#1f77b4'] + plt.cm.tab10(np.linspace(0.1, 0.9, len(methods) - 1)).tolist()

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)

    # Highlight best method
    valid_means = [m for m in means if not np.isnan(m)]
    if valid_means:
        if metric in ['rmse', 'rrmse', 'spectral_distortion']:
            best_idx = np.nanargmin(means)
        else:  # Higher is better for SNR, correlation
            best_idx = np.nanargmax(means)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel(metric.upper().replace('_', ' '))

    if title is None:
        title = f'{metric.upper()} Comparison Across Methods'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_convergence(training_history: Dict, title: str = "Training Convergence",
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training convergence curves.

    Args:
        training_history: Dictionary with training losses for each method
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Main loss plot
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(training_history)))

    for (name, history), color in zip(training_history.items(), colors):
        epochs = range(len(history['main_loss']))
        ax1.semilogy(epochs, history['main_loss'], label=name, color=color, linewidth=1.5)

        # Add confidence interval if available
        if 'main_loss_std' in history:
            loss = np.array(history['main_loss'])
            std = np.array(history['main_loss_std'])
            ax1.fill_between(epochs, loss - std, loss + std, alpha=0.2, color=color)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('(a) Main Reconstruction Loss', fontsize=11, fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Add phase annotations
    if any('pretrain_epochs' in h for h in training_history.values()):
        pretrain_epochs = 5  # Default
        ax1.axvline(x=pretrain_epochs, color='gray', linestyle='--', alpha=0.5)
        ax1.text(pretrain_epochs / 2, ax1.get_ylim()[1] * 0.5, 'Pre-training',
                ha='center', fontsize=8, alpha=0.7)
        ax1.text(pretrain_epochs + 2, ax1.get_ylim()[1] * 0.5, 'Main Training',
                ha='center', fontsize=8, alpha=0.7)

    # Load balancing loss plot
    ax2 = axes[1]
    for (name, history), color in zip(training_history.items(), colors):
        if 'lb_loss' in history:
            epochs = range(len(history['lb_loss']))
            ax2.plot(epochs, history['lb_loss'], label=name, color=color, linewidth=1.5)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Load Balancing Loss')
    ax2.set_title('(b) Expert Load Balancing', fontsize=11, fontweight='bold', loc='left')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_expert_utilization(gate_probs: np.ndarray, expert_names: List[str] = None,
                            title: str = "Expert Utilization",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot expert utilization over time/epochs.

    Args:
        gate_probs: Gate probabilities of shape (epochs, num_experts) or (time, num_experts)
        expert_names: Names for each expert
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    num_experts = gate_probs.shape[1]
    if expert_names is None:
        expert_names = [f'Expert {i}' for i in range(num_experts)]

    # Heatmap of utilization over time
    ax1 = axes[0]
    im = ax1.imshow(gate_probs.T * 100, aspect='auto', cmap='YlOrRd', vmin=0, vmax=50)
    ax1.set_xlabel('Time Step / Epoch')
    ax1.set_ylabel('Expert')
    ax1.set_yticks(range(num_experts))
    ax1.set_yticklabels(expert_names)
    ax1.set_title('(a) Expert Utilization Over Time', fontsize=11, fontweight='bold', loc='left')

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Utilization (%)')

    # Bar chart of average utilization
    ax2 = axes[1]
    avg_util = np.mean(gate_probs, axis=0) * 100
    std_util = np.std(gate_probs, axis=0) * 100

    colors = plt.cm.Set2(np.linspace(0, 1, num_experts))
    bars = ax2.barh(expert_names, avg_util, xerr=std_util, capsize=5, color=colors)

    ax2.set_xlabel('Average Utilization (%)')
    ax2.set_title('(b) Average Expert Utilization', fontsize=11, fontweight='bold', loc='left')
    ax2.set_xlim(0, 100)

    # Add percentage labels
    for bar, val in zip(bars, avg_util):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    # Add ideal line
    ideal = 100 / num_experts
    ax2.axvline(x=ideal, color='red', linestyle='--', alpha=0.7, label='Ideal (equal)')
    ax2.legend(loc='lower right', fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_multi_experiment_comparison(all_results: Dict,
                                     metrics: List[str] = ['rmse', 'snr', 'correlation'],
                                     title: str = "Multi-Experiment Comparison",
                                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare results across multiple experiments (EEG+EOG, EEG+EMG, EEG+EOG+EMG).

    Args:
        all_results: Dictionary with results from multiple experiments
        metrics: List of metrics to compare
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    experiments = list(all_results.keys())
    methods = ['ICA-MoE']

    # Get all unique methods from baselines
    for exp_results in all_results.values():
        for baseline_name in exp_results.get('baselines', {}).keys():
            if baseline_name not in methods:
                methods.append(baseline_name)

    x = np.arange(len(experiments))
    width = 0.8 / len(methods)

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]

        for method_idx, method in enumerate(methods):
            values = []
            errors = []

            for exp in experiments:
                exp_results = all_results[exp]

                if method == 'ICA-MoE':
                    agg = exp_results.get('ica_moe', {}).get('aggregated', {})
                else:
                    agg = exp_results.get('baselines', {}).get(method, {}).get('aggregated', {})

                if metric in agg:
                    values.append(agg[metric]['mean'])
                    errors.append(agg[metric]['std'])
                else:
                    values.append(np.nan)
                    errors.append(0)

            offset = (method_idx - len(methods) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=method, yerr=errors, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([exp.upper().replace('_', '+') for exp in experiments])
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        if ax_idx == 0:
            ax.legend(loc='upper left', fontsize=8, ncol=2)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_summary_figure(results: Dict, clean: np.ndarray, noisy: np.ndarray,
                          denoised: np.ndarray, fs: float = 256.0,
                          sample_idx: int = 0,
                          title: str = "EEG Denoising Summary",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive summary figure for publication.

    Args:
        results: Experiment results dictionary
        clean: Clean EEG signal
        noisy: Noisy EEG signal
        denoised: Denoised EEG signal
        fs: Sampling frequency
        sample_idx: Sample index to visualize
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Get signals
    clean_sig = clean[sample_idx] if clean.ndim > 1 else clean
    noisy_sig = noisy[sample_idx] if noisy.ndim > 1 else noisy
    denoised_sig = denoised[sample_idx] if denoised.ndim > 1 else denoised
    n = len(clean_sig)
    t = np.arange(n) / fs

    # Panel A: Signal comparison (spanning 2 columns)
    ax_sig = fig.add_subplot(gs[0, :2])
    ax_sig.plot(t, clean_sig, 'g-', label='Clean', linewidth=0.8, alpha=0.8)
    ax_sig.plot(t, noisy_sig, 'r-', label='Noisy', linewidth=0.5, alpha=0.5)
    ax_sig.plot(t, denoised_sig, 'b-', label='Denoised', linewidth=0.8)
    ax_sig.set_xlabel('Time (s)')
    ax_sig.set_ylabel('Amplitude (a.u.)')
    ax_sig.set_title('(a) Signal Comparison', fontsize=11, fontweight='bold', loc='left')
    ax_sig.legend(loc='upper right', fontsize=8)
    ax_sig.set_xlim(t[0], min(t[-1], 2.0))  # Show first 2 seconds

    # Panel B: Metrics bar chart
    ax_metrics = fig.add_subplot(gs[0, 2])
    metrics_to_plot = ['rmse', 'correlation']
    ica_moe_agg = results.get('ica_moe', {}).get('aggregated', {})

    metric_values = []
    metric_names = []
    for m in metrics_to_plot:
        if m in ica_moe_agg:
            metric_values.append(ica_moe_agg[m]['mean'])
            metric_names.append(m.upper())

    bars = ax_metrics.barh(metric_names, metric_values, color=['#1f77b4', '#2ca02c'])
    ax_metrics.set_xlabel('Value')
    ax_metrics.set_title('(b) ICA-MoE Metrics', fontsize=11, fontweight='bold', loc='left')

    for bar, val in zip(bars, metric_values):
        ax_metrics.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', va='center', fontsize=9)

    # Panel C: PSD comparison
    from scipy import signal as scipy_signal
    ax_psd = fig.add_subplot(gs[1, 0])
    f, psd_clean = scipy_signal.welch(clean_sig, fs=fs, nperseg=min(256, n//2))
    _, psd_noisy = scipy_signal.welch(noisy_sig, fs=fs, nperseg=min(256, n//2))
    _, psd_denoised = scipy_signal.welch(denoised_sig, fs=fs, nperseg=min(256, n//2))

    freq_mask = (f >= 0.5) & (f <= 50)
    ax_psd.semilogy(f[freq_mask], psd_clean[freq_mask], 'g-', label='Clean', linewidth=1.2)
    ax_psd.semilogy(f[freq_mask], psd_noisy[freq_mask], 'r--', label='Noisy', linewidth=0.8, alpha=0.7)
    ax_psd.semilogy(f[freq_mask], psd_denoised[freq_mask], 'b-', label='Denoised', linewidth=1.2)
    ax_psd.set_xlabel('Frequency (Hz)')
    ax_psd.set_ylabel('PSD (V²/Hz)')
    ax_psd.set_title('(c) Power Spectral Density', fontsize=11, fontweight='bold', loc='left')
    ax_psd.legend(loc='upper right', fontsize=8)
    ax_psd.grid(True, alpha=0.3)

    # Panel D: Spectrogram - Clean
    ax_spec1 = fig.add_subplot(gs[1, 1])
    nperseg = min(64, n // 4)
    f_spec, t_spec, Sxx_clean = scipy_signal.spectrogram(clean_sig, fs=fs, nperseg=nperseg)
    freq_mask_spec = (f_spec >= 0.5) & (f_spec <= 50)
    ax_spec1.pcolormesh(t_spec, f_spec[freq_mask_spec],
                        10 * np.log10(Sxx_clean[freq_mask_spec] + 1e-10),
                        shading='gouraud', cmap='viridis')
    ax_spec1.set_xlabel('Time (s)')
    ax_spec1.set_ylabel('Frequency (Hz)')
    ax_spec1.set_title('(d) Spectrogram - Clean', fontsize=11, fontweight='bold', loc='left')

    # Panel E: Spectrogram - Denoised
    ax_spec2 = fig.add_subplot(gs[1, 2])
    _, _, Sxx_denoised = scipy_signal.spectrogram(denoised_sig, fs=fs, nperseg=nperseg)
    im = ax_spec2.pcolormesh(t_spec, f_spec[freq_mask_spec],
                              10 * np.log10(Sxx_denoised[freq_mask_spec] + 1e-10),
                              shading='gouraud', cmap='viridis')
    ax_spec2.set_xlabel('Time (s)')
    ax_spec2.set_ylabel('Frequency (Hz)')
    ax_spec2.set_title('(e) Spectrogram - Denoised', fontsize=11, fontweight='bold', loc='left')

    # Panel F: Methods comparison
    ax_compare = fig.add_subplot(gs[2, :2])
    methods = ['ICA-MoE']
    rmse_values = []
    rmse_stds = []

    if 'rmse' in ica_moe_agg:
        rmse_values.append(ica_moe_agg['rmse']['mean'])
        rmse_stds.append(ica_moe_agg['rmse']['std'])

    for name, baseline_results in results.get('baselines', {}).items():
        methods.append(name)
        agg = baseline_results.get('aggregated', {})
        if 'rmse' in agg:
            rmse_values.append(agg['rmse']['mean'])
            rmse_stds.append(agg['rmse']['std'])

    x_pos = np.arange(len(methods))
    colors = ['#1f77b4'] + list(plt.cm.tab10(np.linspace(0.1, 0.9, len(methods) - 1)))

    bars = ax_compare.bar(x_pos, rmse_values, yerr=rmse_stds, capsize=5,
                          color=colors, edgecolor='black', linewidth=0.5)

    # Highlight best
    if rmse_values:
        best_idx = np.argmin(rmse_values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

    ax_compare.set_xticks(x_pos)
    ax_compare.set_xticklabels(methods, rotation=45, ha='right')
    ax_compare.set_ylabel('RMSE')
    ax_compare.set_title('(f) RMSE Comparison Across Methods', fontsize=11, fontweight='bold', loc='left')
    ax_compare.grid(axis='y', alpha=0.3)

    # Panel G: Summary statistics table
    ax_table = fig.add_subplot(gs[2, 2])
    ax_table.axis('off')

    # Create summary text
    if ica_moe_agg:
        summary_text = "ICA-MoE Performance Summary\n" + "="*30 + "\n\n"
        for metric, values in ica_moe_agg.items():
            if isinstance(values, dict) and 'mean' in values:
                summary_text += f"{metric.upper()}: {values['mean']:.4f} +/- {values['std']:.4f}\n"

        # Add SNR improvement
        noise_power = np.mean((noisy_sig - clean_sig) ** 2)
        residual_power = np.mean((denoised_sig - clean_sig) ** 2)
        snr_imp = 10 * np.log10(noise_power / (residual_power + 1e-10))
        summary_text += f"\nSNR Improvement: {snr_imp:.2f} dB"

        ax_table.text(0.1, 0.9, summary_text, transform=ax_table.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def generate_latex_table(results: Dict, caption: str = "Comparison of denoising methods",
                         label: str = "tab:results") -> str:
    """
    Generate LaTeX table code for results.

    Args:
        results: Experiment results dictionary
        caption: Table caption
        label: Table label for referencing

    Returns:
        LaTeX table code as string
    """
    metrics = ['rmse', 'snr', 'correlation']

    # Header
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{{caption}}}\n"
    latex += f"\\label{{{label}}}\n"
    latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
    latex += "\\toprule\n"
    latex += "Method & " + " & ".join([m.upper() for m in metrics]) + " \\\\\n"
    latex += "\\midrule\n"

    # ICA-MoE row
    ica_moe_agg = results.get('ica_moe', {}).get('aggregated', {})
    row = "ICA-MoE (Ours)"
    for metric in metrics:
        if metric in ica_moe_agg:
            mean = ica_moe_agg[metric]['mean']
            std = ica_moe_agg[metric]['std']
            row += f" & \\textbf{{{mean:.4f}}} $\\pm$ {std:.4f}"
        else:
            row += " & -"
    latex += row + " \\\\\n"

    # Baseline rows
    for name, baseline_results in results.get('baselines', {}).items():
        agg = baseline_results.get('aggregated', {})
        row = name
        for metric in metrics:
            if metric in agg:
                mean = agg[metric]['mean']
                std = agg[metric]['std']
                row += f" & {mean:.4f} $\\pm$ {std:.4f}"
            else:
                row += " & -"
        latex += row + " \\\\\n"

    # Footer
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex
