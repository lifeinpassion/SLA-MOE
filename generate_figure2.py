#!/usr/bin/env python3
"""
Generate Figure 2: Qualitative comparison of denoising results.

This script produces a publication-quality figure showing:
- (a) Contaminated input signal
- (b) Ground-truth clean signal
- (c) SimpleCNN output
- (d) EEGDnoiseNet output
- (e) RNN-EEG output
- (f) ICA-MoE output (ours)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.baselines import (
    SimpleCNN, EEGDnoiseNet, RNNEEG,
    train_baseline_model, apply_baseline_model
)
from src.models.moe_models import ICAMoEFilter, train_ica_moe
from src.utils.data_utils import generate_synthetic_data, preprocess_data
from src.utils.seed_utils import set_all_seeds

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 0.8,
})


def compute_metrics(clean: np.ndarray, denoised: np.ndarray) -> dict:
    """Compute RMSE, correlation, and SNR."""
    rmse = np.sqrt(np.mean((denoised - clean) ** 2))
    corr = np.corrcoef(clean.flatten(), denoised.flatten())[0, 1]
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum((denoised - clean) ** 2) + 1e-10
    snr = 10 * np.log10(signal_power / noise_power)
    return {'rmse': rmse, 'corr': corr, 'snr': snr}


def generate_figure2(
    sample_idx: int = 0,
    time_range: tuple = (0.0, 2.0),
    save_path: str = 'paper/media/fig2_visualization.png',
    seed: int = 42
):
    """
    Generate Figure 2: Qualitative denoising comparison.

    Args:
        sample_idx: Index of sample to visualize
        time_range: Time range to display (start, end) in seconds
        save_path: Path to save the figure
        seed: Random seed for reproducibility
    """
    set_all_seeds(seed)

    print("=" * 60)
    print("Generating Figure 2: Denoising Methods Comparison")
    print("=" * 60)

    # Generate or load data
    print("\n[1/5] Generating synthetic data...")
    n_samples = 500
    seq_length = 512
    fs = 256.0  # Sampling frequency

    data = generate_synthetic_data(n_samples=n_samples, seq_length=seq_length, seed=seed)
    eeg_data = data['eeg']
    eog_data = data['eog']
    emg_data = data['emg']

    # Preprocess
    eeg_norm, eog_norm, emg_norm, noisy_eeg, eeg_means, eeg_stds = preprocess_data(
        eeg_data, eog_data, emg_data, noise_scale=0.01
    )

    # Reshape for proper broadcasting (1, timepoints) for element-wise operations
    # StandardScaler returns (n_features,) = (512,) shaped arrays
    eeg_means = eeg_means.reshape(1, -1)
    eeg_stds = eeg_stds.reshape(1, -1)

    print(f"   Data shape: {noisy_eeg.shape}")
    print(f"   Mean shape: {eeg_means.shape}, Std shape: {eeg_stds.shape}")
    print(f"   Sample index: {sample_idx}")

    # Train SimpleCNN
    print("\n[2/5] Training SimpleCNN...")
    simple_cnn = SimpleCNN(input_size=seq_length)
    simple_cnn = train_baseline_model(
        simple_cnn, noisy_eeg, eeg_norm,
        epochs=30, batch_size=32, lr=0.001, seed=seed
    )
    # Get normalized output and denormalize manually
    simple_cnn.eval()
    with torch.no_grad():
        simple_cnn_norm = simple_cnn(torch.FloatTensor(noisy_eeg)).numpy()
    simple_cnn_output = simple_cnn_norm * eeg_stds + eeg_means

    # Train EEGDnoiseNet
    print("\n[3/5] Training EEGDnoiseNet...")
    eegdnoisenet = EEGDnoiseNet(input_size=seq_length, num_layers=10)
    eegdnoisenet = train_baseline_model(
        eegdnoisenet, noisy_eeg, eeg_norm,
        epochs=30, batch_size=32, lr=0.001, seed=seed
    )
    eegdnoisenet.eval()
    with torch.no_grad():
        eegdnoisenet_norm = eegdnoisenet(torch.FloatTensor(noisy_eeg)).numpy()
    eegdnoisenet_output = eegdnoisenet_norm * eeg_stds + eeg_means

    # Train RNN-EEG
    print("\n[4/5] Training RNN-EEG...")
    rnn_eeg = RNNEEG(input_size=seq_length, hidden_size=128, num_layers=2)
    rnn_eeg = train_baseline_model(
        rnn_eeg, noisy_eeg, eeg_norm, eog=eog_norm, emg=emg_norm,
        epochs=30, batch_size=32, lr=0.001, seed=seed
    )
    rnn_eeg.eval()
    with torch.no_grad():
        rnn_eeg_norm = rnn_eeg(torch.FloatTensor(noisy_eeg)).numpy()
    rnn_eeg_output = rnn_eeg_norm * eeg_stds + eeg_means

    # Train ICA-MoE
    print("\n[5/5] Training ICA-MoE (Ours)...")
    ica_moe = ICAMoEFilter(
        num_experts=4,
        hidden_size=128,
        n_ica_components=4
    )
    ica_moe_output = train_ica_moe(
        ica_moe, noisy_eeg, eeg_norm, eog_norm, emg_norm,
        eeg_stds, eeg_means,
        epochs=10, pretrain_epochs=5, batch_size=32, seed=seed
    )

    # Denormalize ground truth and noisy for display
    clean_display = eeg_norm * eeg_stds + eeg_means
    noisy_display = noisy_eeg * eeg_stds + eeg_means

    # Create time axis
    t = np.arange(seq_length) / fs

    # Apply time range
    start_idx = int(time_range[0] * fs)
    end_idx = int(time_range[1] * fs)
    t_display = t[start_idx:end_idx]

    # Get signals for the selected sample
    signals = {
        'noisy': noisy_display[sample_idx, start_idx:end_idx],
        'clean': clean_display[sample_idx, start_idx:end_idx],
        'SimpleCNN': simple_cnn_output[sample_idx, start_idx:end_idx],
        'EEGDnoiseNet': eegdnoisenet_output[sample_idx, start_idx:end_idx],
        'RNN-EEG': rnn_eeg_output[sample_idx, start_idx:end_idx],
        'ICA-MoE (Ours)': ica_moe_output[sample_idx, start_idx:end_idx],
    }

    # Compute metrics for each method
    clean_full = clean_display[sample_idx]
    metrics = {}
    for name in ['SimpleCNN', 'EEGDnoiseNet', 'RNN-EEG', 'ICA-MoE (Ours)']:
        if name == 'SimpleCNN':
            output = simple_cnn_output[sample_idx]
        elif name == 'EEGDnoiseNet':
            output = eegdnoisenet_output[sample_idx]
        elif name == 'RNN-EEG':
            output = rnn_eeg_output[sample_idx]
        else:
            output = ica_moe_output[sample_idx]
        metrics[name] = compute_metrics(clean_full, output)

    print("\n" + "=" * 60)
    print("Metrics Summary:")
    print("-" * 60)
    for name, m in metrics.items():
        print(f"{name:20s} | RMSE: {m['rmse']:.4f} | Corr: {m['corr']:.4f} | SNR: {m['snr']:.2f} dB")
    print("=" * 60)

    # Create figure
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.35)

    # Color scheme
    colors = {
        'noisy': '#d62728',      # Red
        'clean': '#2ca02c',      # Green
        'SimpleCNN': '#1f77b4',  # Blue
        'EEGDnoiseNet': '#ff7f0e',  # Orange
        'RNN-EEG': '#9467bd',    # Purple
        'ICA-MoE (Ours)': '#17becf',  # Cyan
    }

    labels = [
        ('(a) Contaminated Input', 'noisy'),
        ('(b) Ground-Truth Clean', 'clean'),
        ('(c) SimpleCNN', 'SimpleCNN'),
        ('(d) EEGDnoiseNet', 'EEGDnoiseNet'),
        ('(e) RNN-EEG', 'RNN-EEG'),
        ('(f) ICA-MoE (Ours)', 'ICA-MoE (Ours)'),
    ]

    for i, (title, key) in enumerate(labels):
        ax = fig.add_subplot(gs[i])
        ax.plot(t_display, signals[key], color=colors[key], linewidth=0.8)
        ax.set_ylabel('Amplitude\n(a.u.)', fontsize=9)
        ax.set_xlim(t_display[0], t_display[-1])

        # Add title with metrics for denoised methods
        if key in metrics:
            m = metrics[key]
            title_text = f"{title} (RMSE: {m['rmse']:.3f}, r: {m['corr']:.4f})"
        else:
            title_text = title
        ax.set_title(title_text, fontsize=10, fontweight='bold', loc='left')

        # Only show x-label on bottom plot
        if i == 5:
            ax.set_xlabel('Time (s)', fontsize=10)
        else:
            ax.set_xticklabels([])

        # Add light grid
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

        # Highlight ICA-MoE with background
        if key == 'ICA-MoE (Ours)':
            ax.set_facecolor('#f0f8ff')

    # Add overall title
    fig.suptitle('Figure 2: Qualitative Comparison of Denoising Results',
                 fontsize=12, fontweight='bold', y=0.98)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {save_path}")

    # Also save as PDF for LaTeX
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_path}")

    plt.close()

    return fig, metrics


def generate_figure2_multiple_samples(
    sample_indices: list = [0, 10, 50],
    time_range: tuple = (0.5, 1.5),
    save_path: str = 'paper/media/fig2_visualization_multi.png',
    seed: int = 42
):
    """
    Generate Figure 2 with multiple sample rows for more comprehensive comparison.

    Args:
        sample_indices: List of sample indices to visualize
        time_range: Time range to display
        save_path: Path to save the figure
        seed: Random seed
    """
    set_all_seeds(seed)

    print("=" * 60)
    print("Generating Multi-Sample Figure 2")
    print("=" * 60)

    # Generate data
    n_samples = 500
    seq_length = 512
    fs = 256.0

    data = generate_synthetic_data(n_samples=n_samples, seq_length=seq_length, seed=seed)
    eeg_norm, eog_norm, emg_norm, noisy_eeg, eeg_means, eeg_stds = preprocess_data(
        data['eeg'], data['eog'], data['emg'], noise_scale=0.01
    )

    eeg_means = eeg_means.reshape(-1, 1)
    eeg_stds = eeg_stds.reshape(-1, 1)

    # Train all models (same as before)
    print("\nTraining models...")

    simple_cnn = SimpleCNN(input_size=seq_length)
    simple_cnn = train_baseline_model(simple_cnn, noisy_eeg, eeg_norm, epochs=30, seed=seed)
    simple_cnn_output = apply_baseline_model(simple_cnn, noisy_eeg, eeg_stds, eeg_means)

    eegdnoisenet = EEGDnoiseNet(input_size=seq_length, num_layers=10)
    eegdnoisenet = train_baseline_model(eegdnoisenet, noisy_eeg, eeg_norm, epochs=30, seed=seed)
    eegdnoisenet_output = apply_baseline_model(eegdnoisenet, noisy_eeg, eeg_stds, eeg_means)

    rnn_eeg = RNNEEG(input_size=seq_length, hidden_size=128)
    rnn_eeg = train_baseline_model(rnn_eeg, noisy_eeg, eeg_norm, eog=eog_norm, emg=emg_norm, epochs=30, seed=seed)
    rnn_eeg_output = apply_baseline_model(rnn_eeg, noisy_eeg, eeg_stds, eeg_means, eog=eog_norm, emg=emg_norm)

    ica_moe = ICAMoEFilter(num_experts=4, hidden_size=128, n_ica_components=4, seed=seed)
    ica_moe_output = ica_moe(noisy_eeg, eog_norm, emg_norm, eeg_norm, eeg_stds, eeg_means, pretrain_epochs=5, train_epochs=10)

    clean_display = eeg_norm * eeg_stds + eeg_means
    noisy_display = noisy_eeg * eeg_stds + eeg_means

    # Create figure with multiple rows
    n_rows = len(sample_indices)
    fig, axes = plt.subplots(n_rows, 6, figsize=(16, 2.5 * n_rows), sharex=True)

    t = np.arange(seq_length) / fs
    start_idx = int(time_range[0] * fs)
    end_idx = int(time_range[1] * fs)
    t_display = t[start_idx:end_idx]

    methods = ['Noisy', 'Clean', 'SimpleCNN', 'EEGDnoiseNet', 'RNN-EEG', 'ICA-MoE']
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#17becf']

    for row, sample_idx in enumerate(sample_indices):
        outputs = [
            noisy_display[sample_idx],
            clean_display[sample_idx],
            simple_cnn_output[sample_idx],
            eegdnoisenet_output[sample_idx],
            rnn_eeg_output[sample_idx],
            ica_moe_output[sample_idx],
        ]

        for col, (method, output, color) in enumerate(zip(methods, outputs, colors)):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.plot(t_display, output[start_idx:end_idx], color=color, linewidth=0.7)

            if row == 0:
                ax.set_title(method, fontsize=10, fontweight='bold')

            if col == 0:
                ax.set_ylabel(f'Sample {sample_idx}', fontsize=9)

            if row == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=9)

            ax.grid(True, alpha=0.3, linewidth=0.5)

            # Highlight ICA-MoE column
            if method == 'ICA-MoE':
                ax.set_facecolor('#f0f8ff')

    plt.suptitle('Figure 2: Qualitative Comparison Across Multiple Samples',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nMulti-sample figure saved to: {save_path}")

    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate Figure 2 for the paper')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='paper/media/fig2_visualization.png',
                        help='Output path for the figure')
    parser.add_argument('--multi', action='store_true', help='Generate multi-sample version')

    args = parser.parse_args()

    if args.multi:
        generate_figure2_multiple_samples(
            sample_indices=[0, 25, 50],
            save_path=args.output.replace('.png', '_multi.png'),
            seed=args.seed
        )
    else:
        generate_figure2(
            sample_idx=args.sample,
            save_path=args.output,
            seed=args.seed
        )
