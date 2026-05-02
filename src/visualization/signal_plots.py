"""
Signal processing visualizations following best practices.

This module provides publication-quality figures for:
- Signal comparison (clean, noisy, denoised)
- Frequency analysis (PSD, spectrogram)
- Time-frequency analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from typing import Optional, List, Tuple, Dict
from pathlib import Path

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
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def plot_signal_comparison(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray,
                           fs: float = 256.0, sample_idx: int = 0,
                           time_range: Tuple[float, float] = None,
                           title: str = "EEG Signal Comparison",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of clean, noisy, and denoised signals.

    Args:
        clean: Clean EEG signal (n_samples, n_timepoints)
        noisy: Noisy EEG signal
        denoised: Denoised EEG signal
        fs: Sampling frequency
        sample_idx: Index of sample to plot
        time_range: Time range to plot (start, end) in seconds
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    # Get single sample
    clean_sig = clean[sample_idx] if clean.ndim > 1 else clean
    noisy_sig = noisy[sample_idx] if noisy.ndim > 1 else noisy
    denoised_sig = denoised[sample_idx] if denoised.ndim > 1 else denoised

    # Create time axis
    n_samples = len(clean_sig)
    t = np.arange(n_samples) / fs

    # Apply time range if specified
    if time_range is not None:
        start_idx = int(time_range[0] * fs)
        end_idx = int(time_range[1] * fs)
        t = t[start_idx:end_idx]
        clean_sig = clean_sig[start_idx:end_idx]
        noisy_sig = noisy_sig[start_idx:end_idx]
        denoised_sig = denoised_sig[start_idx:end_idx]

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Color scheme
    colors = {
        'clean': '#2ca02c',
        'noisy': '#d62728',
        'denoised': '#1f77b4',
        'residual': '#7f7f7f'
    }

    # Plot clean signal
    axes[0].plot(t, clean_sig, color=colors['clean'], linewidth=0.8)
    axes[0].set_ylabel('Clean EEG\n(a.u.)')
    axes[0].set_title('(a) Clean Reference Signal', fontsize=11, fontweight='bold', loc='left')
    axes[0].set_xlim(t[0], t[-1])

    # Plot noisy signal
    axes[1].plot(t, noisy_sig, color=colors['noisy'], linewidth=0.8)
    axes[1].set_ylabel('Noisy EEG\n(a.u.)')
    axes[1].set_title('(b) Noisy Input Signal', fontsize=11, fontweight='bold', loc='left')

    # Plot denoised signal
    axes[2].plot(t, denoised_sig, color=colors['denoised'], linewidth=0.8)
    axes[2].set_ylabel('Denoised EEG\n(a.u.)')
    axes[2].set_title('(c) Denoised Output Signal', fontsize=11, fontweight='bold', loc='left')

    # Plot residual (noise removed)
    residual = noisy_sig - denoised_sig
    axes[3].plot(t, residual, color=colors['residual'], linewidth=0.8)
    axes[3].set_ylabel('Residual\n(a.u.)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('(d) Removed Noise (Residual)', fontsize=11, fontweight='bold', loc='left')

    # Add RMS values
    for ax, sig, name in zip(axes[:3], [clean_sig, noisy_sig, denoised_sig],
                              ['Clean', 'Noisy', 'Denoised']):
        rms = np.sqrt(np.mean(sig ** 2))
        ax.text(0.98, 0.95, f'RMS: {rms:.3f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Compute and display SNR improvement
    noise_power = np.mean((noisy_sig - clean_sig) ** 2)
    residual_power = np.mean((denoised_sig - clean_sig) ** 2)
    snr_improvement = 10 * np.log10(noise_power / (residual_power + 1e-10))

    fig.text(0.98, 0.02, f'SNR Improvement: {snr_improvement:.2f} dB',
            ha='right', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_denoising_results(clean: np.ndarray, noisy: np.ndarray,
                           denoised_dict: Dict[str, np.ndarray],
                           fs: float = 256.0, sample_idx: int = 0,
                           time_range: Tuple[float, float] = None,
                           title: str = "Denoising Method Comparison",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple denoising methods.

    Args:
        clean: Clean EEG signal
        noisy: Noisy EEG signal
        denoised_dict: Dictionary of {method_name: denoised_signal}
        fs: Sampling frequency
        sample_idx: Index of sample to plot
        time_range: Time range to plot
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    n_methods = len(denoised_dict) + 2  # +2 for clean and noisy

    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 2.5 * n_methods), sharex=True)

    # Get single samples
    clean_sig = clean[sample_idx] if clean.ndim > 1 else clean
    noisy_sig = noisy[sample_idx] if noisy.ndim > 1 else noisy

    n_samples = len(clean_sig)
    t = np.arange(n_samples) / fs

    if time_range is not None:
        start_idx = int(time_range[0] * fs)
        end_idx = int(time_range[1] * fs)
        t = t[start_idx:end_idx]
        clean_sig = clean_sig[start_idx:end_idx]
        noisy_sig = noisy_sig[start_idx:end_idx]

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    # Plot clean
    axes[0].plot(t, clean_sig, color='#2ca02c', linewidth=0.8)
    axes[0].set_ylabel('Clean')
    axes[0].set_title('(a) Reference', fontsize=10, loc='left')

    # Plot noisy
    axes[1].plot(t, noisy_sig, color='#d62728', linewidth=0.8)
    axes[1].set_ylabel('Noisy')
    axes[1].set_title('(b) Input', fontsize=10, loc='left')

    # Plot each denoised method
    for i, (name, denoised) in enumerate(denoised_dict.items()):
        denoised_sig = denoised[sample_idx] if denoised.ndim > 1 else denoised
        if time_range is not None:
            denoised_sig = denoised_sig[start_idx:end_idx]

        axes[i + 2].plot(t, denoised_sig, color=colors[i + 2], linewidth=0.8)
        axes[i + 2].set_ylabel(name)

        # Compute metrics
        rmse = np.sqrt(np.mean((denoised_sig - clean_sig) ** 2))
        corr = np.corrcoef(denoised_sig, clean_sig)[0, 1]
        axes[i + 2].set_title(f'({chr(99 + i)}) {name} (RMSE: {rmse:.4f}, r: {corr:.4f})',
                             fontsize=10, loc='left')

    axes[-1].set_xlabel('Time (s)')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_frequency_analysis(signal: np.ndarray, fs: float = 256.0,
                            sample_idx: int = 0,
                            freq_range: Tuple[float, float] = (0.5, 50),
                            title: str = "Frequency Analysis",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot frequency analysis of a signal.

    Args:
        signal: Input signal
        fs: Sampling frequency
        sample_idx: Index of sample to analyze
        freq_range: Frequency range to display
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    sig = signal[sample_idx] if signal.ndim > 1 else signal
    n = len(sig)
    t = np.arange(n) / fs

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.2])

    # Time domain
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, sig, 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.set_title('(a) Time Domain Signal', fontsize=11, fontweight='bold', loc='left')
    ax1.set_xlim(t[0], t[-1])

    # FFT magnitude spectrum
    ax2 = fig.add_subplot(gs[1, 0])
    freqs = fftfreq(n, 1/fs)
    fft_vals = np.abs(fft(sig)) / n
    pos_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    ax2.plot(freqs[pos_mask], fft_vals[pos_mask], 'b-', linewidth=0.8)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('(b) FFT Magnitude Spectrum', fontsize=11, fontweight='bold', loc='left')

    # Power Spectral Density (Welch method)
    ax3 = fig.add_subplot(gs[1, 1])
    f_welch, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, n//2))
    freq_mask = (f_welch >= freq_range[0]) & (f_welch <= freq_range[1])
    ax3.semilogy(f_welch[freq_mask], psd[freq_mask], 'b-', linewidth=0.8)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('PSD (V²/Hz)')
    ax3.set_title('(c) Power Spectral Density', fontsize=11, fontweight='bold', loc='left')

    # Add EEG band annotations
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }
    band_colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99', '#ff99ff']

    for (band_name, (f_low, f_high)), color in zip(bands.items(), band_colors):
        if f_low >= freq_range[0] and f_high <= freq_range[1]:
            ax3.axvspan(f_low, f_high, alpha=0.2, color=color, label=band_name)

    ax3.legend(loc='upper right', fontsize=8)

    # Spectrogram
    ax4 = fig.add_subplot(gs[2, :])
    nperseg = min(128, n // 4)
    f_spec, t_spec, Sxx = scipy_signal.spectrogram(sig, fs=fs, nperseg=nperseg,
                                                    noverlap=nperseg // 2)
    freq_mask = (f_spec >= freq_range[0]) & (f_spec <= freq_range[1])

    im = ax4.pcolormesh(t_spec, f_spec[freq_mask], 10 * np.log10(Sxx[freq_mask] + 1e-10),
                        shading='gouraud', cmap='viridis')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('(d) Spectrogram', fontsize=11, fontweight='bold', loc='left')

    cbar = plt.colorbar(im, ax=ax4, label='Power (dB)')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_psd_comparison(clean: np.ndarray, noisy: np.ndarray,
                        denoised_dict: Dict[str, np.ndarray],
                        fs: float = 256.0, sample_idx: int = 0,
                        freq_range: Tuple[float, float] = (0.5, 50),
                        title: str = "PSD Comparison",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare Power Spectral Density across methods.

    Args:
        clean: Clean signal
        noisy: Noisy signal
        denoised_dict: Dictionary of denoised signals
        fs: Sampling frequency
        sample_idx: Index of sample
        freq_range: Frequency range to display
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get signals
    clean_sig = clean[sample_idx] if clean.ndim > 1 else clean
    noisy_sig = noisy[sample_idx] if noisy.ndim > 1 else noisy
    n = len(clean_sig)

    # Compute PSDs
    f, psd_clean = scipy_signal.welch(clean_sig, fs=fs, nperseg=min(256, n//2))
    _, psd_noisy = scipy_signal.welch(noisy_sig, fs=fs, nperseg=min(256, n//2))

    freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])

    # Left plot: PSD comparison
    ax1 = axes[0]
    ax1.semilogy(f[freq_mask], psd_clean[freq_mask], 'g-', label='Clean', linewidth=1.5)
    ax1.semilogy(f[freq_mask], psd_noisy[freq_mask], 'r--', label='Noisy', linewidth=1.2, alpha=0.7)

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(denoised_dict)))
    for i, (name, denoised) in enumerate(denoised_dict.items()):
        denoised_sig = denoised[sample_idx] if denoised.ndim > 1 else denoised
        _, psd_denoised = scipy_signal.welch(denoised_sig, fs=fs, nperseg=min(256, n//2))
        ax1.semilogy(f[freq_mask], psd_denoised[freq_mask], '-',
                    color=colors[i], label=name, linewidth=1.2)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (V²/Hz)')
    ax1.set_title('(a) Power Spectral Density', fontsize=11, fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right plot: Spectral distortion
    ax2 = axes[1]
    methods = []
    distortions = []

    for name, denoised in denoised_dict.items():
        denoised_sig = denoised[sample_idx] if denoised.ndim > 1 else denoised
        _, psd_denoised = scipy_signal.welch(denoised_sig, fs=fs, nperseg=min(256, n//2))

        # Log spectral distortion
        log_diff = np.log10(psd_clean[freq_mask] + 1e-10) - np.log10(psd_denoised[freq_mask] + 1e-10)
        distortion = np.sqrt(np.mean(log_diff ** 2))

        methods.append(name)
        distortions.append(distortion)

    # Add noisy baseline
    methods.insert(0, 'Noisy')
    log_diff_noisy = np.log10(psd_clean[freq_mask] + 1e-10) - np.log10(psd_noisy[freq_mask] + 1e-10)
    distortions.insert(0, np.sqrt(np.mean(log_diff_noisy ** 2)))

    bars = ax2.barh(methods, distortions, color=['#d62728'] + list(colors))
    ax2.set_xlabel('Log Spectral Distortion')
    ax2.set_title('(b) Spectral Distortion (lower is better)',
                  fontsize=11, fontweight='bold', loc='left')
    ax2.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, distortions):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_time_frequency_comparison(clean: np.ndarray, noisy: np.ndarray,
                                   denoised: np.ndarray, fs: float = 256.0,
                                   sample_idx: int = 0,
                                   freq_range: Tuple[float, float] = (0.5, 50),
                                   title: str = "Time-Frequency Analysis",
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time-frequency comparison using spectrograms.

    Args:
        clean: Clean signal
        noisy: Noisy signal
        denoised: Denoised signal
        fs: Sampling frequency
        sample_idx: Index of sample
        freq_range: Frequency range
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    signals = {
        '(a) Clean Reference': clean,
        '(b) Noisy Input': noisy,
        '(c) Denoised Output': denoised,
        '(d) Removed Noise': None  # Will compute
    }

    # Get single samples
    clean_sig = clean[sample_idx] if clean.ndim > 1 else clean
    noisy_sig = noisy[sample_idx] if noisy.ndim > 1 else noisy
    denoised_sig = denoised[sample_idx] if denoised.ndim > 1 else denoised
    residual_sig = noisy_sig - denoised_sig

    all_sigs = [clean_sig, noisy_sig, denoised_sig, residual_sig]
    n = len(clean_sig)
    nperseg = min(128, n // 4)

    # Find global min/max for consistent colorbar
    vmin, vmax = float('inf'), float('-inf')
    for sig in all_sigs[:3]:  # Exclude residual
        f, t, Sxx = scipy_signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-10)
        vmin = min(vmin, Sxx_db.min())
        vmax = max(vmax, Sxx_db.max())

    for ax, (name, _), sig in zip(axes.flat, signals.items(), all_sigs):
        f, t_spec, Sxx = scipy_signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])

        im = ax.pcolormesh(t_spec, f[freq_mask], 10 * np.log10(Sxx[freq_mask] + 1e-10),
                           shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(name, fontsize=11, fontweight='bold', loc='left')

    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Power (dB)', shrink=0.6)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
