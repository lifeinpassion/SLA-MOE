"""
Evaluation metrics for EEG denoising.
"""
import numpy as np
from typing import Dict, Tuple
from scipy import signal as scipy_signal


def compute_snr(clean_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio improvement.

    Args:
        clean_signal: Original clean EEG signal
        denoised_signal: Denoised EEG signal

    Returns:
        SNR value in dB
    """
    noise = denoised_signal - clean_signal
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return np.inf

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_rmse(clean_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    """
    Compute Root Mean Square Error.

    Args:
        clean_signal: Original clean EEG signal
        denoised_signal: Denoised EEG signal

    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((clean_signal - denoised_signal) ** 2))


def compute_correlation(clean_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        clean_signal: Original clean EEG signal
        denoised_signal: Denoised EEG signal

    Returns:
        Correlation coefficient
    """
    clean_flat = clean_signal.flatten()
    denoised_flat = denoised_signal.flatten()

    correlation = np.corrcoef(clean_flat, denoised_flat)[0, 1]
    return correlation


def compute_relative_rmse(clean_signal: np.ndarray, denoised_signal: np.ndarray) -> float:
    """
    Compute Relative RMSE (RRMSE).

    Args:
        clean_signal: Original clean EEG signal
        denoised_signal: Denoised EEG signal

    Returns:
        RRMSE value
    """
    rmse = compute_rmse(clean_signal, denoised_signal)
    rms_clean = np.sqrt(np.mean(clean_signal ** 2))

    if rms_clean < 1e-10:
        return np.inf

    return rmse / rms_clean


def compute_spectral_distortion(clean_signal: np.ndarray, denoised_signal: np.ndarray,
                                fs: float = 256.0) -> float:
    """
    Compute spectral distortion between clean and denoised signals.

    Args:
        clean_signal: Original clean EEG signal
        denoised_signal: Denoised EEG signal
        fs: Sampling frequency

    Returns:
        Spectral distortion value
    """
    # Compute power spectral density
    f_clean, psd_clean = scipy_signal.welch(clean_signal.flatten(), fs=fs, nperseg=256)
    f_denoised, psd_denoised = scipy_signal.welch(denoised_signal.flatten(), fs=fs, nperseg=256)

    # Compute log spectral distortion
    psd_clean = np.maximum(psd_clean, 1e-10)
    psd_denoised = np.maximum(psd_denoised, 1e-10)

    log_diff = np.log10(psd_clean) - np.log10(psd_denoised)
    spectral_distortion = np.sqrt(np.mean(log_diff ** 2))

    return spectral_distortion


def compute_metrics(clean_signal: np.ndarray, denoised_signal: np.ndarray,
                   noisy_signal: np.ndarray = None, fs: float = 256.0) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        clean_signal: Original clean EEG signal
        denoised_signal: Denoised EEG signal
        noisy_signal: Original noisy signal (optional, for SNR improvement)
        fs: Sampling frequency

    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'rmse': compute_rmse(clean_signal, denoised_signal),
        'rrmse': compute_relative_rmse(clean_signal, denoised_signal),
        'correlation': compute_correlation(clean_signal, denoised_signal),
        'snr': compute_snr(clean_signal, denoised_signal),
        'spectral_distortion': compute_spectral_distortion(clean_signal, denoised_signal, fs)
    }

    # Compute SNR improvement if noisy signal is provided
    if noisy_signal is not None:
        snr_input = compute_snr(clean_signal, noisy_signal)
        snr_output = metrics['snr']
        metrics['snr_improvement'] = snr_output - snr_input
        metrics['snr_input'] = snr_input

    return metrics


def aggregate_metrics(metrics_list: list) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics from multiple runs (mean and std).

    Args:
        metrics_list: List of metric dictionaries from multiple runs

    Returns:
        Dictionary with (mean, std) tuples for each metric
    """
    if not metrics_list:
        return {}

    aggregated = {}
    keys = metrics_list[0].keys()

    for key in keys:
        values = [m[key] for m in metrics_list if not np.isinf(m[key]) and not np.isnan(m[key])]
        if values:
            aggregated[key] = (np.mean(values), np.std(values))
        else:
            aggregated[key] = (np.nan, np.nan)

    return aggregated
