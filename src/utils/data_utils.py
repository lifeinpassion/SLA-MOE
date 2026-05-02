"""
Data preprocessing utilities for EEG denoising experiments.

This module replaces the original noise_scale-based contamination with the
standard EEGdenoiseNet protocol: contaminate at a *target input SNR* (in dB)
by computing the per-segment scaling factor lambda such that the resulting
input SNR matches a requested value. This is the convention used in
Zhang et al., 2021 and in every method that benchmarks against EEGdenoiseNet.

Backward-compatibility: the old preprocess_data / preprocess_eeg_eog /
preprocess_eeg_emg functions are kept and now accept either `noise_scale`
(deprecated) OR `target_snr_db`. If `target_snr_db` is provided, it takes
precedence.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from typing import Tuple, Dict, List, Optional, Iterator
import warnings


# =============================================================================
# Core contamination utilities (SNR-based)
# =============================================================================

def _signal_power(x: np.ndarray) -> np.ndarray:
    """Mean square power per row. Accepts shape (N, T)."""
    return np.mean(x ** 2, axis=-1, keepdims=False)


def lambda_for_target_snr(clean: np.ndarray,
                          artifact: np.ndarray,
                          target_snr_db: float) -> np.ndarray:
    """
    Compute per-segment scaling lambda so that x = clean + lambda * artifact
    has SNR = target_snr_db (computed against `clean` as the signal of
    interest and `lambda * artifact` as the noise).

    SNR_dB = 10 * log10( P_signal / P_noise )
       =>  P_noise = P_signal / 10^(SNR/10)
       =>  lambda^2 * P_artifact = P_signal / 10^(SNR/10)
       =>  lambda = sqrt( P_signal / (P_artifact * 10^(SNR/10)) )

    Args:
        clean:    (N, T) clean EEG segments
        artifact: (N, T) artifact segments (EOG, EMG, or sum)
        target_snr_db: desired input SNR in dB (negative = noisier)

    Returns:
        lambda values, shape (N,)
    """
    if clean.shape != artifact.shape:
        raise ValueError(f"shape mismatch: clean {clean.shape} vs artifact {artifact.shape}")

    p_signal = _signal_power(clean)
    p_artifact = _signal_power(artifact)
    p_artifact = np.maximum(p_artifact, 1e-12)
    ratio = p_signal / (p_artifact * (10 ** (target_snr_db / 10.0)))
    return np.sqrt(np.maximum(ratio, 0.0))


def contaminate(clean: np.ndarray,
                artifact: np.ndarray,
                target_snr_db: Optional[float] = None,
                noise_scale: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Contaminate clean signals with an artifact at either a target SNR (preferred)
    or a fixed noise_scale (legacy behavior).

    Returns:
        noisy: clean + scale * artifact
        scale: per-segment scaling factor used, shape (N,)
    """
    if target_snr_db is not None:
        lam = lambda_for_target_snr(clean, artifact, target_snr_db)  # (N,)
        noisy = clean + lam[:, None] * artifact
        return noisy, lam
    if noise_scale is None:
        raise ValueError("Provide either target_snr_db or noise_scale")
    noisy = clean + noise_scale * artifact
    return noisy, np.full(clean.shape[0], float(noise_scale))


# =============================================================================
# Backward-compatible preprocess_* functions (now SNR-aware)
# =============================================================================

def preprocess_data(eeg_data: np.ndarray,
                    eog_data: np.ndarray,
                    emg_data: np.ndarray,
                    noise_scale: Optional[float] = None,
                    *,
                    target_snr_db: Optional[float] = None) -> Tuple[np.ndarray, ...]:
    """
    Preprocess EEG, EOG, and EMG segments and synthesize contaminated EEG.

    Two contamination modes:
      - target_snr_db: synthesize at a target input SNR (recommended; the
        EEGdenoiseNet protocol).
      - noise_scale:   legacy behavior, fixed multiplicative scaling (kept
        for backward compatibility with the old runners).
    If neither is given, defaults to target_snr_db = -3 dB.

    Returns:
        eeg_normalized, eog_normalized, emg_normalized,
        noisy_eeg, eeg_means, eeg_stds  (6-tuple, matches legacy API)
    """
    if target_snr_db is None and noise_scale is None:
        target_snr_db = -3.0
    elif target_snr_db is not None and noise_scale is not None:
        warnings.warn("Both noise_scale and target_snr_db given; using target_snr_db.")
        noise_scale = None

    scaler_eeg = StandardScaler()
    scaler_eog = StandardScaler()
    scaler_emg = StandardScaler()

    eeg_normalized = scaler_eeg.fit_transform(eeg_data)
    eeg_means, eeg_stds = scaler_eeg.mean_, scaler_eeg.scale_

    eog_normalized = scaler_eog.fit_transform(eog_data)
    emg_normalized = scaler_emg.fit_transform(emg_data)

    artifact = eog_normalized + emg_normalized
    noisy_eeg, _scale = contaminate(eeg_normalized, artifact,
                                    target_snr_db=target_snr_db,
                                    noise_scale=noise_scale)

    # Return signature matches the original 6-tuple for backward-compat.
    return (eeg_normalized, eog_normalized, emg_normalized,
            noisy_eeg, eeg_means, eeg_stds)


def preprocess_eeg_eog(eeg_data: np.ndarray,
                       eog_data: np.ndarray,
                       noise_scale: Optional[float] = None,
                       *,
                       target_snr_db: Optional[float] = None) -> Tuple[np.ndarray, ...]:
    """EEG + EOG only. See preprocess_data for the SNR convention."""
    if target_snr_db is None and noise_scale is None:
        target_snr_db = -3.0
    elif target_snr_db is not None and noise_scale is not None:
        noise_scale = None

    scaler_eeg = StandardScaler()
    scaler_eog = StandardScaler()

    eeg_normalized = scaler_eeg.fit_transform(eeg_data)
    eeg_means, eeg_stds = scaler_eeg.mean_, scaler_eeg.scale_
    eog_normalized = scaler_eog.fit_transform(eog_data)

    noisy_eeg, _scale = contaminate(eeg_normalized, eog_normalized,
                                    target_snr_db=target_snr_db,
                                    noise_scale=noise_scale)
    emg_normalized = np.zeros_like(eog_normalized)

    return (eeg_normalized, eog_normalized, emg_normalized,
            noisy_eeg, eeg_means, eeg_stds)


def preprocess_eeg_emg(eeg_data: np.ndarray,
                       emg_data: np.ndarray,
                       noise_scale: Optional[float] = None,
                       *,
                       target_snr_db: Optional[float] = None) -> Tuple[np.ndarray, ...]:
    """EEG + EMG only. See preprocess_data for the SNR convention."""
    if target_snr_db is None and noise_scale is None:
        target_snr_db = -3.0
    elif target_snr_db is not None and noise_scale is not None:
        noise_scale = None

    scaler_eeg = StandardScaler()
    scaler_emg = StandardScaler()

    eeg_normalized = scaler_eeg.fit_transform(eeg_data)
    eeg_means, eeg_stds = scaler_eeg.mean_, scaler_eeg.scale_
    emg_normalized = scaler_emg.fit_transform(emg_data)

    noisy_eeg, _scale = contaminate(eeg_normalized, emg_normalized,
                                    target_snr_db=target_snr_db,
                                    noise_scale=noise_scale)
    eog_normalized = np.zeros_like(emg_normalized)

    return (eeg_normalized, eog_normalized, emg_normalized,
            noisy_eeg, eeg_means, eeg_stds)


# =============================================================================
# Subject-aware splits (LOSO / GroupKFold)
# =============================================================================

def load_subject_ids(subject_id_path: Optional[str],
                     n_segments: int,
                     fallback_n_subjects: int = 67) -> np.ndarray:
    """
    Load a per-segment subject id array of length n_segments.

    EEGdenoiseNet (Zhang et al. 2021) ships 4,514 EEG segments from 67 subjects.
    The original release does NOT distribute the per-segment subject mapping
    in the .npy files; the mapping is in their MATLAB source. If you have it
    saved as a .npy file (one int per segment), pass its path.

    If no path is given, we fall back to *grouped chunking*: split the
    n_segments into `fallback_n_subjects` near-equal contiguous groups. This
    is a pragmatic LOSO surrogate that papers in this space commonly use when
    the precise mapping is unavailable. It is not a true LOSO and we
    document this limitation in the paper.

    Args:
        subject_id_path: path to .npy with shape (n_segments,)
        n_segments: total number of segments
        fallback_n_subjects: pseudo-subject count for the fallback

    Returns:
        ids: int array of shape (n_segments,)
    """
    if subject_id_path is not None:
        ids = np.load(subject_id_path)
        if ids.shape[0] != n_segments:
            raise ValueError(
                f"subject_id length {ids.shape[0]} != n_segments {n_segments}")
        return ids.astype(int)

    warnings.warn(
        "No subject_id file provided; falling back to grouped chunking "
        f"into {fallback_n_subjects} pseudo-subjects. This is documented as a "
        "limitation in the paper. To use true LOSO, save your subject mapping "
        "as a .npy and pass it via --subject-ids.")
    chunk = np.linspace(0, fallback_n_subjects, n_segments + 1, dtype=int)[:-1]
    return chunk


def subject_kfold_splits(subject_ids: np.ndarray,
                         n_splits: int = 5,
                         seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Subject-grouped K-fold splits. Returns a list of (train_idx, test_idx)
    tuples; test fold consists of segments from a disjoint subset of subjects.

    Args:
        subject_ids: (n_segments,) integer subject id per segment
        n_splits: number of folds (default 5)
        seed: RNG seed for fold assignment shuffling
    """
    n = len(subject_ids)
    rng = np.random.RandomState(seed)
    unique_subjects = np.unique(subject_ids)
    rng.shuffle(unique_subjects)

    fold_assignment = {s: i % n_splits for i, s in enumerate(unique_subjects)}
    fold_per_segment = np.array([fold_assignment[s] for s in subject_ids])

    splits = []
    for k in range(n_splits):
        test_idx = np.where(fold_per_segment == k)[0]
        train_idx = np.where(fold_per_segment != k)[0]
        splits.append((train_idx, test_idx))
    return splits


def loso_splits(subject_ids: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
    """
    True leave-one-subject-out iterator. Yields (train_idx, test_idx, held_out_subject).
    Heavy: produces N_subjects folds.
    """
    unique_subjects = np.unique(subject_ids)
    for s in unique_subjects:
        test_idx = np.where(subject_ids == s)[0]
        train_idx = np.where(subject_ids != s)[0]
        yield train_idx, test_idx, int(s)


def segment_kfold_splits(n_segments: int,
                         n_splits: int = 5,
                         seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Plain segment-level K-fold (no subject grouping). Used only as a
    baseline / sanity check; should NOT be used for the headline results."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_segments)))


# =============================================================================
# Synthetic generator (unchanged, for unit tests)
# =============================================================================

def generate_synthetic_data(n_samples: int = 100, seq_length: int = 512,
                            seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate small synthetic EEG/EOG/EMG for unit tests and CI runs."""
    np.random.seed(seed)
    t = np.linspace(0, 1, seq_length)

    clean_eeg = np.array([
        np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        + 0.1 * np.random.randn(seq_length)
        for _ in range(n_samples)
    ])
    eog = np.array([
        0.5 * np.sin(2 * np.pi * 0.3 * t + np.random.rand() * 2 * np.pi)
        + 0.3 * np.random.randn(seq_length)
        for _ in range(n_samples)
    ])
    emg = np.array([
        0.2 * np.random.randn(seq_length) * np.abs(np.sin(2 * np.pi * 2 * t))
        for _ in range(n_samples)
    ])

    return {'eeg': clean_eeg, 'eog': eog, 'emg': emg}
