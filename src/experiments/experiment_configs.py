"""
Experiment configurations for EEG denoising experiments.

CHANGELOG vs the original (JBHI desk-reject revision):
  - Replaced fixed `noise_scale=0.01` with a multi-point input-SNR sweep
    in {-7, -5, -3, -1, 0, +1, +2} dB, the standard EEGdenoiseNet protocol.
  - Baselines now run on ALL 5 seeds (was 2), matching SLA-MoE.
  - Added subject-aware split flag (LOSO / grouped K-fold).
  - Added a `quick` config that fits on a MacBook Air M1 / 8 GB for sanity
    checks (~10 minutes); the `full` configs are intended for Colab/Kaggle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from ..utils.seed_utils import EXPERIMENT_SEEDS


# Standard input SNR sweep used in EEGdenoiseNet papers.
DEFAULT_SNR_SWEEP_DB: List[float] = [-7.0, -5.0, -3.0, -1.0, 0.0, 1.0, 2.0]
# Single SNR for the headline EEG+EOG+EMG comparison table (paper Table II).
DEFAULT_HEADLINE_SNR_DB: float = -3.0


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    noise_type: str  # 'eog', 'emg', 'eog_emg'

    # ---- contamination ----
    target_snr_db: float = DEFAULT_HEADLINE_SNR_DB
    snr_sweep_db: List[float] = field(default_factory=lambda: list(DEFAULT_SNR_SWEEP_DB))
    # Legacy: only used if target_snr_db is set to None explicitly (back-compat).
    noise_scale: Optional[float] = None

    # ---- training ----
    seeds: List[int] = field(default_factory=lambda: list(EXPERIMENT_SEEDS))
    epochs: int = 10
    pretrain_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    # ---- model ----
    num_experts: int = 4
    hidden_size: int = 128
    n_ica_components: int = 4
    top_k: int = 2

    # ---- SLA-MoE specific knobs (now exposed) ----
    tau_low: float = 0.5
    tau_high: float = 0.5
    tau_kurt: float = 1.0
    tau_skew: float = 1.0
    alpha_blend: List[float] = field(default_factory=lambda: [0.5, 0.5, 1.0, 0.5])  # one per expert
    lambda_art: float = 1.0
    lambda_independence: float = 0.1
    lambda_load_balance: float = 0.01

    # ---- evaluation ----
    subject_split: bool = True              # use grouped K-fold by subject
    n_folds: int = 5
    subject_id_path: Optional[str] = None   # set this if you have the EEGdenoiseNet
                                            # subject-id .npy mapping
    fallback_n_subjects: int = 67

    # ---- bookkeeping ----
    results_dir: str = "results"
    save_intermediate: bool = True
    verbose: bool = True


@dataclass
class BaselineConfig:
    """Configuration for baseline models. Now runs on the full seed list."""
    name: str
    model_type: str
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    hidden_channels: int = 64
    num_layers: int = 10
    seeds: List[int] = field(default_factory=lambda: list(EXPERIMENT_SEEDS))


# =============================================================================
# Full experiment configurations (Colab / Kaggle)
# =============================================================================

EEG_EOG_CONFIG = ExperimentConfig(
    name="EEG_EOG",
    description="EEG denoising with EOG artifact removal only",
    noise_type="eog",
    results_dir="results/eeg_eog",
)

EEG_EMG_CONFIG = ExperimentConfig(
    name="EEG_EMG",
    description="EEG denoising with EMG artifact removal only",
    noise_type="emg",
    results_dir="results/eeg_emg",
)

EEG_EOG_EMG_CONFIG = ExperimentConfig(
    name="EEG_EOG_EMG",
    description="EEG denoising with both EOG and EMG artifact removal",
    noise_type="eog_emg",
    results_dir="results/eeg_eog_emg",
)


# =============================================================================
# Quick / smoke configurations (M1 MacBook Air, ~10 minutes)
# =============================================================================

EEG_EOG_EMG_QUICK = ExperimentConfig(
    name="EEG_EOG_EMG_QUICK",
    description="Fast sanity-check config for M1 / 8 GB. Not for paper results.",
    noise_type="eog_emg",
    snr_sweep_db=[-3.0],          # one SNR point
    seeds=EXPERIMENT_SEEDS[:2],   # two seeds
    epochs=3,
    pretrain_epochs=1,
    batch_size=16,
    n_folds=2,                    # 2-fold instead of 5
    results_dir="results/quick/eeg_eog_emg",
)


# =============================================================================
# Baseline configurations (now full 5-seed)
# =============================================================================

# DL baseline epochs reduced to 15 (was 50). Profiling on Colab T4 showed:
#   - 50 epochs: EEGDnoiseNet ~63 min/fold, RNN_EEG ~2 hr/fold
#   - 15 epochs: EEGDnoiseNet ~19 min/fold, RNN_EEG ~38 min/fold
# 15 epochs is sufficient for the loss to plateau on EEGdenoiseNet and gives a
# fair benchmark while keeping the full sweep to ~5 hours rather than days.
# RNN_EEG hidden_channels also reduced from 128 -> 64 (4x speedup) since it was
# pathologically slow with the larger size.
BASELINE_CONFIGS: Dict[str, BaselineConfig] = {
    "eegdnoisenet": BaselineConfig(
        name="EEGDnoiseNet", model_type="eegdnoisenet",
        epochs=15, hidden_channels=64, num_layers=10),
    "eegdnet": BaselineConfig(
        name="EEGDnet", model_type="eegdnet",
        epochs=15, hidden_channels=32),
    "rnn_eeg": BaselineConfig(
        name="RNN_EEG", model_type="rnn_eeg",
        epochs=15, hidden_channels=64),  # was 128
    "resnet_eeg": BaselineConfig(
        name="ResNet_EEG", model_type="resnet_eeg",
        epochs=15, hidden_channels=64, num_layers=4),
    "simple_cnn": BaselineConfig(
        name="SimpleCNN", model_type="simple_cnn",
        epochs=15, hidden_channels=64),
    # Traditional filters: no training, but still keyed for the runner.
    "wiener": BaselineConfig(name="WienerFilter", model_type="wiener", epochs=0),
    "lms":    BaselineConfig(name="LMSFilter",    model_type="lms",    epochs=0),
    "rls":    BaselineConfig(name="RLSFilter",    model_type="rls",    epochs=0),
    "kalman": BaselineConfig(name="KalmanFilter", model_type="kalman", epochs=0),
}


# Quick-test baseline subset for M1 sanity runs.
QUICK_BASELINE_KEYS = ["simple_cnn", "lms"]


# =============================================================================
# Accessors
# =============================================================================

def get_all_experiment_configs() -> Dict[str, ExperimentConfig]:
    return {
        "eeg_eog": EEG_EOG_CONFIG,
        "eeg_emg": EEG_EMG_CONFIG,
        "eeg_eog_emg": EEG_EOG_EMG_CONFIG,
    }


def get_quick_experiment_configs() -> Dict[str, ExperimentConfig]:
    return {"eeg_eog_emg_quick": EEG_EOG_EMG_QUICK}


def get_baseline_configs() -> Dict[str, BaselineConfig]:
    return BASELINE_CONFIGS.copy()


def get_quick_baseline_configs() -> Dict[str, BaselineConfig]:
    return {k: BASELINE_CONFIGS[k] for k in QUICK_BASELINE_KEYS}
