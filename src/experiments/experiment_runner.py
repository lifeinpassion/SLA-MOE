"""
Experiment runner for EEG denoising experiments.

This module provides:
- ExperimentRunner: Class for running single experiments
- run_multi_seed_experiment: Function for running experiments with multiple seeds
"""
import os
import json
import logging
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..utils.seed_utils import set_all_seeds, EXPERIMENT_SEEDS
from ..utils.data_utils import preprocess_data, preprocess_eeg_eog, preprocess_eeg_emg
from ..utils.metrics import compute_metrics, aggregate_metrics
from ..models.baselines import (
    EEGDnoiseNet, EEGDnet, RNNEEG, ResNetEEG, SimpleCNN,
    WienerFilter, LMSFilter, RLSFilter, KalmanFilter,
    train_baseline_model, apply_baseline_model
)
from ..models.moe_models import (
    RNNMoEFilter, ICAMoEFilter, ReservoirMoEFilter,
    train_ica_moe
)
from .experiment_configs import ExperimentConfig, BaselineConfig


class ExperimentRunner:
    """
    Class for running EEG denoising experiments.
    """

    def __init__(self, config: ExperimentConfig, base_dir: str = "."):
        self.config = config
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        self.results = {}

    def _setup_logging(self):
        """Setup logging for the experiment."""
        log_file = self.results_dir / f"{self.config.name}_experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def preprocess(self, eeg_data: np.ndarray, eog_data: np.ndarray,
                   emg_data: np.ndarray) -> Tuple:
        """
        Preprocess data based on experiment configuration.
        """
        if self.config.noise_type == "eog":
            return preprocess_eeg_eog(eeg_data, eog_data, self.config.noise_scale)
        elif self.config.noise_type == "emg":
            return preprocess_eeg_emg(eeg_data, emg_data, self.config.noise_scale)
        else:  # eog_emg
            return preprocess_data(eeg_data, eog_data, emg_data, self.config.noise_scale)

    def run_ica_moe(self, noisy_eeg: np.ndarray, clean_eeg: np.ndarray,
                    eog: np.ndarray, emg: np.ndarray,
                    eeg_stds: np.ndarray, eeg_means: np.ndarray,
                    seed: int) -> Tuple[np.ndarray, Dict]:
        """
        Run ICA-MoE model.
        """
        self.logger.info(f"Running ICA-MoE with seed {seed}")
        set_all_seeds(seed)

        model = ICAMoEFilter(
            num_experts=self.config.num_experts,
            hidden_size=self.config.hidden_size,
            n_ica_components=self.config.n_ica_components,
            top_k=self.config.top_k
        )

        denoised = train_ica_moe(
            model, noisy_eeg, clean_eeg, eog, emg, eeg_stds, eeg_means,
            epochs=self.config.epochs,
            pretrain_epochs=self.config.pretrain_epochs,
            batch_size=self.config.batch_size,
            seed=seed
        )

        # Denormalize clean for metrics
        clean_denorm = clean_eeg * eeg_stds + eeg_means
        noisy_denorm = noisy_eeg * eeg_stds + eeg_means

        metrics = compute_metrics(clean_denorm, denoised, noisy_denorm)

        return denoised, metrics

    def run_baseline(self, baseline_config: BaselineConfig,
                     noisy_eeg: np.ndarray, clean_eeg: np.ndarray,
                     eog: np.ndarray, emg: np.ndarray,
                     eeg_stds: np.ndarray, eeg_means: np.ndarray,
                     seed: int) -> Tuple[np.ndarray, Dict]:
        """
        Run a baseline model.
        """
        self.logger.info(f"Running {baseline_config.name} with seed {seed}")
        set_all_seeds(seed)

        input_size = noisy_eeg.shape[1]

        # Create model based on type
        if baseline_config.model_type == "eegdnoisenet":
            model = EEGDnoiseNet(
                input_size=input_size,
                hidden_channels=baseline_config.hidden_channels,
                num_layers=baseline_config.num_layers
            )
        elif baseline_config.model_type == "eegdnet":
            model = EEGDnet(
                input_size=input_size,
                base_channels=baseline_config.hidden_channels
            )
        elif baseline_config.model_type == "rnn_eeg":
            model = RNNEEG(
                input_size=input_size,
                hidden_size=baseline_config.hidden_channels
            )
        elif baseline_config.model_type == "resnet_eeg":
            model = ResNetEEG(
                input_size=input_size,
                base_channels=baseline_config.hidden_channels,
                num_blocks=baseline_config.num_layers
            )
        elif baseline_config.model_type == "simple_cnn":
            model = SimpleCNN(input_size=input_size)
        elif baseline_config.model_type == "wiener":
            wiener_filter = WienerFilter()
            denoised = wiener_filter(noisy_eeg, eeg_stds, eeg_means)
            clean_denorm = clean_eeg * eeg_stds + eeg_means
            noisy_denorm = noisy_eeg * eeg_stds + eeg_means
            metrics = compute_metrics(clean_denorm, denoised, noisy_denorm)
            return denoised, metrics
        elif baseline_config.model_type == "lms":
            lms_filter = LMSFilter()
            denoised = lms_filter(noisy_eeg, eog, emg, eeg_stds, eeg_means)
            clean_denorm = clean_eeg * eeg_stds + eeg_means
            noisy_denorm = noisy_eeg * eeg_stds + eeg_means
            metrics = compute_metrics(clean_denorm, denoised, noisy_denorm)
            return denoised, metrics
        elif baseline_config.model_type == "rls":
            rls_filter = RLSFilter()
            denoised = rls_filter(noisy_eeg, eog, emg, eeg_stds, eeg_means)
            clean_denorm = clean_eeg * eeg_stds + eeg_means
            noisy_denorm = noisy_eeg * eeg_stds + eeg_means
            metrics = compute_metrics(clean_denorm, denoised, noisy_denorm)
            return denoised, metrics
        elif baseline_config.model_type == "kalman":
            kalman_filter = KalmanFilter()
            denoised = kalman_filter(noisy_eeg, eeg_stds, eeg_means)
            clean_denorm = clean_eeg * eeg_stds + eeg_means
            noisy_denorm = noisy_eeg * eeg_stds + eeg_means
            metrics = compute_metrics(clean_denorm, denoised, noisy_denorm)
            return denoised, metrics
        else:
            raise ValueError(f"Unknown baseline model type: {baseline_config.model_type}")

        # Train deep learning model
        model = train_baseline_model(
            model, noisy_eeg, clean_eeg, eog, emg,
            epochs=baseline_config.epochs,
            batch_size=self.config.batch_size,
            lr=baseline_config.learning_rate,
            seed=seed
        )

        # Apply model
        denoised = apply_baseline_model(model, noisy_eeg, eeg_stds, eeg_means, eog, emg)

        # Compute metrics
        clean_denorm = clean_eeg * eeg_stds + eeg_means
        noisy_denorm = noisy_eeg * eeg_stds + eeg_means
        metrics = compute_metrics(clean_denorm, denoised, noisy_denorm)

        return denoised, metrics

    def run_experiment(self, eeg_data: np.ndarray, eog_data: np.ndarray,
                      emg_data: np.ndarray,
                      baseline_configs: Dict[str, BaselineConfig] = None) -> Dict:
        """
        Run the full experiment with multiple seeds.

        Args:
            eeg_data: Clean EEG data
            eog_data: EOG data
            emg_data: EMG data
            baseline_configs: Optional dictionary of baseline configurations

        Returns:
            Dictionary with all results
        """
        self.logger.info(f"Starting experiment: {self.config.name}")
        self.logger.info(f"Description: {self.config.description}")
        self.logger.info(f"Seeds: {self.config.seeds}")

        results = {
            "config": {
                "name": self.config.name,
                "description": self.config.description,
                "noise_type": self.config.noise_type,
                "seeds": self.config.seeds
            },
            "ica_moe": {},
            "baselines": {},
            "timestamp": datetime.now().isoformat()
        }

        # Run ICA-MoE for each seed
        ica_moe_metrics_list = []
        for seed in self.config.seeds:
            # Preprocess data
            clean_eeg, eog_norm, emg_norm, noisy_eeg, eeg_means, eeg_stds = \
                self.preprocess(eeg_data, eog_data, emg_data)

            denoised, metrics = self.run_ica_moe(
                noisy_eeg, clean_eeg, eog_norm, emg_norm, eeg_stds, eeg_means, seed
            )

            results["ica_moe"][f"seed_{seed}"] = {
                "metrics": metrics,
                "denoised_shape": denoised.shape
            }
            ica_moe_metrics_list.append(metrics)

            # Save intermediate results
            if self.config.save_intermediate:
                np.save(
                    self.results_dir / f"ica_moe_denoised_seed{seed}.npy",
                    denoised
                )

            self.logger.info(f"ICA-MoE seed {seed}: RMSE={metrics['rmse']:.4f}, "
                           f"SNR={metrics['snr']:.2f}dB, Corr={metrics['correlation']:.4f}")

        # Aggregate ICA-MoE results
        results["ica_moe"]["aggregated"] = {
            k: {"mean": v[0], "std": v[1]}
            for k, v in aggregate_metrics(ica_moe_metrics_list).items()
        }

        # Run baselines if provided
        if baseline_configs:
            for name, baseline_config in baseline_configs.items():
                baseline_metrics_list = []
                results["baselines"][name] = {}

                for seed in self.config.seeds:
                    clean_eeg, eog_norm, emg_norm, noisy_eeg, eeg_means, eeg_stds = \
                        self.preprocess(eeg_data, eog_data, emg_data)

                    try:
                        denoised, metrics = self.run_baseline(
                            baseline_config, noisy_eeg, clean_eeg,
                            eog_norm, emg_norm, eeg_stds, eeg_means, seed
                        )

                        results["baselines"][name][f"seed_{seed}"] = {
                            "metrics": metrics
                        }
                        baseline_metrics_list.append(metrics)

                        self.logger.info(f"{name} seed {seed}: RMSE={metrics['rmse']:.4f}, "
                                       f"SNR={metrics['snr']:.2f}dB")

                    except Exception as e:
                        self.logger.error(f"Error running {name} with seed {seed}: {e}")
                        results["baselines"][name][f"seed_{seed}"] = {"error": str(e)}

                # Aggregate baseline results
                if baseline_metrics_list:
                    results["baselines"][name]["aggregated"] = {
                        k: {"mean": v[0], "std": v[1]}
                        for k, v in aggregate_metrics(baseline_metrics_list).items()
                    }

        # Save final results
        self._save_results(results)

        return results

    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        results_file = self.results_dir / f"{self.config.name}_results.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_serializable(results)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {results_file}")


def run_multi_seed_experiment(experiment_type: str,
                              eeg_data: np.ndarray,
                              eog_data: np.ndarray,
                              emg_data: np.ndarray,
                              seeds: List[int] = None,
                              include_baselines: bool = True,
                              base_dir: str = ".") -> Dict:
    """
    Run experiment with multiple seeds.

    Args:
        experiment_type: One of 'eeg_eog', 'eeg_emg', 'eeg_eog_emg'
        eeg_data: Clean EEG data
        eog_data: EOG data
        emg_data: EMG data
        seeds: List of seeds to use (default: EXPERIMENT_SEEDS)
        include_baselines: Whether to run baseline models
        base_dir: Base directory for results

    Returns:
        Dictionary with experiment results
    """
    from .experiment_configs import (
        EEG_EOG_CONFIG, EEG_EMG_CONFIG, EEG_EOG_EMG_CONFIG,
        BASELINE_CONFIGS
    )

    # Select configuration
    if experiment_type == "eeg_eog":
        config = EEG_EOG_CONFIG
    elif experiment_type == "eeg_emg":
        config = EEG_EMG_CONFIG
    elif experiment_type == "eeg_eog_emg":
        config = EEG_EOG_EMG_CONFIG
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Override seeds if provided
    if seeds is not None:
        config.seeds = seeds

    # Create runner
    runner = ExperimentRunner(config, base_dir)

    # Run experiment
    baseline_configs = BASELINE_CONFIGS if include_baselines else None
    results = runner.run_experiment(eeg_data, eog_data, emg_data, baseline_configs)

    return results


def compare_all_experiments(base_dir: str = ".") -> Dict:
    """
    Compare results across all experiment types.

    Args:
        base_dir: Base directory containing results

    Returns:
        Dictionary with comparison results
    """
    base_path = Path(base_dir)
    comparison = {}

    for exp_type in ["eeg_eog", "eeg_emg", "eeg_eog_emg"]:
        results_file = base_path / f"results/{exp_type}/{exp_type.upper()}_results.json"

        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                comparison[exp_type] = {
                    "ica_moe": results.get("ica_moe", {}).get("aggregated", {}),
                    "baselines": {
                        name: data.get("aggregated", {})
                        for name, data in results.get("baselines", {}).items()
                    }
                }

    return comparison
