#!/usr/bin/env python3
"""
Main experiment runner script for EEG denoising experiments.

This script runs all experiments (EEG+EOG, EEG+EMG, EEG+EOG+EMG) with multiple seeds
and generates comparison figures.

Usage:
    python run_experiments.py --experiment all --seeds 40 41 42 43 44
    python run_experiments.py --experiment eeg_eog --seeds 42
    python run_experiments.py --experiment eeg_emg --baselines
"""
import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.seed_utils import set_all_seeds, EXPERIMENT_SEEDS
from src.utils.data_utils import generate_synthetic_data, preprocess_data
from src.experiments.experiment_runner import ExperimentRunner, run_multi_seed_experiment
from src.experiments.experiment_configs import (
    EEG_EOG_CONFIG, EEG_EMG_CONFIG, EEG_EOG_EMG_CONFIG,
    BASELINE_CONFIGS, get_all_experiment_configs
)
from src.visualization.signal_plots import (
    plot_signal_comparison, plot_psd_comparison, plot_frequency_analysis
)
from src.visualization.results_plots import (
    plot_metrics_comparison, plot_multi_experiment_comparison,
    create_summary_figure, generate_latex_table
)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )


def load_data(data_path: str = None) -> dict:
    """
    Load EEG, EOG, and EMG data.

    Args:
        data_path: Path to data directory. If None, generates synthetic data.

    Returns:
        Dictionary with 'eeg', 'eog', 'emg' arrays
    """
    if data_path and Path(data_path).exists():
        data_dir = Path(data_path)

        # Try to load numpy files
        data = {}
        for name in ['eeg', 'eog', 'emg']:
            file_path = data_dir / f"{name}.npy"
            if file_path.exists():
                data[name] = np.load(file_path)
                logging.info(f"Loaded {name} data from {file_path}")
            else:
                logging.warning(f"{name} data not found at {file_path}")

        if all(k in data for k in ['eeg', 'eog', 'emg']):
            return data

    # Generate synthetic data if real data not available
    logging.info("Generating synthetic data for experiments...")
    return generate_synthetic_data(n_samples=200, seq_length=512, seed=42)


def run_single_experiment(experiment_type: str, data: dict, seeds: list,
                          include_baselines: bool, base_dir: str) -> dict:
    """
    Run a single experiment type.

    Args:
        experiment_type: One of 'eeg_eog', 'eeg_emg', 'eeg_eog_emg'
        data: Data dictionary
        seeds: List of seeds
        include_baselines: Whether to run baselines
        base_dir: Base directory for results

    Returns:
        Results dictionary
    """
    logging.info(f"Running {experiment_type} experiment with seeds {seeds}")

    results = run_multi_seed_experiment(
        experiment_type=experiment_type,
        eeg_data=data['eeg'],
        eog_data=data['eog'],
        emg_data=data['emg'],
        seeds=seeds,
        include_baselines=include_baselines,
        base_dir=base_dir
    )

    return results


def generate_visualizations(results: dict, data: dict, experiment_type: str,
                            output_dir: Path):
    """
    Generate visualization figures for experiment results.

    Args:
        results: Experiment results
        data: Data dictionary
        experiment_type: Experiment type
        output_dir: Output directory for figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess data for visualization
    from src.utils.data_utils import preprocess_data, preprocess_eeg_eog, preprocess_eeg_emg

    if experiment_type == 'eeg_eog':
        clean, eog, emg, noisy, means, stds = preprocess_eeg_eog(data['eeg'], data['eog'])
    elif experiment_type == 'eeg_emg':
        clean, eog, emg, noisy, means, stds = preprocess_eeg_emg(data['eeg'], data['emg'])
    else:
        clean, eog, emg, noisy, means, stds = preprocess_data(data['eeg'], data['eog'], data['emg'])

    # Denormalize for plotting
    clean_denorm = clean * stds + means
    noisy_denorm = noisy * stds + means

    # Load denoised data if available
    denoised_file = output_dir.parent / f"ica_moe_denoised_seed{EXPERIMENT_SEEDS[0]}.npy"
    if denoised_file.exists():
        denoised = np.load(denoised_file)
    else:
        # Use clean as placeholder
        denoised = clean_denorm.copy()
        logging.warning(f"Denoised data not found, using clean signal as placeholder")

    # Generate figures
    logging.info("Generating visualization figures...")

    # Signal comparison
    try:
        fig = plot_signal_comparison(
            clean_denorm, noisy_denorm, denoised,
            title=f"EEG Signal Comparison - {experiment_type.upper()}",
            save_path=str(output_dir / f"{experiment_type}_signal_comparison.png")
        )
        logging.info(f"Saved signal comparison figure")
    except Exception as e:
        logging.error(f"Failed to generate signal comparison: {e}")

    # Metrics comparison
    try:
        fig = plot_metrics_comparison(
            results, metric='rmse',
            title=f"RMSE Comparison - {experiment_type.upper()}",
            save_path=str(output_dir / f"{experiment_type}_rmse_comparison.png")
        )
        logging.info(f"Saved RMSE comparison figure")
    except Exception as e:
        logging.error(f"Failed to generate metrics comparison: {e}")

    # Frequency analysis
    try:
        fig = plot_frequency_analysis(
            clean_denorm,
            title=f"Frequency Analysis - {experiment_type.upper()}",
            save_path=str(output_dir / f"{experiment_type}_frequency_analysis.png")
        )
        logging.info(f"Saved frequency analysis figure")
    except Exception as e:
        logging.error(f"Failed to generate frequency analysis: {e}")

    # Summary figure
    try:
        fig = create_summary_figure(
            results, clean_denorm, noisy_denorm, denoised,
            title=f"EEG Denoising Summary - {experiment_type.upper()}",
            save_path=str(output_dir / f"{experiment_type}_summary.png")
        )
        logging.info(f"Saved summary figure")
    except Exception as e:
        logging.error(f"Failed to generate summary figure: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run EEG denoising experiments with ICA-MoE model"
    )
    parser.add_argument(
        '--experiment', '-e',
        choices=['eeg_eog', 'eeg_emg', 'eeg_eog_emg', 'all'],
        default='all',
        help='Experiment type to run'
    )
    parser.add_argument(
        '--seeds', '-s',
        nargs='+',
        type=int,
        default=EXPERIMENT_SEEDS,
        help='Random seeds for experiments'
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Output directory for results'
    )
    parser.add_argument(
        '--baselines', '-b',
        action='store_true',
        help='Include baseline model comparisons'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization figures'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--latex-table',
        action='store_true',
        help='Generate LaTeX table for results'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logging.info("="*60)
    logging.info("EEG Denoising Experiment Runner")
    logging.info("="*60)

    # Load data
    data = load_data(args.data_path)
    logging.info(f"Data loaded: EEG shape={data['eeg'].shape}, "
                f"EOG shape={data['eog'].shape}, EMG shape={data['emg'].shape}")

    # Determine experiments to run
    if args.experiment == 'all':
        experiments = ['eeg_eog', 'eeg_emg', 'eeg_eog_emg']
    else:
        experiments = [args.experiment]

    # Run experiments
    all_results = {}
    for exp_type in experiments:
        logging.info(f"\n{'='*40}")
        logging.info(f"Running {exp_type.upper()} experiment")
        logging.info(f"{'='*40}")

        results = run_single_experiment(
            experiment_type=exp_type,
            data=data,
            seeds=args.seeds,
            include_baselines=args.baselines,
            base_dir=args.output_dir
        )
        all_results[exp_type] = results

        # Generate visualizations
        if args.visualize:
            output_dir = Path(args.output_dir) / 'figures' / exp_type
            generate_visualizations(results, data, exp_type, output_dir)

    # Generate multi-experiment comparison
    if len(experiments) > 1 and args.visualize:
        logging.info("\nGenerating multi-experiment comparison...")
        output_dir = Path(args.output_dir) / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig = plot_multi_experiment_comparison(
                all_results,
                title="Comparison Across Noise Types",
                save_path=str(output_dir / "multi_experiment_comparison.png")
            )
            logging.info("Saved multi-experiment comparison figure")
        except Exception as e:
            logging.error(f"Failed to generate comparison: {e}")

    # Generate LaTeX tables
    if args.latex_table:
        logging.info("\nGenerating LaTeX tables...")
        output_dir = Path(args.output_dir) / 'tables'
        output_dir.mkdir(parents=True, exist_ok=True)

        for exp_type, results in all_results.items():
            latex_code = generate_latex_table(
                results,
                caption=f"Performance comparison for {exp_type.upper()} experiment",
                label=f"tab:{exp_type}_results"
            )
            table_file = output_dir / f"{exp_type}_results.tex"
            with open(table_file, 'w') as f:
                f.write(latex_code)
            logging.info(f"Saved LaTeX table to {table_file}")

    # Print summary
    logging.info("\n" + "="*60)
    logging.info("EXPERIMENT SUMMARY")
    logging.info("="*60)

    for exp_type, results in all_results.items():
        logging.info(f"\n{exp_type.upper()}:")
        ica_moe_agg = results.get('ica_moe', {}).get('aggregated', {})
        if ica_moe_agg:
            for metric, values in ica_moe_agg.items():
                if isinstance(values, dict) and 'mean' in values:
                    logging.info(f"  ICA-MoE {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")

    logging.info("\n" + "="*60)
    logging.info("Experiments completed!")
    logging.info("="*60)


if __name__ == "__main__":
    main()
