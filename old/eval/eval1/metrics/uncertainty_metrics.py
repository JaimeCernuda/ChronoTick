#!/usr/bin/env python3
"""
ChronoTick Evaluation 1 - Uncertainty Metrics

Analyze uncertainty calibration and quantification performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class UncertaintyMetricsCalculator:
    """Calculate uncertainty calibration and coverage metrics"""

    def __init__(self):
        self.results = {}

    def calculate_coverage_probability(self, predictions: np.ndarray, uncertainties: np.ndarray,
                                     ground_truth: np.ndarray, confidence_levels: List[float]) -> Dict[str, float]:
        """Calculate empirical coverage probability for different confidence levels"""
        coverage_results = {}

        for conf_level in confidence_levels:
            # Calculate confidence intervals
            z_score = stats.norm.ppf((1 + conf_level) / 2)  # Two-sided confidence interval
            lower_bounds = predictions - z_score * uncertainties
            upper_bounds = predictions + z_score * uncertainties

            # Check if ground truth falls within bounds
            within_bounds = (ground_truth >= lower_bounds) & (ground_truth <= upper_bounds)
            empirical_coverage = np.mean(within_bounds)

            coverage_results[f'coverage_{int(conf_level*100)}'] = empirical_coverage
            coverage_results[f'expected_{int(conf_level*100)}'] = conf_level
            coverage_results[f'miscalibration_{int(conf_level*100)}'] = abs(empirical_coverage - conf_level)

        return coverage_results

    def calculate_uncertainty_quality(self, predictions: np.ndarray, uncertainties: np.ndarray,
                                    ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate uncertainty quality metrics"""
        errors = np.abs(predictions - ground_truth)

        # Sharpness: average uncertainty width
        mean_uncertainty = np.mean(uncertainties)

        # Reliability: correlation between uncertainty and actual error
        uncertainty_error_correlation = np.corrcoef(uncertainties, errors)[0, 1]

        # Calibration error (expected calibration error)
        # Bin uncertainties and check coverage in each bin
        n_bins = 10
        bin_boundaries = np.linspace(0, np.max(uncertainties), n_bins + 1)
        calibration_errors = []

        for i in range(n_bins):
            bin_mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 5:  # At least 5 samples in bin
                bin_uncertainties = uncertainties[bin_mask]
                bin_errors = errors[bin_mask]
                bin_predictions = predictions[bin_mask]
                bin_ground_truth = ground_truth[bin_mask]

                # Expected vs empirical coverage for 68% confidence (1 sigma)
                within_1sigma = np.abs(bin_ground_truth - bin_predictions) <= bin_uncertainties
                empirical_coverage = np.mean(within_1sigma)
                expected_coverage = 0.68  # 1 sigma
                calibration_errors.append(abs(empirical_coverage - expected_coverage))

        expected_calibration_error = np.mean(calibration_errors) if calibration_errors else np.nan

        # Uncertainty reduction vs baseline
        baseline_uncertainty = np.std(predictions - ground_truth)  # Use prediction std as baseline
        uncertainty_reduction = (baseline_uncertainty - mean_uncertainty) / baseline_uncertainty

        return {
            'mean_uncertainty_seconds': mean_uncertainty,
            'mean_uncertainty_microseconds': mean_uncertainty * 1e6,
            'uncertainty_error_correlation': uncertainty_error_correlation,
            'expected_calibration_error': expected_calibration_error,
            'uncertainty_reduction_vs_baseline': uncertainty_reduction,
            'sharpness_score': 1.0 / mean_uncertainty if mean_uncertainty > 0 else 0,  # Higher is better
        }

    def analyze_uncertainty_vs_error(self, predictions: np.ndarray, uncertainties: np.ndarray,
                                   ground_truth: np.ndarray) -> Dict[str, float]:
        """Analyze relationship between predicted uncertainty and actual errors"""
        errors = np.abs(predictions - ground_truth)

        # Rank correlation (Spearman) - more robust than Pearson
        spearman_corr, spearman_p = stats.spearmanr(uncertainties, errors)

        # Mutual information
        # Discretize for mutual information calculation
        n_bins = 20
        uncertainty_bins = pd.cut(uncertainties, bins=n_bins, labels=False)
        error_bins = pd.cut(errors, bins=n_bins, labels=False)

        # Calculate mutual information using contingency table
        contingency = pd.crosstab(uncertainty_bins, error_bins)
        mutual_info = 0.0
        total_samples = len(uncertainties)

        for i in range(n_bins):
            for j in range(n_bins):
                if contingency.iloc[i, j] > 0:
                    p_ij = contingency.iloc[i, j] / total_samples
                    p_i = np.sum(contingency.iloc[i, :]) / total_samples
                    p_j = np.sum(contingency.iloc[:, j]) / total_samples
                    mutual_info += p_ij * np.log(p_ij / (p_i * p_j))

        # Quantile-based analysis
        uncertainty_quantiles = np.percentile(uncertainties, [25, 50, 75, 90, 95])
        error_quantiles = np.percentile(errors, [25, 50, 75, 90, 95])

        return {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'mutual_information': mutual_info,
            'uncertainty_q25': uncertainty_quantiles[0] * 1e6,
            'uncertainty_q50': uncertainty_quantiles[1] * 1e6,
            'uncertainty_q75': uncertainty_quantiles[2] * 1e6,
            'uncertainty_q90': uncertainty_quantiles[3] * 1e6,
            'uncertainty_q95': uncertainty_quantiles[4] * 1e6,
            'error_q25': error_quantiles[0] * 1e6,
            'error_q50': error_quantiles[1] * 1e6,
            'error_q75': error_quantiles[2] * 1e6,
            'error_q90': error_quantiles[3] * 1e6,
            'error_q95': error_quantiles[4] * 1e6,
        }

    def calculate_ntp_baseline_uncertainty(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline uncertainty metrics from NTP measurements"""
        ntp_uncertainties = df['measurement_uncertainty'].values
        ntp_delays = df['ntp_delay'].values

        return {
            'ntp_mean_uncertainty_microseconds': np.mean(ntp_uncertainties) * 1e6,
            'ntp_std_uncertainty_microseconds': np.std(ntp_uncertainties) * 1e6,
            'ntp_mean_delay_microseconds': np.mean(ntp_delays) * 1e6,
            'ntp_median_uncertainty_microseconds': np.median(ntp_uncertainties) * 1e6,
            'ntp_p95_uncertainty_microseconds': np.percentile(ntp_uncertainties, 95) * 1e6,
        }

    def generate_uncertainty_plots(self, predictions: np.ndarray, uncertainties: np.ndarray,
                                 ground_truth: np.ndarray, output_dir: Path, model_name: str = "ChronoTick"):
        """Generate uncertainty analysis plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        errors = np.abs(predictions - ground_truth) * 1e6  # Microseconds
        uncertainties_us = uncertainties * 1e6  # Microseconds

        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)

        # 1. Uncertainty vs Error scatter plot
        plt.figure(figsize=fig_size)
        plt.scatter(uncertainties_us, errors, alpha=0.5, s=1)
        plt.plot([0, np.max(uncertainties_us)], [0, np.max(uncertainties_us)], 'r--',
                label='Perfect Calibration (error = uncertainty)')
        plt.xlabel('Predicted Uncertainty (μs)')
        plt.ylabel('Actual Error (μs)')
        plt.title(f'{model_name} - Uncertainty vs Actual Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{model_name}_uncertainty_vs_error.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Calibration plot
        plt.figure(figsize=fig_size)
        confidence_levels = np.arange(0.1, 1.0, 0.05)
        coverage_probs = []

        for conf_level in confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            within_bounds = np.abs(predictions - ground_truth) <= z_score * uncertainties
            coverage_probs.append(np.mean(within_bounds))

        plt.plot(confidence_levels, coverage_probs, 'b-', label='Empirical Coverage')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Expected Coverage Probability')
        plt.ylabel('Empirical Coverage Probability')
        plt.title(f'{model_name} - Uncertainty Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{model_name}_calibration.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Uncertainty distribution
        plt.figure(figsize=fig_size)
        plt.hist(uncertainties_us, bins=50, alpha=0.7, edgecolor='black', density=True)
        plt.axvline(np.mean(uncertainties_us), color='red', linestyle='--',
                   label=f'Mean: {np.mean(uncertainties_us):.1f}μs')
        plt.axvline(np.median(uncertainties_us), color='green', linestyle='--',
                   label=f'Median: {np.median(uncertainties_us):.1f}μs')
        plt.xlabel('Predicted Uncertainty (μs)')
        plt.ylabel('Density')
        plt.title(f'{model_name} - Uncertainty Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{model_name}_uncertainty_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Reliability diagram (binned calibration plot)
        plt.figure(figsize=fig_size)
        n_bins = 10
        bin_boundaries = np.linspace(0, np.max(uncertainties_us), n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        empirical_coverage = []
        bin_counts = []

        for i in range(n_bins):
            bin_mask = (uncertainties_us >= bin_boundaries[i]) & (uncertainties_us < bin_boundaries[i + 1])
            if np.sum(bin_mask) > 0:
                bin_errors = errors[bin_mask]
                bin_uncertainties = uncertainties_us[bin_mask]
                # 68% confidence interval (1 sigma)
                within_1sigma = bin_errors <= bin_uncertainties
                empirical_coverage.append(np.mean(within_1sigma))
                bin_counts.append(np.sum(bin_mask))
            else:
                empirical_coverage.append(0)
                bin_counts.append(0)

        # Plot reliability diagram
        bar_widths = (bin_boundaries[1:] - bin_boundaries[:-1]) * 0.8
        plt.bar(bin_centers, empirical_coverage, width=bar_widths, alpha=0.7,
               label='Empirical Coverage')
        plt.axhline(0.68, color='red', linestyle='--', label='Expected Coverage (68%)')
        plt.xlabel('Predicted Uncertainty Bin (μs)')
        plt.ylabel('Empirical Coverage Probability')
        plt.title(f'{model_name} - Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{model_name}_reliability_diagram.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Uncertainty plots saved to {output_dir}")

    def run_full_uncertainty_analysis(self, dataset_path: Path, predictions: np.ndarray,
                                    uncertainties: np.ndarray, output_dir: Path,
                                    model_name: str = "ChronoTick") -> Dict:
        """Run complete uncertainty analysis"""
        logger.info(f"Running uncertainty analysis for {model_name}")

        # Load dataset
        if str(dataset_path).endswith('.gz'):
            df = pd.read_csv(dataset_path, compression='gzip')
        else:
            df = pd.read_csv(dataset_path)

        # Filter valid measurements
        valid_mask = (
            df['ground_truth_offset'].notna() &
            (df['measurement_uncertainty'] < 0.1) &
            (abs(df['clock_offset']) < 10.0)
        )
        df = df[valid_mask].copy()
        ground_truth = df['ground_truth_offset'].values

        # Ensure arrays match
        min_len = min(len(predictions), len(uncertainties), len(ground_truth))
        predictions = predictions[:min_len]
        uncertainties = uncertainties[:min_len]
        ground_truth = ground_truth[:min_len]

        # Calculate coverage probabilities
        confidence_levels = [0.68, 0.90, 0.95, 0.99]
        coverage_metrics = self.calculate_coverage_probability(
            predictions, uncertainties, ground_truth, confidence_levels
        )

        # Calculate uncertainty quality metrics
        quality_metrics = self.calculate_uncertainty_quality(
            predictions, uncertainties, ground_truth
        )

        # Analyze uncertainty vs error relationship
        relationship_metrics = self.analyze_uncertainty_vs_error(
            predictions, uncertainties, ground_truth
        )

        # Calculate NTP baseline
        ntp_baseline = self.calculate_ntp_baseline_uncertainty(df)

        # Generate plots
        self.generate_uncertainty_plots(
            predictions, uncertainties, ground_truth, output_dir, model_name
        )

        # Combine results
        results = {
            'coverage_metrics': coverage_metrics,
            'uncertainty_quality': quality_metrics,
            'uncertainty_error_relationship': relationship_metrics,
            'ntp_baseline': ntp_baseline,
            'analysis_metadata': {
                'dataset_path': str(dataset_path),
                'model_name': model_name,
                'total_measurements': len(predictions),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }

        # Save report
        import json
        report_path = output_dir / f'{model_name}_uncertainty_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_results = {k: convert_numpy(v) if isinstance(v, dict)
                              else {sk: convert_numpy(sv) for sk, sv in v.items()}
                              for k, v in results.items()}

        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Uncertainty analysis complete. Report saved to {report_path}")

        return results


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate ChronoTick uncertainty metrics")
    parser.add_argument('--dataset', type=Path, required=True, help='Dataset CSV file')
    parser.add_argument('--predictions', type=Path, required=True, help='Predictions numpy file')
    parser.add_argument('--uncertainties', type=Path, required=True, help='Uncertainties numpy file')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--model-name', default='ChronoTick', help='Model name')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load predictions and uncertainties
    predictions = np.load(args.predictions)
    uncertainties = np.load(args.uncertainties)

    # Run analysis
    calculator = UncertaintyMetricsCalculator()
    results = calculator.run_full_uncertainty_analysis(
        args.dataset, predictions, uncertainties, args.output, args.model_name
    )

    print(f"Uncertainty analysis complete. Results saved to {args.output}")
    print(f"Coverage 95%: {results['coverage_metrics']['coverage_95']:.3f}")
    print(f"Mean uncertainty: {results['uncertainty_quality']['mean_uncertainty_microseconds']:.2f} μs")
    print(f"Uncertainty-error correlation: {results['uncertainty_quality']['uncertainty_error_correlation']:.3f}")


if __name__ == "__main__":
    main()