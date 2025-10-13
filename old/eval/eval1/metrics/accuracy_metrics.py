#!/usr/bin/env python3
"""
ChronoTick Evaluation 1 - Accuracy Metrics

Calculate prediction accuracy metrics including MAE, RMSE, and percentile errors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AccuracyMetricsCalculator:
    """Calculate accuracy metrics for ChronoTick predictions vs ground truth"""

    def __init__(self):
        self.results = {}

    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Load measurement dataset from CSV"""
        try:
            if str(dataset_path).endswith('.gz'):
                df = pd.read_csv(dataset_path, compression='gzip')
            else:
                df = pd.read_csv(dataset_path)

            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

            # Filter out invalid measurements
            valid_mask = (
                df['ground_truth_offset'].notna() &
                (df['measurement_uncertainty'] < 0.1) &  # < 100ms uncertainty
                (abs(df['clock_offset']) < 10.0)  # < 10s offset
            )

            filtered_df = df[valid_mask].copy()
            logger.info(f"Loaded {len(df)} measurements, {len(filtered_df)} valid")

            return filtered_df

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            raise

    def calculate_prediction_errors(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate various error metrics"""
        errors = predictions - ground_truth

        metrics = {
            # Basic metrics
            'mae_seconds': np.mean(np.abs(errors)),
            'rmse_seconds': np.sqrt(np.mean(errors**2)),
            'mean_error_seconds': np.mean(errors),
            'std_error_seconds': np.std(errors),

            # Convert to microseconds for readability
            'mae_microseconds': np.mean(np.abs(errors)) * 1e6,
            'rmse_microseconds': np.sqrt(np.mean(errors**2)) * 1e6,
            'mean_error_microseconds': np.mean(errors) * 1e6,
            'std_error_microseconds': np.std(errors) * 1e6,

            # Percentile errors
            'p50_error_microseconds': np.percentile(np.abs(errors), 50) * 1e6,
            'p90_error_microseconds': np.percentile(np.abs(errors), 90) * 1e6,
            'p95_error_microseconds': np.percentile(np.abs(errors), 95) * 1e6,
            'p99_error_microseconds': np.percentile(np.abs(errors), 99) * 1e6,

            # Maximum errors
            'max_abs_error_microseconds': np.max(np.abs(errors)) * 1e6,
            'max_positive_error_microseconds': np.max(errors) * 1e6,
            'max_negative_error_microseconds': np.min(errors) * 1e6,

            # Count statistics
            'total_predictions': len(predictions),
            'error_count': np.sum(np.isnan(errors)),
        }

        return metrics

    def analyze_model_performance(self, df: pd.DataFrame, model_predictions: np.ndarray,
                                model_name: str = "ChronoTick") -> Dict[str, float]:
        """Analyze performance of a specific model"""
        ground_truth = df['ground_truth_offset'].values

        # Calculate basic accuracy metrics
        accuracy_metrics = self.calculate_prediction_errors(model_predictions, ground_truth)

        # Add model-specific metrics
        accuracy_metrics['model_name'] = model_name

        # Calculate improvement vs raw NTP
        ntp_errors = df['clock_offset'].values - ground_truth
        ntp_mae = np.mean(np.abs(ntp_errors))
        model_mae = accuracy_metrics['mae_seconds']

        accuracy_metrics['improvement_vs_ntp'] = (ntp_mae - model_mae) / ntp_mae * 100
        accuracy_metrics['ntp_baseline_mae_microseconds'] = ntp_mae * 1e6

        return accuracy_metrics

    def compare_baseline_methods(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compare different baseline methods"""
        ground_truth = df['ground_truth_offset'].values

        baselines = {
            'raw_ntp': df['clock_offset'].values,
            'system_clock': np.zeros_like(ground_truth),  # Assume system clock = 0 offset
        }

        # Add simple moving average if we have enough data
        if len(df) > 10:
            baselines['moving_average'] = df['clock_offset'].rolling(window=10, center=True).mean().fillna(0).values

        results = {}
        for baseline_name, predictions in baselines.items():
            if len(predictions) == len(ground_truth):
                results[baseline_name] = self.calculate_prediction_errors(predictions, ground_truth)

        return results

    def analyze_error_distribution(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """Analyze the distribution of prediction errors"""
        errors = (predictions - ground_truth) * 1e6  # Convert to microseconds

        # Remove outliers for distribution analysis (beyond 3 sigma)
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        mask = np.abs(errors - mean_err) <= 3 * std_err
        clean_errors = errors[mask]

        distribution_metrics = {
            'skewness': float(pd.Series(clean_errors).skew()),
            'kurtosis': float(pd.Series(clean_errors).kurtosis()),
            'outlier_percentage': (len(errors) - len(clean_errors)) / len(errors) * 100,

            # Error bounds analysis
            'within_1us_percentage': np.sum(np.abs(errors) <= 1.0) / len(errors) * 100,
            'within_10us_percentage': np.sum(np.abs(errors) <= 10.0) / len(errors) * 100,
            'within_100us_percentage': np.sum(np.abs(errors) <= 100.0) / len(errors) * 100,
            'within_1ms_percentage': np.sum(np.abs(errors) <= 1000.0) / len(errors) * 100,
        }

        return distribution_metrics

    def generate_accuracy_plots(self, df: pd.DataFrame, predictions: np.ndarray,
                              output_dir: Path, model_name: str = "ChronoTick"):
        """Generate accuracy visualization plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ground_truth = df['ground_truth_offset'].values
        errors = (predictions - ground_truth) * 1e6  # Microseconds

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)

        # 1. Error distribution histogram
        plt.figure(figsize=fig_size)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
        plt.axvline(np.mean(errors), color='green', linestyle='--', label=f'Mean Error: {np.mean(errors):.1f}μs')
        plt.xlabel('Prediction Error (μs)')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Prediction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{model_name}_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Error vs Time plot
        plt.figure(figsize=fig_size)
        time_hours = (df['timestamp'] - df['timestamp'].iloc[0]) / 3600
        plt.scatter(time_hours, errors, alpha=0.5, s=1)
        plt.axhline(0, color='red', linestyle='--', label='Perfect Prediction')
        plt.xlabel('Time (hours)')
        plt.ylabel('Prediction Error (μs)')
        plt.title(f'{model_name} - Prediction Error vs Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{model_name}_error_vs_time.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Cumulative error distribution
        plt.figure(figsize=fig_size)
        sorted_abs_errors = np.sort(np.abs(errors))
        percentiles = np.linspace(0, 100, len(sorted_abs_errors))
        plt.plot(percentiles, sorted_abs_errors)
        plt.axhline(10, color='red', linestyle='--', label='10μs Target')
        plt.axvline(95, color='green', linestyle='--', label='95th Percentile')
        plt.xlabel('Percentile')
        plt.ylabel('Absolute Error (μs)')
        plt.title(f'{model_name} - Cumulative Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{model_name}_cumulative_error.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Prediction vs Ground Truth scatter
        plt.figure(figsize=fig_size)
        plt.scatter(ground_truth * 1e6, predictions * 1e6, alpha=0.5, s=1)
        min_val = min(np.min(ground_truth), np.min(predictions)) * 1e6
        max_val = max(np.max(ground_truth), np.max(predictions)) * 1e6
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        plt.xlabel('Ground Truth Offset (μs)')
        plt.ylabel('Predicted Offset (μs)')
        plt.title(f'{model_name} - Prediction vs Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f'{model_name}_prediction_vs_truth.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Accuracy plots saved to {output_dir}")

    def save_metrics_report(self, metrics: Dict, output_path: Path):
        """Save metrics to JSON file"""
        import json

        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_metrics = {k: convert_numpy(v) for k, v in metrics.items()}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        logger.info(f"Metrics report saved to {output_path}")

    def run_full_accuracy_analysis(self, dataset_path: Path, predictions: np.ndarray,
                                 output_dir: Path, model_name: str = "ChronoTick") -> Dict:
        """Run complete accuracy analysis"""
        logger.info(f"Running accuracy analysis for {model_name}")

        # Load dataset
        df = self.load_dataset(dataset_path)

        # Ensure predictions match dataset length
        if len(predictions) != len(df):
            logger.warning(f"Predictions length ({len(predictions)}) != dataset length ({len(df)})")
            min_len = min(len(predictions), len(df))
            predictions = predictions[:min_len]
            df = df.iloc[:min_len]

        # Calculate accuracy metrics
        accuracy_metrics = self.analyze_model_performance(df, predictions, model_name)

        # Analyze error distribution
        distribution_metrics = self.analyze_error_distribution(predictions, df['ground_truth_offset'].values)

        # Compare with baselines
        baseline_metrics = self.compare_baseline_methods(df)

        # Generate plots
        self.generate_accuracy_plots(df, predictions, output_dir, model_name)

        # Combine all results
        results = {
            'model_accuracy': accuracy_metrics,
            'error_distribution': distribution_metrics,
            'baseline_comparison': baseline_metrics,
            'analysis_metadata': {
                'dataset_path': str(dataset_path),
                'model_name': model_name,
                'total_measurements': len(df),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }

        # Save report
        report_path = output_dir / f'{model_name}_accuracy_report.json'
        self.save_metrics_report(results, report_path)

        return results


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate ChronoTick accuracy metrics")
    parser.add_argument('--dataset', type=Path, required=True, help='Dataset CSV file')
    parser.add_argument('--predictions', type=Path, required=True, help='Predictions numpy file')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--model-name', default='ChronoTick', help='Model name')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load predictions
    predictions = np.load(args.predictions)

    # Run analysis
    calculator = AccuracyMetricsCalculator()
    results = calculator.run_full_accuracy_analysis(
        args.dataset, predictions, args.output, args.model_name
    )

    print(f"Accuracy analysis complete. Results saved to {args.output}")
    print(f"MAE: {results['model_accuracy']['mae_microseconds']:.2f} μs")
    print(f"RMSE: {results['model_accuracy']['rmse_microseconds']:.2f} μs")
    print(f"95th percentile error: {results['model_accuracy']['p95_error_microseconds']:.2f} μs")


if __name__ == "__main__":
    main()