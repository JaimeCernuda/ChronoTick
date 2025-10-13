#!/usr/bin/env python3
"""
ChronoTick Evaluation 1 - Dataset Replay Experiment

Replay historical datasets through ChronoTick and compare predictions with ground truth.
"""

import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import threading
import queue
from dataclasses import dataclass

# Add ChronoTick to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tsfm"))

# Import ChronoTick components
try:
    from chronotick_inference.real_data_pipeline import RealDataPipeline
    from chronotick_inference.ntp_client import NTPMeasurement
    from chronotick_inference.engine import ChronoTickInferenceEngine
    from chronotick_inference.config_selector import ConfigSelector
    CHRONOTICK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ChronoTick components not available: {e}")
    CHRONOTICK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Results from dataset replay"""
    timestamp: float
    prediction: float
    uncertainty: float
    ground_truth: float
    input_offset: float
    model_used: str
    inference_time_ms: float


class DatasetReplayer:
    """Replay historical dataset through ChronoTick inference pipeline"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "tsfm" / "chronotick_inference" / "config.yaml"
        self.results = []
        self.engine = None
        self.pipeline = None

    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        """Load and prepare dataset for replay"""
        logger.info(f"Loading dataset from {dataset_path}")

        if str(dataset_path).endswith('.gz'):
            df = pd.read_csv(dataset_path, compression='gzip')
        else:
            df = pd.read_csv(dataset_path)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Filter valid measurements
        valid_mask = (
            df['ground_truth_offset'].notna() &
            (df['measurement_uncertainty'] < 0.1) &  # < 100ms uncertainty
            (abs(df['clock_offset']) < 10.0) &  # < 10s offset
            df['quality_flags'].str.len() < 50  # Basic quality filter
        )

        filtered_df = df[valid_mask].copy()
        logger.info(f"Loaded {len(df)} measurements, {len(filtered_df)} valid for replay")

        return filtered_df

    def initialize_chronotick(self, model_name: str = "chronos"):
        """Initialize ChronoTick inference engine"""
        if not CHRONOTICK_AVAILABLE:
            raise RuntimeError("ChronoTick components not available")

        logger.info(f"Initializing ChronoTick with model: {model_name}")

        try:
            # Load configuration
            import yaml
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

            # Override model selection
            config['models']['primary_model'] = model_name

            # Initialize components
            config_selector = ConfigSelector(config)
            selected_config = config_selector.select_optimal_config()

            self.engine = ChronoTickInferenceEngine(selected_config)
            self.pipeline = RealDataPipeline(config)

            logger.info("ChronoTick initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ChronoTick: {e}")
            raise

    def create_ntp_measurement(self, row: pd.Series) -> NTPMeasurement:
        """Create NTP measurement from dataset row"""
        return NTPMeasurement(
            offset=row['clock_offset'],
            delay=row['ntp_delay'],
            stratum=int(row['ntp_stratum']),
            precision=row['measurement_uncertainty'],
            server=row['ntp_server'],
            timestamp=row['timestamp'],
            uncertainty=row['measurement_uncertainty']
        )

    def replay_measurement(self, row: pd.Series, warmup_complete: bool = True) -> Optional[ReplayResult]:
        """Replay single measurement through ChronoTick"""
        try:
            # Create NTP measurement
            ntp_measurement = self.create_ntp_measurement(row)

            # Feed to pipeline
            start_time = time.time()

            if warmup_complete:
                # Get prediction from trained model
                correction = self.pipeline.get_current_correction()
                model_used = correction.source if correction else "none"
                prediction = row['clock_offset'] + (correction.offset_correction if correction else 0.0)
                uncertainty = correction.offset_uncertainty if correction else row['measurement_uncertainty']
            else:
                # During warmup, just use NTP
                prediction = row['clock_offset']
                uncertainty = row['measurement_uncertainty']
                model_used = "ntp_warmup"

            inference_time = (time.time() - start_time) * 1000  # ms

            # Add measurement to pipeline for future predictions
            self.pipeline.add_ntp_measurement(ntp_measurement)

            return ReplayResult(
                timestamp=row['timestamp'],
                prediction=prediction,
                uncertainty=uncertainty,
                ground_truth=row['ground_truth_offset'],
                input_offset=row['clock_offset'],
                model_used=model_used,
                inference_time_ms=inference_time
            )

        except Exception as e:
            logger.warning(f"Failed to replay measurement at {row['timestamp']}: {e}")
            return None

    def run_replay_experiment(self, dataset_path: Path, model_name: str = "chronos",
                            warmup_measurements: int = 300, speed_factor: float = 1.0,
                            max_measurements: Optional[int] = None) -> List[ReplayResult]:
        """Run complete replay experiment"""
        logger.info(f"Starting replay experiment with {model_name}")

        # Load dataset
        df = self.load_dataset(dataset_path)

        if max_measurements:
            df = df.head(max_measurements)

        # Initialize ChronoTick
        self.initialize_chronotick(model_name)

        results = []
        start_time = time.time()

        try:
            for idx, row in df.iterrows():
                # Determine if warmup is complete
                warmup_complete = idx >= warmup_measurements

                # Replay measurement
                result = self.replay_measurement(row, warmup_complete)
                if result:
                    results.append(result)

                # Progress reporting
                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    eta = (len(df) - idx - 1) / rate
                    logger.info(f"Progress: {idx + 1}/{len(df)} ({100 * (idx + 1) / len(df):.1f}%), "
                              f"Rate: {rate:.1f} measurements/s, ETA: {eta:.1f}s")

                # Speed control
                if speed_factor < float('inf'):
                    time.sleep(1.0 / speed_factor)

        except KeyboardInterrupt:
            logger.info("Replay interrupted by user")
        except Exception as e:
            logger.error(f"Replay failed: {e}")
            raise

        finally:
            # Cleanup
            if self.pipeline:
                self.pipeline.stop()

        logger.info(f"Replay complete: {len(results)} results")
        return results

    def save_results(self, results: List[ReplayResult], output_path: Path):
        """Save replay results to file"""
        if not results:
            logger.warning("No results to save")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        results_data = []
        for result in results:
            results_data.append({
                'timestamp': result.timestamp,
                'prediction': result.prediction,
                'uncertainty': result.uncertainty,
                'ground_truth': result.ground_truth,
                'input_offset': result.input_offset,
                'model_used': result.model_used,
                'inference_time_ms': result.inference_time_ms,
                'prediction_error': result.prediction - result.ground_truth,
                'abs_prediction_error': abs(result.prediction - result.ground_truth),
                'input_error': result.input_offset - result.ground_truth,
                'abs_input_error': abs(result.input_offset - result.ground_truth)
            })

        df = pd.DataFrame(results_data)

        # Save as CSV
        if str(output_path).endswith('.gz'):
            df.to_csv(output_path, compression='gzip', index=False)
        else:
            df.to_csv(output_path, index=False)

        logger.info(f"Results saved to {output_path}")

        # Also save numpy arrays for metrics calculation
        predictions_path = output_path.parent / f"{output_path.stem}_predictions.npy"
        uncertainties_path = output_path.parent / f"{output_path.stem}_uncertainties.npy"

        np.save(predictions_path, df['prediction'].values)
        np.save(uncertainties_path, df['uncertainty'].values)

        logger.info(f"Predictions saved to {predictions_path}")
        logger.info(f"Uncertainties saved to {uncertainties_path}")

    def generate_summary_report(self, results: List[ReplayResult]) -> Dict:
        """Generate summary statistics from replay results"""
        if not results:
            return {}

        df = pd.DataFrame([{
            'prediction_error': r.prediction - r.ground_truth,
            'abs_prediction_error': abs(r.prediction - r.ground_truth),
            'input_error': r.input_offset - r.ground_truth,
            'abs_input_error': abs(r.input_offset - r.ground_truth),
            'uncertainty': r.uncertainty,
            'inference_time_ms': r.inference_time_ms,
            'model_used': r.model_used
        } for r in results])

        # Calculate improvement metrics
        prediction_mae = df['abs_prediction_error'].mean()
        input_mae = df['abs_input_error'].mean()
        improvement = (input_mae - prediction_mae) / input_mae * 100

        # Coverage analysis
        within_1sigma = np.sum(df['abs_prediction_error'] <= df['uncertainty']) / len(df)
        within_2sigma = np.sum(df['abs_prediction_error'] <= 2 * df['uncertainty']) / len(df)

        summary = {
            'total_measurements': len(results),
            'prediction_mae_seconds': prediction_mae,
            'prediction_mae_microseconds': prediction_mae * 1e6,
            'input_mae_seconds': input_mae,
            'input_mae_microseconds': input_mae * 1e6,
            'improvement_percentage': improvement,
            'mean_uncertainty_microseconds': df['uncertainty'].mean() * 1e6,
            'coverage_1sigma': within_1sigma,
            'coverage_2sigma': within_2sigma,
            'mean_inference_time_ms': df['inference_time_ms'].mean(),
            'p95_prediction_error_microseconds': df['abs_prediction_error'].quantile(0.95) * 1e6,
            'p99_prediction_error_microseconds': df['abs_prediction_error'].quantile(0.99) * 1e6,
            'model_distribution': df['model_used'].value_counts().to_dict()
        }

        return summary


def main():
    parser = argparse.ArgumentParser(description="ChronoTick Dataset Replay Experiment")
    parser.add_argument('--dataset', type=Path, required=True, help='Dataset CSV file')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--model', default='chronos', help='Model to use (chronos, timesfm, ttm)')
    parser.add_argument('--warmup', type=int, default=300, help='Number of warmup measurements')
    parser.add_argument('--speed', type=float, default=float('inf'), help='Replay speed factor')
    parser.add_argument('--max-measurements', type=int, default=None, help='Maximum measurements to replay')
    parser.add_argument('--config', type=Path, default=None, help='ChronoTick config file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not CHRONOTICK_AVAILABLE:
        logger.error("ChronoTick components not available. Cannot run replay experiment.")
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run experiment
    replayer = DatasetReplayer(args.config)
    results = replayer.run_replay_experiment(
        dataset_path=args.dataset,
        model_name=args.model,
        warmup_measurements=args.warmup,
        speed_factor=args.speed,
        max_measurements=args.max_measurements
    )

    # Save results
    results_file = args.output / f"replay_results_{args.model}.csv.gz"
    replayer.save_results(results, results_file)

    # Generate summary
    summary = replayer.generate_summary_report(results)

    # Save summary
    import json
    summary_file = args.output / f"replay_summary_{args.model}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Experiment complete. Results saved to {args.output}")
    logger.info(f"Summary: {summary['improvement_percentage']:.1f}% improvement, "
               f"MAE: {summary['prediction_mae_microseconds']:.1f}μs")

    return 0


if __name__ == "__main__":
    exit(main())