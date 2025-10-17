#!/usr/bin/env python3
"""
ChronoTick Real Data Pipeline

Complete pipeline that integrates NTP measurements, predictive scheduling,
and model fusion to provide real clock corrections with proper error bounds.

Replaces the synthetic ClockDataGenerator with real measurements.
"""

import time
import threading
import logging
import math
import json
import functools
import inspect
import statistics
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

try:
    # Try relative imports first (when used as module)
    from .ntp_client import ClockMeasurementCollector, NTPMeasurement
    from .predictive_scheduler import PredictiveScheduler, CorrectionWithBounds
    from .utils import SystemMetricsCollector
except ImportError:
    # Fallback for direct execution
    from ntp_client import ClockMeasurementCollector, NTPMeasurement
    from predictive_scheduler import PredictiveScheduler, CorrectionWithBounds
    from utils import SystemMetricsCollector

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")


def debug_trace_pipeline(include_args=True, include_result=True, include_timing=True):
    """
    Specialized debug logging for pipeline operations with model I/O tracking.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not debug_logger.isEnabledFor(logging.DEBUG):
                return func(*args, **kwargs)
            
            func_name = f"{func.__module__}.{func.__qualname__}"
            call_id = id(args) if args else id(kwargs)
            
            # Log function entry
            debug_info = {
                "function": func_name,
                "call_id": call_id,
                "timestamp": time.time()
            }
            
            if include_args:
                try:
                    debug_args = []
                    for i, arg in enumerate(args):
                        if hasattr(arg, '__dict__'):
                            debug_args.append(f"<{type(arg).__name__} object>")
                        elif isinstance(arg, np.ndarray):
                            debug_args.append(f"<numpy.array shape={arg.shape} dtype={arg.dtype}>")
                        elif isinstance(arg, (str, int, float, bool, type(None))):
                            debug_args.append(repr(arg))
                        elif isinstance(arg, (list, tuple)) and len(arg) > 10:
                            debug_args.append(f"<{type(arg).__name__} len={len(arg)}>")
                        else:
                            debug_args.append(repr(arg))
                    
                    debug_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, np.ndarray):
                            debug_kwargs[k] = f"<numpy.array shape={v.shape} dtype={v.dtype}>"
                        elif hasattr(v, '__dict__'):
                            debug_kwargs[k] = f"<{type(v).__name__} object>"
                        elif isinstance(v, (list, tuple)) and len(v) > 10:
                            debug_kwargs[k] = f"<{type(v).__name__} len={len(v)}>"
                        else:
                            debug_kwargs[k] = v
                    
                    debug_info["args"] = debug_args
                    debug_info["kwargs"] = debug_kwargs
                except Exception as e:
                    debug_info["args_error"] = str(e)
            
            debug_logger.debug(f"PIPELINE_ENTRY: {json.dumps(debug_info, indent=2)}")
            
            # Execute function with timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log function exit
                exit_info = {
                    "function": func_name,
                    "call_id": call_id,
                    "status": "success",
                    "timestamp": time.time()
                }
                
                if include_timing:
                    exit_info["execution_time_ms"] = execution_time * 1000
                
                if include_result:
                    try:
                        if isinstance(result, (str, int, float, bool, type(None))):
                            exit_info["result"] = result
                        elif isinstance(result, np.ndarray):
                            exit_info["result"] = {
                                "type": "numpy.array",
                                "shape": result.shape,
                                "dtype": str(result.dtype),
                                "mean": float(np.mean(result)) if result.size > 0 else None,
                                "std": float(np.std(result)) if result.size > 0 else None,
                                "min": float(np.min(result)) if result.size > 0 else None,
                                "max": float(np.max(result)) if result.size > 0 else None
                            }
                        elif hasattr(result, '__dict__'):
                            # For dataclasses and objects, extract key attributes
                            if hasattr(result, '__dataclass_fields__'):
                                exit_info["result"] = {
                                    "type": type(result).__name__,
                                    "fields": {k: v for k, v in result.__dict__.items() 
                                             if not k.startswith('_')}
                                }
                            else:
                                exit_info["result"] = f"<{type(result).__name__} object>"
                        elif isinstance(result, (dict, list)):
                            result_str = json.dumps(result, default=str)
                            if len(result_str) > 1000:
                                exit_info["result"] = result_str[:1000] + "... (truncated)"
                            else:
                                exit_info["result"] = result
                        else:
                            exit_info["result"] = f"<{type(result).__name__}>"
                    except Exception as e:
                        exit_info["result_error"] = str(e)
                
                debug_logger.debug(f"PIPELINE_EXIT: {json.dumps(exit_info, indent=2, default=str)}")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_info = {
                    "function": func_name,
                    "call_id": call_id,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": time.time()
                }
                if include_timing:
                    error_info["execution_time_ms"] = execution_time * 1000
                
                debug_logger.debug(f"PIPELINE_ERROR: {json.dumps(error_info, indent=2)}")
                raise
        
        return wrapper
    return decorator


@dataclass
class ModelPrediction:
    """Single model prediction result"""
    offset: float               # Predicted offset correction
    drift: float               # Predicted drift rate
    offset_uncertainty: float  # Offset prediction uncertainty
    drift_uncertainty: float   # Drift prediction uncertainty
    confidence: float          # Overall prediction confidence
    timestamp: float           # When prediction was made


class PredictionFusionEngine:
    """
    Implements design.md inverse-variance weighting for model fusion.
    Combines CPU and GPU predictions with temporal and uncertainty weighting.
    """
    
    def __init__(self, config: dict):
        """Initialize fusion engine with configuration"""
        self.temporal_weighting = config.get('temporal_weighting', True)
        self.uncertainty_weighting = config.get('uncertainty_weighting', True)
        
    def extract_uncertainty(self, prediction: ModelPrediction) -> float:
        """
        Extract uncertainty from prediction using design.md equation 49-50:
        σ ≈ (Q₉₀ - Q₁₀) / 2.56
        
        For now, use offset_uncertainty directly from TSFM models
        """
        return prediction.offset_uncertainty
    
    def fuse_predictions(self, cpu_correction: CorrectionWithBounds, 
                        gpu_correction: CorrectionWithBounds,
                        cpu_weight: float, gpu_weight: float) -> CorrectionWithBounds:
        """
        Fuse CPU and GPU predictions using design.md equations 53-54:
        ŷ(t) = Σᵢ wᵢ(t) · ŷᵢ(t), with wᵢ(t) = (1/σᵢ²) / (Σⱼ 1/σⱼ²)
        """
        
        if not self.uncertainty_weighting:
            # Simple temporal weighting only
            fused_offset = cpu_weight * cpu_correction.offset_correction + gpu_weight * gpu_correction.offset_correction
            fused_drift = cpu_weight * cpu_correction.drift_rate + gpu_weight * gpu_correction.drift_rate
            fused_offset_unc = cpu_weight * cpu_correction.offset_uncertainty + gpu_weight * gpu_correction.offset_uncertainty
            fused_drift_unc = cpu_weight * cpu_correction.drift_uncertainty + gpu_weight * gpu_correction.drift_uncertainty
        else:
            # Inverse-variance weighting (design.md equation 54)
            cpu_inv_var = 1.0 / (cpu_correction.offset_uncertainty**2) if cpu_correction.offset_uncertainty > 0 else 0
            gpu_inv_var = 1.0 / (gpu_correction.offset_uncertainty**2) if gpu_correction.offset_uncertainty > 0 else 0
            
            # Apply temporal weighting on top of uncertainty weighting
            weighted_cpu_inv_var = cpu_inv_var * cpu_weight
            weighted_gpu_inv_var = gpu_inv_var * gpu_weight
            
            total_inv_var = weighted_cpu_inv_var + weighted_gpu_inv_var
            
            if total_inv_var > 0:
                final_cpu_weight = weighted_cpu_inv_var / total_inv_var
                final_gpu_weight = weighted_gpu_inv_var / total_inv_var
            else:
                # Fallback to temporal weights
                final_cpu_weight = cpu_weight
                final_gpu_weight = gpu_weight
            
            # Fused prediction (design.md equation 53)
            fused_offset = final_cpu_weight * cpu_correction.offset_correction + final_gpu_weight * gpu_correction.offset_correction
            fused_drift = final_cpu_weight * cpu_correction.drift_rate + final_gpu_weight * gpu_correction.drift_rate
            
            # Combine uncertainties using inverse-variance weighting
            if total_inv_var > 0:
                fused_offset_unc = 1.0 / math.sqrt(total_inv_var)
            else:
                fused_offset_unc = max(cpu_correction.offset_uncertainty, gpu_correction.offset_uncertainty)
            
            # Drift uncertainty combination
            fused_drift_unc = (final_cpu_weight * cpu_correction.drift_uncertainty +
                              final_gpu_weight * gpu_correction.drift_uncertainty)

        # Fuse quantiles if available from both models
        fused_quantiles = None
        if cpu_correction.quantiles and gpu_correction.quantiles:
            # Find common quantile levels
            common_levels = set(cpu_correction.quantiles.keys()) & set(gpu_correction.quantiles.keys())
            if common_levels:
                fused_quantiles = {}
                for q_level in common_levels:
                    # Fuse quantiles using the same weights as predictions
                    if not self.uncertainty_weighting:
                        fused_quantiles[q_level] = (cpu_weight * cpu_correction.quantiles[q_level] +
                                                    gpu_weight * gpu_correction.quantiles[q_level])
                    else:
                        fused_quantiles[q_level] = (final_cpu_weight * cpu_correction.quantiles[q_level] +
                                                    final_gpu_weight * gpu_correction.quantiles[q_level])
        elif cpu_correction.quantiles:
            # Only CPU has quantiles
            fused_quantiles = cpu_correction.quantiles.copy() if cpu_correction.quantiles else None
        elif gpu_correction.quantiles:
            # Only GPU has quantiles
            fused_quantiles = gpu_correction.quantiles.copy() if gpu_correction.quantiles else None

        return CorrectionWithBounds(
            offset_correction=fused_offset,
            drift_rate=fused_drift,
            offset_uncertainty=fused_offset_unc,
            drift_uncertainty=fused_drift_unc,
            prediction_time=time.time(),
            valid_until=max(cpu_correction.valid_until, gpu_correction.valid_until),
            confidence=(cpu_weight * cpu_correction.confidence + gpu_weight * gpu_correction.confidence),
            source="fusion",
            quantiles=fused_quantiles
        )


class DatasetManager:
    """
    Maintains consistent 1-second measurement frequency for TSFM models.
    Applies retrospective correction when new NTP data arrives.
    """
    
    def __init__(self, max_dataset_size=1000):
        """Initialize dataset manager

        Args:
            max_dataset_size: Maximum number of measurements to keep in dataset (sliding window)
        """
        self.measurement_dataset = {}  # timestamp -> {offset, drift, source, uncertainty}
        self.prediction_history = []   # List of (timestamp, prediction) tuples
        self.lock = threading.RLock()  # Use reentrant lock to allow nested lock acquisition
        self.max_dataset_size = max_dataset_size  # FIX #1: Dataset sliding window
        
    def add_ntp_measurement(self, measurement: NTPMeasurement):
        """Add real NTP measurement to dataset"""
        with self.lock:
            self.measurement_dataset[int(measurement.timestamp)] = {
                'timestamp': measurement.timestamp,
                'offset': measurement.offset,
                'drift': 0.0,  # Will be calculated from sequence
                'source': 'ntp_measurement',
                'uncertainty': measurement.uncertainty,
                'corrected': False
            }

            # DETAILED DEBUG: Show what we're storing
            logger.info(f"[DATASET_ADD_NTP] ▶▶▶ STORED NTP measurement:")
            logger.info(f"  Timestamp: {measurement.timestamp:.2f} ({int(measurement.timestamp)})")
            logger.info(f"  Offset: {measurement.offset*1000:.3f}ms")
            logger.info(f"  Uncertainty: {measurement.uncertainty*1000:.3f}ms")
            logger.info(f"  Source: ntp_measurement")
            logger.info(f"  Dataset size: {len(self.measurement_dataset)} entries")

    def add_prediction(self, timestamp: float, offset: float, drift: float,
                      source: str, uncertainty: float, confidence: float, was_capped: bool = False):
        """
        Add ML prediction to dataset for autoregressive training.

        Args:
            timestamp: Prediction timestamp
            offset: Predicted offset correction
            drift: Predicted drift rate
            source: Prediction source ('cpu', 'gpu', 'fusion')
            uncertainty: Prediction uncertainty
            confidence: Prediction confidence [0,1]
            was_capped: Whether this prediction was capped (FIX C: skip backtracking for capped)
        """
        with self.lock:
            self.measurement_dataset[int(timestamp)] = {
                'timestamp': timestamp,
                'offset': offset,
                'drift': drift,
                'source': f'prediction_{source}',  # Tag as prediction
                'uncertainty': uncertainty,
                'confidence': confidence,
                'corrected': False,
                'was_capped': was_capped  # FIX C: Track if prediction was capped
            }

            # DETAILED DEBUG: Show what prediction we're storing
            logger.info(f"[DATASET_ADD_PRED] ▶▶▶ STORED {source} prediction:")
            logger.info(f"  Timestamp: {timestamp:.2f} ({int(timestamp)})")
            logger.info(f"  Offset: {offset*1000:.3f}ms")
            logger.info(f"  Drift: {drift*1e6:.3f}μs/s")
            logger.info(f"  Uncertainty: {uncertainty*1000:.3f}ms")
            logger.info(f"  Confidence: {confidence:.3f}")
            logger.info(f"  Dataset size: {len(self.measurement_dataset)} entries")

            # FIX #1: Dataset Sliding Window - Prevent unbounded growth
            if len(self.measurement_dataset) > self.max_dataset_size:
                # Remove oldest entries (keep most recent max_dataset_size measurements)
                sorted_timestamps = sorted(self.measurement_dataset.keys())
                num_to_remove = len(self.measurement_dataset) - self.max_dataset_size
                timestamps_to_remove = sorted_timestamps[:num_to_remove]

                for ts in timestamps_to_remove:
                    del self.measurement_dataset[ts]

                logger.info(f"[DATASET_SLIDING_WINDOW] ✂️ Trimmed dataset: removed {num_to_remove} old measurements")
                logger.info(f"[DATASET_SLIDING_WINDOW] Dataset now: {len(self.measurement_dataset)} entries (max={self.max_dataset_size})")
    
    def fill_measurement_gaps(self, start_time: float, end_time: float,
                             corrections: List[CorrectionWithBounds]):
        """
        Fill gaps between NTP measurements with fused model predictions.
        Maintains 1-second frequency required by TSFM models.
        """
        with self.lock:
            for i, correction in enumerate(corrections):
                timestamp = int(start_time + i)
                
                if timestamp not in self.measurement_dataset and timestamp < end_time:
                    self.measurement_dataset[timestamp] = {
                        'timestamp': timestamp,
                        'offset': correction.offset_correction,
                        'drift': correction.drift_rate,
                        'source': 'fused_prediction',
                        'uncertainty': correction.offset_uncertainty,
                        'corrected': False
                    }
    
    def apply_ntp_correction(self, ntp_measurement: NTPMeasurement, method: str = 'linear',
                            offset_uncertainty: float = 0.001, drift_uncertainty: float = 0.0001):
        """
        Apply dataset-only NTP correction using one of three methods.

        This corrects the historical dataset so that future autoregressive predictions
        automatically align with NTP ground truth.

        Args:
            ntp_measurement: New NTP measurement (ground truth)
            method: Correction method - 'none', 'linear', 'drift_aware', or 'advanced'
            offset_uncertainty: ML model's offset uncertainty (for drift_aware, advanced)
            drift_uncertainty: ML model's drift uncertainty (for drift_aware, advanced)

        Returns:
            dict: Correction metadata with keys:
                - error: Measured error (NTP_truth - Prediction)
                - interval_start: Start of correction interval
                - interval_end: End of correction interval (NTP timestamp)
                - interval_duration: Duration of interval in seconds
                - method: Correction method used
                Returns None if method='none' or no prediction found
        """
        if method == 'none':
            # Just add NTP measurement, no correction
            self.add_ntp_measurement(ntp_measurement)
            return None

        with self.lock:
            # Find the most recent ML PREDICTION (not NTP!) before NTP time
            # CRITICAL FIX: Must compare NTP to ML predictions, not NTP to NTP!
            prediction_at_ntp = self._get_last_ml_prediction_before(ntp_measurement.timestamp)
            if not prediction_at_ntp:
                logger.warning(f"[NTP_CORRECTION] No ML predictions before NTP time {ntp_measurement.timestamp:.0f}, "
                             f"skipping correction (this is normal during early operation)")
                self.add_ntp_measurement(ntp_measurement)
                return None

            # Calculate error: error = NTP_truth - ML_Prediction
            error = ntp_measurement.offset - prediction_at_ntp['offset']

            # DETAILED DEBUG: Show error calculation
            logger.info(f"[NTP_CORRECTION_{method.upper()}] ═══════════════════════════════════════════")
            logger.info(f"[NTP_CORRECTION_{method.upper()}] NEW NTP MEASUREMENT ARRIVED:")
            logger.info(f"  NTP timestamp: {ntp_measurement.timestamp:.2f}")
            logger.info(f"  NTP offset (ground truth): {ntp_measurement.offset*1000:.3f}ms")
            logger.info(f"  NTP uncertainty: {ntp_measurement.uncertainty*1000:.3f}ms")
            logger.info(f"[NTP_CORRECTION_{method.upper()}] PREDICTION AT NTP TIME:")
            logger.info(f"  Prediction offset: {prediction_at_ntp['offset']*1000:.3f}ms")
            logger.info(f"  Prediction source: {prediction_at_ntp['source']}")
            logger.info(f"[NTP_CORRECTION_{method.upper()}] ERROR CALCULATION:")
            logger.info(f"  Error = NTP_truth - Prediction")
            logger.info(f"  Error = {ntp_measurement.offset*1000:.3f}ms - {prediction_at_ntp['offset']*1000:.3f}ms")
            logger.info(f"  Error = {error*1000:.3f}ms")
            if abs(error*1000) < 1.0:
                logger.info(f"  ✓ Error is SMALL (<1ms) - prediction was accurate!")
            else:
                logger.info(f"  ✗ Error is LARGE (≥1ms) - prediction missed the mark!")

            # Find interval start (last NTP or first prediction)
            interval_start = None
            for ts in sorted(self.measurement_dataset.keys()):
                if ts < ntp_measurement.timestamp:
                    data = self.measurement_dataset[ts]
                    if data['source'] == 'ntp_measurement':
                        interval_start = ts

            if interval_start is None:
                # No previous NTP, use first prediction
                interval_start = min(self.measurement_dataset.keys())

            interval_duration = ntp_measurement.timestamp - interval_start

            logger.info(f"[NTP_CORRECTION_{method.upper()}] CORRECTION INTERVAL:")
            logger.info(f"  Start: {interval_start:.0f}")
            logger.info(f"  End: {ntp_measurement.timestamp:.0f}")
            logger.info(f"  Duration: {interval_duration:.0f}s")
            logger.info(f"  Method: {method}")
            logger.info(f"[NTP_CORRECTION_{method.upper()}] ═══════════════════════════════════════════")

            # Apply correction based on method
            if method == 'linear':
                self._apply_linear_correction(interval_start, ntp_measurement.timestamp, error)
            elif method == 'drift_aware':
                self._apply_drift_aware_correction(
                    interval_start, ntp_measurement.timestamp, error,
                    offset_uncertainty, drift_uncertainty, interval_duration
                )
            elif method == 'advanced':
                self._apply_advanced_correction(
                    interval_start, ntp_measurement.timestamp, error,
                    ntp_measurement.uncertainty, offset_uncertainty, drift_uncertainty
                )
            elif method == 'advance_absolute':
                self._apply_advance_absolute_correction(
                    interval_start, ntp_measurement.timestamp, error,
                    ntp_measurement.uncertainty, offset_uncertainty, drift_uncertainty, interval_duration
                )
            elif method == 'backtracking':
                self._apply_backtracking_correction(
                    interval_start, ntp_measurement.timestamp, error, interval_duration
                )
            else:
                logger.error(f"Unknown correction method: {method}")
                return None

            # Add the new NTP measurement
            self.add_ntp_measurement(ntp_measurement)

            logger.info(f"[NTP_CORRECTION_{method.upper()}] Correction applied to "
                       f"{int(ntp_measurement.timestamp - interval_start)} measurements")

            # Return correction metadata for logging
            return {
                'error': error,
                'interval_start': interval_start,
                'interval_end': ntp_measurement.timestamp,
                'interval_duration': interval_duration,
                'method': method
            }

    def _apply_linear_correction(self, start_time: float, end_time: float, error: float):
        """
        Apply linear retrospective correction.

        Distributes error linearly across time:
        correction(t) = (t - start) / (end - start) × error
        """
        interval_duration = end_time - start_time

        for timestamp in range(int(start_time), int(end_time)):
            if timestamp in self.measurement_dataset:
                # Linear weight: α = (t_i - t_start) / (t_end - t_start)
                alpha = (timestamp - start_time) / interval_duration if interval_duration > 0 else 0

                # Apply correction: ô_t_i'' ← ô_t_i + α · δ
                self.measurement_dataset[timestamp]['offset'] += alpha * error
                self.measurement_dataset[timestamp]['corrected'] = True

    def _apply_drift_aware_correction(self, start_time: float, end_time: float, error: float,
                                     σ_offset: float, σ_drift: float, Δt: float):
        """
        Apply drift-aware retrospective correction.

        Attributes error to both offset and drift based on uncertainty:
        - offset_correction: based on offset uncertainty
        - drift_correction: based on drift uncertainty accumulated over time

        correction(t) = offset_corr + drift_corr × (t - start)
        """
        # Variance contributions
        var_offset = σ_offset ** 2
        var_drift = (σ_drift * Δt) ** 2
        var_total = var_offset + var_drift

        if var_total == 0:
            # Fallback to linear if no uncertainty info
            self._apply_linear_correction(start_time, end_time, error)
            return

        # Weight allocation based on uncertainty
        w_offset = var_offset / var_total
        w_drift = var_drift / var_total

        # Attribute error to offset and drift
        offset_correction = w_offset * error
        drift_correction = (w_drift * error) / Δt if Δt > 0 else 0.0

        logger.info(f"[DRIFT_AWARE] w_offset={w_offset:.3f}, w_drift={w_drift:.3f}, "
                   f"offset_corr={offset_correction*1000:.2f}ms, "
                   f"drift_corr={drift_correction*1e6:.2f}μs/s")

        # Apply to each timestamp
        for timestamp in range(int(start_time), int(end_time)):
            if timestamp in self.measurement_dataset:
                # Time since interval start
                t_elapsed = timestamp - start_time

                # Combined correction: offset + drift × time
                correction = offset_correction + (drift_correction * t_elapsed)

                self.measurement_dataset[timestamp]['offset'] += correction
                self.measurement_dataset[timestamp]['drift'] += drift_correction  # Also adjust drift
                self.measurement_dataset[timestamp]['corrected'] = True

    def _apply_advanced_correction(self, start_time: float, end_time: float, error: float,
                                   sigma_measurement: float, sigma_prediction: float, sigma_drift: float):
        """
        Advanced correction using full temporal uncertainty model with confidence degradation.

        This method models how measurement confidence degrades over time since the last
        NTP synchronization. Points further from the last sync have accumulated more
        uncertainty and are corrected more aggressively toward the new NTP ground truth.

        Uncertainty model:
            sigma_total^2(t) = sigma_measurement^2 + sigma_prediction^2 + (sigma_drift * delta_t)^2

        where delta_t is time elapsed since last NTP sync. The quadratic growth of drift
        uncertainty reflects physical clock behavior. Corrections are weighted inversely
        to confidence: high uncertainty → low confidence → more correction.

        Args:
            start_time: Start of interval (last NTP sync)
            end_time: End of interval (new NTP measurement)
            error: Measured error at end_time (NTP_truth - Prediction)
            sigma_measurement: NTP measurement uncertainty (seconds)
            sigma_prediction: ML model prediction uncertainty (seconds)
            sigma_drift: Drift rate uncertainty (seconds/second)
        """
        interval_duration = end_time - start_time

        # Base uncertainty (non-temporal components)
        sigma_squared_base = sigma_measurement**2 + sigma_prediction**2

        logger.info(f"[ADVANCED] Starting advanced correction over {interval_duration:.0f}s interval")
        logger.info(f"[ADVANCED] sigma_measurement={sigma_measurement*1000:.2f}ms, "
                   f"sigma_prediction={sigma_prediction*1000:.2f}ms, "
                   f"sigma_drift={sigma_drift*1e6:.2f}μs/s")

        # Calculate time-dependent uncertainty and weights
        total_weight = 0.0
        weights = {}
        max_sigma_squared_total = 0.0
        min_sigma_squared_total = float('inf')

        # FIXED: Include end_time by adding +1 to range
        for timestamp in range(int(start_time), int(end_time) + 1):
            if timestamp in self.measurement_dataset:
                # Time elapsed since last sync
                delta_t = timestamp - start_time

                # Temporal uncertainty growth (quadratic with time)
                sigma_squared_temporal = (sigma_drift * delta_t)**2

                # Total uncertainty at this timestamp
                sigma_squared_total = sigma_squared_base + sigma_squared_temporal

                # FIXED: Use DIRECT variance weighting for advanced method
                # Concept: Points with HIGH uncertainty (far from last NTP) get MORE correction
                # Points with LOW uncertainty (near last NTP) get LESS correction
                # This is opposite of inverse-variance used in fusion!
                weight = sigma_squared_total
                weights[timestamp] = weight
                total_weight += weight
                max_sigma_squared_total = max(max_sigma_squared_total, sigma_squared_total)
                min_sigma_squared_total = min(min_sigma_squared_total, sigma_squared_total)

        logger.info(f"[ADVANCED] Uncertainty range: {math.sqrt(min_sigma_squared_total)*1000:.2f}ms "
                   f"to {math.sqrt(max_sigma_squared_total)*1000:.2f}ms")

        # Normalize and apply corrections
        correction_count = 0
        max_correction = 0.0
        min_correction = float('inf')

        # FIXED: Include end_time by adding +1 to range
        for timestamp in range(int(start_time), int(end_time) + 1):
            if timestamp in self.measurement_dataset:
                # Normalized weight (ensures corrections sum to total error)
                alpha = weights[timestamp] / total_weight if total_weight > 0 else 0

                # Apply weighted correction
                correction = alpha * error
                self.measurement_dataset[timestamp]['offset'] += correction
                self.measurement_dataset[timestamp]['corrected'] = True
                correction_count += 1
                max_correction = max(max_correction, abs(correction))
                min_correction = min(min_correction, abs(correction))

                # Log sample of corrections (first 3, last 3, and every 30th)
                if correction_count <= 3 or correction_count % 30 == 0:
                    delta_t = timestamp - start_time
                    sigma_squared_total = weights[timestamp]
                    confidence = 1.0 / (1.0 + sigma_squared_total * 1e6)  # Scale for readability
                    logger.debug(f"[ADVANCED] t={timestamp}, delta_t={delta_t:3.0f}s, "
                               f"sigma={math.sqrt(sigma_squared_total)*1000:.2f}ms, "
                               f"confidence={confidence:.3f}, alpha={alpha:.4f}, "
                               f"correction={correction*1000:+.2f}ms")

        logger.info(f"[ADVANCED] Applied to {correction_count} measurements")
        logger.info(f"[ADVANCED] Correction range: {min_correction*1000:.2f}ms "
                   f"to {max_correction*1000:.2f}ms")

    def _apply_advance_absolute_correction(self, start_time: float, end_time: float, error: float,
                                          sigma_measurement: float, sigma_prediction: float,
                                          sigma_drift: float, interval_duration: float):
        """
        Advanced Absolute correction: Per-point directional correction toward target line.

        Goal: "Nudge the system closer to the line between two successive NTP measurements proportionally.
        If we went above that line we go down, if we went below we go up. Avoids over-corrections
        that push errors that went up more up (if error is positive) or errors that went down more
        down (if error is negative)."

        Key innovation: PER-POINT directional correction
        - Calculate target line: y_target(t) = E × (t - t_start) / Δt (linear from 0 to E)
        - For each point: deviation = y_current - y_target
        - If deviation > 0 (ABOVE line): correction is NEGATIVE (push DOWN toward line)
        - If deviation < 0 (BELOW line): correction is POSITIVE (push UP toward line)

        This prevents over-correction by bringing ALL points closer to the ideal line, not pushing
        everything in the same direction based on endpoint error.

        Uses uncertainty weighting (same as 'advanced'):
        - Points with HIGH uncertainty get MORE correction
        - Points with LOW uncertainty get LESS correction

        Args:
            start_time: Start of interval (last NTP sync, where target line = 0)
            end_time: End of interval (new NTP measurement, where target line = E)
            error: Measured point error at end_time (NTP_truth - Prediction)
            sigma_measurement: NTP measurement uncertainty (seconds)
            sigma_prediction: ML model prediction uncertainty (seconds)
            sigma_drift: Drift rate uncertainty (seconds/second)
            interval_duration: Duration of interval in seconds
        """
        logger.info(f"[ADVANCE_ABSOLUTE] Starting advance_absolute correction (v4 - per-point directional)")
        logger.info(f"[ADVANCE_ABSOLUTE] Endpoint error: {error*1000:.2f}ms over {interval_duration:.0f}s")
        logger.info(f"[ADVANCE_ABSOLUTE] Target: Linear line from 0 to {error*1000:.2f}ms")
        logger.info(f"[ADVANCE_ABSOLUTE] Method: Bring each point closer to target line")

        # Base uncertainty (non-temporal components)
        sigma_squared_base = sigma_measurement**2 + sigma_prediction**2

        # Calculate per-point deviations from target line and uncertainty weights
        total_weight = 0.0
        weights = {}
        deviations = {}
        total_absolute_deviation = 0.0

        for timestamp in range(int(start_time), int(end_time) + 1):
            if timestamp in self.measurement_dataset:
                delta_t = timestamp - start_time

                # Target offset on the ideal line (linear from 0 to E)
                target_offset = error * delta_t / interval_duration if interval_duration > 0 else 0

                # Current offset at this timestamp
                current_offset = self.measurement_dataset[timestamp]['offset']

                # Deviation from target line
                # Positive deviation = point is ABOVE line (need to push DOWN)
                # Negative deviation = point is BELOW line (need to push UP)
                deviation = current_offset - target_offset
                deviations[timestamp] = deviation
                total_absolute_deviation += abs(deviation)

                # Uncertainty weighting (same as 'advanced')
                sigma_squared_temporal = (sigma_drift * delta_t)**2
                sigma_squared_total = sigma_squared_base + sigma_squared_temporal
                weight = sigma_squared_total
                weights[timestamp] = weight
                total_weight += weight

        logger.info(f"[ADVANCE_ABSOLUTE] Total absolute deviation from line: {total_absolute_deviation*1000:.2f}ms")
        logger.info(f"[ADVANCE_ABSOLUTE] Goal: Reduce this by correcting each point toward line")

        # Apply corrections: bring each point toward the target line
        correction_count = 0
        max_correction = 0.0
        min_correction = float('inf')
        total_correction_applied = 0.0
        corrections_up = 0
        corrections_down = 0

        for timestamp in range(int(start_time), int(end_time) + 1):
            if timestamp in self.measurement_dataset:
                delta_t = timestamp - start_time
                deviation = deviations[timestamp]

                # Normalized uncertainty weight
                alpha = weights[timestamp] / total_weight if total_weight > 0 else 0

                # Correction = weighted share of deviation (with OPPOSITE sign to bring toward line)
                # If deviation > 0 (above line): correction < 0 (push down)
                # If deviation < 0 (below line): correction > 0 (push up)
                correction = -alpha * deviation * (len(weights) / 2.0)  # Scale by N/2 for accumulated error concept

                # Apply correction
                self.measurement_dataset[timestamp]['offset'] += correction
                self.measurement_dataset[timestamp]['corrected'] = True

                correction_count += 1
                max_correction = max(max_correction, abs(correction))
                if abs(correction) > 0:
                    min_correction = min(min_correction, abs(correction))
                total_correction_applied += correction

                if correction > 0:
                    corrections_up += 1
                elif correction < 0:
                    corrections_down += 1

                # Log sample
                if correction_count <= 5 or correction_count % 30 == 0:
                    target_offset = error * delta_t / interval_duration if interval_duration > 0 else 0
                    direction = "↓ DOWN" if correction < 0 else "↑ UP  "
                    logger.debug(f"[ADVANCE_ABSOLUTE] t={timestamp}, Δt={delta_t:3.0f}s, "
                               f"target={target_offset*1000:+.2f}ms, dev={deviation*1000:+.2f}ms, "
                               f"corr={correction*1000:+.2f}ms {direction}")

        logger.info(f"[ADVANCE_ABSOLUTE] Applied to {correction_count} measurements")
        logger.info(f"[ADVANCE_ABSOLUTE] Corrections UP: {corrections_up}, DOWN: {corrections_down}")
        logger.info(f"[ADVANCE_ABSOLUTE] Correction range: {min_correction*1000:.2f}ms to {max_correction*1000:.2f}ms")
        logger.info(f"[ADVANCE_ABSOLUTE] Total correction applied: {total_correction_applied*1000:.2f}ms")
        logger.info(f"[ADVANCE_ABSOLUTE] This brings all points closer to the ideal NTP line")

    def _apply_backtracking_correction(self, start_time: float, end_time: float, error: float, interval_duration: float):
        """
        Backtracking Learning Correction: REPLACE predictions with interpolated NTP ground truth.

        CRITICAL FIX: Corrects FULL CONTEXT WINDOW (512s), not just NTP interval (180s).
        This ensures model trains on corrected data, not contaminated predictions.

        Key Innovation: Make the dataset look like "what NTP would have measured" at each point.

        This method REPLACES all predictions in the context window with linearly
        interpolated values based on NTP boundaries. This gives the ML model
        an NTP-aligned dataset to learn from for future predictions.

        Example:
        - Context window: 512s
        - NTP measurement at t=0: offset=10ms
        - NTP measurement at t=180: offset=20ms
        - ML predicted at t=[1...511]: various values
        - After correction: REPLACE all with linear interpolation 10ms → 20ms

        Philosophy:
        - STRONGER correction (not weaker) for better NTP alignment
        - Dataset becomes "NTP-like" so future predictions stay aligned
        - Covers FULL context window so model sees corrected data

        Args:
            start_time: Start of interval (last NTP measurement timestamp)
            end_time: End of interval (new NTP measurement timestamp)
            error: Measured error at end_time (NTP_truth - Prediction)
            interval_duration: Duration of interval in seconds
        """
        # CRITICAL: Get context window size from config
        context_window_size = 512  # TimesFM default context length

        # FIX: Expand correction window to cover full context
        # OLD: Only correct between last NTP and current NTP (180s)
        # NEW: Correct full context window before current NTP (512s)
        # CRITICAL: Don't limit by start_time - always go back full 512s!
        correction_start = end_time - context_window_size
        correction_end = end_time
        correction_duration = correction_end - correction_start

        logger.info(f"[BACKTRACKING] Starting backtracking learning correction (CONTEXT WINDOW FIX)")
        logger.info(f"[BACKTRACKING] Error: {error*1000:.2f}ms over NTP interval: {interval_duration:.0f}s")
        logger.info(f"[BACKTRACKING] CONTEXT WINDOW COVERAGE:")
        logger.info(f"  Context size: {context_window_size}s")
        logger.info(f"  NTP interval: [{start_time:.0f}, {end_time:.0f}] = {interval_duration:.0f}s")
        logger.info(f"  Correction window: [{correction_start:.0f}, {correction_end:.0f}] = {correction_duration:.0f}s")
        logger.info(f"  Expected samples: ~{int(correction_duration)}")
        logger.info(f"[BACKTRACKING] Strategy: REPLACE predictions with interpolated NTP")

        # CRITICAL FIX: Find NTP measurement BEFORE correction window starts
        # We need to interpolate from the NTP before correction_start to the NTP at end_time
        ntp_before_offset = None
        ntp_before_time = None

        # Search for the most recent NTP measurement before correction_start
        for ts in sorted(self.measurement_dataset.keys()):
            if ts < correction_start:
                data = self.measurement_dataset[ts]
                if data['source'] == 'ntp_measurement':
                    ntp_before_offset = data['offset']
                    ntp_before_time = ts

        # Get the PREDICTION at end_time to calculate the new NTP offset
        end_prediction_offset = None
        if int(end_time) in self.measurement_dataset:
            end_prediction_offset = self.measurement_dataset[int(end_time)]['offset']

        # If no prediction at exact end_time, find closest
        if end_prediction_offset is None:
            for ts in sorted(self.measurement_dataset.keys(), reverse=True):
                if ts <= int(end_time):
                    end_prediction_offset = self.measurement_dataset[ts]['offset']
                    break

        if end_prediction_offset is None:
            logger.warning(f"[BACKTRACKING] No prediction near end_time {end_time:.0f}, skipping correction")
            return

        # Calculate the NTP offset at end_time: NTP_truth = Prediction + error
        ntp_after_offset = end_prediction_offset + error
        ntp_after_time = end_time

        # Handle first NTP case (no previous NTP before correction window)
        if ntp_before_offset is None:
            logger.info(f"[BACKTRACKING] First NTP or no NTP before correction window - using current NTP for whole window")
            ntp_before_offset = ntp_after_offset
            ntp_before_time = correction_start

        logger.info(f"[BACKTRACKING] NTP BOUNDARIES FOR INTERPOLATION:")
        logger.info(f"  Before: {ntp_before_offset*1000:.2f}ms @ t={ntp_before_time:.0f}")
        logger.info(f"  After: {ntp_after_offset*1000:.2f}ms @ t={ntp_after_time:.0f}")
        logger.info(f"  Interpolating across: {ntp_after_time - ntp_before_time:.0f}s")
        logger.info(f"  Delta: {(ntp_after_offset - ntp_before_offset)*1000:.2f}ms over {ntp_after_time - ntp_before_time:.0f}s")
        logger.info(f"  Drift rate: {(ntp_after_offset - ntp_before_offset)/(ntp_after_time - ntp_before_time)*1e6:.2f}μs/s")
        logger.info(f"[BACKTRACKING] Replacing ALL predictions in context window with NTP-interpolated values")

        # CRITICAL DEBUG: Show what predictions exist in dataset before correction
        pred_count_before_corr = 0
        pred_range_before = []
        for ts in sorted(self.measurement_dataset.keys()):
            if self.measurement_dataset[ts]['source'].startswith('prediction_'):
                pred_count_before_corr += 1
                if len(pred_range_before) < 5 or ts > int(correction_end):
                    pred_range_before.append((ts, self.measurement_dataset[ts]['offset']*1000))
        logger.info(f"[BACKTRACKING] Dataset has {pred_count_before_corr} predictions before correction")
        logger.info(f"[BACKTRACKING] Sample predictions: {pred_range_before[:10]}")

        # Collect all replacements for detailed logging
        replacements = []  # List of (timestamp, old_value, new_value, delta)

        # FIX C: Count skipped capped predictions
        skipped_capped_count = 0

        # REPLACE all predictions with linearly interpolated NTP values
        correction_count = 0
        max_replacement = 0.0
        min_replacement = float('inf')
        sum_before = 0.0
        sum_after = 0.0
        sum_delta = 0.0

        # CRITICAL FIX: Loop from correction_start to end_time (full context window)
        # NOT from start_time to end_time (just NTP interval)
        for timestamp in range(int(correction_start), int(correction_end)):
            if timestamp in self.measurement_dataset:
                # FIX C: Skip backtracking for capped predictions (prevents toxic feedback loop)
                was_capped = self.measurement_dataset[timestamp].get('was_capped', False)
                if was_capped:
                    skipped_capped_count += 1
                    logger.debug(f"[BACKTRACKING] Skipping capped prediction at t={timestamp} (FIX C)")
                    continue  # Skip this prediction - don't apply backtracking to capped values

                # Calculate interpolation weight using NTP boundaries
                total_duration = ntp_after_time - ntp_before_time
                if total_duration > 0:
                    alpha = (timestamp - ntp_before_time) / total_duration
                else:
                    alpha = 0

                # Calculate what NTP "would have measured" at this point
                ntp_interpolated = ntp_before_offset + alpha * (ntp_after_offset - ntp_before_offset)

                # Get current prediction
                current_offset = self.measurement_dataset[timestamp]['offset']

                # Calculate replacement delta for logging
                replacement_delta = ntp_interpolated - current_offset

                # Store replacement details
                replacements.append((timestamp, current_offset, ntp_interpolated, replacement_delta))

                # REPLACE prediction with interpolated NTP value
                self.measurement_dataset[timestamp]['offset'] = ntp_interpolated
                self.measurement_dataset[timestamp]['corrected'] = True

                # Statistics
                correction_count += 1
                max_replacement = max(max_replacement, abs(replacement_delta))
                if abs(replacement_delta) > 0:
                    min_replacement = min(min_replacement, abs(replacement_delta))
                sum_before += current_offset
                sum_after += ntp_interpolated
                sum_delta += replacement_delta

        # CRITICAL FIX: Delete all future predictions beyond backtracking window
        # These were made with uncorrected data and will poison future training
        future_predictions_deleted = 0
        for timestamp in list(self.measurement_dataset.keys()):
            if timestamp > int(correction_end):
                data = self.measurement_dataset[timestamp]
                if data['source'].startswith('prediction_'):
                    logger.debug(f"[BACKTRACKING] Deleting future prediction at t={timestamp} (beyond correction window)")
                    del self.measurement_dataset[timestamp]
                    future_predictions_deleted += 1

        if future_predictions_deleted > 0:
            logger.warning(f"[BACKTRACKING] 🗑️ Deleted {future_predictions_deleted} future predictions (made with uncorrected data)")
            logger.warning(f"[BACKTRACKING] These will be regenerated with clean corrected data")

        # Log detailed before/after summary
        if correction_count > 0:
            mean_before = sum_before / correction_count
            mean_after = sum_after / correction_count
            mean_delta = sum_delta / correction_count

            logger.info(f"[BACKTRACKING] ═══════════════════════════════════════════")
            logger.info(f"[BACKTRACKING] REPLACEMENT SUMMARY ({correction_count} predictions)")
            logger.info(f"[BACKTRACKING] ═══════════════════════════════════════════")
            logger.info(f"[BACKTRACKING] BEFORE correction:")
            logger.info(f"  Mean offset: {mean_before*1000:.3f}ms")
            logger.info(f"  Range: {min([r[1] for r in replacements])*1000:.3f}ms to {max([r[1] for r in replacements])*1000:.3f}ms")
            logger.info(f"[BACKTRACKING] AFTER correction (NTP-interpolated):")
            logger.info(f"  Mean offset: {mean_after*1000:.3f}ms")
            logger.info(f"  Range: {min([r[2] for r in replacements])*1000:.3f}ms to {max([r[2] for r in replacements])*1000:.3f}ms")
            logger.info(f"[BACKTRACKING] DELTA (after - before):")
            logger.info(f"  Mean delta: {mean_delta*1000:+.3f}ms")
            logger.info(f"  Min delta: {min([r[3] for r in replacements])*1000:+.3f}ms")
            logger.info(f"  Max delta: {max([r[3] for r in replacements])*1000:+.3f}ms")
            logger.info(f"[BACKTRACKING] ═══════════════════════════════════════════")

            # Log first 5 and last 5 replacements
            logger.info(f"[BACKTRACKING] First 5 replacements:")
            for i, (ts, old, new, delta) in enumerate(replacements[:5], 1):
                logger.info(f"  {i}. t={ts}: {old*1000:.3f}ms → {new*1000:.3f}ms (Δ={delta*1000:+.3f}ms)")

            if len(replacements) > 10:
                logger.info(f"[BACKTRACKING] ... ({len(replacements)-10} more replacements) ...")

            if len(replacements) > 5:
                logger.info(f"[BACKTRACKING] Last 5 replacements:")
                for i, (ts, old, new, delta) in enumerate(replacements[-5:], len(replacements)-4):
                    logger.info(f"  {i}. t={ts}: {old*1000:.3f}ms → {new*1000:.3f}ms (Δ={delta*1000:+.3f}ms)")

            logger.info(f"[BACKTRACKING] ═══════════════════════════════════════════")

        # CRITICAL: Log context window coverage metrics
        context_coverage_pct = (correction_count / context_window_size * 100) if context_window_size > 0 else 0

        logger.info(f"[BACKTRACKING] ═══════════════════════════════════════════")
        logger.info(f"[BACKTRACKING] ✅ REPLACED {correction_count} predictions with NTP-interpolated values")
        if skipped_capped_count > 0:
            logger.info(f"[BACKTRACKING] ⚠️ SKIPPED {skipped_capped_count} capped predictions (FIX C: prevents toxic feedback)")
        if correction_count > 0:
            logger.info(f"[BACKTRACKING] Replacement range: {min_replacement*1000:.2f}ms to {max_replacement*1000:.2f}ms")

        logger.info(f"[BACKTRACKING] CONTEXT WINDOW COVERAGE:")
        logger.info(f"  Corrected: {correction_count} samples")
        logger.info(f"  Context size: {context_window_size} samples")
        logger.info(f"  Coverage: {context_coverage_pct:.1f}%")
        if context_coverage_pct >= 95:
            logger.info(f"  Status: ✅ EXCELLENT (≥95% coverage)")
        elif context_coverage_pct >= 80:
            logger.info(f"  Status: ✅ GOOD (≥80% coverage)")
        elif context_coverage_pct >= 50:
            logger.info(f"  Status: ⚠️ PARTIAL (<80% coverage)")
        else:
            logger.info(f"  Status: ❌ POOR (<50% coverage - model contamination likely!)")

        logger.info(f"[BACKTRACKING] Dataset now looks like 'what NTP would have measured'")
        logger.info(f"[BACKTRACKING] Future predictions will learn from this NTP-aligned dataset")
        logger.info(f"[BACKTRACKING] ═══════════════════════════════════════════")

    def apply_retrospective_correction(self, ntp_measurement: NTPMeasurement,
                                     interval_start: float):
        """
        DEPRECATED: Use apply_ntp_correction() instead.
        Kept for backward compatibility.
        """
        self.apply_ntp_correction(ntp_measurement, method='linear')
    
    def _get_last_ml_prediction_before(self, timestamp: float) -> Optional[dict]:
        """
        Get the most recent ML PREDICTION (not NTP measurement) before the given timestamp.

        This is critical for NTP correction - we must compare NTP to ML predictions,
        NOT NTP to NTP! This was the root cause of small corrections.

        Args:
            timestamp: Target timestamp to find ML prediction before

        Returns:
            Most recent ML prediction before timestamp, or None if none found
        """
        with self.lock:
            # Find ML predictions only (source starts with 'prediction_')
            candidates = []
            for ts, data in self.measurement_dataset.items():
                if ts <= int(timestamp):  # Must be at or before NTP time
                    source = data.get('source', '')
                    # Only ML predictions, not NTP measurements
                    if source.startswith('prediction_'):
                        candidates.append((ts, data))

            if not candidates:
                return None

            # Return the most recent (highest timestamp)
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

    def get_measurement_at_time(self, timestamp: float) -> Optional[dict]:
        """
        Get the most recent measurement (any source) before the given timestamp.

        DEPRECATED: Use _get_last_ml_prediction_before() for NTP correction to avoid
        comparing NTP to NTP!

        Args:
            timestamp: Target timestamp to find measurement before

        Returns:
            Most recent measurement before timestamp, or None if none found
        """
        with self.lock:
            # Find the most recent measurement (any source) before timestamp
            candidates = []
            for ts, data in self.measurement_dataset.items():
                if ts <= int(timestamp):  # Must be at or before NTP time
                    candidates.append((ts, data))

            if not candidates:
                return None

            # Return the most recent (highest timestamp)
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
    
    def get_recent_ntp_baseline(self, lookback_count: int = 10) -> Optional[float]:
        """
        Calculate the recent NTP baseline for offset normalization.

        Phase 2 Enhancement: Uses exponential smoothing to prevent sudden baseline jumps.

        If smoothing is enabled (default):
            - Applies EMA: smoothed = α * new + (1-α) * previous
            - Prevents wild oscillations (e.g., 849ms → 70ms jumps)
            - Trades slight lag for stability

        Otherwise falls back to simple mean of recent N measurements.

        Args:
            lookback_count: Number of recent NTP measurements to average (fallback mode)

        Returns:
            Smoothed NTP baseline, or None if insufficient data
        """
        with self.lock:
            # Find recent NTP measurements
            ntp_measurements = []
            for ts in sorted(self.measurement_dataset.keys(), reverse=True):
                data = self.measurement_dataset[ts]
                if data['source'] == 'ntp_measurement':
                    ntp_measurements.append(data['offset'])
                    if len(ntp_measurements) >= lookback_count:
                        break

            if len(ntp_measurements) < 2:
                return None

            # Calculate raw baseline from recent measurements
            raw_baseline = float(np.mean(ntp_measurements))

            # Apply exponential smoothing if enabled
            if self.baseline_smoothing_enabled:
                if self.smoothed_baseline is None:
                    # Initialize on first measurement
                    self.smoothed_baseline = raw_baseline
                    logger.info(f"[BASELINE_SMOOTHING] ✓ Initialized baseline: {raw_baseline*1000:.3f}ms")
                else:
                    # EMA: new = α * current + (1-α) * previous
                    previous_smoothed = self.smoothed_baseline
                    self.smoothed_baseline = (
                        self.baseline_smoothing_alpha * raw_baseline +
                        (1 - self.baseline_smoothing_alpha) * self.smoothed_baseline
                    )

                    # Log significant changes (>10ms delta)
                    delta = abs(self.smoothed_baseline - previous_smoothed) * 1000  # ms
                    if delta > 10.0:
                        logger.info(f"[BASELINE_SMOOTHING] Raw: {raw_baseline*1000:.3f}ms, "
                                   f"Smoothed: {self.smoothed_baseline*1000:.3f}ms "
                                   f"(Δ{delta:.1f}ms from prev)")
                    else:
                        logger.debug(f"[BASELINE_SMOOTHING] Raw: {raw_baseline*1000:.3f}ms, "
                                    f"Smoothed: {self.smoothed_baseline*1000:.3f}ms")

                baseline = self.smoothed_baseline
            else:
                # Fallback: simple mean
                baseline = raw_baseline
                logger.debug(f"[NORMALIZATION] NTP baseline from {len(ntp_measurements)} recent measurements: {baseline*1000:.3f}ms")

            return baseline

    def get_recent_measurements(self, window_seconds: int = None, normalize: bool = True) -> Tuple[List[Tuple[float, float]], Optional[float]]:
        """
        Get recent offset measurements for TSFM model input.

        Args:
            window_seconds: Time window in seconds. If None, returns ALL measurements (no filtering).
                          This prevents the dataset from being artificially truncated over time.
            normalize: If True, subtract NTP baseline to create zero-centered training data.
                      This prevents the model from learning absolute bias and focuses learning
                      on drift patterns and relative changes.

        Returns:
            Tuple of (measurements, normalization_bias):
            - measurements: List of (timestamp, offset) tuples sorted by timestamp
            - normalization_bias: The value subtracted from offsets (None if not normalized)
        """
        with self.lock:
            if window_seconds is None:
                # No time filtering - return ALL accumulated measurements
                # This is the default to prevent dataset shrinkage over time
                measurements = [(ts, data['offset']) for ts, data in self.measurement_dataset.items()]
            else:
                # Apply time window filtering
                current_time = time.time()
                cutoff_time = current_time - window_seconds

                measurements = []
                for timestamp, data in self.measurement_dataset.items():
                    if timestamp >= cutoff_time:
                        measurements.append((timestamp, data['offset']))

            # Sort by timestamp
            measurements.sort(key=lambda x: x[0])

            # Apply offset normalization if requested
            normalization_bias = None
            if normalize and measurements:
                # Get recent NTP baseline for normalization
                normalization_bias = self.get_recent_ntp_baseline(lookback_count=10)

                if normalization_bias is not None:
                    # Subtract baseline from all offsets to create zero-centered data
                    measurements = [(ts, offset - normalization_bias) for ts, offset in measurements]
                    logger.info(f"[NORMALIZATION] ✓ Applied offset normalization: baseline={normalization_bias*1000:.3f}ms")

            # DETAILED DEBUG: Show what we're returning for ML input
            if measurements:
                offsets_ms = [m[1]*1000 for m in measurements]
                logger.info(f"[DATASET_GET_MEASUREMENTS] ◀◀◀ RETURNING data for ML model:")
                logger.info(f"  Count: {len(measurements)} measurements")
                logger.info(f"  Time range: {measurements[0][0]:.0f} - {measurements[-1][0]:.0f}")
                logger.info(f"  Duration: {measurements[-1][0] - measurements[0][0]:.0f}s")
                logger.info(f"  Offset range: {min(offsets_ms):.3f}ms - {max(offsets_ms):.3f}ms")
                logger.info(f"  Offset mean: {np.mean(offsets_ms):.3f}ms")
                logger.info(f"  Offset std: {np.std(offsets_ms):.3f}ms")
                logger.info(f"  Normalized: {normalize} (baseline: {normalization_bias*1000:.3f}ms)" if normalization_bias else f"  Normalized: {normalize}")
                logger.info(f"  Window filter: {window_seconds}s" if window_seconds else "  Window filter: None (all data)")
                # Show first 5 and last 5 measurements
                logger.info(f"  First 5: {[(t, o*1000) for t, o in measurements[:5]]}")
                logger.info(f"  Last 5: {[(t, o*1000) for t, o in measurements[-5:]]}")
            else:
                logger.warning(f"[DATASET_GET_MEASUREMENTS] ◀◀◀ EMPTY dataset! No measurements to return")

            return measurements, normalization_bias
    
    def calculate_drift_from_measurements(self, measurements: List[Tuple[float, float]]) -> float:
        """Calculate drift rate from recent offset measurements"""
        if len(measurements) < 2:
            return 0.0
        
        # Simple linear regression to estimate drift
        timestamps = np.array([m[0] for m in measurements])
        offsets = np.array([m[1] for m in measurements])
        
        # Calculate drift as slope of offset vs time
        if len(timestamps) > 1:
            drift = np.polyfit(timestamps, offsets, 1)[0]  # Slope of linear fit
            return float(drift)
        
        return 0.0


class RealDataPipeline:
    """
    Complete pipeline replacing synthetic ClockDataGenerator.
    
    Integrates:
    - Real NTP measurements
    - Predictive scheduling
    - Model fusion  
    - Error bounds calculation
    - Dataset management
    """
    
    def __init__(self, config_path: str):
        """Initialize real data pipeline with configuration"""
        self.config_path = config_path
        
        # Initialize components
        self.ntp_collector = ClockMeasurementCollector(config_path)
        self.predictive_scheduler = PredictiveScheduler(config_path)
        self.system_metrics = SystemMetricsCollector(collection_interval=1.0)

        # Load configuration first
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # FIX #1: Load max_dataset_size from config for sliding window
        max_dataset_size = config.get('prediction_scheduling', {}).get('dataset', {}).get('max_history_size', 1000)
        self.dataset_manager = DatasetManager(max_dataset_size=max_dataset_size)
        logger.info(f"Dataset Manager: max_history_size={max_dataset_size} (sliding window)")

        # Baseline Smoothing (Phase 2) - Exponential Moving Average
        baseline_smoothing_config = config.get('prediction_scheduling', {}).get('baseline_smoothing', {})
        self.baseline_smoothing_enabled = baseline_smoothing_config.get('enabled', True)
        self.baseline_smoothing_alpha = baseline_smoothing_config.get('alpha', 0.3)  # 0.3 = moderate smoothing
        self.smoothed_baseline = None  # Will be initialized on first NTP measurement
        logger.info(f"[BASELINE_SMOOTHING] Enabled: {self.baseline_smoothing_enabled}, "
                   f"alpha={self.baseline_smoothing_alpha} ({'moderate' if self.baseline_smoothing_alpha == 0.3 else 'light' if self.baseline_smoothing_alpha > 0.3 else 'heavy'} smoothing)")

        # LAYER 1: Ultra-Aggressive Capping (simplified formula)
        adaptive_capping_config = config.get('prediction_scheduling', {}).get('adaptive_capping', {})
        self.max_multiplier = adaptive_capping_config.get('max_multiplier', 1.5)  # 1.5x NTP default
        self.absolute_max = adaptive_capping_config.get('absolute_max', 0.300)  # 300ms hard limit
        self.absolute_min = adaptive_capping_config.get('absolute_min', 0.020)  # 20ms minimum
        logger.info(f"[LAYER 1] Ultra-Aggressive Capping: max_multiplier={self.max_multiplier}x, "
                   f"absolute_max={self.absolute_max*1000:.0f}ms, absolute_min={self.absolute_min*1000:.0f}ms")

        # LAYER 2: Adaptive Sanity Check Filter
        sanity_check_config = config.get('prediction_scheduling', {}).get('sanity_check', {})
        self.sanity_check_enabled = sanity_check_config.get('enabled', True)

        # Adaptive range multiplier (replaces hardcoded absolute_limit)
        self.adaptive_range_multiplier = sanity_check_config.get('adaptive_range_multiplier', 10.0)
        self.absolute_min_bound = sanity_check_config.get('absolute_min_bound', 0.001)  # 1ms floor
        self.absolute_max_bound = sanity_check_config.get('absolute_max_bound', 10.0)   # 10s ceiling

        # Fallback limits (used if not enough NTP data for adaptive)
        self.sanity_relative_limit = sanity_check_config.get('relative_limit', 5.0)  # 5x NTP
        self.sanity_statistical_limit = sanity_check_config.get('statistical_limit', 3.0)  # 3 sigma

        # Adaptive bounds (calculated from NTP observations)
        self.adaptive_lower_bound = self.absolute_min_bound  # Start at floor
        self.adaptive_upper_bound = self.absolute_max_bound  # Start at ceiling

        logger.info(f"[LAYER 2] Adaptive Sanity Check: enabled={self.sanity_check_enabled}")
        logger.info(f"  Range multiplier: {self.adaptive_range_multiplier}x (predictions within avg ± {self.adaptive_range_multiplier}×avg)")
        logger.info(f"  Hard bounds: [{self.absolute_min_bound*1000:.1f}ms, {self.absolute_max_bound*1000:.0f}ms]")
        logger.info(f"  Fallback: relative={self.sanity_relative_limit}x, statistical={self.sanity_statistical_limit}σ")

        # LAYER 3: Confidence-based capping (DISABLED)
        confidence_capping_config = config.get('prediction_scheduling', {}).get('confidence_capping', {})
        self.confidence_capping_enabled = confidence_capping_config.get('enabled', False)
        logger.info(f"[LAYER 3] Confidence Capping: DISABLED (simplified to single capping method)")

        fusion_config = config.get('prediction_scheduling', {}).get('fusion', {})
        self.fusion_engine = PredictionFusionEngine(fusion_config)

        # NTP Correction Configuration - Dataset-Only Correction
        ntp_correction_config = config.get('prediction_scheduling', {}).get('ntp_correction', {})
        self.ntp_correction_enabled = ntp_correction_config.get('enabled', True)
        self.ntp_correction_method = ntp_correction_config.get('method', 'linear')
        self.ntp_offset_uncertainty = ntp_correction_config.get('offset_uncertainty', 0.001)
        self.ntp_drift_uncertainty = ntp_correction_config.get('drift_uncertainty', 0.0001)

        logger.info(f"NTP Correction (Dataset-Only): enabled={self.ntp_correction_enabled}, "
                   f"method={self.ntp_correction_method}, "
                   f"offset_unc={self.ntp_offset_uncertainty*1000:.2f}ms, "
                   f"drift_unc={self.ntp_drift_uncertainty*1e6:.2f}μs/s")

        # Pipeline state
        self.initialized = False
        self.last_ntp_time = 0
        self.last_ntp_offset = None
        self.last_ntp_uncertainty = None
        self.last_processed_ntp_count = 0  # Track how many NTP measurements we've processed
        self.warm_up_complete = False
        self.lock = threading.Lock()

        # FIX A: Track recent NTP measurements for robust adaptive capping
        self.recent_ntp_measurements = []  # List of (timestamp, offset) tuples
        self.max_recent_ntp_count = 5  # Keep last 5 NTP measurements

        # Statistics
        self.stats = {
            'total_corrections': 0,
            'ntp_measurements': 0,
            'prediction_cache_hits': 0,
            'prediction_cache_misses': 0,
            'retrospective_corrections': 0,
            'dataset_writes_from_scheduler': 0,  # NEW: Track scheduler writes
            'dataset_writes_from_api': 0,  # NEW: Should stay at 0 after fix
            'ntp_corrections_applied': 0,
            'ntp_corrections_skipped': 0
        }
        
    def initialize(self, cpu_model=None, gpu_model=None):
        """
        Initialize the pipeline with model interfaces.

        Args:
            cpu_model: REQUIRED - Short-term CPU model for frequent predictions
            gpu_model: OPTIONAL - Long-term model for infrequent predictions (can run on CPU or GPU)

        Raises:
            ValueError: If cpu_model is not provided
        """
        logger.info("Initializing real data pipeline...")

        # CRITICAL: Short-term CPU model is REQUIRED for ChronoTick to work
        if cpu_model is None:
            error_msg = (
                "ChronoTick requires at least a short-term CPU model to function. "
                "The short-term model provides frequent (1Hz) predictions for immediate time corrections. "
                "Without ML models, ChronoTick degrades to pure NTP with simple linear extrapolation. "
                "\n\nTo fix this, initialize the pipeline with a CPU model:"
                "\n  factory = TSFMFactory()"
                "\n  cpu_model = factory.load_model('chronos', device='cpu')"
                "\n  pipeline.initialize(cpu_model=cpu_model_wrapper)"
                "\n\nThe long-term GPU model is optional but recommended for better long-range predictions."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # GPU model is optional but recommended
        if gpu_model is None:
            logger.warning(
                "Long-term model not provided - running in short-term only mode. "
                "For best results, provide both cpu_model and gpu_model for dual-model fusion."
            )

        # Set model interfaces for predictive scheduler
        # CRITICAL FIX: Inject dataset_manager so scheduler writes predictions at 1Hz for autoregressive training
        self.predictive_scheduler.set_model_interfaces(
            cpu_model, gpu_model, self.fusion_engine, dataset_manager=self.dataset_manager
        )
        logger.info(f"Models configured: cpu_model={'Yes' if cpu_model else 'No'}, "
                   f"gpu_model={'Yes' if gpu_model else 'No'}, "
                   f"fusion={'Enabled' if cpu_model and gpu_model else 'Disabled'}, "
                   f"dataset_manager_injected={'Yes' if self.dataset_manager else 'No'}")

        # Start components (but NOT scheduler yet - wait for data)
        self.system_metrics.start_collection()
        self.ntp_collector.start_collection()
        # DON'T start scheduler yet - wait until warmup completes and we have data

        self.initialized = True
        logger.info("Real data pipeline initialized successfully")

        # Wait for warm-up phase
        warm_up_duration = self.ntp_collector.warm_up_duration
        logger.info(f"Starting {warm_up_duration}s warm-up phase...")
        logger.info(f"Scheduler will start AFTER warmup completes to ensure sufficient data")

        # Use actual warmup duration from config
        threading.Timer(warm_up_duration, self._mark_warm_up_complete).start()

    def _mark_warm_up_complete(self):
        """Mark warm-up phase as complete and START SCHEDULER"""
        logger.info("Warm-up timer fired - checking if ready to switch to predictive mode")

        # CRITICAL: Populate dataset with NTP measurements BEFORE starting scheduler
        logger.info("Populating dataset with collected NTP measurements...")
        self._check_for_ntp_updates(time.time())

        dataset_size = len(self.dataset_manager.get_recent_measurements())
        logger.info(f"Dataset populated with {dataset_size} NTP measurements")

        # CRITICAL: Need >= 10 measurements for ML models to work
        MIN_DATASET_SIZE = 10
        if dataset_size < MIN_DATASET_SIZE:
            logger.warning(f"Dataset has only {dataset_size} measurements (need >= {MIN_DATASET_SIZE} for ML).")
            logger.warning(f"Waiting for more NTP measurements before starting scheduler...")
            logger.warning(f"STAYING IN WARMUP MODE until sufficient data is collected")

            # Schedule retry in 5 seconds to check again
            threading.Timer(5.0, self._retry_scheduler_start).start()
            return  # DON'T start scheduler yet! STAY IN WARMUP MODE

        # Dataset is sufficient - start scheduler now and mark warmup complete
        self._start_scheduler_with_data()

    def _retry_scheduler_start(self):
        """Retry starting scheduler after waiting for more NTP data"""
        logger.info("Checking if enough NTP data has been collected...")
        self._check_for_ntp_updates(time.time())

        # FIX: get_recent_measurements() returns (measurements, normalization_bias) tuple
        measurements, _ = self.dataset_manager.get_recent_measurements(normalize=False)
        dataset_size = len(measurements)
        logger.info(f"Dataset now has {dataset_size} measurements")

        MIN_DATASET_SIZE = 10
        if dataset_size < MIN_DATASET_SIZE:
            logger.warning(f"Still waiting... (have {dataset_size}, need {MIN_DATASET_SIZE})")
            # Keep retrying every 5 seconds
            threading.Timer(5.0, self._retry_scheduler_start).start()
        else:
            logger.info(f"Sufficient data collected! Starting scheduler...")
            self._start_scheduler_with_data()

    def _start_scheduler_with_data(self):
        """Actually start the scheduler once we have sufficient data"""
        # FIX: get_recent_measurements() returns (measurements, normalization_bias) tuple
        measurements, _ = self.dataset_manager.get_recent_measurements(normalize=False)
        dataset_size = len(measurements)
        logger.info(f"Starting predictive scheduler with {dataset_size} measurements...")
        self.predictive_scheduler.start_scheduler()
        logger.info("Predictive scheduler started - ML predictions now active")

        # CRITICAL: Only NOW mark warmup as complete (scheduler is running)
        self.warm_up_complete = True
        logger.info("Warm-up phase complete - switching to predictive mode")
    
    def shutdown(self):
        """Shutdown the pipeline"""
        logger.info("Shutting down real data pipeline...")

        self.ntp_collector.stop_collection()

        # Only stop scheduler if it was started
        if self.warm_up_complete and self.predictive_scheduler.scheduler_running:
            self.predictive_scheduler.stop_scheduler()

        self.system_metrics.stop_collection()

        self.initialized = False
        logger.info("Real data pipeline shutdown complete")
    
    @debug_trace_pipeline(include_args=True, include_result=True, include_timing=True)
    def get_real_clock_correction(self, current_time: float) -> CorrectionWithBounds:
        """
        Main function replacing synthetic data generation.
        Returns real clock correction with proper error bounds.
        
        This is the function that replaces:
        data_generator.generate_offset_sequence(...)
        """
        if not self.initialized:
            return self._fallback_correction(current_time)
        
        with self.lock:
            self.stats['total_corrections'] += 1
        
        # Check for new NTP measurements and apply retrospective correction
        self._check_for_ntp_updates(current_time)
        
        # During warm-up, use direct NTP measurements
        if not self.warm_up_complete:
            return self._get_warm_up_correction(current_time)
        
        # Normal operation: use predictive corrections
        return self._get_predictive_correction(current_time)
    
    @debug_trace_pipeline(include_args=True, include_result=False, include_timing=True)
    def _check_for_ntp_updates(self, current_time: float):
        """
        Check for new NTP measurements and apply dataset-only correction.

        This is the ONLY place where NTP correction happens - by correcting
        the historical dataset. NO real-time blending!
        """
        logger.debug(f"_check_for_ntp_updates called: current_time={current_time}, "
                    f"last_processed_count={self.last_processed_ntp_count}")

        # Get ALL recent measurements from collector (not just the last one!)
        all_measurements = self.ntp_collector.get_recent_measurements(window_seconds=500)

        logger.debug(f"NTP collector has {len(all_measurements)} total measurements, "
                    f"already processed {self.last_processed_ntp_count}")

        # Process any new measurements we haven't seen yet
        if len(all_measurements) > self.last_processed_ntp_count:
            new_measurements = all_measurements[self.last_processed_ntp_count:]
            logger.info(f"Found {len(new_measurements)} new NTP measurements to process")

            # Apply dataset correction for each new NTP measurement
            for timestamp, offset, uncertainty in new_measurements:
                # Create NTPMeasurement object
                ntp_measurement = NTPMeasurement(
                    offset=offset,
                    delay=0.0,  # Not stored in offset_measurements
                    stratum=1,  # Assume good quality
                    precision=uncertainty,
                    server="ntp",
                    timestamp=timestamp,
                    uncertainty=uncertainty
                )

                # Apply dataset-only correction using configured method
                # SKIP correction during warmup (no predictions to correct yet)
                if self.ntp_correction_enabled and self.warm_up_complete:
                    # Use uncertainties from configuration
                    self.dataset_manager.apply_ntp_correction(
                        ntp_measurement,
                        method=self.ntp_correction_method,
                        offset_uncertainty=self.ntp_offset_uncertainty,
                        drift_uncertainty=self.ntp_drift_uncertainty
                    )
                else:
                    # Just add NTP without correction (warmup or correction disabled)
                    self.dataset_manager.add_ntp_measurement(ntp_measurement)
                    if not self.warm_up_complete:
                        logger.debug(f"[NTP_WARMUP] Skipping correction during warmup phase")

                # Track latest NTP
                with self.lock:
                    self.stats['ntp_measurements'] += 1
                    if self.ntp_correction_enabled and self.warm_up_complete:
                        self.stats['ntp_corrections_applied'] += 1
                    else:
                        self.stats['ntp_corrections_skipped'] += 1
                    self.last_ntp_time = timestamp
                    self.last_ntp_offset = offset
                    self.last_ntp_uncertainty = uncertainty

                # PERIODIC HEALTH CHECK: Log dataset health after every NTP measurement
                # This verifies 1Hz coverage and detects issues early
                if self.warm_up_complete:
                    self.log_dataset_health()

                    # FIX A: Track recent NTP measurements for robust adaptive capping
                    self.recent_ntp_measurements.append((timestamp, offset))
                    # Keep only the last N measurements
                    if len(self.recent_ntp_measurements) > self.max_recent_ntp_count:
                        self.recent_ntp_measurements = self.recent_ntp_measurements[-self.max_recent_ntp_count:]

                    # Update adaptive sanity bounds based on new NTP measurement
                    self._update_adaptive_sanity_bounds()

            # Update count of processed measurements
            self.last_processed_ntp_count = len(all_measurements)

            logger.info(f"Dataset now has {len(self.dataset_manager.get_recent_measurements())} measurements")
            logger.info(f"Latest NTP: t={self.last_ntp_time:.0f}, offset={self.last_ntp_offset*1000:.2f}ms")

    # Old real-time blending methods removed - now using dataset-only correction


    @debug_trace_pipeline(include_args=True, include_result=True, include_timing=True)
    def _get_warm_up_correction(self, current_time: float) -> CorrectionWithBounds:
        """Get correction during warm-up phase using direct NTP measurements"""
        logger.debug(f"_get_warm_up_correction called: current_time={current_time}, warm_up_complete={self.warm_up_complete}")
        latest_offset = self.ntp_collector.get_latest_offset()

        if latest_offset is not None:
            # Use latest NTP measurement with conservative uncertainty
            return CorrectionWithBounds(
                offset_correction=latest_offset,
                drift_rate=0.0,  # No drift during warm-up
                offset_uncertainty=0.005,  # Conservative 5ms uncertainty
                drift_uncertainty=0.000001,  # Conservative drift uncertainty
                prediction_time=current_time,
                valid_until=current_time + 60,
                confidence=0.7,  # Lower confidence during warm-up
                source="ntp_warm_up"
            )
        else:
            return self._fallback_correction(current_time)
    
    @debug_trace_pipeline(include_args=True, include_result=True, include_timing=True)
    def _get_predictive_correction(self, current_time: float) -> CorrectionWithBounds:
        """Get correction using predictive scheduling and model fusion"""
        logger.info(f"[CACHE_LOOKUP] _get_predictive_correction called: current_time={current_time:.2f}")

        # Log cache state BEFORE lookup
        cache_size = len(self.predictive_scheduler.prediction_cache)
        if cache_size > 0:
            cache_keys = sorted(self.predictive_scheduler.prediction_cache.keys())
            logger.info(f"[CACHE_STATE] Cache has {cache_size} entries, range: {cache_keys[0]:.0f} - {cache_keys[-1]:.0f}")
        else:
            logger.info(f"[CACHE_STATE] Cache is EMPTY")

        # Try to get fused correction from scheduler
        logger.info(f"[CACHE_LOOKUP] Calling get_fused_correction...")
        correction = self.predictive_scheduler.get_fused_correction(current_time)
        logger.info(f"[CACHE_RESULT] get_fused_correction returned: {correction is not None}, source={correction.source if correction else None}")

        if correction:
            with self.lock:
                self.stats['prediction_cache_hits'] += 1
            logger.info(f"[CACHE_HIT] Using fused correction from cache")

            # LAYER 1 & 2: Apply ultra-aggressive capping + sanity check
            correction, adaptive_capped, sanity_passed = self._apply_adaptive_capping(correction, current_time)

            # LAYER 3: Confidence capping DISABLED (skip)
            confidence_capped = False

            # Combine capping flags: adaptive_capped OR sanity_failed
            was_capped = adaptive_capped or (not sanity_passed)

            # CRITICAL: DO NOT write to dataset here!
            # Scheduler already wrote this prediction at 1Hz for autoregressive training.
            # Writing here would:
            #   1. Create duplicate/conflicting writes (race condition)
            #   2. Overwrite raw predictions with capped versions
            #   3. Confuse backtracking (was_capped flag inconsistency)
            # Capping only affects the API RESPONSE, not the training dataset.
            logger.debug(f"[API_SERVE] Serving prediction: offset={correction.offset_correction*1000:.2f}ms, "
                        f"capped={was_capped}, source={correction.source}")

            return correction

        # Fallback: get individual model predictions
        logger.info(f"[CACHE_LOOKUP] Fused correction missed, trying get_correction_at_time...")
        cpu_correction = self.predictive_scheduler.get_correction_at_time(current_time)
        logger.info(f"[CACHE_RESULT] get_correction_at_time returned: {cpu_correction is not None}, source={cpu_correction.source if cpu_correction else None}")

        if cpu_correction:
            with self.lock:
                self.stats['prediction_cache_hits'] += 1
            logger.info(f"[CACHE_HIT] Using CPU correction from cache")

            # FIX #2: Apply adaptive capping BEFORE returning to API (FIX C: returns tuple)
            cpu_correction, adaptive_capped = self._apply_adaptive_capping(cpu_correction, current_time)

            # FIX #3: Apply confidence-based capping BEFORE returning to API (FIX C: returns tuple)
            cpu_correction, confidence_capped = self._apply_confidence_based_capping(cpu_correction)

            # FIX C: Combine capping flags (True if either capped)
            was_capped = adaptive_capped or confidence_capped

            # CRITICAL: DO NOT write to dataset here!
            # Scheduler already wrote this prediction at 1Hz for autoregressive training.
            # See comment above for rationale.
            logger.debug(f"[API_SERVE] Serving prediction: offset={cpu_correction.offset_correction*1000:.2f}ms, "
                        f"capped={was_capped}, source={cpu_correction.source}")

            return cpu_correction

        # Cache miss - use fallback
        with self.lock:
            self.stats['prediction_cache_misses'] += 1

        logger.error(f"[CACHE_MISS] Both cache lookups failed - calling fallback (will raise RuntimeError)")
        return self._fallback_correction(current_time)
    
    @debug_trace_pipeline(include_args=True, include_result=True, include_timing=True)
    def _fallback_correction(self, current_time: float) -> CorrectionWithBounds:
        """FAIL LOUDLY - no fallbacks in research mode"""
        logger.debug(f"_fallback_correction called: current_time={current_time}, initialized={self.initialized}")

        # Get diagnostic info for error message
        latest_offset = self.ntp_collector.get_latest_offset()
        dataset_size = len(self.dataset_manager.get_recent_measurements())
        cache_size = len(self.predictive_scheduler.prediction_cache)

        # RESEARCH MODE: NO FALLBACKS - CRASH LOUDLY
        error_msg = (
            f"CRITICAL: ML prediction cache miss - NO FALLBACKS IN RESEARCH MODE!\n"
            f"Time: {current_time}\n"
            f"Initialized: {self.initialized}\n"
            f"Warm-up complete: {self.warm_up_complete}\n"
            f"Dataset size: {dataset_size} measurements\n"
            f"Prediction cache size: {cache_size} entries\n"
            f"Latest NTP offset: {latest_offset}\n"
            f"Scheduler running: {self.predictive_scheduler.scheduler_running}\n"
            f"\n"
            f"REFUSING to serve NTP fallback data - this is a RESEARCH system.\n"
            f"We need ML predictions or nothing. Crash = bug to fix, not hide.\n"
            f"\n"
            f"Possible causes:\n"
            f"1. Scheduler not started yet (check warm_up_complete)\n"
            f"2. ML model failing silently (check logs for CRITICAL errors)\n"
            f"3. Prediction cache not being populated (check scheduler stats)\n"
            f"4. Dataset has insufficient data for ML (need >= 10 measurements)\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _apply_adaptive_capping(self, correction: CorrectionWithBounds, current_time: float) -> tuple:
        """
        LAYER 1: Adaptive Capping (follows NTP baseline).

        Cap formula: Cap = max(NTP_magnitude * multiplier, absolute_min)
        - Cap grows with NTP baseline (allows correcting unsynchronized clocks)
        - No absolute_max here (that's for sanity check layer only)
        - Minimum cap prevents predictions from going too close to zero

        Returns:
            Tuple of (capped_correction, was_capped, sanity_passed)
        """
        if self.last_ntp_offset is None:
            logger.debug(f"[LAYER 1] No NTP reference yet, skipping capping")
            return correction, False, True

        # Calculate cap RELATIVE to NTP baseline (not absolute)
        last_ntp_magnitude = abs(self.last_ntp_offset)
        cap = last_ntp_magnitude * self.max_multiplier
        cap = max(cap, self.absolute_min)  # At least min_cap to prevent near-zero predictions
        # NO absolute_max ceiling here! Let cap follow NTP baseline for unsynchronized clocks

        prediction_magnitude = abs(correction.offset_correction)

        # LAYER 2: Sanity check BEFORE capping
        sanity_passed, sanity_reason = self._sanity_check_prediction(correction, current_time)

        if not sanity_passed:
            # Prediction failed sanity check - use last NTP as fallback
            logger.error(f"[LAYER 2] ❌ SANITY CHECK FAILED: {sanity_reason}")
            logger.error(f"[LAYER 2] Replacing insane prediction ({correction.offset_correction*1000:.1f}ms) with last NTP ({self.last_ntp_offset*1000:.1f}ms)")

            # Replace with last NTP (safe fallback)
            safe_correction = CorrectionWithBounds(
                offset_correction=self.last_ntp_offset,
                drift_rate=0.0,
                offset_uncertainty=self.last_ntp_uncertainty if self.last_ntp_uncertainty else 0.010,
                drift_uncertainty=0.0001,
                prediction_time=correction.prediction_time,
                valid_until=correction.valid_until,
                confidence=0.5,  # Low confidence for fallback
                source="sanity_fallback_ntp",
                quantiles=None
            )
            return safe_correction, True, False  # Capped, sanity failed

        # Sanity passed - check if prediction is below minimum (FIX: enforce absolute_min)
        if prediction_magnitude < self.absolute_min:
            original_prediction = correction.offset_correction
            boosted_prediction = np.sign(correction.offset_correction) * self.absolute_min
            boosted_confidence = correction.confidence * 0.9  # Slight confidence reduction for boosting

            logger.warning(f"[LAYER 1] ⬆️ BOOSTING PREDICTION TO MINIMUM:")
            logger.warning(f"  Original: {original_prediction*1000:.1f}ms")
            logger.warning(f"  Minimum: {self.absolute_min*1000:.1f}ms")
            logger.warning(f"  Boosted to: {boosted_prediction*1000:.1f}ms")
            logger.warning(f"  Confidence: {correction.confidence:.2f} → {boosted_confidence:.2f}")

            boosted_correction = CorrectionWithBounds(
                offset_correction=boosted_prediction,
                drift_rate=correction.drift_rate,
                offset_uncertainty=correction.offset_uncertainty,
                drift_uncertainty=correction.drift_uncertainty,
                prediction_time=correction.prediction_time,
                valid_until=correction.valid_until,
                confidence=boosted_confidence,
                source=correction.source,
                quantiles=correction.quantiles
            )
            return boosted_correction, True, True  # Boosted (counted as capped), sanity passed
        # Check if prediction exceeds cap maximum
        elif prediction_magnitude > cap:
            original_prediction = correction.offset_correction
            capped_prediction = np.sign(correction.offset_correction) * cap
            capped_confidence = correction.confidence * 0.7

            logger.warning(f"[LAYER 1] ✂️ CAPPING PREDICTION:")
            logger.warning(f"  Original: {original_prediction*1000:.1f}ms")
            logger.warning(f"  Cap: {cap*1000:.1f}ms (NTP={last_ntp_magnitude*1000:.1f}ms × {self.max_multiplier})")
            logger.warning(f"  Capped to: {capped_prediction*1000:.1f}ms")
            logger.warning(f"  Confidence: {correction.confidence:.2f} → {capped_confidence:.2f}")

            capped_correction = CorrectionWithBounds(
                offset_correction=capped_prediction,
                drift_rate=correction.drift_rate,
                offset_uncertainty=correction.offset_uncertainty,
                drift_uncertainty=correction.drift_uncertainty,
                prediction_time=correction.prediction_time,
                valid_until=correction.valid_until,
                confidence=capped_confidence,
                source=correction.source,
                quantiles=correction.quantiles
            )
            return capped_correction, True, True  # Capped, sanity passed
        else:
            logger.debug(f"[LAYER 1] ✓ Prediction OK: {self.absolute_min*1000:.1f}ms <= {prediction_magnitude*1000:.1f}ms <= {cap*1000:.1f}ms")
            return correction, False, True  # Not capped, sanity passed

    def _update_adaptive_sanity_bounds(self):
        """
        Update adaptive sanity bounds based on recent NTP measurements.
        Called after each NTP correction to adapt to system clock behavior.

        Formula:
          avg_error = mean(|recent_ntp_offsets|)
          lower_bound = max(absolute_min_bound, avg_error - multiplier × avg_error)
          upper_bound = min(absolute_max_bound, avg_error + multiplier × avg_error)
        """
        if len(self.recent_ntp_measurements) < 2:
            # Not enough data - use hard bounds
            logger.debug(f"[ADAPTIVE_SANITY] Not enough NTP data ({len(self.recent_ntp_measurements)} measurements), using hard bounds")
            return

        # Calculate average NTP error magnitude
        recent_magnitudes = [abs(offset) for _, offset in self.recent_ntp_measurements]
        avg_error = statistics.mean(recent_magnitudes)

        # Calculate adaptive range: avg ± (multiplier × avg)
        # Lower: avg - (multiplier × avg) = avg(1 - multiplier)
        # Upper: avg + (multiplier × avg) = avg(1 + multiplier)
        lower_bound = avg_error * (1 - self.adaptive_range_multiplier)
        upper_bound = avg_error * (1 + self.adaptive_range_multiplier)

        # Enforce hard bounds (never go beyond these)
        self.adaptive_lower_bound = max(self.absolute_min_bound, lower_bound)
        self.adaptive_upper_bound = min(self.absolute_max_bound, upper_bound)

        logger.info(f"[ADAPTIVE_SANITY] ✅ Updated adaptive bounds from {len(self.recent_ntp_measurements)} NTP measurements:")
        logger.info(f"  Avg NTP error: {avg_error*1000:.2f}ms")
        logger.info(f"  Calculated range: [{lower_bound*1000:.2f}ms, {upper_bound*1000:.2f}ms]")
        logger.info(f"  Applied range: [{self.adaptive_lower_bound*1000:.2f}ms, {self.adaptive_upper_bound*1000:.2f}ms]")
        logger.info(f"  (Hard bounds: [{self.absolute_min_bound*1000:.2f}ms, {self.absolute_max_bound*1000:.0f}ms])")

    def _sanity_check_prediction(self, correction: CorrectionWithBounds, current_time: float) -> tuple:
        """
        LAYER 2: Adaptive sanity check filter to catch catastrophic predictions.

        Uses adaptive bounds learned from system clock behavior:
        - Lower bound: avg_error - (multiplier × avg_error)
        - Upper bound: avg_error + (multiplier × avg_error)

        Fallback checks (if adaptive bounds not yet calculated):
        1. Relative limit: prediction < 5x average NTP
        2. Statistical limit: prediction < 3σ from NTP mean

        Returns:
            (passed, reason) tuple
        """
        if not self.sanity_check_enabled:
            return True, "Sanity check disabled"

        if self.last_ntp_offset is None:
            return True, "No NTP reference yet"

        pred_magnitude = abs(correction.offset_correction)

        # Check 1: Adaptive absolute bounds (primary check)
        if pred_magnitude < self.adaptive_lower_bound:
            return False, f"Below adaptive lower bound: {pred_magnitude*1000:.1f}ms < {self.adaptive_lower_bound*1000:.1f}ms"
        if pred_magnitude > self.adaptive_upper_bound:
            return False, f"Exceeds adaptive upper bound: {pred_magnitude*1000:.1f}ms > {self.adaptive_upper_bound*1000:.1f}ms"

        # Check 2: Relative to NTP (5x threshold)
        if len(self.recent_ntp_measurements) >= 2:
            recent_magnitudes = [abs(offset) for _, offset in self.recent_ntp_measurements]
            ntp_avg = statistics.mean(recent_magnitudes)

            if pred_magnitude > self.sanity_relative_limit * ntp_avg:
                return False, f"{self.sanity_relative_limit}x larger than NTP avg ({pred_magnitude/ntp_avg:.1f}x)"

            # Check 3: Statistical outlier (3 sigma)
            if len(recent_magnitudes) >= 3:
                ntp_std = statistics.stdev(recent_magnitudes)
                z_score = (pred_magnitude - ntp_avg) / ntp_std if ntp_std > 0 else 0

                if abs(z_score) > self.sanity_statistical_limit:
                    return False, f"{abs(z_score):.1f}σ outlier (threshold={self.sanity_statistical_limit}σ)"

        return True, "All checks passed"

    def _apply_confidence_based_capping(self, correction: CorrectionWithBounds) -> tuple:
        """
        FIX #3: Apply confidence-based prediction capping to reduce contamination from uncertain predictions.

        ALWAYS adds to dataset (maintains evenly-spaced data for TimesFM), but caps low-confidence predictions:
        - High confidence (>=0.8): Use as-is (already went through adaptive cap)
        - Medium confidence (0.5-0.8): Cap to 1.5x last NTP magnitude
        - Low confidence (<0.5): Cap to 1.2x last NTP magnitude

        Args:
            correction: The correction to cap (already adaptive-capped)

        Returns:
            Tuple of (capped_correction, was_capped)
        """
        if self.last_ntp_offset is None:
            # No NTP reference yet - allow prediction through
            logger.debug(f"[CONFIDENCE_CAP] No NTP reference yet, skipping confidence capping")
            return correction, False

        confidence = correction.confidence

        # High confidence: Use prediction as-is (already adaptive-capped)
        if confidence >= self.high_confidence_threshold:
            logger.debug(f"[CONFIDENCE_CAP] ✓ High confidence ({confidence:.2f}), using as-is")
            return correction, False

        # Medium or low confidence: Apply additional capping relative to last NTP
        last_ntp_magnitude = abs(self.last_ntp_offset)
        prediction_magnitude = abs(correction.offset_correction)

        # Determine cap multiplier based on confidence level
        if confidence >= self.medium_confidence_threshold:
            # Medium confidence
            cap_multiplier = self.medium_confidence_multiplier
            confidence_level = "MEDIUM"
        else:
            # Low confidence
            cap_multiplier = self.low_confidence_multiplier
            confidence_level = "LOW"

        cap = last_ntp_magnitude * cap_multiplier

        # Apply cap if prediction exceeds it
        if prediction_magnitude > cap:
            original_prediction = correction.offset_correction
            capped_prediction = np.sign(correction.offset_correction) * cap

            logger.info(f"[CONFIDENCE_CAP] ✂️ CAPPING {confidence_level} CONFIDENCE PREDICTION:")
            logger.info(f"  Confidence: {confidence:.2f} ({confidence_level})")
            logger.info(f"  Original prediction: {original_prediction*1000:.1f}ms")
            logger.info(f"  Capped to: {capped_prediction*1000:.1f}ms")
            logger.info(f"  Last NTP magnitude: {last_ntp_magnitude*1000:.1f}ms")
            logger.info(f"  Cap multiplier: {cap_multiplier}x")
            logger.info(f"  Cap limit: {cap*1000:.1f}ms")

            # Return capped correction (STILL ADDED TO DATASET! FIX C: with was_capped=True)
            capped_correction = CorrectionWithBounds(
                offset_correction=capped_prediction,
                drift_rate=correction.drift_rate,
                offset_uncertainty=correction.offset_uncertainty,
                drift_uncertainty=correction.drift_uncertainty,
                prediction_time=correction.prediction_time,
                valid_until=correction.valid_until,
                confidence=confidence,  # Keep original confidence
                source=correction.source,
                quantiles=correction.quantiles
            )
            return capped_correction, True  # FIX C: Signal that confidence capping was applied
        else:
            # No additional capping needed
            logger.debug(f"[CONFIDENCE_CAP] ✓ {confidence_level} confidence ({confidence:.2f}), "
                        f"within {cap_multiplier}x bound: {prediction_magnitude*1000:.1f}ms <= {cap*1000:.1f}ms")
            return correction, False  # FIX C: No capping

    def log_dataset_health(self):
        """
        Log dataset health metrics to verify 1Hz autoregressive training.

        This helps detect issues like:
        - Sparse dataset (not 1Hz)
        - API calls overwriting scheduler predictions
        - Missing autoregressive feedback
        """
        try:
            # Get dataset info
            dataset_size = len(self.dataset_manager.measurement_dataset)

            if dataset_size == 0:
                logger.warning("[DATASET_HEALTH] Dataset is EMPTY - no training data available!")
                return

            # Get timestamp range
            timestamps = sorted(self.dataset_manager.measurement_dataset.keys())
            oldest_ts = timestamps[0]
            newest_ts = timestamps[-1]
            time_span = newest_ts - oldest_ts

            # Calculate expected vs actual samples at 1Hz
            expected_samples = int(time_span) + 1 if time_span > 0 else 1
            coverage_pct = (dataset_size / expected_samples * 100) if expected_samples > 0 else 0

            # Count sources
            ntp_count = sum(1 for ts in timestamps
                           if self.dataset_manager.measurement_dataset[ts].get('source') == 'ntp_measurement')
            pred_count = dataset_size - ntp_count

            # Log health metrics
            logger.info(f"[DATASET_HEALTH] ===== Dataset Health Check =====")
            logger.info(f"[DATASET_HEALTH] Total samples: {dataset_size}")
            logger.info(f"[DATASET_HEALTH] Time span: {time_span:.0f}s ({time_span/60:.1f}min)")
            logger.info(f"[DATASET_HEALTH] Timestamp range: [{oldest_ts:.0f}, {newest_ts:.0f}]")
            logger.info(f"[DATASET_HEALTH] Expected at 1Hz: {expected_samples} samples")
            logger.info(f"[DATASET_HEALTH] Coverage: {coverage_pct:.1f}% (should be ~100% for 1Hz)")
            logger.info(f"[DATASET_HEALTH] Sources: {ntp_count} NTP, {pred_count} predictions")
            logger.info(f"[DATASET_HEALTH] Scheduler writes: {self.stats.get('dataset_writes_from_scheduler', 0)}")
            logger.info(f"[DATASET_HEALTH] API writes: {self.stats.get('dataset_writes_from_api', 0)} (should be 0)")

            if coverage_pct < 90:
                logger.warning(f"[DATASET_HEALTH] ⚠️ LOW COVERAGE: {coverage_pct:.1f}% < 90% - autoregressive training degraded!")
            elif coverage_pct > 110:
                logger.warning(f"[DATASET_HEALTH] ⚠️ OVER-COVERAGE: {coverage_pct:.1f}% > 110% - possible duplicate writes!")
            else:
                logger.info(f"[DATASET_HEALTH] ✅ Healthy 1Hz coverage for autoregressive training")

        except Exception as e:
            logger.error(f"[DATASET_HEALTH] Error checking dataset health: {e}")

    def get_stats(self) -> dict:
        """Get pipeline statistics"""
        with self.lock:
            pipeline_stats = self.stats.copy()

        # Add component statistics
        if self.predictive_scheduler:
            pipeline_stats.update(self.predictive_scheduler.get_stats())

        if self.ntp_collector:
            pipeline_stats.update(self.ntp_collector.ntp_client.get_measurement_statistics())

        if self.system_metrics:
            pipeline_stats.update(self.system_metrics.get_metrics_summary())

        return pipeline_stats


def create_real_data_pipeline(config_path: str) -> RealDataPipeline:
    """Create a real data pipeline instance"""
    return RealDataPipeline(config_path)


if __name__ == "__main__":
    # Test the real data pipeline
    # Config files are now in project root after reorganization
    project_root = Path(__file__).parent.parent.parent.parent.parent
    config_path = project_root / "configs" / "config_enhanced_features.yaml"

    pipeline = create_real_data_pipeline(str(config_path))
    
    print("Testing Real Data Pipeline...")
    print("Initializing...")
    
    pipeline.initialize()
    
    # Wait a moment for initialization
    time.sleep(2)
    
    # Test getting corrections
    for i in range(5):
        current_time = time.time()
        correction = pipeline.get_real_clock_correction(current_time)
        
        print(f"Correction {i+1}:")
        print(f"  Offset: {correction.offset_correction*1e6:.1f}μs")
        print(f"  Drift: {correction.drift_rate*1e6:.1f}μs/s")
        print(f"  Uncertainty: ±{correction.offset_uncertainty*1e6:.1f}μs")
        print(f"  Source: {correction.source}")
        print(f"  Confidence: {correction.confidence:.2f}")
        print()
        
        time.sleep(1)
    
    # Show statistics
    stats = pipeline.get_stats()
    print("Pipeline Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    pipeline.shutdown()
    print("Real data pipeline test completed!")