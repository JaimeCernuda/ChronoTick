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
    
    def __init__(self):
        """Initialize dataset manager"""
        self.measurement_dataset = {}  # timestamp -> {offset, drift, source, uncertainty}
        self.prediction_history = []   # List of (timestamp, prediction) tuples
        self.lock = threading.RLock()  # Use reentrant lock to allow nested lock acquisition
        
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

    def add_prediction(self, timestamp: float, offset: float, drift: float,
                      source: str, uncertainty: float, confidence: float):
        """
        Add ML prediction to dataset for autoregressive training.

        Args:
            timestamp: Prediction timestamp
            offset: Predicted offset correction
            drift: Predicted drift rate
            source: Prediction source ('cpu', 'gpu', 'fusion')
            uncertainty: Prediction uncertainty
            confidence: Prediction confidence [0,1]
        """
        with self.lock:
            self.measurement_dataset[int(timestamp)] = {
                'timestamp': timestamp,
                'offset': offset,
                'drift': drift,
                'source': f'prediction_{source}',  # Tag as prediction
                'uncertainty': uncertainty,
                'confidence': confidence,
                'corrected': False
            }

            logger.debug(f"[DATASET_STORE] Stored {source} prediction: t={timestamp:.0f}, "
                        f"offset={offset*1000:.2f}ms, uncertainty={uncertainty*1000:.2f}ms")
    
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
            # Find the most recent prediction before NTP
            prediction_at_ntp = self.get_measurement_at_time(ntp_measurement.timestamp)
            if not prediction_at_ntp:
                logger.warning(f"[NTP_CORRECTION] No prediction at NTP time {ntp_measurement.timestamp}, just adding NTP")
                self.add_ntp_measurement(ntp_measurement)
                return None

            # Calculate error: error = NTP_truth - Prediction
            error = ntp_measurement.offset - prediction_at_ntp['offset']

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

            logger.info(f"[NTP_CORRECTION_{method.upper()}] Applying correction: "
                       f"error={error*1000:.2f}ms over {interval_duration:.0f}s "
                       f"[{interval_start:.0f} → {ntp_measurement.timestamp:.0f}]")

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

        Key Innovation: Make the dataset look like "what NTP would have measured" at each point.

        This method REPLACES all predictions between NTP measurements with linearly
        interpolated values based on the two NTP boundaries. This gives the ML model
        an NTP-aligned dataset to learn from for future predictions.

        Example:
        - NTP measurement at t=0: offset=10ms
        - NTP measurement at t=5: offset=20ms
        - ML predicted at t=[1,2,3,4]: [11,12,13,14]ms
        - After correction: REPLACE with [12,14,16,18]ms (linear interpolation)

        Philosophy:
        - STRONGER correction (not weaker) for better NTP alignment
        - Dataset becomes "NTP-like" so future predictions stay aligned
        - Combined with enhanced NTP (better accuracy) = winning algorithm

        Args:
            start_time: Start of interval (last NTP measurement timestamp)
            end_time: End of interval (new NTP measurement timestamp)
            error: Measured error at end_time (NTP_truth - Prediction)
            interval_duration: Duration of interval in seconds
        """
        logger.info(f"[BACKTRACKING] Starting backtracking learning correction")
        logger.info(f"[BACKTRACKING] Error: {error*1000:.2f}ms over {interval_duration:.0f}s")
        logger.info(f"[BACKTRACKING] Strategy: REPLACE predictions with interpolated NTP")

        # Get the NTP offset at the start of the interval
        # This is the last NTP measurement that was added
        start_ntp_offset = None
        if int(start_time) in self.measurement_dataset:
            start_ntp_offset = self.measurement_dataset[int(start_time)]['offset']
        else:
            logger.warning(f"[BACKTRACKING] No measurement at start_time {start_time:.0f}, skipping correction")
            return

        # Calculate the NTP offset at the end (current NTP measurement)
        # end_ntp_offset = start_ntp_offset + error
        # But we need to get the PREDICTION at end_time first
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

        # The new NTP measurement is: end_ntp_offset = end_prediction_offset + error
        end_ntp_offset = end_prediction_offset + error

        logger.info(f"[BACKTRACKING] NTP boundaries: start={start_ntp_offset*1000:.2f}ms @ t={start_time:.0f}, "
                   f"end={end_ntp_offset*1000:.2f}ms @ t={end_time:.0f}")
        logger.info(f"[BACKTRACKING] Interpolating between these values for all predictions in interval")

        # REPLACE all predictions with linearly interpolated NTP values
        correction_count = 0
        max_replacement = 0.0
        min_replacement = float('inf')

        for timestamp in range(int(start_time) + 1, int(end_time)):
            if timestamp in self.measurement_dataset:
                # Calculate interpolation weight
                alpha = (timestamp - start_time) / interval_duration if interval_duration > 0 else 0

                # Calculate what NTP "would have measured" at this point
                ntp_interpolated = start_ntp_offset + alpha * (end_ntp_offset - start_ntp_offset)

                # Get current prediction
                current_offset = self.measurement_dataset[timestamp]['offset']

                # Calculate replacement delta for logging
                replacement_delta = ntp_interpolated - current_offset

                # REPLACE prediction with interpolated NTP value
                self.measurement_dataset[timestamp]['offset'] = ntp_interpolated
                self.measurement_dataset[timestamp]['corrected'] = True

                correction_count += 1
                max_replacement = max(max_replacement, abs(replacement_delta))
                if abs(replacement_delta) > 0:
                    min_replacement = min(min_replacement, abs(replacement_delta))

                # Log sample of replacements (first 3, last 3, and every 30th)
                if correction_count <= 3 or correction_count % 30 == 0:
                    logger.debug(f"[BACKTRACKING] t={timestamp}, alpha={alpha:.3f}, "
                               f"was={current_offset*1000:.2f}ms → now={ntp_interpolated*1000:.2f}ms "
                               f"(delta={replacement_delta*1000:+.2f}ms)")

        logger.info(f"[BACKTRACKING] REPLACED {correction_count} predictions with NTP-interpolated values")
        if correction_count > 0:
            logger.info(f"[BACKTRACKING] Replacement range: {min_replacement*1000:.2f}ms to {max_replacement*1000:.2f}ms")
        logger.info(f"[BACKTRACKING] Dataset now looks like 'what NTP would have measured'")
        logger.info(f"[BACKTRACKING] Future predictions will learn from this NTP-aligned dataset")

    def apply_retrospective_correction(self, ntp_measurement: NTPMeasurement,
                                     interval_start: float):
        """
        DEPRECATED: Use apply_ntp_correction() instead.
        Kept for backward compatibility.
        """
        self.apply_ntp_correction(ntp_measurement, method='linear')
    
    def get_measurement_at_time(self, timestamp: float) -> Optional[dict]:
        """
        Get the most recent prediction measurement before the given timestamp.

        This is used by apply_ntp_correction() to find the prediction that should
        be compared against the new NTP ground truth.

        Args:
            timestamp: Target timestamp to find measurement before

        Returns:
            Most recent prediction measurement before timestamp, or None if none found
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
    
    def get_recent_measurements(self, window_seconds: int = None) -> List[Tuple[float, float]]:
        """
        Get recent offset measurements for TSFM model input.

        Args:
            window_seconds: Time window in seconds. If None, returns ALL measurements (no filtering).
                          This prevents the dataset from being artificially truncated over time.

        Returns:
            List of (timestamp, offset) tuples sorted by timestamp
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
            return measurements
    
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
        self.dataset_manager = DatasetManager()
        
        # Load fusion configuration
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
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

        # Statistics
        self.stats = {
            'total_corrections': 0,
            'ntp_measurements': 0,
            'prediction_cache_hits': 0,
            'prediction_cache_misses': 0,
            'retrospective_corrections': 0,
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
        self.predictive_scheduler.set_model_interfaces(cpu_model, gpu_model, self.fusion_engine)
        logger.info(f"Models configured: cpu_model={'Yes' if cpu_model else 'No'}, "
                   f"gpu_model={'Yes' if gpu_model else 'No'}, "
                   f"fusion={'Enabled' if cpu_model and gpu_model else 'Disabled'}")

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
        self.warm_up_complete = True
        logger.info("Warm-up phase complete - switching to predictive mode")

        # CRITICAL: Populate dataset with NTP measurements BEFORE starting scheduler
        logger.info("Populating dataset with collected NTP measurements...")
        self._check_for_ntp_updates(time.time())

        dataset_size = len(self.dataset_manager.get_recent_measurements())
        logger.info(f"Dataset populated with {dataset_size} NTP measurements")

        if dataset_size < 10:
            logger.warning(f"Dataset has only {dataset_size} measurements (need >= 10 for ML). "
                          "Models may fail until more data is collected.")

        # NOW start the scheduler with populated dataset
        logger.info("Starting predictive scheduler with populated dataset...")
        self.predictive_scheduler.start_scheduler()
        logger.info("Predictive scheduler started - ML predictions now active")
    
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

            # NO real-time NTP blending! Dataset correction handles everything.
            # Just store the prediction and return it.
            self.dataset_manager.add_prediction(
                timestamp=current_time,
                offset=correction.offset_correction,
                drift=correction.drift_rate,
                source=correction.source,
                uncertainty=correction.offset_uncertainty,
                confidence=correction.confidence
            )

            return correction

        # Fallback: get individual model predictions
        logger.info(f"[CACHE_LOOKUP] Fused correction missed, trying get_correction_at_time...")
        cpu_correction = self.predictive_scheduler.get_correction_at_time(current_time)
        logger.info(f"[CACHE_RESULT] get_correction_at_time returned: {cpu_correction is not None}, source={cpu_correction.source if cpu_correction else None}")

        if cpu_correction:
            with self.lock:
                self.stats['prediction_cache_hits'] += 1
            logger.info(f"[CACHE_HIT] Using CPU correction from cache")

            # NO real-time NTP blending! Dataset correction handles everything.
            # Just store the prediction and return it.
            self.dataset_manager.add_prediction(
                timestamp=current_time,
                offset=cpu_correction.offset_correction,
                drift=cpu_correction.drift_rate,
                source=cpu_correction.source,
                uncertainty=cpu_correction.offset_uncertainty,
                confidence=cpu_correction.confidence
            )

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
    config_path = Path(__file__).parent / "configs" / "hybrid_timesfm_chronos.yaml"
    
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