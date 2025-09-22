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
        
        return CorrectionWithBounds(
            offset_correction=fused_offset,
            drift_rate=fused_drift,
            offset_uncertainty=fused_offset_unc,
            drift_uncertainty=fused_drift_unc,
            prediction_time=time.time(),
            valid_until=max(cpu_correction.valid_until, gpu_correction.valid_until),
            confidence=(cpu_weight * cpu_correction.confidence + gpu_weight * gpu_correction.confidence),
            source="fusion"
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
        self.lock = threading.Lock()
        
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
    
    def apply_retrospective_correction(self, ntp_measurement: NTPMeasurement,
                                     interval_start: float):
        """
        Apply design.md Algorithm 1: Retrospective Bias Correction
        
        δ ← o_t - ô_t
        For i ← 0 to n:
            α ← (t_i - t_start) / (t_end - t_start)  # Linear weighting
            ô_t_i'' ← ô_t_i + α · δ
        """
        with self.lock:
            # Calculate prediction error
            prediction_at_ntp_time = self.get_measurement_at_time(ntp_measurement.timestamp)
            if not prediction_at_ntp_time:
                logger.warning("No prediction available for retrospective correction")
                return
            
            prediction_error = ntp_measurement.offset - prediction_at_ntp_time['offset']
            
            logger.info(f"Applying retrospective correction: error={prediction_error*1e6:.1f}μs "
                       f"over interval [{interval_start}, {ntp_measurement.timestamp}]")
            
            # Apply linear weighting correction (design.md Algorithm 1)
            interval_duration = ntp_measurement.timestamp - interval_start
            
            for timestamp in range(int(interval_start), int(ntp_measurement.timestamp)):
                if timestamp in self.measurement_dataset:
                    # Calculate linear weight: α = (t_i - t_start) / (t_end - t_start)
                    alpha = (timestamp - interval_start) / interval_duration if interval_duration > 0 else 0
                    
                    # Apply correction: ô_t_i'' ← ô_t_i + α · δ
                    self.measurement_dataset[timestamp]['offset'] += alpha * prediction_error
                    self.measurement_dataset[timestamp]['corrected'] = True
            
            # Add the new NTP measurement
            self.add_ntp_measurement(ntp_measurement)
            
            logger.debug(f"Retrospective correction applied to {int(ntp_measurement.timestamp - interval_start)} measurements")
    
    def get_measurement_at_time(self, timestamp: float) -> Optional[dict]:
        """Get measurement data at specific timestamp"""
        with self.lock:
            return self.measurement_dataset.get(int(timestamp))
    
    def get_recent_measurements(self, window_seconds: int = 300) -> List[Tuple[float, float]]:
        """Get recent offset measurements for TSFM model input"""
        with self.lock:
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
        
        # Pipeline state
        self.initialized = False
        self.last_ntp_time = 0
        self.warm_up_complete = False
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_corrections': 0,
            'ntp_measurements': 0,
            'prediction_cache_hits': 0,
            'prediction_cache_misses': 0,
            'retrospective_corrections': 0
        }
        
    def initialize(self, cpu_model=None, gpu_model=None):
        """Initialize the pipeline with model interfaces"""
        logger.info("Initializing real data pipeline...")
        
        # Set model interfaces for predictive scheduler
        if cpu_model or gpu_model:
            self.predictive_scheduler.set_model_interfaces(cpu_model, gpu_model, self.fusion_engine)
        
        # Start components
        self.system_metrics.start_collection()
        self.ntp_collector.start_collection()
        self.predictive_scheduler.start_scheduler()
        
        self.initialized = True
        logger.info("Real data pipeline initialized successfully")
        
        # Wait for warm-up phase
        warm_up_duration = self.ntp_collector.warm_up_duration
        logger.info(f"Starting {warm_up_duration}s warm-up phase...")
        
        # In a real implementation, we'd wait for warm-up
        # For now, mark as complete after a short delay
        threading.Timer(5.0, self._mark_warm_up_complete).start()
    
    def _mark_warm_up_complete(self):
        """Mark warm-up phase as complete"""
        self.warm_up_complete = True
        logger.info("Warm-up phase complete - switching to predictive mode")
    
    def shutdown(self):
        """Shutdown the pipeline"""
        logger.info("Shutting down real data pipeline...")
        
        self.ntp_collector.stop_collection()
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
    
    def _check_for_ntp_updates(self, current_time: float):
        """Check for new NTP measurements and apply retrospective correction"""
        latest_ntp = self.ntp_collector.last_measurement
        
        if (latest_ntp and 
            latest_ntp.timestamp > self.last_ntp_time and
            self.last_ntp_time > 0):  # Not the first measurement
            
            # Apply retrospective correction
            self.dataset_manager.apply_retrospective_correction(
                latest_ntp, self.last_ntp_time
            )
            
            with self.lock:
                self.stats['retrospective_corrections'] += 1
            
            self.last_ntp_time = latest_ntp.timestamp
        elif latest_ntp and self.last_ntp_time == 0:
            # First NTP measurement
            self.last_ntp_time = latest_ntp.timestamp
            self.dataset_manager.add_ntp_measurement(latest_ntp)
            
            with self.lock:
                self.stats['ntp_measurements'] += 1
    
    def _get_warm_up_correction(self, current_time: float) -> CorrectionWithBounds:
        """Get correction during warm-up phase using direct NTP measurements"""
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
    
    def _get_predictive_correction(self, current_time: float) -> CorrectionWithBounds:
        """Get correction using predictive scheduling and model fusion"""
        # Try to get fused correction from scheduler
        correction = self.predictive_scheduler.get_fused_correction(current_time)
        
        if correction:
            with self.lock:
                self.stats['prediction_cache_hits'] += 1
            return correction
        
        # Fallback: get individual model predictions
        cpu_correction = self.predictive_scheduler.get_correction_at_time(current_time)
        if cpu_correction:
            with self.lock:
                self.stats['prediction_cache_hits'] += 1
            return cpu_correction
        
        # Cache miss - use fallback
        with self.lock:
            self.stats['prediction_cache_misses'] += 1
        
        return self._fallback_correction(current_time)
    
    def _fallback_correction(self, current_time: float) -> CorrectionWithBounds:
        """Fallback correction when predictions not available"""
        # Try to use recent NTP measurement
        latest_offset = self.ntp_collector.get_latest_offset()
        
        if latest_offset is not None:
            # Calculate time since last NTP measurement
            time_since_ntp = current_time - self.ntp_collector.last_measurement_time
            
            # Use simple linear extrapolation with conservative uncertainty
            estimated_drift = 1e-6  # 1μs/s conservative drift estimate
            extrapolated_offset = latest_offset + estimated_drift * time_since_ntp
            
            # Uncertainty grows with time since NTP
            base_uncertainty = 0.005  # 5ms base
            drift_uncertainty = time_since_ntp * 1e-6  # 1μs per second since NTP
            total_uncertainty = base_uncertainty + drift_uncertainty
            
            return CorrectionWithBounds(
                offset_correction=extrapolated_offset,
                drift_rate=estimated_drift,
                offset_uncertainty=total_uncertainty,
                drift_uncertainty=1e-6,
                prediction_time=current_time,
                valid_until=current_time + 30,
                confidence=max(0.1, 1.0 - time_since_ntp / 300),  # Confidence decreases with time
                source="ntp_fallback"
            )
        
        # Last resort: no correction
        return CorrectionWithBounds(
            offset_correction=0.0,
            drift_rate=0.0,
            offset_uncertainty=0.010,  # 10ms uncertainty when no data
            drift_uncertainty=1e-5,
            prediction_time=current_time,
            valid_until=current_time + 10,
            confidence=0.0,
            source="no_data"
        )
    
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