#!/usr/bin/env python3
"""
ChronoTick Predictive Scheduler

Schedules ML model predictions ahead of time to avoid real-time inference delays.
When chronotick.time() is called, corrections are immediately available.
"""

import time
import threading
import heapq
import logging
from typing import Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import math
from pathlib import Path
import yaml
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled prediction task"""
    execution_time: float
    task_id: str
    task_function: Callable
    args: tuple
    kwargs: dict = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.execution_time < other.execution_time


@dataclass
class CorrectionWithBounds:
    """Clock correction with simplified error bounds (ML only)"""

    # Primary corrections
    offset_correction: float
    drift_rate: float

    # ML model uncertainties only
    offset_uncertainty: float    # From TSFM offset prediction interval
    drift_uncertainty: float     # From TSFM drift prediction interval

    # Metadata
    prediction_time: float       # When this prediction was made
    valid_until: float          # When this prediction expires
    confidence: float           # Overall confidence [0,1]
    source: str                 # "cpu", "gpu", "fusion"

    # Optional quantiles for confidence intervals
    quantiles: Optional[Dict[str, float]] = None  # e.g., {'0.1': val, '0.5': val, '0.9': val}
    
    def get_time_uncertainty(self, time_delta: float) -> float:
        """Calculate time uncertainty using error propagation"""
        return math.sqrt(
            self.offset_uncertainty**2 + 
            (self.drift_uncertainty * time_delta)**2
        )
    
    def get_corrected_time_bounds(self, system_time: float, time_delta: float = 0.0) -> Tuple[float, float]:
        """Get bounds for corrected time using mathematical error propagation"""
        corrected_time = system_time + self.offset_correction + self.drift_rate * time_delta
        uncertainty = self.get_time_uncertainty(time_delta)
        
        return (corrected_time - uncertainty, corrected_time + uncertainty)
    
    def is_valid(self, current_time: float) -> bool:
        """Check if this prediction is still valid"""
        return current_time <= self.valid_until

    def get_confidence_interval(self, confidence_level: float = 0.9) -> Optional[Tuple[float, float]]:
        """
        Get confidence interval for the offset correction based on quantiles.

        Args:
            confidence_level: Desired confidence level (e.g., 0.9 for 90%)

        Returns:
            Tuple of (lower_bound, upper_bound) for offset correction, or None if quantiles unavailable

        Example:
            90% confidence (0.9) → uses quantiles [0.05, 0.95] (middle 90%)
            80% confidence (0.8) → uses quantiles [0.1, 0.9] (middle 80%)
        """
        if not self.quantiles:
            return None

        # Map confidence level to quantile bounds
        # For confidence_level C, we want the middle C of the distribution
        # This means excluding (1-C)/2 from each tail
        tail = (1.0 - confidence_level) / 2.0
        lower_quantile = tail
        upper_quantile = 1.0 - tail

        # Find the closest available quantiles
        # TimesFM typically provides: 0.1, 0.5, 0.9
        available_quantiles = sorted([float(q) for q in self.quantiles.keys()])

        # Find closest quantile to lower bound
        lower_q_str = min(available_quantiles, key=lambda q: abs(q - lower_quantile))
        # Find closest quantile to upper bound
        upper_q_str = max(available_quantiles, key=lambda q: abs(q - upper_quantile))

        # Get quantile values
        lower_bound = self.quantiles[str(lower_q_str)]
        upper_bound = self.quantiles[str(upper_q_str)]

        return (lower_bound, upper_bound)


class PredictiveScheduler:
    """
    Schedules ML model predictions ahead of time to ensure zero-latency corrections.
    
    Timeline example:
    - t=205: Start CPU prediction for t=210-240 (5s lead time)
    - t=207: CPU prediction ready, cached
    - t=210: chronotick.time() gets immediate correction
    """
    
    def __init__(self, config_path: str):
        """Initialize scheduler with configuration"""
        self.config = self._load_config(config_path)
        
        # Scheduling parameters
        self.cpu_interval = self.config['prediction_scheduling']['cpu_model']['prediction_interval']
        self.cpu_horizon = self.config['prediction_scheduling']['cpu_model']['prediction_horizon']
        self.cpu_lead_time = self.config['prediction_scheduling']['cpu_model']['prediction_lead_time']
        self.cpu_max_inference = self.config['prediction_scheduling']['cpu_model']['max_inference_time']
        
        self.gpu_interval = self.config['prediction_scheduling']['gpu_model']['prediction_interval']
        self.gpu_horizon = self.config['prediction_scheduling']['gpu_model']['prediction_horizon']
        self.gpu_lead_time = self.config['prediction_scheduling']['gpu_model']['prediction_lead_time']
        self.gpu_max_inference = self.config['prediction_scheduling']['gpu_model']['max_inference_time']

        # NTP check interval from config
        self.ntp_check_interval = self.config['clock_measurement']['timing']['normal_operation']['measurement_interval']
        logger.info(f"[SCHEDULER_NTP] NTP check interval: {self.ntp_check_interval}s")

        # Prediction cache
        self.cache_size = self.config['prediction_scheduling']['dataset']['prediction_cache_size']
        self.prediction_cache: Dict[float, CorrectionWithBounds] = {}

        # CRITICAL: Load capping parameters to prevent runaway predictions
        adaptive_capping = self.config['prediction_scheduling'].get('adaptive_capping', {})
        self.absolute_max = adaptive_capping.get('absolute_max', 0.300)  # 300ms hard limit
        self.absolute_min = adaptive_capping.get('absolute_min', 0.001)  # 1ms floor
        logger.info(f"[SCHEDULER_CAPPING] Loaded capping limits: absolute_max={self.absolute_max*1000:.0f}ms, absolute_min={self.absolute_min*1000:.0f}ms")

        # Scheduler state
        self.task_queue = []  # Priority queue of scheduled tasks
        self.scheduler_thread = None
        self.scheduler_running = False
        self.lock = threading.Lock()
        
        # Model interfaces (to be injected)
        self.cpu_model = None
        self.gpu_model = None
        self.fusion_engine = None
        self.dataset_manager = None  # NEW: For autoregressive feedback at 1Hz
        
        # Statistics
        self.stats = {
            'predictions_scheduled': 0,
            'predictions_completed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'late_predictions': 0
        }
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # DEBUG: Log what we actually loaded
            logger.info(f"PredictiveScheduler loaded config from: {config_path}")
            logger.info(f"Config keys found: {list(config.keys())}")
            logger.info(f"Has prediction_scheduling: {'prediction_scheduling' in config}")

            return config
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise
    
    def set_model_interfaces(self, cpu_model, gpu_model, fusion_engine, dataset_manager=None, pipeline=None):
        """Inject model interfaces for making predictions"""
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.fusion_engine = fusion_engine
        self.dataset_manager = dataset_manager  # NEW: For autoregressive feedback
        self.pipeline = pipeline  # NEW: For accessing NTP state (last_ntp_offset, max_multiplier)
    
    def start_scheduler(self):
        """Start the predictive scheduler thread"""
        if self.scheduler_running:
            logger.warning("Scheduler already running")
            return
            
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Predictive scheduler started")
        
        # Schedule initial predictions
        current_time = time.time()
        self._schedule_initial_predictions(current_time)
    
    def stop_scheduler(self):
        """Stop the predictive scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Predictive scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop - executes scheduled prediction tasks"""
        while self.scheduler_running:
            try:
                current_time = time.time()

                # Get ready tasks (with lock)
                tasks_to_execute = []
                with self.lock:
                    while self.task_queue and self.task_queue[0].execution_time <= current_time:
                        task = heapq.heappop(self.task_queue)
                        tasks_to_execute.append(task)

                # Execute tasks (without lock to avoid deadlock)
                for task in tasks_to_execute:
                    try:
                        # Execute the prediction task
                        logger.debug(f"Executing prediction task: {task.task_id}")
                        task.task_function(*task.args, **task.kwargs)

                    except Exception as e:
                        logger.error(f"Prediction task {task.task_id} failed: {e}")

                # Small sleep to avoid busy waiting
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(1.0)
    
    def _schedule_initial_predictions(self, current_time: float):
        """Schedule the first set of predictions"""
        # Schedule first CPU prediction
        cpu_target_time = self._next_cpu_target_time(current_time)
        self._schedule_cpu_prediction(cpu_target_time)

        # Schedule first GPU prediction
        gpu_target_time = self._next_gpu_target_time(current_time)
        self._schedule_gpu_prediction(gpu_target_time)

        # Schedule first NTP check (CRITICAL: enables backtracking correction)
        ntp_check_time = current_time + self.ntp_check_interval
        self._schedule_ntp_check(ntp_check_time)
        logger.info(f"[SCHEDULER_NTP] Initial NTP check scheduled for {ntp_check_time:.0f} (in {self.ntp_check_interval}s)")
    
    def _next_cpu_target_time(self, current_time: float) -> float:
        """Calculate next CPU prediction target time"""
        return math.ceil(current_time / self.cpu_interval) * self.cpu_interval
    
    def _next_gpu_target_time(self, current_time: float) -> float:
        """Calculate next GPU prediction target time"""
        return math.ceil(current_time / self.gpu_interval) * self.gpu_interval
    
    def _schedule_cpu_prediction(self, target_time: float):
        """Schedule CPU prediction to be ready before target_time"""
        execution_time = target_time - self.cpu_lead_time
        
        task = ScheduledTask(
            execution_time=execution_time,
            task_id=f"cpu_pred_{target_time}",
            task_function=self._execute_cpu_prediction,
            args=(target_time, target_time + self.cpu_horizon)
        )
        
        with self.lock:
            heapq.heappush(self.task_queue, task)
            self.stats['predictions_scheduled'] += 1
            
        logger.debug(f"Scheduled CPU prediction for t={target_time} (exec at t={execution_time})")
    
    def _schedule_gpu_prediction(self, target_time: float):
        """Schedule GPU prediction to be ready before target_time"""
        execution_time = target_time - self.gpu_lead_time

        task = ScheduledTask(
            execution_time=execution_time,
            task_id=f"gpu_pred_{target_time}",
            task_function=self._execute_gpu_prediction,
            args=(target_time, target_time + self.gpu_horizon)
        )

        with self.lock:
            heapq.heappush(self.task_queue, task)
            self.stats['predictions_scheduled'] += 1

        logger.debug(f"Scheduled GPU prediction for t={target_time} (exec at t={execution_time})")

    def _schedule_ntp_check(self, check_time: float):
        """
        Schedule periodic NTP check to enable backtracking correction.

        CRITICAL: This is what triggers apply_ntp_correction() which does:
        1. Checks for new NTP measurements
        2. Applies backtracking to correct recent predictions
        3. Adds NTP measurements to dataset as ground truth anchors

        Without this, dataset fills with 100% predictions → feedback loop!
        """
        task = ScheduledTask(
            execution_time=check_time,
            task_id=f"ntp_check_{check_time}",
            task_function=self._execute_ntp_check,
            args=(check_time,)
        )

        with self.lock:
            heapq.heappush(self.task_queue, task)

        logger.info(f"[SCHEDULER_NTP] Scheduled NTP check for t={check_time:.0f}")

    def _cap_prediction_offset(self, offset: float) -> tuple:
        """
        Cap prediction offset to prevent runaway predictions.
        Capping is RELATIVE to current NTP baseline, not absolute.

        Args:
            offset: Raw prediction offset (seconds)

        Returns:
            (capped_offset, was_capped) tuple
        """
        magnitude = abs(offset)

        # Get current NTP baseline from pipeline (if available)
        last_ntp_offset = None
        max_multiplier = 2.5  # Default
        if self.pipeline and hasattr(self.pipeline, 'last_ntp_offset'):
            last_ntp_offset = self.pipeline.last_ntp_offset
            logger.debug(f"[SCHEDULER_CAP_DEBUG] Retrieved from pipeline: last_ntp_offset={last_ntp_offset}, type={type(last_ntp_offset)}")
        else:
            logger.warning(f"[SCHEDULER_CAP_DEBUG] Pipeline access failed: pipeline={self.pipeline}, has_attr={hasattr(self.pipeline, 'last_ntp_offset') if self.pipeline else 'N/A'}")
        if self.pipeline and hasattr(self.pipeline, 'max_multiplier'):
            max_multiplier = self.pipeline.max_multiplier

        # Calculate cap based on NTP baseline
        if last_ntp_offset is not None:
            # Cap is RELATIVE to NTP baseline: allow up to NTP_magnitude * max_multiplier
            last_ntp_magnitude = abs(last_ntp_offset)
            cap = last_ntp_magnitude * max_multiplier
            cap = max(cap, self.absolute_min)  # At least min_cap
            # NOTE: absolute_max NOT applied - cap scales with NTP baseline
        else:
            # No NTP reference yet - use absolute_max as fallback during initialization
            cap = self.absolute_max
            logger.debug(f"[SCHEDULER_CAP] No NTP baseline yet, using absolute_max={self.absolute_max*1000:.0f}ms as cap")

        # Apply cap
        if magnitude > cap:
            capped_offset = math.copysign(cap, offset)
            logger.warning(f"[SCHEDULER_CAP] Capped prediction from {offset*1000:.2f}ms to {capped_offset*1000:.2f}ms (exceeded {cap*1000:.0f}ms limit, NTP baseline={(last_ntp_offset*1000 if last_ntp_offset else 0):.0f}ms)")
            return capped_offset, True
        elif magnitude < self.absolute_min:
            capped_offset = math.copysign(self.absolute_min, offset)
            return capped_offset, True
        else:
            return offset, False

    def _execute_cpu_prediction(self, start_time: float, end_time: float):
        """Execute CPU model prediction and cache results"""
        try:
            logger.info(f"[SCHEDULER_TASK] _execute_cpu_prediction ENTRY: start_time={start_time:.0f}, end_time={end_time:.0f}, cpu_model={self.cpu_model is not None}")
            if not self.cpu_model:
                logger.error("[SCHEDULER_TASK] CPU model not available")
                return

            logger.info(f"[SCHEDULER_TASK] Executing CPU prediction for t={start_time:.0f}-{end_time:.0f}")

            # Make CPU prediction
            horizon = int(end_time - start_time)
            logger.info(f"[SCHEDULER_TASK] Calling cpu_model.predict_with_uncertainty(horizon={horizon})")
            cpu_prediction = self.cpu_model.predict_with_uncertainty(horizon=horizon)
            logger.info(f"[SCHEDULER_TASK] CPU model returned {len(cpu_prediction) if cpu_prediction else 0} predictions")

            # Cache prediction for each time step AND write to dataset for autoregressive feedback
            dataset_writes = 0
            for i, pred in enumerate(cpu_prediction):
                timestamp = start_time + i

                # CRITICAL: Cap predictions BEFORE caching and dataset storage
                capped_offset, was_capped = self._cap_prediction_offset(pred.offset)

                correction = CorrectionWithBounds(
                    offset_correction=capped_offset,  # Use capped offset
                    drift_rate=pred.drift,
                    offset_uncertainty=pred.offset_uncertainty,
                    drift_uncertainty=pred.drift_uncertainty,
                    prediction_time=timestamp,  # Time this prediction is FOR, not when cached
                    valid_until=end_time,
                    confidence=pred.confidence,
                    source="cpu",
                    quantiles=pred.quantiles  # Pass quantiles from prediction
                )

                # Cache for API serving
                self._cache_prediction(timestamp, correction)

                # CRITICAL FIX: Write to dataset for autoregressive training at 1Hz
                # This ensures model trains on recent predictions, not just NTP
                # IMPORTANT: Store UNCAPPED predictions so model learns from its actual output
                # Capping is only for safety when serving to clients (cache above)
                if self.dataset_manager:
                    self.dataset_manager.add_prediction(
                        timestamp=timestamp,
                        offset=pred.offset,  # Use UNCAPPED offset for training
                        drift=pred.drift,
                        source="cpu",
                        uncertainty=pred.offset_uncertainty,
                        confidence=pred.confidence,
                        was_capped=was_capped  # Track if it was capped
                    )
                    dataset_writes += 1

            logger.info(f"[SCHEDULER_DATASET] Wrote {dataset_writes}/{len(cpu_prediction)} predictions to dataset at 1Hz")
            logger.info(f"[SCHEDULER_DATASET] Timestamp range: [{start_time:.0f}, {end_time-1:.0f}]")
            logger.info(f"[SCHEDULER_DATASET] This provides autoregressive training data for next model run")

            # Update stats
            with self.lock:
                self.stats['predictions_completed'] += 1

        except Exception as e:
            logger.error(f"CPU prediction execution failed: {e}")
            with self.lock:
                self.stats['late_predictions'] += 1

        finally:
            # CRITICAL: Schedule next CPU prediction even if this one failed
            # This prevents the self-scheduling chain from breaking
            next_target = start_time + self.cpu_interval
            self._schedule_cpu_prediction(next_target)
    
    def _execute_gpu_prediction(self, start_time: float, end_time: float):
        """Execute GPU model prediction and cache results"""
        try:
            if not self.gpu_model:
                logger.error("GPU model not available")
                return

            logger.debug(f"Executing GPU prediction for t={start_time}-{end_time}")

            # Make GPU prediction
            gpu_prediction = self.gpu_model.predict_with_uncertainty(
                horizon=int(end_time - start_time)
            )

            # Cache prediction for each time step AND write to dataset for autoregressive feedback
            for i, pred in enumerate(gpu_prediction):
                timestamp = start_time + i

                # CRITICAL: Cap predictions BEFORE caching and dataset storage
                capped_offset, was_capped = self._cap_prediction_offset(pred.offset)

                correction = CorrectionWithBounds(
                    offset_correction=capped_offset,  # Use capped offset
                    drift_rate=pred.drift,
                    offset_uncertainty=pred.offset_uncertainty,
                    drift_uncertainty=pred.drift_uncertainty,
                    prediction_time=timestamp,  # Time this prediction is FOR, not when cached
                    valid_until=end_time,
                    confidence=pred.confidence,
                    source="gpu",
                    quantiles=pred.quantiles  # Pass quantiles from prediction
                )

                # Cache for API serving
                self._cache_prediction(timestamp, correction)

                # CRITICAL FIX: Write to dataset for autoregressive training at 1Hz
                # This ensures model trains on recent predictions, not just NTP
                # IMPORTANT: Store UNCAPPED predictions so model learns from its actual output
                # Capping is only for safety when serving to clients (cache above)
                if self.dataset_manager:
                    self.dataset_manager.add_prediction(
                        timestamp=timestamp,
                        offset=pred.offset,  # Use UNCAPPED offset for training
                        drift=pred.drift,
                        source="gpu",
                        uncertainty=pred.offset_uncertainty,
                        confidence=pred.confidence,
                        was_capped=was_capped  # Track if it was capped
                    )

            # Update stats
            with self.lock:
                self.stats['predictions_completed'] += 1

        except Exception as e:
            logger.error(f"GPU prediction execution failed: {e}")
            with self.lock:
                self.stats['late_predictions'] += 1

        finally:
            # CRITICAL: Schedule next GPU prediction even if this one failed
            # This prevents the self-scheduling chain from breaking
            next_target = start_time + self.gpu_interval
            self._schedule_gpu_prediction(next_target)

    def _execute_ntp_check(self, check_time: float):
        """
        Execute NTP check to trigger backtracking correction.

        This is the CRITICAL missing link that was causing 100% prediction feedback loop!
        """
        try:
            logger.info(f"[SCHEDULER_NTP] ═══════════════════════════════════════════")
            logger.info(f"[SCHEDULER_NTP] EXECUTING NTP CHECK at t={check_time:.0f}")

            if not self.pipeline:
                logger.error("[SCHEDULER_NTP] Pipeline not available, cannot check for NTP updates")
                return

            # Call pipeline's NTP check which triggers apply_ntp_correction()
            logger.info(f"[SCHEDULER_NTP] Calling pipeline._check_for_ntp_updates()...")
            self.pipeline._check_for_ntp_updates(check_time)

            logger.info(f"[SCHEDULER_NTP] NTP check completed")
            logger.info(f"[SCHEDULER_NTP] ═══════════════════════════════════════════")

        except Exception as e:
            logger.error(f"[SCHEDULER_NTP] NTP check failed: {e}", exc_info=True)

        finally:
            # CRITICAL: Schedule next NTP check to keep the chain alive
            next_check_time = check_time + self.ntp_check_interval
            self._schedule_ntp_check(next_check_time)
            logger.info(f"[SCHEDULER_NTP] Next NTP check scheduled for t={next_check_time:.0f} (in {self.ntp_check_interval}s)")

    def _cache_prediction(self, timestamp: float, correction: CorrectionWithBounds):
        """Cache prediction with size management"""
        logger.info(f"[SCHEDULER_CACHE] Caching prediction: timestamp={timestamp:.0f}, source={correction.source}, offset={correction.offset_correction*1000:.2f}ms")
        with self.lock:
            self.prediction_cache[timestamp] = correction

            # Manage cache size - keep predictions around CURRENT time, not furthest future
            if len(self.prediction_cache) > self.cache_size:
                current_time = time.time()

                # Sort keys by distance from current time
                sorted_keys = sorted(
                    self.prediction_cache.keys(),
                    key=lambda t: abs(t - current_time)
                )

                # Keep the closest N entries to current time
                keys_to_keep = set(sorted_keys[:self.cache_size])
                keys_to_remove = [k for k in self.prediction_cache.keys() if k not in keys_to_keep]

                for key in keys_to_remove:
                    del self.prediction_cache[key]

                logger.info(f"[SCHEDULER_CACHE] Cache trimmed: removed {len(keys_to_remove)} entries far from current time ({current_time:.0f})")
    
    def _interpolate_correction(self, correction: CorrectionWithBounds, current_time: float) -> CorrectionWithBounds:
        """
        Apply drift-based interpolation to a cached correction for continuous time accuracy.

        This ensures that two client queries 1ms apart return different interpolated results
        instead of the same stale 1-second-granularity prediction.

        Args:
            correction: Base correction from cache (at integer timestamp)
            current_time: Actual query time (may have sub-second component)

        Returns:
            Interpolated correction with offset adjusted for drift over delta_t
        """
        # Calculate time elapsed since this prediction was made
        delta_t = current_time - correction.prediction_time

        # Interpolate offset using drift rate
        # offset(t) = offset(t0) + drift_rate * Δt
        interpolated_offset = correction.offset_correction + (correction.drift_rate * delta_t)

        logger.debug(f"[SCHEDULER_INTERP] t={current_time:.3f}, base_t={correction.prediction_time:.0f}, "
                    f"Δt={delta_t:.3f}s, base_offset={correction.offset_correction*1000:.3f}ms, "
                    f"drift={correction.drift_rate*1e6:.3f}μs/s, "
                    f"interpolated_offset={interpolated_offset*1000:.3f}ms "
                    f"(Δ={abs(interpolated_offset - correction.offset_correction)*1000:.3f}ms)")

        # Create new correction with interpolated offset
        return CorrectionWithBounds(
            offset_correction=interpolated_offset,  # Interpolated offset
            drift_rate=correction.drift_rate,  # Drift rate unchanged
            offset_uncertainty=correction.offset_uncertainty,
            drift_uncertainty=correction.drift_uncertainty,
            prediction_time=correction.prediction_time,
            valid_until=correction.valid_until,
            confidence=correction.confidence,
            source=correction.source,
            quantiles=correction.quantiles
        )

    def get_correction_at_time(self, current_time: float) -> Optional[CorrectionWithBounds]:
        """
        Get correction for current time - should be immediately available
        from pre-computed predictions
        """
        logger.info(f"[SCHEDULER] get_correction_at_time called: current_time={current_time:.2f}")
        timestamp = math.floor(current_time)  # Round to nearest second
        logger.info(f"[SCHEDULER] Rounded timestamp: {timestamp}, cache_size={len(self.prediction_cache)}")

        with self.lock:
            # Check cache for exact match
            if timestamp in self.prediction_cache:
                correction = self.prediction_cache[timestamp]
                logger.info(f"[SCHEDULER] Found exact match at timestamp={timestamp}, valid={correction.is_valid(current_time)}, source={correction.source}")
                if correction.is_valid(current_time):
                    self.stats['cache_hits'] += 1
                    # CRITICAL: Apply drift-based interpolation for continuous time accuracy
                    return self._interpolate_correction(correction, current_time)

            # Check for nearby cached predictions
            logger.info(f"[SCHEDULER] No exact match, checking nearby timestamps...")
            for offset in [-1, 0, 1, 2]:  # Check nearby seconds
                check_time = timestamp + offset
                if check_time in self.prediction_cache:
                    correction = self.prediction_cache[check_time]
                    logger.info(f"[SCHEDULER] Found nearby match at timestamp={check_time}, valid={correction.is_valid(current_time)}, source={correction.source}")
                    if correction.is_valid(current_time):
                        self.stats['cache_hits'] += 1
                        # CRITICAL: Apply drift-based interpolation for continuous time accuracy
                        return self._interpolate_correction(correction, current_time)

            # Cache miss - this shouldn't happen with proper scheduling
            self.stats['cache_misses'] += 1
            cache_keys = sorted(self.prediction_cache.keys())
            cache_range = f"{cache_keys[0]:.0f} - {cache_keys[-1]:.0f}" if cache_keys else "EMPTY"
            logger.warning(f"[SCHEDULER] Cache miss for t={current_time:.2f} (timestamp={timestamp}). Cache range: {cache_range}, looking for: {timestamp}")
            return None
    
    def get_fused_correction(self, current_time: float) -> Optional[CorrectionWithBounds]:
        """
        Get fused CPU+GPU correction with temporal weighting
        """
        logger.info(f"[SCHEDULER] get_fused_correction called: current_time={current_time:.2f}")
        timestamp = math.floor(current_time)
        logger.info(f"[SCHEDULER] Fused correction timestamp: {timestamp}, cache_size={len(self.prediction_cache)}")

        with self.lock:
            cpu_correction = None
            gpu_correction = None

            # Find CPU and GPU predictions for this time
            for ts, correction in self.prediction_cache.items():
                if abs(ts - timestamp) <= 1 and correction.is_valid(current_time):
                    if correction.source == "cpu":
                        cpu_correction = correction
                        logger.info(f"[SCHEDULER] Found CPU correction at ts={ts}")
                    elif correction.source == "gpu":
                        gpu_correction = correction
                        logger.info(f"[SCHEDULER] Found GPU correction at ts={ts}")

            # Apply fusion if both available
            logger.info(f"[SCHEDULER] Fusion check: cpu={cpu_correction is not None}, gpu={gpu_correction is not None}, fusion_engine={self.fusion_engine is not None}")
            if cpu_correction and gpu_correction and self.fusion_engine:
                logger.info("[SCHEDULER] Applying fusion of CPU and GPU predictions")

                # CRITICAL: Interpolate both predictions BEFORE fusion
                cpu_interp = self._interpolate_correction(cpu_correction, current_time)
                gpu_interp = self._interpolate_correction(gpu_correction, current_time)

                # Calculate temporal weights based on CPU prediction window
                cpu_window_start = timestamp - (timestamp % self.cpu_interval)
                time_in_window = timestamp - cpu_window_start
                cpu_progress = min(time_in_window / self.cpu_interval, 1.0)

                # Progressive weighting: start CPU-heavy, move to GPU-heavy
                cpu_weight = 1.0 - cpu_progress
                gpu_weight = cpu_progress

                # Fuse interpolated predictions
                return self.fusion_engine.fuse_predictions(
                    cpu_interp, gpu_interp, cpu_weight, gpu_weight
                )

            # Fallback to available prediction (with interpolation)
            result = cpu_correction or gpu_correction
            logger.info(f"[SCHEDULER] Returning fallback prediction: {result.source if result else None}")
            if result:
                # CRITICAL: Apply drift-based interpolation for continuous time accuracy
                return self._interpolate_correction(result, current_time)
            return None
    
    def get_stats(self) -> dict:
        """Get scheduler statistics"""
        with self.lock:
            return self.stats.copy()


def create_test_scheduler():
    """Create a test scheduler for development"""
    config_path = Path(__file__).parent / "configs" / "hybrid_timesfm_chronos.yaml"
    return PredictiveScheduler(str(config_path))


if __name__ == "__main__":
    # Test the predictive scheduler
    scheduler = create_test_scheduler()
    
    print("Testing Predictive Scheduler...")
    print(f"CPU interval: {scheduler.cpu_interval}s")
    print(f"CPU lead time: {scheduler.cpu_lead_time}s") 
    print(f"GPU interval: {scheduler.gpu_interval}s")
    print(f"GPU lead time: {scheduler.gpu_lead_time}s")
    
    # This would normally be integrated with real models
    print("Scheduler created successfully!")