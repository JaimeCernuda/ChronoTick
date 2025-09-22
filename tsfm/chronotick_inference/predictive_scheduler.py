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
        
        # Prediction cache
        self.cache_size = self.config['prediction_scheduling']['dataset']['prediction_cache_size']
        self.prediction_cache: Dict[float, CorrectionWithBounds] = {}
        
        # Scheduler state
        self.task_queue = []  # Priority queue of scheduled tasks
        self.scheduler_thread = None
        self.scheduler_running = False
        self.lock = threading.Lock()
        
        # Model interfaces (to be injected)
        self.cpu_model = None
        self.gpu_model = None
        self.fusion_engine = None
        
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
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise
    
    def set_model_interfaces(self, cpu_model, gpu_model, fusion_engine):
        """Inject model interfaces for making predictions"""
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.fusion_engine = fusion_engine
    
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
                
                with self.lock:
                    # Execute all ready tasks
                    while self.task_queue and self.task_queue[0].execution_time <= current_time:
                        task = heapq.heappop(self.task_queue)
                        
                        try:
                            # Execute the prediction task
                            logger.debug(f"Executing prediction task: {task.task_id}")
                            task.task_function(*task.args, **task.kwargs)
                            self.stats['predictions_completed'] += 1
                            
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
    
    def _execute_cpu_prediction(self, start_time: float, end_time: float):
        """Execute CPU model prediction and cache results"""
        try:
            if not self.cpu_model:
                logger.error("CPU model not available")
                return
                
            logger.debug(f"Executing CPU prediction for t={start_time}-{end_time}")
            
            # Make CPU prediction
            cpu_prediction = self.cpu_model.predict_with_uncertainty(
                horizon=int(end_time - start_time)
            )
            
            # Cache prediction for each time step
            for i, pred in enumerate(cpu_prediction):
                timestamp = start_time + i
                
                correction = CorrectionWithBounds(
                    offset_correction=pred.offset,
                    drift_rate=pred.drift,
                    offset_uncertainty=pred.offset_uncertainty,
                    drift_uncertainty=pred.drift_uncertainty,
                    prediction_time=time.time(),
                    valid_until=end_time,
                    confidence=pred.confidence,
                    source="cpu"
                )
                
                self._cache_prediction(timestamp, correction)
            
            # Update stats
            with self.lock:
                self.stats['predictions_completed'] += 1
            
            # Schedule next CPU prediction
            next_target = start_time + self.cpu_interval
            self._schedule_cpu_prediction(next_target)
            
        except Exception as e:
            logger.error(f"CPU prediction execution failed: {e}")
            with self.lock:
                self.stats['late_predictions'] += 1
    
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
            
            # Cache prediction for each time step
            for i, pred in enumerate(gpu_prediction):
                timestamp = start_time + i
                
                correction = CorrectionWithBounds(
                    offset_correction=pred.offset,
                    drift_rate=pred.drift,
                    offset_uncertainty=pred.offset_uncertainty,
                    drift_uncertainty=pred.drift_uncertainty,
                    prediction_time=time.time(),
                    valid_until=end_time,
                    confidence=pred.confidence,
                    source="gpu"
                )
                
                self._cache_prediction(timestamp, correction)
            
            # Update stats
            with self.lock:
                self.stats['predictions_completed'] += 1
            
            # Schedule next GPU prediction
            next_target = start_time + self.gpu_interval
            self._schedule_gpu_prediction(next_target)
            
        except Exception as e:
            logger.error(f"GPU prediction execution failed: {e}")
            with self.lock:
                self.stats['late_predictions'] += 1
    
    def _cache_prediction(self, timestamp: float, correction: CorrectionWithBounds):
        """Cache prediction with size management"""
        with self.lock:
            self.prediction_cache[timestamp] = correction
            
            # Manage cache size
            if len(self.prediction_cache) > self.cache_size:
                # Remove oldest entries
                sorted_keys = sorted(self.prediction_cache.keys())
                keys_to_remove = sorted_keys[:-self.cache_size]
                for key in keys_to_remove:
                    del self.prediction_cache[key]
    
    def get_correction_at_time(self, current_time: float) -> Optional[CorrectionWithBounds]:
        """
        Get correction for current time - should be immediately available
        from pre-computed predictions
        """
        timestamp = math.floor(current_time)  # Round to nearest second
        
        with self.lock:
            # Check cache for exact match
            if timestamp in self.prediction_cache:
                correction = self.prediction_cache[timestamp]
                if correction.is_valid(current_time):
                    self.stats['cache_hits'] += 1
                    return correction
            
            # Check for nearby cached predictions
            for offset in [-1, 0, 1, 2]:  # Check nearby seconds
                check_time = timestamp + offset
                if check_time in self.prediction_cache:
                    correction = self.prediction_cache[check_time]
                    if correction.is_valid(current_time):
                        self.stats['cache_hits'] += 1
                        return correction
            
            # Cache miss - this shouldn't happen with proper scheduling
            self.stats['cache_misses'] += 1
            logger.warning(f"Cache miss for t={current_time} - prediction not ready")
            return None
    
    def get_fused_correction(self, current_time: float) -> Optional[CorrectionWithBounds]:
        """
        Get fused CPU+GPU correction with temporal weighting
        """
        timestamp = math.floor(current_time)
        
        with self.lock:
            cpu_correction = None
            gpu_correction = None
            
            # Find CPU and GPU predictions for this time
            for ts, correction in self.prediction_cache.items():
                if abs(ts - timestamp) <= 1 and correction.is_valid(current_time):
                    if correction.source == "cpu":
                        cpu_correction = correction
                    elif correction.source == "gpu":
                        gpu_correction = correction
            
            # Apply fusion if both available
            if cpu_correction and gpu_correction and self.fusion_engine:
                # Calculate temporal weights based on CPU prediction window
                cpu_window_start = timestamp - (timestamp % self.cpu_interval)
                time_in_window = timestamp - cpu_window_start
                cpu_progress = min(time_in_window / self.cpu_interval, 1.0)
                
                # Progressive weighting: start CPU-heavy, move to GPU-heavy
                cpu_weight = 1.0 - cpu_progress
                gpu_weight = cpu_progress
                
                return self.fusion_engine.fuse_predictions(
                    cpu_correction, gpu_correction, cpu_weight, gpu_weight
                )
            
            # Fallback to available prediction
            return cpu_correction or gpu_correction
    
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