"""
ChronoTick - Software-Defined Clock with ML-Based Drift Correction

This package provides corrected timestamps using machine learning inference
to predict and compensate for clock drift and environmental factors.

Main API:
    chronotick.time() - Get corrected timestamp
    chronotick.start() - Start the inference daemon
    chronotick.stop() - Stop the inference daemon
"""

import time as _time
import threading
import os
from pathlib import Path
from typing import Optional, Tuple, NamedTuple
import atexit
import logging

# Import daemon functionality
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference.daemon import ChronoTickDaemon, TimeResponse


class CorrectedTime(NamedTuple):
    """
    A corrected timestamp with uncertainty information.
    
    Attributes:
        timestamp: The corrected timestamp
        raw_timestamp: The original system timestamp
        offset_correction: The applied offset correction (seconds)
        uncertainty: Uncertainty in the correction (seconds)
        confidence: Confidence in the prediction (0.0 to 1.0)
        lower_bound: Lower bound of 95% confidence interval
        upper_bound: Upper bound of 95% confidence interval
    """
    timestamp: float
    raw_timestamp: float
    offset_correction: float
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ChronoTick:
    """
    Main ChronoTick interface providing corrected timestamps.
    
    This class manages the inference daemon and provides a clean API
    for getting corrected timestamps with uncertainty bounds.
    """
    
    def __init__(self):
        """Initialize ChronoTick interface."""
        self.daemon: Optional[ChronoTickDaemon] = None
        self._lock = threading.Lock()
        self._started = False
        self._config_path = None
        self._cpu_affinity = None
        
        # Statistics
        self._total_calls = 0
        self._successful_calls = 0
        self._fallback_calls = 0
        
        # Fallback cache for when daemon is unavailable
        self._last_known_offset = 0.0
        self._last_update_time = 0.0
        self._fallback_timeout = 10.0  # Use cached offset for 10 seconds
        
        # Setup logging
        self.logger = logging.getLogger('ChronoTick')
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
    
    def start(self, config_path: Optional[str] = None, 
              cpu_affinity: Optional[list] = None,
              auto_config: bool = True) -> bool:
        """
        Start the ChronoTick inference daemon.
        
        Args:
            config_path: Path to configuration file
            cpu_affinity: List of CPU cores to bind daemon to
            auto_config: Automatically select optimal configuration
            
        Returns:
            True if started successfully
        """
        with self._lock:
            if self._started:
                self.logger.warning("ChronoTick already started")
                return True
            
            # Auto-select configuration if not provided
            if config_path is None or auto_config:
                config_path = self._auto_select_config()
            
            if config_path is None:
                self.logger.error("No configuration available")
                return False
            
            # Auto-select CPU affinity if not provided
            if cpu_affinity is None:
                cpu_affinity = self._auto_select_cpu_affinity()
            
            try:
                self.daemon = ChronoTickDaemon(config_path, cpu_affinity)
                
                if self.daemon.start_daemon():
                    self._started = True
                    self._config_path = config_path
                    self._cpu_affinity = cpu_affinity
                    
                    self.logger.info(f"ChronoTick started with config: {config_path}")
                    if cpu_affinity:
                        self.logger.info(f"CPU affinity: {cpu_affinity}")
                    
                    # Print to terminal what configuration was selected
                    from pathlib import Path
                    config_name = Path(config_path).stem
                    print(f"🕒 ChronoTick: Using configuration '{config_name}'")
                    print(f"📁 Config file: {config_path}")
                    if cpu_affinity:
                        print(f"⚙️  CPU affinity: {cpu_affinity}")
                    
                    return True
                else:
                    self.logger.error("Failed to start daemon")
                    self.daemon = None
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error starting ChronoTick: {e}")
                self.daemon = None
                return False
    
    def stop(self) -> bool:
        """
        Stop the ChronoTick inference daemon.
        
        Returns:
            True if stopped successfully
        """
        with self._lock:
            if not self._started or self.daemon is None:
                return True
            
            try:
                success = self.daemon.stop_daemon()
                self._started = False
                self.daemon = None
                
                if success:
                    self.logger.info("ChronoTick stopped")
                else:
                    self.logger.warning("ChronoTick stop may have failed")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Error stopping ChronoTick: {e}")
                return False
    
    def time(self, include_uncertainty: bool = True, 
             timeout: float = 0.1) -> float:
        """
        Get corrected timestamp (compatible with time.time()).
        
        Args:
            include_uncertainty: Include uncertainty calculation
            timeout: Timeout for daemon communication
            
        Returns:
            Corrected timestamp as float
        """
        result = self.time_detailed(include_uncertainty, timeout)
        return result.timestamp
    
    def time_detailed(self, include_uncertainty: bool = True,
                     timeout: float = 0.1) -> CorrectedTime:
        """
        Get detailed corrected timestamp with uncertainty information.
        
        Args:
            include_uncertainty: Include uncertainty calculation
            timeout: Timeout for daemon communication
            
        Returns:
            CorrectedTime with detailed information
        """
        self._total_calls += 1
        raw_time = _time.time()
        
        # Check if daemon is available
        if not self._started or self.daemon is None:
            self._fallback_calls += 1
            return self._fallback_time(raw_time)
        
        try:
            # Get corrected time from daemon
            response = self.daemon.get_corrected_time(
                include_uncertainty=include_uncertainty,
                include_bounds=True,
                timeout=timeout
            )
            
            if response.status == "success":
                self._successful_calls += 1
                
                # Update fallback cache
                self._last_known_offset = response.offset_correction
                self._last_update_time = raw_time
                
                return CorrectedTime(
                    timestamp=response.corrected_time,
                    raw_timestamp=response.raw_time,
                    offset_correction=response.offset_correction,
                    uncertainty=response.uncertainty,
                    confidence=response.confidence,
                    lower_bound=response.lower_bound,
                    upper_bound=response.upper_bound
                )
            else:
                # Daemon error, use fallback
                self.logger.warning(f"Daemon error: {response.error}")
                self._fallback_calls += 1
                return self._fallback_time(raw_time)
                
        except Exception as e:
            self.logger.warning(f"Communication error: {e}")
            self._fallback_calls += 1
            return self._fallback_time(raw_time)
    
    def _fallback_time(self, raw_time: float) -> CorrectedTime:
        """
        Provide fallback timestamp when daemon is unavailable.
        
        Uses cached offset if recent enough, otherwise returns raw time.
        """
        # Check if we have a recent cached offset
        if (self._last_known_offset != 0.0 and 
            raw_time - self._last_update_time < self._fallback_timeout):
            
            # Use cached offset with degraded confidence
            age = raw_time - self._last_update_time
            confidence_degradation = min(age / self._fallback_timeout, 1.0)
            
            return CorrectedTime(
                timestamp=raw_time + self._last_known_offset,
                raw_timestamp=raw_time,
                offset_correction=self._last_known_offset,
                uncertainty=None,  # Unknown uncertainty in fallback mode
                confidence=max(0.1, 1.0 - confidence_degradation),
                lower_bound=None,
                upper_bound=None
            )
        else:
            # No valid cached offset, return raw time
            return CorrectedTime(
                timestamp=raw_time,
                raw_timestamp=raw_time,
                offset_correction=0.0,
                uncertainty=None,
                confidence=0.0,  # No confidence without correction
                lower_bound=None,
                upper_bound=None
            )
    
    def status(self) -> dict:
        """
        Get ChronoTick status and statistics.
        
        Returns:
            Dictionary with status information
        """
        base_status = {
            'started': self._started,
            'config_path': self._config_path,
            'cpu_affinity': self._cpu_affinity,
            'total_calls': self._total_calls,
            'successful_calls': self._successful_calls,
            'fallback_calls': self._fallback_calls,
            'success_rate': self._successful_calls / max(self._total_calls, 1),
            'last_known_offset_us': self._last_known_offset * 1e6,
            'cache_age': _time.time() - self._last_update_time if self._last_update_time > 0 else None
        }
        
        if self.daemon:
            try:
                daemon_status = self.daemon.get_daemon_status(timeout=0.5)
                base_status.update({
                    'daemon_running': daemon_status.running,
                    'daemon_pid': daemon_status.pid,
                    'daemon_uptime': daemon_status.uptime,
                    'daemon_memory_mb': daemon_status.memory_usage_mb,
                    'daemon_requests': daemon_status.total_requests,
                    'daemon_success_rate': daemon_status.successful_requests / max(daemon_status.total_requests, 1),
                    'avg_inference_time_ms': daemon_status.avg_inference_time * 1000 if daemon_status.avg_inference_time else None
                })
            except Exception as e:
                base_status['daemon_error'] = str(e)
        
        return base_status
    
    def _auto_select_config(self) -> Optional[str]:
        """Automatically select the best configuration."""
        try:
            from chronotick_inference.config_selector import recommend_config, detect_hardware
            
            hardware = detect_hardware()
            config_name, _ = recommend_config(hardware)
            
            if config_name:
                # Use the selector to apply the configuration
                import subprocess
                result = subprocess.run([
                    'python', 'chronotick_inference/config_selector.py', 
                    '--select', config_name
                ], capture_output=True)
                
                if result.returncode == 0:
                    config_path = Path(__file__).parent.parent / "chronotick_inference" / "config.yaml"
                    return str(config_path)
        except Exception as e:
            self.logger.warning(f"Auto-config failed: {e}")
        
        # Fallback to default
        config_path = Path(__file__).parent.parent / "chronotick_inference" / "config.yaml"
        if config_path.exists():
            return str(config_path)
        
        return None
    
    def _auto_select_cpu_affinity(self) -> Optional[list]:
        """Automatically select CPU affinity for optimal performance."""
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            
            if cpu_count >= 4:
                # Use cores 0-1 for inference (avoid core 0 if possible due to system overhead)
                return [1, 2] if cpu_count > 2 else [0, 1]
            elif cpu_count >= 2:
                # Use core 1 (avoid core 0)
                return [1]
            else:
                # Single core system
                return None
                
        except Exception:
            return None
    
    def _cleanup(self):
        """Cleanup on exit."""
        if self._started:
            self.stop()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Global ChronoTick instance
_chronotick = ChronoTick()

# Public API functions
def time() -> float:
    """
    Get corrected timestamp (drop-in replacement for time.time()).
    
    Returns:
        Corrected timestamp as float
        
    Example:
        >>> import chronotick
        >>> chronotick.start()
        >>> corrected_timestamp = chronotick.time()
        >>> print(f"Corrected time: {corrected_timestamp}")
    """
    return _chronotick.time()

def time_detailed() -> CorrectedTime:
    """
    Get detailed corrected timestamp with uncertainty information.
    
    Returns:
        CorrectedTime with detailed information
        
    Example:
        >>> import chronotick
        >>> chronotick.start()
        >>> ct = chronotick.time_detailed()
        >>> print(f"Corrected: {ct.timestamp}")
        >>> print(f"Offset: {ct.offset_correction*1e6:.3f}μs")
        >>> print(f"Uncertainty: ±{ct.uncertainty*1e6:.3f}μs")
    """
    return _chronotick.time_detailed()

def start(config_path: Optional[str] = None, 
          cpu_affinity: Optional[list] = None) -> bool:
    """
    Start ChronoTick inference daemon.
    
    Args:
        config_path: Path to configuration file (auto-selected if None)
        cpu_affinity: CPU cores to bind to (auto-selected if None)
        
    Returns:
        True if started successfully
        
    Example:
        >>> import chronotick
        >>> chronotick.start(cpu_affinity=[1, 2])
        >>> # Now chronotick.time() returns corrected timestamps
    """
    return _chronotick.start(config_path, cpu_affinity)

def stop() -> bool:
    """
    Stop ChronoTick inference daemon.
    
    Returns:
        True if stopped successfully
    """
    return _chronotick.stop()

def status() -> dict:
    """
    Get ChronoTick status and statistics.
    
    Returns:
        Dictionary with status information
    """
    return _chronotick.status()

# Version information
__version__ = "1.0.0"
__author__ = "ChronoTick Team"

# Export public API
__all__ = [
    'time',
    'time_detailed', 
    'start',
    'stop',
    'status',
    'CorrectedTime',
    'ChronoTick'
]