"""
Debug logging utility for ChronoTick inference.

Provides comprehensive function call logging with inputs/outputs.
Can be toggled on/off via environment variable or config.
"""

import functools
import logging
import time
import os
from typing import Any, Callable
import numpy as np

# Global debug flag (can be controlled via environment or config)
DEBUG_ENABLED = os.environ.get('CHRONOTICK_DEBUG', 'false').lower() == 'true'

# Logger setup
logger = logging.getLogger('chronotick.debug')
logger.setLevel(logging.DEBUG if DEBUG_ENABLED else logging.INFO)


def enable_debug():
    """Enable debug logging globally."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = True
    logger.setLevel(logging.DEBUG)
    logger.info("✓ Debug logging ENABLED")


def disable_debug():
    """Disable debug logging globally."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = False
    logger.setLevel(logging.INFO)
    logger.info("✓ Debug logging DISABLED")


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return DEBUG_ENABLED


def format_value(value: Any, max_len: int = 100) -> str:
    """
    Format a value for logging (handle arrays, large objects, etc.).

    Args:
        value: Value to format
        max_len: Maximum string length

    Returns:
        Formatted string representation
    """
    if isinstance(value, np.ndarray):
        shape = value.shape
        dtype = value.dtype
        if value.size <= 10:
            return f"array({value.tolist()}, shape={shape}, dtype={dtype})"
        else:
            return f"array(shape={shape}, dtype={dtype}, min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f})"

    elif isinstance(value, (list, tuple)) and len(value) > 10:
        return f"{type(value).__name__}(len={len(value)}, first={value[0]}, last={value[-1]})"

    elif isinstance(value, dict):
        if len(value) <= 5:
            return str(value)
        else:
            return f"dict(keys={list(value.keys())}, len={len(value)})"

    else:
        value_str = str(value)
        if len(value_str) > max_len:
            return value_str[:max_len] + "..."
        return value_str


def debug_log_call(func: Callable) -> Callable:
    """
    Decorator to log function calls with inputs and outputs.

    Only logs if DEBUG_ENABLED is True.

    Usage:
        @debug_log_call
        def my_function(arg1, arg2):
            return result
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DEBUG_ENABLED:
            # Fast path: no logging overhead
            return func(*args, **kwargs)

        # Build function signature
        func_name = f"{func.__module__}.{func.__qualname__}"

        # Log inputs
        logger.debug("=" * 80)
        logger.debug(f"→ CALL: {func_name}")

        # Log positional args
        if args:
            logger.debug(f"  Args ({len(args)}):")
            for i, arg in enumerate(args):
                # Skip 'self' for methods
                if i == 0 and hasattr(arg, '__class__'):
                    logger.debug(f"    [0] self: {arg.__class__.__name__} instance")
                else:
                    logger.debug(f"    [{i}] {format_value(arg)}")

        # Log keyword args
        if kwargs:
            logger.debug(f"  Kwargs ({len(kwargs)}):")
            for key, value in kwargs.items():
                logger.debug(f"    {key} = {format_value(value)}")

        # Execute function and measure time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            # Log output
            logger.debug(f"← RETURN: {func_name} (took {elapsed:.4f}s)")
            logger.debug(f"  Result: {format_value(result)}")
            logger.debug("=" * 80)

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.debug(f"✗ EXCEPTION: {func_name} (after {elapsed:.4f}s)")
            logger.debug(f"  Error: {type(e).__name__}: {e}")
            logger.debug("=" * 80)
            raise

    return wrapper


def debug_log_section(section_name: str):
    """
    Log a section separator for better readability.

    Usage:
        debug_log_section("Model Loading")
    """
    if DEBUG_ENABLED:
        logger.debug("")
        logger.debug("#" * 80)
        logger.debug(f"# {section_name}")
        logger.debug("#" * 80)


def debug_log_variable(name: str, value: Any):
    """
    Log a variable value.

    Usage:
        debug_log_variable("prediction", predictions)
    """
    if DEBUG_ENABLED:
        logger.debug(f"  {name} = {format_value(value)}")


def debug_log_metrics(metrics: dict):
    """
    Log a dictionary of metrics in a readable format.

    Usage:
        debug_log_metrics({
            'mae': 0.123,
            'rmse': 0.456,
            'predictions': predictions_array
        })
    """
    if DEBUG_ENABLED:
        logger.debug("  Metrics:")
        for key, value in metrics.items():
            logger.debug(f"    - {key}: {format_value(value)}")


class DebugTimer:
    """
    Context manager for timing code blocks.

    Usage:
        with DebugTimer("Loading model"):
            model.load()
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        if DEBUG_ENABLED:
            logger.debug(f"⏱ START: {self.name}")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if DEBUG_ENABLED:
            if exc_type is None:
                logger.debug(f"⏱ END: {self.name} (took {elapsed:.4f}s)")
            else:
                logger.debug(f"⏱ FAILED: {self.name} (after {elapsed:.4f}s)")


# Example usage in code:
if __name__ == '__main__':
    # Enable debug logging
    enable_debug()

    # Test decorator
    @debug_log_call
    def test_function(x, y, mode='standard'):
        """Test function for debug logging."""
        result = x + y
        time.sleep(0.1)  # Simulate work
        return result

    # Test section logging
    debug_log_section("Testing Debug Logger")

    # Test function call
    result = test_function(10, 20, mode='enhanced')

    # Test variable logging
    debug_log_variable("computed_result", result)

    # Test metrics logging
    debug_log_metrics({
        'accuracy': 0.95,
        'loss': 0.05,
        'predictions': np.array([1, 2, 3, 4, 5])
    })

    # Test timer
    with DebugTimer("Simulated work"):
        time.sleep(0.2)

    print(f"\nResult: {result}")
