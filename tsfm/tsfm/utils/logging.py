"""
Logging setup utilities for TSFM library.
"""

import logging
import sys
import time
import functools
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager


def setup_logging(level: str = "INFO",
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Set up logging configuration for TSFM library.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to write logs
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific loggers to reduce noise
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, file={log_file}")


def get_logger(name: str, config: Optional[Dict] = None) -> logging.Logger:
    """Get a logger with the specified name."""
    logger = logging.getLogger(name)
    
    if config:
        level = config.get('level', 'INFO')
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        format_str = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if logger.handlers:
            formatter = logging.Formatter(format_str)
            for handler in logger.handlers:
                handler.setFormatter(formatter)
    
    return logger


class LogLevel(Enum):
    """Logging levels enum."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def log_performance(logger: logging.Logger) -> Callable:
    """
    Decorator to log function performance.
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__
            
            try:
                logger.info(f"{func_name} started")
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func_name} completed successfully in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func_name} failed after {execution_time:.4f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


class ModelLogger:
    """Logger class specifically for model operations."""
    
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize model logger.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(f"tsfm.{model_name}")
    
    def log_model_operation(self, operation: str, status: str, details: Optional[Dict] = None):
        """Log model operations."""
        message = f"Model {self.model_name} - {operation}: {status}"
        if details:
            detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            message += f" ({detail_str})"
        self.logger.info(message)
    
    def log_error(self, operation: str, error: Exception, context: Optional[Dict] = None):
        """Log errors with context."""
        message = f"Model {self.model_name} - {operation} failed: {str(error)}"
        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            message += f" (context: {context_str})"
        self.logger.error(message)
    
    def log_performance_metrics(self, operation: str, metrics: Dict):
        """Log performance metrics."""
        metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        message = f"Model {self.model_name} - {operation} performance: {metrics_str}"
        self.logger.info(message)
    
    def log_warning(self, operation: str, message: str):
        """Log warnings."""
        warning_msg = f"Model {self.model_name} - {operation}: {message}"
        self.logger.warning(warning_msg)
    
    def log_debug(self, operation: str, debug_info: Dict):
        """Log debug information."""
        debug_str = ", ".join([f"{k}={v}" for k, v in debug_info.items()])
        message = f"Model {self.model_name} - {operation} debug: {debug_str}"
        self.logger.debug(message)
    
    @contextmanager
    def operation_context(self, operation: str):
        """Context manager for logging operation start/end."""
        self.logger.info(f"Model {self.model_name} - {operation} started")
        start_time = time.time()
        
        try:
            yield
            execution_time = time.time() - start_time
            self.logger.info(f"Model {self.model_name} - {operation} completed in {execution_time:.4f}s")
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Model {self.model_name} - {operation} failed after {execution_time:.4f}s: {str(e)}")
            raise