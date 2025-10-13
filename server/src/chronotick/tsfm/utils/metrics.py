"""
Metrics calculation utilities for time series forecasting.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for time series forecasting metrics."""
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Square Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(numerator / denominator) * 100
    
    @staticmethod
    def msis(y_true: np.ndarray, y_pred: np.ndarray, 
             seasonal_period: int = 24) -> float:
        """Calculate Mean Scaled Interval Score (for seasonal data)."""
        if len(y_true) < seasonal_period:
            seasonal_period = 1
        
        # Calculate seasonal naive forecast error
        if seasonal_period > 1:
            seasonal_error = np.mean(np.abs(y_true[seasonal_period:] - y_true[:-seasonal_period]))
        else:
            seasonal_error = np.mean(np.abs(np.diff(y_true)))
        
        if seasonal_error == 0:
            seasonal_error = 1e-8
        
        # Calculate MSIS
        forecast_error = np.mean(np.abs(y_true - y_pred))
        return forecast_error / seasonal_error
    
    @staticmethod
    def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        return np.corrcoef(y_true, y_pred)[0, 1]
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct trend predictions)."""
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction) * 100
    
    def calculate_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                     metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate all or specified metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metric names to calculate. If None, calculates all.
            
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = ["mae", "rmse", "mse", "mape", "smape", "msis", 
                      "correlation", "r2_score", "directional_accuracy"]
        
        results = {}
        
        for metric in metrics:
            try:
                if hasattr(self, metric):
                    value = getattr(self, metric)(y_true, y_pred)
                    results[metric] = float(value) if not np.isnan(value) else 0.0
                else:
                    logger.warning(f"Unknown metric: {metric}")
            except Exception as e:
                logger.warning(f"Error calculating {metric}: {e}")
                results[metric] = 0.0
        
        return results


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate forecasting metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        metrics: List of metric names to calculate
        
    Returns:
        Dictionary of metric names and values
    """
    calculator = MetricsCalculator()
    return calculator.calculate_all(y_true, y_pred, metrics)


def validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate inputs for metric calculation."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")
    
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        logger.warning("NaN values detected in inputs")
    
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        logger.warning("Infinite values detected in inputs")


def benchmark_metrics(results: Dict[str, Dict[str, float]], 
                     tolerance: Dict[str, float]) -> Dict[str, bool]:
    """
    Benchmark metrics against tolerance thresholds.
    
    Args:
        results: Dictionary of model results
        tolerance: Dictionary of metric tolerances
        
    Returns:
        Dictionary indicating which models pass benchmarks
    """
    benchmark_results = {}
    
    for model_name, metrics in results.items():
        passes = True
        for metric_name, threshold in tolerance.items():
            if metric_name in metrics:
                if metric_name in ["mae", "rmse", "mse", "mape", "smape", "msis"]:
                    # Lower is better metrics
                    if metrics[metric_name] > threshold:
                        passes = False
                        break
                else:
                    # Higher is better metrics (correlation, r2, directional_accuracy)
                    if metrics[metric_name] < threshold:
                        passes = False
                        break
        
        benchmark_results[model_name] = passes
    
    return benchmark_results