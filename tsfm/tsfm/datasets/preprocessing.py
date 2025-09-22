"""
Data preprocessing utilities for time series data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_data(data: Union[np.ndarray, pd.DataFrame], 
                  method: str = "zscore") -> Tuple[Union[np.ndarray, pd.DataFrame], Dict]:
    """
    Normalize time series data.
    
    Args:
        data: Input time series data
        method: Normalization method ('zscore', 'minmax')
        
    Returns:
        Tuple of (normalized_data, normalization_stats)
    """
    if isinstance(data, pd.DataFrame):
        # Handle DataFrame
        normalized_data = data.copy()
        stats = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            if method == "zscore":
                mean = data[col].mean()
                std = data[col].std() + 1e-8
                normalized_data[col] = (data[col] - mean) / std
                stats[col] = {"method": "zscore", "mean": mean, "std": std}
            elif method == "minmax":
                min_val = data[col].min()
                max_val = data[col].max()
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val + 1e-8)
                stats[col] = {"method": "minmax", "min": min_val, "max": max_val}
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized_data, stats
    
    else:
        # Handle numpy array
        if method == "zscore":
            mean = np.mean(data)
            std = np.std(data) + 1e-8
            normalized = (data - mean) / std
            stats = {"method": "zscore", "mean": mean, "std": std}
        elif method == "minmax":
            min_val = np.min(data)
            max_val = np.max(data)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            stats = {"method": "minmax", "min": min_val, "max": max_val}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.debug(f"Normalized data using {method} method")
        return normalized, stats


def denormalize_data(normalized_data: np.ndarray, stats: Dict, method: str = "zscore") -> np.ndarray:
    """
    Denormalize data using provided statistics.
    
    Args:
        normalized_data: Normalized data
        stats: Normalization statistics from normalize_data
        method: Denormalization method ('zscore', 'minmax')
        
    Returns:
        Denormalized data
    """
    if method == "zscore":
        if "mean" not in stats or "std" not in stats:
            raise KeyError("Missing 'mean' or 'std' in stats for zscore denormalization")
        return normalized_data * stats["std"] + stats["mean"]
    elif method == "minmax":
        if "min" not in stats or "max" not in stats:
            raise KeyError("Missing 'min' or 'max' in stats for minmax denormalization")
        return normalized_data * (stats["max"] - stats["min"]) + stats["min"]
    else:
        raise ValueError(f"Unknown denormalization method: {method}")


def create_sequences(data: np.ndarray, 
                    sequence_length: int, 
                    forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series forecasting.
    
    Args:
        data: Input time series data
        sequence_length: Length of input sequences
        forecast_horizon: Length of forecast sequences
        
    Returns:
        Tuple of (X, y) where X is input sequences and y is target sequences
    """
    if len(data) < sequence_length + forecast_horizon:
        raise ValueError(f"Data length {len(data)} is too short for sequence_length={sequence_length} + forecast_horizon={forecast_horizon}")
    
    # Handle multivariatet data
    if data.ndim == 1:
        # Univariate case
        n_sequences = len(data) - sequence_length - forecast_horizon + 1
        X = np.zeros((n_sequences, sequence_length))
        y = np.zeros((n_sequences, forecast_horizon))
        
        for i in range(n_sequences):
            X[i] = data[i:i + sequence_length]
            y[i] = data[i + sequence_length:i + sequence_length + forecast_horizon]
    
    else:
        # Multivariate case
        n_sequences = len(data) - sequence_length - forecast_horizon + 1
        n_features = data.shape[1]
        X = np.zeros((n_sequences, sequence_length, n_features))
        y = np.zeros((n_sequences, forecast_horizon, n_features))
        
        for i in range(n_sequences):
            X[i] = data[i:i + sequence_length]
            y[i] = data[i + sequence_length:i + sequence_length + forecast_horizon]
    
    logger.info(f"Created {n_sequences} sequences: X={X.shape}, y={y.shape}")
    return X, y


def split_data(data: np.ndarray, 
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split time series data into train/validation/test sets.
    
    Args:
        data: Input time series data
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return train_data, val_data, test_data


def create_sliding_windows(data: np.ndarray,
                          context_length: int,
                          horizon_length: int,
                          stride: int = 1,
                          min_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series forecasting.
    
    Args:
        data: Input time series data
        context_length: Length of historical context
        horizon_length: Length of forecast horizon
        stride: Step size between windows
        min_samples: Minimum number of samples required
        
    Returns:
        Tuple of (contexts, targets) arrays
    """
    total_window_length = context_length + horizon_length
    
    if len(data) < total_window_length:
        raise ValueError(f"Data length {len(data)} is too short for window length {total_window_length}")
    
    # Calculate number of windows
    num_windows = (len(data) - total_window_length) // stride + 1
    
    if num_windows < min_samples:
        logger.warning(f"Only {num_windows} windows available, less than minimum {min_samples}")
    
    contexts = []
    targets = []
    
    for i in range(0, len(data) - total_window_length + 1, stride):
        context = data[i:i + context_length]
        target = data[i + context_length:i + total_window_length]
        
        contexts.append(context)
        targets.append(target)
    
    contexts = np.array(contexts)
    targets = np.array(targets)
    
    logger.info(f"Created {len(contexts)} sliding windows: context={context_length}, horizon={horizon_length}")
    
    return contexts, targets


def remove_outliers(data: np.ndarray, 
                   method: str = "iqr", 
                   threshold: float = 1.5) -> np.ndarray:
    """
    Remove outliers from time series data.
    
    Args:
        data: Input time series data
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Data with outliers removed (replaced with median)
    """
    data_clean = data.copy()
    
    if method == "iqr":
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        outliers = z_scores > threshold
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Replace outliers with median
    median = np.median(data)
    data_clean[outliers] = median
    
    num_outliers = np.sum(outliers)
    logger.info(f"Removed {num_outliers} outliers ({num_outliers/len(data)*100:.1f}%) using {method} method")
    
    return data_clean


def fill_missing_values(data: np.ndarray, method: str = "interpolate") -> np.ndarray:
    """
    Fill missing values in time series data.
    
    Args:
        data: Input time series data (may contain NaN)
        method: Fill method ('interpolate', 'forward', 'backward', 'mean', 'median')
        
    Returns:
        Data with missing values filled
    """
    if not np.any(np.isnan(data)):
        return data
    
    data_filled = data.copy()
    missing_mask = np.isnan(data_filled)
    
    if method == "interpolate":
        # Linear interpolation
        valid_indices = np.where(~missing_mask)[0]
        if len(valid_indices) > 1:
            data_filled[missing_mask] = np.interp(
                np.where(missing_mask)[0], 
                valid_indices, 
                data_filled[valid_indices]
            )
    elif method == "forward":
        # Forward fill
        for i in range(1, len(data_filled)):
            if np.isnan(data_filled[i]) and not np.isnan(data_filled[i-1]):
                data_filled[i] = data_filled[i-1]
    elif method == "backward":
        # Backward fill
        for i in range(len(data_filled)-2, -1, -1):
            if np.isnan(data_filled[i]) and not np.isnan(data_filled[i+1]):
                data_filled[i] = data_filled[i+1]
    elif method == "mean":
        mean_val = np.nanmean(data_filled)
        data_filled[missing_mask] = mean_val
    elif method == "median":
        median_val = np.nanmedian(data_filled)
        data_filled[missing_mask] = median_val
    else:
        raise ValueError(f"Unknown fill method: {method}")
    
    num_filled = np.sum(missing_mask)
    logger.info(f"Filled {num_filled} missing values using {method} method")
    
    return data_filled


def detect_anomalies(data: np.ndarray, 
                    window_size: int = 50,
                    threshold: float = 3.0) -> np.ndarray:
    """
    Detect anomalies in time series data using moving statistics.
    
    Args:
        data: Input time series data
        window_size: Size of the moving window
        threshold: Threshold for anomaly detection (in standard deviations)
        
    Returns:
        Boolean array indicating anomalies
    """
    anomalies = np.zeros(len(data), dtype=bool)
    
    for i in range(window_size, len(data)):
        window = data[i-window_size:i]
        window_mean = np.mean(window)
        window_std = np.std(window)
        
        if window_std > 0:
            z_score = abs(data[i] - window_mean) / window_std
            if z_score > threshold:
                anomalies[i] = True
    
    num_anomalies = np.sum(anomalies)
    logger.info(f"Detected {num_anomalies} anomalies ({num_anomalies/len(data)*100:.1f}%)")
    
    return anomalies