"""
Base classes for Time Series Foundation Models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model loading and operational status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class ForecastOutput:
    """Standard output format for all time series models."""
    predictions: np.ndarray
    quantiles: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __repr__(self):
        return (f"ForecastOutput(predictions_shape={self.predictions.shape}, "
                f"has_quantiles={self.quantiles is not None}, "
                f"metadata_keys={list(self.metadata.keys()) if self.metadata else None})")


@dataclass
class MultivariateInput:
    """
    Container for multivariate time series input.
    
    Args:
        data: Multivariate time series data (n_variables, n_timesteps)
        variable_names: Names of the variables
        target_variables: Specific variables to forecast (if None, forecast all)
        metadata: Additional information about the data
    """
    data: np.ndarray
    variable_names: List[str]
    target_variables: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CovariatesInput:
    """
    Container for covariates/exogenous variables input.
    
    Args:
        target: Main target time series to forecast
        covariates: Dictionary of covariate name -> values (historical)
        future_covariates: Dictionary of known future covariate values
        categorical_covariates: Dictionary of categorical covariate mappings
        metadata: Additional information about the covariates
    """
    target: np.ndarray
    covariates: Dict[str, np.ndarray]
    future_covariates: Optional[Dict[str, np.ndarray]] = None
    categorical_covariates: Optional[Dict[str, List[Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class FrequencyInfo:
    """
    Container for frequency information.
    
    Args:
        freq_str: String representation of frequency ('D', 'H', 'M', etc.)
        freq_value: Numeric frequency code (for models that need it)
        is_regular: Whether the time series has regular intervals
        detected_freq: Auto-detected frequency if available
    """
    freq_str: Optional[str] = None
    freq_value: Optional[int] = None
    is_regular: bool = True
    detected_freq: Optional[str] = None
    
    def __repr__(self):
        return (f"FrequencyInfo(freq_str='{self.freq_str}', "
                f"freq_value={self.freq_value}, "
                f"is_regular={self.is_regular}, "
                f"detected_freq='{self.detected_freq}')")


class BaseTimeSeriesModel(ABC):
    """Base class for all time series foundation models."""
    
    def __init__(self, model_name: str, device: str = "cpu", **kwargs):
        """
        Initialize base time series model.
        
        Args:
            model_name: Name identifier for the model
            device: Device to run the model on ('cpu', 'cuda', etc.)
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.device = device
        self.config = kwargs
        self.status = ModelStatus.UNLOADED
        self.model = None
        
        # Model metadata
        self._model_info = {
            "name": model_name,
            "device": device,
            "status": self.status.value
        }
        
        logger.info(f"Initialized {self.__class__.__name__} with name: {model_name}, device: {device}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights and prepare for inference."""
        pass
    
    @abstractmethod
    def forecast(self, 
                 context: Union[np.ndarray, List[float]], 
                 horizon: int,
                 **kwargs) -> ForecastOutput:
        """
        Generate forecasts for the given context and horizon.
        
        Args:
            context: Historical time series data
            horizon: Number of steps to forecast
            **kwargs: Model-specific parameters
            
        Returns:
            ForecastOutput containing predictions and metadata
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model and free resources."""
        pass
    
    def forecast_multivariate(self, 
                             multivariate_input: MultivariateInput,
                             horizon: int,
                             **kwargs) -> Dict[str, ForecastOutput]:
        """
        Generate forecasts for multivariate time series.
        
        Args:
            multivariate_input: Multivariate time series data container
            horizon: Number of steps to forecast
            **kwargs: Model-specific parameters
            
        Returns:
            Dictionary of variable_name -> ForecastOutput
        """
        # Default implementation: forecast each variable separately
        if not hasattr(self, 'multivariate_support') or not self.multivariate_support:
            logger.warning(f"{self.__class__.__name__} doesn't support native multivariate forecasting, falling back to univariate")
        
        results = {}
        target_vars = multivariate_input.target_variables or multivariate_input.variable_names
        
        for i, var_name in enumerate(multivariate_input.variable_names):
            if var_name in target_vars:
                # Extract univariate series for this variable
                var_data = multivariate_input.data[i, :]
                
                # Generate forecast
                forecast_result = self.forecast(var_data, horizon, **kwargs)
                
                # Add variable name to metadata
                if forecast_result.metadata is None:
                    forecast_result.metadata = {}
                forecast_result.metadata['variable_name'] = var_name
                
                results[var_name] = forecast_result
        
        return results
    
    def forecast_with_covariates(self,
                                covariates_input: CovariatesInput,
                                horizon: int,
                                frequency: Optional[FrequencyInfo] = None,
                                **kwargs) -> ForecastOutput:
        """
        Generate forecasts using covariates/exogenous variables.
        
        Args:
            covariates_input: Target time series with covariates
            horizon: Number of steps to forecast
            frequency: Frequency information for the time series
            **kwargs: Model-specific parameters
            
        Returns:
            ForecastOutput containing predictions enhanced by covariates
        """
        # Default implementation: use only target, ignore covariates
        if not hasattr(self, 'covariates_support') or not self.covariates_support:
            logger.warning(f"{self.__class__.__name__} doesn't support covariates, using target only")
            
        # Add frequency information if provided
        forecast_kwargs = kwargs.copy()
        if frequency and frequency.freq_value is not None:
            forecast_kwargs['freq'] = frequency.freq_value
        
        result = self.forecast(covariates_input.target, horizon, **forecast_kwargs)
        
        # Add covariates info to metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata['covariates_available'] = list(covariates_input.covariates.keys())
        result.metadata['covariates_used'] = hasattr(self, 'covariates_support') and self.covariates_support
        
        return result
    
    def validate_input(self, context: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Validate and preprocess input data.
        
        Args:
            context: Input time series data
            
        Returns:
            Validated and preprocessed numpy array
        """
        if isinstance(context, list):
            context = np.array(context, dtype=np.float32)
        elif isinstance(context, np.ndarray):
            context = context.astype(np.float32)
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
        
        # Check for empty arrays
        if context.size == 0:
            raise ValueError("Input array is empty")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(context)) or np.any(np.isinf(context)):
            raise ValueError("Input contains NaN or Inf values")
        
        return context
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model.
        
        Returns:
            Dictionary containing health status information
        """
        health_status = {
            "model_name": self.model_name,
            "status": self.status.value,
            "device": self.device,
            "model_loaded": self.model is not None,
            "class": self.__class__.__name__
        }
        
        # Test basic functionality if model is loaded
        if self.status == ModelStatus.LOADED:
            try:
                test_input = np.random.randn(100).astype(np.float32)
                test_output = self.forecast(test_input, horizon=10)
                health_status["test_forecast_success"] = True
                health_status["test_output_shape"] = test_output.predictions.shape
            except Exception as e:
                health_status["test_forecast_success"] = False
                health_status["test_error"] = str(e)
        
        return health_status
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        return {
            **self._model_info,
            "status": self.status.value,
            "config": self.config
        }
    
    @staticmethod
    def detect_frequency(timestamps: Optional[List] = None, 
                        data_points: Optional[int] = None) -> FrequencyInfo:
        """
        Detect frequency from timestamps or infer from data characteristics.
        
        Args:
            timestamps: List of timestamp strings or datetime objects
            data_points: Number of data points (for inference)
            
        Returns:
            FrequencyInfo object with detected frequency information
        """
        freq_info = FrequencyInfo()
        
        if timestamps is not None and len(timestamps) > 1:
            try:
                # Try to parse timestamps and detect frequency
                import pandas as pd
                ts_series = pd.to_datetime(timestamps)
                inferred_freq = pd.infer_freq(ts_series)
                
                if inferred_freq:
                    freq_info.freq_str = inferred_freq
                    freq_info.detected_freq = inferred_freq
                    freq_info.is_regular = True
                    
                    # Map to numeric codes that models might expect
                    freq_mapping = {
                        'D': 1, 'H': 2, 'M': 3, 'Q': 4, 'Y': 5,
                        'B': 6, 'W': 7, 'T': 8, 'S': 9
                    }
                    
                    # Extract base frequency
                    base_freq = inferred_freq.rstrip('0123456789-')
                    freq_info.freq_value = freq_mapping.get(base_freq, 0)
                    
                else:
                    freq_info.is_regular = False
                    freq_info.freq_value = 0  # Unknown/irregular
                    
            except Exception as e:
                logger.warning(f"Could not detect frequency from timestamps: {e}")
                freq_info.is_regular = False
                freq_info.freq_value = 0
        else:
            # No timestamps provided, use default
            freq_info.freq_value = 0  # Unknown frequency
            
        return freq_info
    
    @staticmethod
    def convert_frequency_format(freq_input: Union[str, int, FrequencyInfo]) -> FrequencyInfo:
        """
        Convert various frequency formats to standardized FrequencyInfo.
        
        Args:
            freq_input: Frequency in various formats
            
        Returns:
            Standardized FrequencyInfo object
        """
        if isinstance(freq_input, FrequencyInfo):
            return freq_input
            
        freq_info = FrequencyInfo()
        
        if isinstance(freq_input, str):
            freq_info.freq_str = freq_input
            # Common frequency mappings
            freq_mapping = {
                'd': 1, 'daily': 1,
                'h': 2, 'hourly': 2, 
                'm': 3, 'monthly': 3,
                'q': 4, 'quarterly': 4,
                'y': 5, 'yearly': 5, 'annual': 5,
                'b': 6, 'business': 6,
                'w': 7, 'weekly': 7,
                't': 8, 'minute': 8, 'min': 8,
                's': 9, 'second': 9, 'sec': 9
            }
            freq_info.freq_value = freq_mapping.get(freq_input.lower(), 0)
            
        elif isinstance(freq_input, int):
            freq_info.freq_value = freq_input
            # Reverse mapping for display
            reverse_mapping = {
                1: 'D', 2: 'H', 3: 'M', 4: 'Q', 5: 'Y',
                6: 'B', 7: 'W', 8: 'T', 9: 'S'
            }
            freq_info.freq_str = reverse_mapping.get(freq_input, 'Unknown')
            
        return freq_info
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.model_name}', status={self.status.value}, device='{self.device}')"
    
    def __str__(self):
        return self.__repr__()