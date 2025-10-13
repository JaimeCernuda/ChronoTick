"""
TSFM Factory: Main factory class for managing time series foundation models.
"""

import logging
from typing import Any, Dict, List, Optional, Type
from pathlib import Path
import yaml

from .base import BaseTimeSeriesModel, ModelStatus

logger = logging.getLogger(__name__)


class TSFMFactory:
    """Factory class for creating and managing time series foundation models."""
    
    # Registry of available models
    _models_registry: Dict[str, Type[BaseTimeSeriesModel]] = {}
    _model_aliases: Dict[str, str] = {}
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the TSFM Factory.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = {}
        self.loaded_models = {}
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        else:
            # Try to load default config
            default_config = Path(__file__).parent / "config" / "default.yaml"
            if default_config.exists():
                self.load_config(str(default_config))
        
        # Import and register all models
        self._register_all_models()
        
        logger.info(f"TSFMFactory initialized with {len(self._models_registry)} models")
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseTimeSeriesModel], 
                      aliases: Optional[List[str]] = None) -> None:
        """
        Register a model class with the factory.
        
        Args:
            name: Primary name for the model
            model_class: Model class (must inherit from BaseTimeSeriesModel)
            aliases: Optional list of alternative names
        """
        if not issubclass(model_class, BaseTimeSeriesModel):
            raise ValueError(f"{model_class} must inherit from BaseTimeSeriesModel")
        
        cls._models_registry[name.lower()] = model_class
        
        # Register aliases
        if aliases:
            for alias in aliases:
                cls._model_aliases[alias.lower()] = name.lower()
        
        logger.debug(f"Registered model: {name} ({model_class.__name__})")
    
    def _register_all_models(self) -> None:
        """Import and register all available models."""
        try:
            from .LLM import (
                TimesFMModel,
                TTMModel,
                ChronosBoltModel,
                TotoModel,
                TimeMoEModel
            )
            
            # Register each model
            self.register_model("timesfm", TimesFMModel, ["times-fm", "timesfoundation"])
            self.register_model("ttm", TTMModel, ["tiny-time-mixer"])
            self.register_model("chronos", ChronosBoltModel, ["chronos-bolt", "chronos-t5"])
            self.register_model("toto", TotoModel, ["datadog-toto", "toto-open"])
            self.register_model("time_moe", TimeMoEModel, ["timemoe", "time-mixture-of-experts"])
            
        except ImportError as e:
            logger.warning(f"Some models could not be imported: {e}")
    
    def list_models(self) -> List[str]:
        """Get list of all available models."""
        return list(self._models_registry.keys())
    
    def list_available_models(self) -> List[str]:
        """Get list of all available models (alias for compatibility)."""
        return self.list_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_name = model_name.lower()
        
        # Check for alias
        if model_name in self._model_aliases:
            model_name = self._model_aliases[model_name]
        
        if model_name not in self._models_registry:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.list_models()}")
        
        model_class = self._models_registry[model_name]
        
        # Get enhanced capabilities by creating a temporary instance
        try:
            temp_model = model_class(model_name=model_name, device="cpu")
            capabilities = self._get_model_capabilities(temp_model)
        except Exception:
            capabilities = {}
        
        return {
            "name": model_name,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "docstring": model_class.__doc__,
            "capabilities": capabilities
        }
    
    def create_model(self, model_name: str, **kwargs) -> BaseTimeSeriesModel:
        """
        Create a new model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance
        """
        model_name = model_name.lower()
        
        # Check for alias
        if model_name in self._model_aliases:
            model_name = self._model_aliases[model_name]
        
        if model_name not in self._models_registry:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.list_models()}")
        
        # Get model configuration from config file
        model_config = self.config.get("models", {}).get(model_name, {})
        
        # Merge with provided kwargs (kwargs override config)
        final_config = {**model_config, **kwargs}
        
        # Create model instance
        model_class = self._models_registry[model_name]
        model = model_class(model_name=model_name, **final_config)
        
        logger.info(f"Created model instance: {model_name}")
        return model
    
    def load_model(self, model_name: str, **kwargs) -> BaseTimeSeriesModel:
        """
        Create and load a model, ready for inference.
        
        Args:
            model_name: Name of the model to load
            **kwargs: Model-specific parameters
            
        Returns:
            Loaded model instance
        """
        # Check if model is already loaded
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded, returning cached instance")
            return self.loaded_models[model_name]
        
        # Create model
        model = self.create_model(model_name, **kwargs)
        
        # Load model weights
        try:
            model.load_model()
            self.loaded_models[model_name] = model
            logger.info(f"Model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
        
        return model
    
    def unload_model(self, model_name: str) -> None:
        """Unload a specific model and free resources."""
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            model.unload_model()
            del self.loaded_models[model_name]
            logger.info(f"Model '{model_name}' unloaded")
        else:
            logger.warning(f"Model '{model_name}' not in loaded models")
    
    def unload_all_models(self) -> None:
        """Unload all loaded models."""
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        logger.info("All models unloaded")
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self.loaded_models.keys())
    
    def list_loaded_models(self) -> List[str]:
        """Get list of currently loaded models (alias for compatibility)."""
        return self.get_loaded_models()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all loaded models."""
        health_report = {
            "factory_status": "operational",
            "registered_models": self.list_models(),
            "loaded_models": self.get_loaded_models(),
            "models_health": {}
        }
        
        for model_name, model in self.loaded_models.items():
            health_report["models_health"][model_name] = model.health_check()
        
        return health_report
    
    def _get_model_capabilities(self, model: BaseTimeSeriesModel) -> Dict[str, Any]:
        """Get enhanced capabilities of a model."""
        capabilities = {
            "multivariate_support": getattr(model, 'multivariate_support', False),
            "covariates_support": getattr(model, 'covariates_support', False),
            "exogenous_support": getattr(model, 'exogenous_support', False),
            "covariates_adaptable": getattr(model, 'covariates_adaptable', False),
            "mixture_of_experts": getattr(model, 'mixture_of_experts', False),
            "direct_multistep": getattr(model, 'direct_multistep', False),
            "enhanced_methods": []
        }
        
        # Check for enhanced methods
        if hasattr(model, 'forecast_multivariate'):
            capabilities["enhanced_methods"].append("forecast_multivariate")
        if hasattr(model, 'forecast_with_covariates'):
            capabilities["enhanced_methods"].append("forecast_with_covariates")
        
        return capabilities
    
    def list_models_by_capability(self, capability: str) -> List[str]:
        """List models that support a specific capability."""
        supported_models = []
        
        for model_name in self.list_models():
            try:
                model_class = self._models_registry[model_name]
                temp_model = model_class(model_name=model_name, device="cpu")
                
                if getattr(temp_model, capability, False):
                    supported_models.append(model_name)
            except Exception:
                continue
        
        return supported_models
    
    def get_multivariate_models(self) -> List[str]:
        """Get list of models that support multivariate forecasting."""
        return self.list_models_by_capability('multivariate_support')
    
    def get_covariates_models(self) -> List[str]:
        """Get list of models that support covariates/exogenous variables."""
        models_with_covariates = []
        models_with_covariates.extend(self.list_models_by_capability('covariates_support'))
        models_with_covariates.extend(self.list_models_by_capability('exogenous_support'))
        models_with_covariates.extend(self.list_models_by_capability('covariates_adaptable'))
        return list(set(models_with_covariates))  # Remove duplicates
    
    def forecast_multivariate(self, model_name: str, multivariate_input, horizon: int, **kwargs):
        """
        Convenience method for multivariate forecasting.
        
        Args:
            model_name: Name of the model to use
            multivariate_input: MultivariateInput data structure
            horizon: Forecast horizon
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of forecasts for each variable
        """
        if model_name not in self.loaded_models:
            raise ValueError(f"Model '{model_name}' not loaded. Call load_model() first.")
        
        model = self.loaded_models[model_name]
        
        if not hasattr(model, 'forecast_multivariate'):
            raise ValueError(f"Model '{model_name}' does not support multivariate forecasting")
        
        return model.forecast_multivariate(multivariate_input, horizon, **kwargs)
    
    def forecast_with_covariates(self, model_name: str, covariates_input, horizon: int, 
                                frequency=None, **kwargs):
        """
        Convenience method for covariates forecasting.
        
        Args:
            model_name: Name of the model to use
            covariates_input: CovariatesInput data structure
            horizon: Forecast horizon
            frequency: Optional frequency information
            **kwargs: Additional parameters
            
        Returns:
            ForecastOutput with covariates-enhanced predictions
        """
        if model_name not in self.loaded_models:
            raise ValueError(f"Model '{model_name}' not loaded. Call load_model() first.")
        
        model = self.loaded_models[model_name]
        
        if not hasattr(model, 'forecast_with_covariates'):
            raise ValueError(f"Model '{model_name}' does not support covariates forecasting")
        
        return model.forecast_with_covariates(covariates_input, horizon, frequency, **kwargs)
    
    def get_enhanced_features_summary(self) -> Dict[str, Any]:
        """Get summary of enhanced features across all models."""
        summary = {
            "total_models": len(self.list_models()),
            "multivariate_models": len(self.get_multivariate_models()),
            "covariates_models": len(self.get_covariates_models()),
            "enhanced_capabilities": {
                "multivariate_support": self.get_multivariate_models(),
                "covariates_support": self.get_covariates_models(),
                "frequency_detection": True,  # Available in base class
                "enhanced_data_structures": ["MultivariateInput", "CovariatesInput", "FrequencyInfo"]
            }
        }
        
        return summary
    
    def create_multivariate_input(self, data, variable_names, target_variables=None, metadata=None):
        """
        Convenience method to create MultivariateInput.
        
        Args:
            data: Multivariate time series data (n_variables, n_timesteps)
            variable_names: Names of the variables
            target_variables: Specific variables to forecast (if None, forecast all)
            metadata: Additional information about the data
            
        Returns:
            MultivariateInput instance
        """
        from .base import MultivariateInput
        
        return MultivariateInput(
            data=data,
            variable_names=variable_names,
            target_variables=target_variables,
            metadata=metadata
        )
    
    def create_covariates_input(self, target, covariates, future_covariates=None, 
                               categorical_covariates=None, metadata=None):
        """
        Convenience method to create CovariatesInput.
        
        Args:
            target: Main target time series to forecast
            covariates: Dictionary of covariate name -> values (historical)
            future_covariates: Dictionary of known future covariate values
            categorical_covariates: Dictionary of categorical covariate mappings
            metadata: Additional information about the covariates
            
        Returns:
            CovariatesInput instance
        """
        from .base import CovariatesInput
        
        return CovariatesInput(
            target=target,
            covariates=covariates,
            future_covariates=future_covariates,
            categorical_covariates=categorical_covariates,
            metadata=metadata
        )
    
    def create_frequency_info(self, freq_str=None, freq_value=None, is_regular=True, detected_freq=None):
        """
        Convenience method to create FrequencyInfo.
        
        Args:
            freq_str: String representation of frequency ('D', 'H', 'M', etc.)
            freq_value: Numeric frequency code (for models that need it)
            is_regular: Whether the time series has regular intervals
            detected_freq: Auto-detected frequency if available
            
        Returns:
            FrequencyInfo instance
        """
        from .base import FrequencyInfo
        
        return FrequencyInfo(
            freq_str=freq_str,
            freq_value=freq_value,
            is_regular=is_regular,
            detected_freq=detected_freq
        )
    
    def _create_model(self, model_name: str, **kwargs) -> BaseTimeSeriesModel:
        """Create a model instance without loading (for internal use and testing)."""
        return self.create_model(model_name, **kwargs)
    
    def __repr__(self):
        return (f"TSFMFactory(registered={len(self._models_registry)}, "
                f"loaded={len(self.loaded_models)})")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload all models."""
        self.unload_all_models()