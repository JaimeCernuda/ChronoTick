"""
Chronos-Bolt implementation.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import warnings

from ..base import BaseTimeSeriesModel, ForecastOutput, ModelStatus

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class ChronosBoltModel(BaseTimeSeriesModel):
    """
    Chronos-Bolt implementation.
    
    Chronos-Bolt is Amazon's enhanced T5 encoder-decoder foundation model for probabilistic forecasting.
    Features 250x faster inference, 20x memory efficiency, multivariate support, and direct multi-step forecasting.
    """
    
    def __init__(self, model_name: str = "chronos", device: str = "cpu", **kwargs):
        super().__init__(model_name, device, **kwargs)
        
        # Chronos-Bolt specific configuration
        self.model_size = kwargs.get('model_size', 'base')  # Upgraded from tiny: tiny, small, base
        self.model_repo = kwargs.get('model_repo', f'amazon/chronos-bolt-{self.model_size}')  # NEW: chronos-bolt
        self.prediction_length = kwargs.get('prediction_length', 96)
        self.num_samples = kwargs.get('num_samples', 20)
        self.multivariate_support = kwargs.get('multivariate_support', True)  # NEW
        self.direct_multistep = kwargs.get('direct_multistep', True)  # NEW: 250x faster
        
        # Model components
        self.pipeline = None
        self._dependency_error = None
        
        logger.info(f"Initialized Chronos-Bolt with repo: {self.model_repo}")
        logger.info(f"Size: {self.model_size}, Multivariate: {self.multivariate_support}")
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import chronos
            from chronos import ChronosBoltPipeline
            return True
        except ImportError as e:
            self._dependency_error = f"Chronos package not available: {e}. Install with: pip install chronos-forecasting"
            return False
    
    def load_model(self) -> None:
        """Load the Chronos-Bolt model."""
        try:
            logger.info(f"Loading Chronos-Bolt model from {self.model_repo}...")
            self.status = ModelStatus.LOADING
            
            # Check dependencies
            if not self._check_dependencies():
                self.status = ModelStatus.ERROR
                raise RuntimeError(self._dependency_error)
            
            from chronos import ChronosBoltPipeline
            import torch
            
            # Determine device
            device_map = "cpu" if self.device == "cpu" else "auto"
            torch_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
            
            # Load pipeline - use ChronosBoltPipeline for Chronos-Bolt models
            self.pipeline = ChronosBoltPipeline.from_pretrained(
                self.model_repo,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
            
            logger.info(f"Chronos-Bolt model loaded successfully:")
            logger.info(f"  Model repo: {self.model_repo}")
            logger.info(f"  Model size: {self.model_size}")
            logger.info(f"  Prediction length: {self.prediction_length}")
            logger.info(f"  Device: {self.device}")
            
            self.status = ModelStatus.LOADED
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            error_msg = f"Chronos-Bolt model loading failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def forecast(self, context: Union[np.ndarray, List[float]], 
                 horizon: int, **kwargs) -> ForecastOutput:
        """
        Generate forecasts using Chronos-Bolt.
        
        Args:
            context: Historical time series data
            horizon: Number of steps to forecast
            **kwargs: Additional parameters (num_samples, etc.)
            
        Returns:
            ForecastOutput with predictions and metadata
        """
        if self.status != ModelStatus.LOADED or self.pipeline is None:
            if self._dependency_error:
                raise RuntimeError(self._dependency_error)
            else:
                raise RuntimeError("Chronos-Bolt model not loaded. Call load_model() first.")
        
        try:
            # Validate input
            context = self.validate_input(context)
            
            logger.info(f"Generating Chronos-Bolt forecast for {len(context)} context points, horizon {horizon}")
            
            import torch
            
            # Convert to tensor and add batch dimension
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            
            # Generate forecast - ChronosBoltPipeline returns quantiles directly
            forecast = self.pipeline.predict(
                context=context_tensor,
                prediction_length=horizon
            )
            
            # ChronosBoltPipeline returns shape (batch_size, num_quantiles, prediction_length)
            # Extract predictions and quantiles
            forecast_np = forecast[0].numpy()  # Remove batch dimension: (num_quantiles, prediction_length)
            
            # Use median quantile (0.5) as main prediction
            if forecast_np.ndim > 1 and forecast_np.shape[0] >= 5:  # Should have 9 quantiles [0.1, 0.2, ..., 0.9]
                predictions = forecast_np[4]  # Index 4 = 0.5 quantile (median)
                
                # Extract all quantiles
                quantiles = {}
                quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                for i, q in enumerate(quantile_levels):
                    if i < forecast_np.shape[0]:
                        quantiles[str(q)] = forecast_np[i]
            else:
                # Fallback if quantile structure is different
                predictions = forecast_np.flatten() if forecast_np.ndim > 1 else forecast_np
                quantiles = None
            
            logger.info(f"Generated {len(predictions)} predictions")
            logger.info(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
            
            return ForecastOutput(
                predictions=predictions,
                quantiles=quantiles,
                metadata={
                    'model_name': self.model_name,
                    'model_repo': self.model_repo,
                    'model_size': self.model_size,
                    'forecast_horizon': horizon,
                    'context_length': len(context),
                    'device': self.device,
                    'prediction_method': 'quantile_median',
                    'input_shape': context.shape,
                    'output_shape': predictions.shape,
                    'has_quantiles': quantiles is not None,
                    'multivariate_support': self.multivariate_support,
                    'direct_multistep': self.direct_multistep
                }
            )
            
        except Exception as e:
            logger.error(f"Chronos-Bolt forecasting error: {e}")
            raise RuntimeError(f"Chronos-Bolt forecasting failed: {e}")
    
    def forecast_multivariate(self, multivariate_input, horizon: int, **kwargs):
        """
        Chronos-Bolt multivariate forecasting implementation.
        
        Since Chronos-Bolt is primarily a univariate model, we forecast each variable 
        separately but return a unified result format.
        """
        if self.status != ModelStatus.LOADED:
            raise RuntimeError("Chronos-Bolt model not loaded. Call load_model() first.")
            
        try:
            logger.info(f"Chronos-Bolt multivariate forecasting for {len(multivariate_input.variable_names)} variables")
            
            # Validate input shape
            if multivariate_input.data.ndim != 2:
                raise ValueError(f"Multivariate data must be 2D (variables, timesteps), got shape {multivariate_input.data.shape}")
            
            results = {}
            target_vars = multivariate_input.target_variables or multivariate_input.variable_names
            
            for i, var_name in enumerate(multivariate_input.variable_names):
                if var_name in target_vars:
                    # Extract univariate series for this variable
                    var_data = multivariate_input.data[i, :].astype(np.float32)
                    
                    # Generate forecast using main forecast method
                    forecast_result = self.forecast(var_data, horizon, **kwargs)
                    
                    # Enhance metadata with multivariate info
                    forecast_result.metadata.update({
                        'variable_name': var_name,
                        'multivariate_variables': multivariate_input.variable_names,
                        'target_variables': list(target_vars),
                        'variable_index': i,
                        'total_variables': len(multivariate_input.variable_names),
                        'chronos_bolt_multivariate': True,
                        'inference_method': 'separate_univariate'
                    })
                    
                    results[var_name] = forecast_result
            
            logger.info(f"Generated multivariate forecasts for {len(results)} variables")
            return results
            
        except Exception as e:
            logger.error(f"Chronos-Bolt multivariate forecasting failed: {e}")
            raise RuntimeError(f"Chronos-Bolt multivariate forecasting failed: {e}")

    def forecast_with_covariates(self, covariates_input, horizon: int, **kwargs):
        """
        Chronos-Bolt covariates forecasting implementation.
        
        Since Chronos-Bolt doesn't natively support covariates, we use the target 
        series for forecasting and include covariates information in metadata.
        """
        if self.status != ModelStatus.LOADED:
            raise RuntimeError("Chronos-Bolt model not loaded. Call load_model() first.")
            
        try:
            logger.info(f"Chronos-Bolt covariates forecasting with {len(covariates_input.covariates)} covariates")
            
            # Use target series for forecasting (Chronos-Bolt is univariate)
            target_data = covariates_input.target.astype(np.float32)
            
            # Generate forecast using main forecast method
            forecast_result = self.forecast(target_data, horizon, **kwargs)
            
            # Enhance metadata with covariates info
            covariate_names = list(covariates_input.covariates.keys())
            future_covariate_names = list(covariates_input.future_covariates.keys()) if covariates_input.future_covariates else []
            
            forecast_result.metadata.update({
                'covariates_available': covariate_names,
                'covariates_used': False,  # Chronos-Bolt doesn't use covariates in inference
                'future_covariates_available': future_covariate_names,
                'future_covariates_used': future_covariate_names if future_covariate_names else False,
                'chronos_bolt_covariates': True,
                'covariates_method': 'metadata_only',
                'target_series_length': len(target_data)
            })
            
            if covariates_input.categorical_covariates:
                forecast_result.metadata['categorical_covariates'] = list(covariates_input.categorical_covariates.keys())
            
            logger.info(f"Generated covariates-informed forecast with {len(covariate_names)} covariates")
            return forecast_result
            
        except Exception as e:
            logger.error(f"Chronos-Bolt covariates forecasting failed: {e}")
            raise RuntimeError(f"Chronos-Bolt covariates forecasting failed: {e}")
    
    def unload_model(self) -> None:
        """Unload the Chronos-Bolt model and free resources."""
        try:
            if self.pipeline is not None:
                # Chronos pipeline doesn't have explicit unload, so we delete the reference
                del self.pipeline
                self.pipeline = None
                
                # Clear GPU cache if using GPU
                if self.device != "cpu":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                
                logger.info("Chronos-Bolt model unloaded successfully")
            
            self.status = ModelStatus.UNLOADED
            
        except Exception as e:
            logger.error(f"Error unloading Chronos-Bolt model: {e}")
            self.status = ModelStatus.ERROR
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the Chronos-Bolt model."""
        health_status = super().health_check()
        
        # Add Chronos-Bolt-specific health information
        health_status.update({
            'model_repo': self.model_repo,
            'model_size': self.model_size,
            'prediction_length': self.prediction_length,
            'num_samples': self.num_samples,
            'dependency_error': self._dependency_error
        })
        
        # Check dependencies
        if not self._check_dependencies():
            health_status['dependency_status'] = 'failed'
            health_status['dependency_error'] = self._dependency_error
        else:
            health_status['dependency_status'] = 'ok'
        
        return health_status