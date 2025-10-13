"""
TimesFM (Time Series Foundation Model) implementation.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import warnings

from ..base import BaseTimeSeriesModel, ForecastOutput, ModelStatus

# Import debug logging utilities (optional, gracefully handle if not available)
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from chronotick.inference.debug_logger import debug_log_call, debug_log_section, debug_log_variable, DebugTimer
    DEBUG_AVAILABLE = True
except ImportError:
    # Fallback: create no-op decorators if debug_logger not available
    def debug_log_call(func):
        return func
    def debug_log_section(name):
        pass
    def debug_log_variable(name, value):
        pass
    class DebugTimer:
        def __init__(self, name):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    DEBUG_AVAILABLE = False

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class TimesFMModel(BaseTimeSeriesModel):
    """
    TimesFM 2.5 (Time Series Foundation Model) implementation.

    TimesFM 2.5 is Google Research's enhanced 200M parameter foundation model for time-series forecasting.
    Features 4x longer context (2048), dynamic covariates support, flexible horizons, and 25% better performance.
    """
    
    def __init__(self, model_name: str = "timesfm", device: str = "cpu", **kwargs):
        super().__init__(model_name, device, **kwargs)
        
        # TimesFM 2.5 specific configuration
        self.model_repo = kwargs.get('model_repo', 'google/timesfm-2.5-200m-pytorch')
        self.context_len = kwargs.get('context_len', 2048)  # 4x increase
        self.horizon_len = kwargs.get('horizon_len', 'flexible')  # Flexible at API level
        # For initialization, use default integer value if flexible
        self._init_horizon_len = 96 if self.horizon_len == 'flexible' else self.horizon_len
        self.input_patch_len = kwargs.get('input_patch_len', 32)
        self.output_patch_len = kwargs.get('output_patch_len', 128)
        # Use default num_layers unless explicitly overridden
        # The checkpoint determines actual layers, setting wrong value causes mismatch
        self.num_layers = kwargs.get('num_layers', 20)  # Default from TimesFM
        self.model_dims = kwargs.get('model_dims', 1280)
        self.covariates_support = kwargs.get('covariates_support', True)  # NEW

        # Minimum context length should be output_patch_len for proper patching
        self.min_context_len = max(self.input_patch_len, self.output_patch_len)
        
        # Model components
        self.tfm = None
        self._dependency_error = None
        
        logger.info(f"Initialized TimesFM 2.5 with repo: {self.model_repo}")
        logger.info(f"Context length: {self.context_len}, Covariates: {self.covariates_support}")
    
    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import timesfm
            import torch
            return True
        except ImportError as e:
            self._dependency_error = f"TimesFM dependencies not available: {e}. Install with: pip install timesfm torch"
            return False

    @debug_log_call
    def load_model(self) -> None:
        """Load the TimesFM 2.5 model using the official timesfm package."""
        try:
            debug_log_section("TimesFM 2.5 Model Loading")
            logger.info(f"Loading TimesFM 2.5 model from {self.model_repo}...")
            self.status = ModelStatus.LOADING

            # Check dependencies
            if not self._check_dependencies():
                self.status = ModelStatus.ERROR
                raise RuntimeError(self._dependency_error)

            import timesfm
            import torch

            # TimesFM 2.5 API: from_pretrained + compile
            logger.info("Using TimesFM 2.5 via official timesfm package")
            logger.info(f"Config: context_len={self.context_len}, device={self.device}")

            # Load model with from_pretrained
            # torch_compile is disabled for CPU for compatibility
            torch_compile_enabled = (self.device != "cpu")
            self.tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_repo,
                torch_compile=torch_compile_enabled
            )

            # Get config values with defaults
            max_context = min(self.context_len, 2048)  # TimesFM 2.5 max
            max_horizon = 256  # TimesFM 2.5 default max horizon

            # Compile with ForecastConfig using config values
            forecast_config = timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,  # Time offsets can be negative
                fix_quantile_crossing=True,
            )
            self.tfm.compile(forecast_config)

            logger.info(f"ForecastConfig: max_context={max_context}, max_horizon={max_horizon}")

            logger.info(f"TimesFM 2.5 model loaded successfully:")
            logger.info(f"  Model repo: {self.model_repo}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Context length: {self.context_len}")
            logger.info(f"  Max horizon: 256")

            self.status = ModelStatus.LOADED

        except Exception as e:
            self.status = ModelStatus.ERROR
            error_msg = f"TimesFM 2.5 model loading failed: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    @debug_log_call
    def forecast(self, context: Union[np.ndarray, List[float]],
                 horizon: int, **kwargs) -> ForecastOutput:
        """
        Generate forecasts using TimesFM.

        Args:
            context: Historical time series data
            horizon: Number of steps to forecast
            **kwargs: Additional parameters (freq for frequency)

        Returns:
            ForecastOutput with predictions and metadata
        """
        debug_log_section("TimesFM Forecast")
        debug_log_variable("horizon", horizon)
        debug_log_variable("kwargs", kwargs)

        if self.status != ModelStatus.LOADED or self.tfm is None:
            if self._dependency_error:
                raise RuntimeError(self._dependency_error)
            else:
                raise RuntimeError("TimesFM model not loaded. Call load_model() first.")
        
        try:
            # Validate input
            context = self.validate_input(context)

            # Get frequency from kwargs or use default
            freq = kwargs.get('freq', 0)  # 0 for unknown frequency

            # TimesFM expects inputs as a list of 1D time series
            # Each time series in the list should be 1D: (sequence_length,)
            # The list provides the batch dimension
            if context.ndim == 2:
                # Already batched - extract first series
                context_1d = context[0]
            else:
                # Already 1D
                context_1d = context

            # TimesFM requires minimum context length based on output_patch_len
            # Pad with zeros if context is too short
            min_context_len = self.min_context_len  # 128 (output_patch_len)
            if len(context_1d) < min_context_len:
                logger.warning(f"Context length {len(context_1d)} < minimum {min_context_len}, padding with zeros")
                # Pad at the beginning with zeros
                padding = np.zeros(min_context_len - len(context_1d), dtype=np.float32)
                context_1d = np.concatenate([padding, context_1d])
                logger.debug(f"After padding: context length = {len(context_1d)}")

            logger.info(f"Generating TimesFM 2.5 forecast for {len(context_1d)} context points, horizon {horizon}")
            logger.debug(f"Context shape before TimesFM: 1D with length {len(context_1d)}")

            # Import torch for tensor operations
            import torch

            # Check for data quality issues
            has_nan = np.isnan(context_1d).any()
            has_inf = np.isinf(context_1d).any()
            if has_nan or has_inf:
                logger.error(f"Context has NaN: {has_nan}, has Inf: {has_inf}")
                raise ValueError(f"Context contains invalid values (NaN: {has_nan}, Inf: {has_inf})")

            logger.debug(f"Context min: {context_1d.min():.6f}, max: {context_1d.max():.6f}, mean: {context_1d.mean():.6f}")

            # TimesFM 2.5 API: model.forecast(horizon=, inputs=)
            # inputs: list of numpy arrays
            # Returns: (point_forecast, quantile_forecast)
            logger.debug(f"Calling TimesFM 2.5 with context shape: {context_1d.shape}, horizon: {horizon}")

            # TimesFM 2.5 forecast method
            point_forecast, quantile_forecast = self.tfm.forecast(
                horizon=horizon,
                inputs=[context_1d]  # List of numpy arrays
            )

            logger.debug(f"TimesFM 2.5 returned forecasts successfully")

            # Extract predictions from output
            # point_forecast is a list/array of predictions
            if isinstance(point_forecast, list) and len(point_forecast) > 0:
                predictions = np.array(point_forecast[0])[:horizon]
            elif isinstance(point_forecast, np.ndarray):
                if point_forecast.ndim > 1:
                    predictions = point_forecast[0][:horizon]
                else:
                    predictions = point_forecast[:horizon]
            else:
                raise ValueError(f"Unexpected point_forecast format: {type(point_forecast)}")

            logger.debug(f"Got {len(predictions)} predictions from model")

            # Extract quantiles if available
            quantiles = None
            if quantile_forecast is not None:
                quantiles = {}
                # TimesFM 2.5 provides quantile forecasts
                # Format depends on model output structure
                if isinstance(quantile_forecast, dict):
                    for q_level, q_values in quantile_forecast.items():
                        if isinstance(q_values, (list, np.ndarray)):
                            q_arr = np.array(q_values)
                            if q_arr.ndim > 1:
                                quantiles[str(q_level)] = q_arr[0][:horizon]
                            else:
                                quantiles[str(q_level)] = q_arr[:horizon]
                logger.debug(f"Extracted {len(quantiles) if quantiles else 0} quantile levels")

            logger.info(f"Generated {len(predictions)} predictions")
            logger.info(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")
            
            return ForecastOutput(
                predictions=predictions,
                quantiles=quantiles,
                metadata={
                    'model_name': self.model_name,
                    'model_repo': self.model_repo,
                    'forecast_horizon': horizon,
                    'context_length': len(context),
                    'device': self.device,
                    'freq': freq,
                    'context_len': self.context_len,
                    'input_patch_len': self.input_patch_len,
                    'output_patch_len': self.output_patch_len,
                    'input_shape': context.shape,
                    'output_shape': predictions.shape,
                    'has_quantiles': quantiles is not None,
                    'covariates_support': self.covariates_support,
                    'timesfm_version': '2.5'
                }
            )
            
        except Exception as e:
            import traceback
            logger.error(f"TimesFM forecasting error: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"TimesFM forecasting failed: {e}")

    def forecast_multivariate(self, multivariate_input, horizon: int, **kwargs):
        """
        TimesFM 2.5 multivariate forecasting implementation.

        Since TimesFM 2.5 is primarily a univariate model, we forecast each variable
        separately but return a unified result format.
        """
        if self.status != ModelStatus.LOADED:
            raise RuntimeError("TimesFM 2.5 model not loaded. Call load_model() first.")
            
        try:
            logger.info(f"TimesFM 2.5 multivariate forecasting for {len(multivariate_input.variable_names)} variables")
            
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
                        'timesfm_2_5_multivariate': True,
                        'inference_method': 'separate_univariate',
                        'context_length_used': len(var_data)
                    })
                    
                    results[var_name] = forecast_result
            
            logger.info(f"Generated multivariate forecasts for {len(results)} variables")
            return results
            
        except Exception as e:
            logger.error(f"TimesFM 2.5 multivariate forecasting failed: {e}")
            raise RuntimeError(f"TimesFM 2.5 multivariate forecasting failed: {e}")
    
    @debug_log_call
    def forecast_with_covariates(self, covariates_input, horizon: int, frequency=None, **kwargs):
        """
        Enhanced TimesFM 2.5 forecasting with dynamic covariates support.

        TimesFM 2.5 supports dynamic covariates that influence the forecasting process.
        This method provides two modes:
        - use_covariates=True: Uses covariates in predictions (calls TimesFM API with covariates)
        - use_covariates=False: Standard forecast, covariates only in metadata

        Args:
            covariates_input: CovariatesInput with target series and covariates
            horizon: Number of steps to forecast
            frequency: Optional FrequencyInfo object
            **kwargs: Additional parameters including:
                - use_covariates (bool): Whether to use covariates in predictions (default: False)
                - xreg_mode (str): 'xreg + timesfm' or 'timesfm + xreg' (default: 'xreg + timesfm')

        Returns:
            ForecastOutput with predictions and metadata
        """
        debug_log_section("TimesFM Forecast with Covariates")

        if not self.covariates_support or self.status != ModelStatus.LOADED:
            # Fallback to base implementation
            logger.debug("Falling back to base implementation (covariates_support=False or not loaded)")
            return super().forecast_with_covariates(covariates_input, horizon, frequency, **kwargs)

        # Check if we should use covariates in predictions
        use_covariates = kwargs.get('use_covariates', False)
        debug_log_variable("use_covariates", use_covariates)
        debug_log_variable("num_covariates", len(covariates_input.covariates))
        debug_log_variable("covariate_names", list(covariates_input.covariates.keys()))

        try:
            # Get frequency information
            freq = 0  # Default
            if frequency and frequency.freq_value is not None:
                freq = frequency.freq_value
            elif 'freq' in kwargs:
                freq = kwargs['freq']
            debug_log_variable("frequency", freq)

            # Validate input
            with DebugTimer("Input validation"):
                context = self.validate_input(covariates_input.target)

            # TimesFM expects inputs as a list of 1D time series
            # Ensure context is 1D for proper formatting
            if context.ndim == 2:
                context = context[0]  # Extract first series if batched

            # TimesFM requires minimum context length based on output_patch_len
            # Pad with zeros if context is too short
            min_context_len = self.min_context_len  # 128 (output_patch_len)
            if len(context) < min_context_len:
                logger.warning(f"Context length {len(context)} < minimum {min_context_len}, padding with zeros")
                # Pad at the beginning with zeros
                padding = np.zeros(min_context_len - len(context), dtype=np.float32)
                context = np.concatenate([padding, context])
                logger.debug(f"After padding: context length = {len(context)}")

            debug_log_variable("context_shape", f"1D with length {len(context)}")
            debug_log_variable("context_length", len(context))

            if use_covariates and len(covariates_input.covariates) > 0:
                # Use TimesFM API with covariates
                logger.info(f"TimesFM 2.5 forecasting WITH {len(covariates_input.covariates)} covariates in predictions")

                # Format covariates for TimesFM API
                # TimesFM expects: dict[str, Sequence[Sequence[float]]]
                # We have: dict[str, np.ndarray] -> need to wrap in list
                with DebugTimer("Formatting covariates"):
                    dynamic_numerical_covariates = {}
                    for var_name, values in covariates_input.covariates.items():
                        # Convert numpy array to list and wrap in outer list
                        dynamic_numerical_covariates[var_name] = [values.tolist()]
                        debug_log_variable(f"covariate_{var_name}_length", len(values))

                # Handle categorical covariates if present
                dynamic_categorical_covariates = None
                if covariates_input.categorical_covariates:
                    dynamic_categorical_covariates = {}
                    for var_name, values in covariates_input.categorical_covariates.items():
                        dynamic_categorical_covariates[var_name] = [values]
                    debug_log_variable("categorical_covariates", list(dynamic_categorical_covariates.keys()))

                # Get xreg_mode from kwargs
                xreg_mode = kwargs.get('xreg_mode', 'xreg + timesfm')
                debug_log_variable("xreg_mode", xreg_mode)

                logger.info(f"Calling TimesFM with dynamic_numerical_covariates: {list(dynamic_numerical_covariates.keys())}")
                logger.info(f"xreg_mode: {xreg_mode}")

                # Call TimesFM API with covariates
                with DebugTimer("TimesFM.forecast_with_covariates"):
                    point_forecast, xreg_outputs = self.tfm.forecast_with_covariates(
                        inputs=[context],  # List of 1D time series
                        dynamic_numerical_covariates=dynamic_numerical_covariates,
                        dynamic_categorical_covariates=dynamic_categorical_covariates,
                        freq=[freq],
                        xreg_mode=xreg_mode
                    )
                debug_log_variable("point_forecast_type", type(point_forecast))
                debug_log_variable("xreg_outputs_available", xreg_outputs is not None)

                # Extract predictions
                with DebugTimer("Extracting predictions"):
                    if isinstance(point_forecast, (list, tuple)) and len(point_forecast) > 0:
                        predictions = np.array(point_forecast[0])[:horizon]
                    elif isinstance(point_forecast, np.ndarray):
                        if point_forecast.ndim > 1:
                            predictions = point_forecast[0][:horizon]
                        else:
                            predictions = point_forecast[:horizon]
                    else:
                        raise ValueError(f"Unexpected point_forecast format: {type(point_forecast)}")

                debug_log_variable("predictions_length", len(predictions))
                debug_log_variable("predictions_min", predictions.min())
                debug_log_variable("predictions_max", predictions.max())
                debug_log_variable("predictions_mean", predictions.mean())

                logger.info(f"Generated {len(predictions)} predictions WITH covariates")
                logger.info(f"Prediction range: {predictions.min():.4f} to {predictions.max():.4f}")

                # Build metadata
                metadata = {
                    'model_name': self.model_name,
                    'model_repo': self.model_repo,
                    'forecast_horizon': horizon,
                    'context_length': len(context),
                    'device': self.device,
                    'freq': freq,
                    'covariates_used_in_prediction': True,  # KEY: Actually used!
                    'covariates_available': list(covariates_input.covariates.keys()),
                    'categorical_covariates': list(covariates_input.categorical_covariates.keys()) if covariates_input.categorical_covariates else [],
                    'xreg_mode': xreg_mode,
                    'xreg_outputs_available': xreg_outputs is not None,
                    'timesfm_version': '2.5',
                    'timesfm_api_used': 'forecast_with_covariates'
                }

                return ForecastOutput(
                    predictions=predictions,
                    quantiles=None,  # Covariates mode may not have quantiles
                    metadata=metadata
                )

            else:
                # Standard forecast without using covariates in predictions
                logger.info(f"TimesFM 2.5 forecasting WITHOUT covariates in predictions (use_covariates=False)")
                debug_log_variable("mode", "standard_forecast_without_covariates")

                # Generate forecast using standard method
                with DebugTimer("Standard forecast (no covariates)"):
                    forecast_result = self.forecast(covariates_input.target, horizon, freq=freq)

                # Enhance metadata with covariates information (metadata-only)
                if forecast_result.metadata is None:
                    forecast_result.metadata = {}

                forecast_result.metadata.update({
                    'covariates_used_in_prediction': False,  # KEY: Not used!
                    'covariates_available': list(covariates_input.covariates.keys()),
                    'future_covariates_available': covariates_input.future_covariates is not None,
                    'categorical_covariates': list(covariates_input.categorical_covariates.keys()) if covariates_input.categorical_covariates else [],
                    'timesfm_version': '2.5',
                    'timesfm_api_used': 'forecast'
                })

                return forecast_result

        except Exception as e:
            logger.warning(f"TimesFM 2.5 covariates forecasting failed: {e}, falling back to base")
            return super().forecast_with_covariates(covariates_input, horizon, frequency, **kwargs)
    
    def unload_model(self) -> None:
        """Unload the TimesFM model and free resources."""
        try:
            if self.tfm is not None:
                # TimesFM doesn't have explicit unload, so we just delete the reference
                del self.tfm
                self.tfm = None
                
                # Clear GPU cache if using GPU
                if self.device != "cpu":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                
                logger.info("TimesFM model unloaded successfully")
            
            self.status = ModelStatus.UNLOADED
            
        except Exception as e:
            logger.error(f"Error unloading TimesFM model: {e}")
            self.status = ModelStatus.ERROR
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the TimesFM model."""
        health_status = super().health_check()
        
        # Add TimesFM-specific health information
        health_status.update({
            'model_repo': self.model_repo,
            'context_len': self.context_len,
            'horizon_len': self.horizon_len,
            'dependency_error': self._dependency_error
        })
        
        # Check dependencies
        if not self._check_dependencies():
            health_status['dependency_status'] = 'failed'
            health_status['dependency_error'] = self._dependency_error
        else:
            health_status['dependency_status'] = 'ok'
        
        return health_status