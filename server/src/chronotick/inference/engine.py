"""
ChronoTick Inference Layer

This module provides the forecasting engine for the ChronoTick system, implementing
both short-term and long-term models with optional fusion capabilities using the
TSFM Factory library.
"""

import logging
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import yaml
from pathlib import Path
import psutil
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# TSFM imports - adjust path as needed
import sys
sys.path.append('../tsfm')
from chronotick.tsfm import TSFMFactory, MultivariateInput, CovariatesInput, FrequencyInfo
from chronotick.tsfm.datasets.preprocessing import remove_outliers, fill_missing_values

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model types supported by ChronoTick."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class PredictionResult:
    """Container for prediction results with batch support for offset and drift (Experiment-14)."""
    predictions: np.ndarray  # Offset predictions
    uncertainty: Optional[np.ndarray] = None  # Offset uncertainty
    quantiles: Optional[Dict[str, np.ndarray]] = None  # Offset quantiles

    # EXPERIMENT-14: Drift prediction support via TimesFM batch forecasting
    drift_predictions: Optional[np.ndarray] = None  # Drift predictions (μs/s)
    drift_uncertainty: Optional[np.ndarray] = None  # Drift uncertainty
    drift_quantiles: Optional[Dict[str, np.ndarray]] = None  # Drift quantiles

    confidence: float = 1.0
    model_type: ModelType = ModelType.SHORT_TERM
    timestamp: float = 0.0
    inference_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FusedPrediction:
    """Container for fused prediction results."""
    prediction: float
    uncertainty: float
    weights: Dict[str, float]
    source_predictions: Dict[str, PredictionResult]
    timestamp: float
    metadata: Dict[str, Any]


class ChronoTickInferenceEngine:
    """
    ChronoTick Inference Engine
    
    Provides short-term and long-term forecasting capabilities for clock drift
    prediction using TSFM foundation models.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the inference engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize TSFM Factory
        self.factory = TSFMFactory()
        
        # Model instances
        self.short_term_model = None
        self.long_term_model = None
        
        # Threading and synchronization
        self.prediction_lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Prediction cache
        cache_size = self.config.get('performance', {}).get('cache_size', 10)
        self.prediction_cache = {
            ModelType.SHORT_TERM: queue.Queue(maxsize=cache_size),
            ModelType.LONG_TERM: queue.Queue(maxsize=cache_size)
        }
        
        # Performance tracking
        self.performance_stats = {
            'short_term_inferences': 0,
            'long_term_inferences': 0,
            'fusion_operations': 0,
            'total_inference_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Frequency info for time series models
        self.frequency_info = self._create_frequency_info()
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("ChronoTick Inference Engine initialized")
        logger.info(f"Configuration: {self.config_path}")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging_config = self.config.get('logging', {})
        level_name = logging_config.get('level', 'INFO')
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _create_frequency_info(self) -> FrequencyInfo:
        """Create frequency information for time series models."""
        clock_config = self.config.get('clock', {})
        freq_type = clock_config.get('frequency_type', 'second')
        freq_code = clock_config.get('frequency_code', 9)  # Default to 9 (second)

        return self.factory.create_frequency_info(
            freq_str=freq_type[0].upper(),  # 'S' for second
            freq_value=freq_code,
            is_regular=True,
            detected_freq=freq_type[0].upper()
        )
    
    def initialize_models(self) -> bool:
        """
        Initialize the forecasting models.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize short-term model
            if self.config['short_term']['enabled']:
                short_config = self.config['short_term']
                logger.info(f"Loading short-term model: {short_config['model_name']}")
                
                self.short_term_model = self.factory.load_model(
                    short_config['model_name'],
                    device=short_config['device'],
                    **short_config.get('model_params', {})
                )
                logger.info("Short-term model loaded successfully")
            
            # Initialize long-term model
            if self.config['long_term']['enabled']:
                long_config = self.config['long_term']
                logger.info(f"Loading long-term model: {long_config['model_name']}")
                
                self.long_term_model = self.factory.load_model(
                    long_config['model_name'],
                    device=long_config['device'],
                    **long_config.get('model_params', {})
                )
                logger.info("Long-term model loaded successfully")
            
            # Verify at least one model is loaded
            if not self.short_term_model and not self.long_term_model:
                raise RuntimeError("No models enabled in configuration")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False
    
    def predict_short_term(self,
                          offset_history: np.ndarray,
                          drift_history: Optional[np.ndarray] = None,  # EXPERIMENT-14
                          covariates: Optional[Dict[str, np.ndarray]] = None) -> Optional[PredictionResult]:
        """
        Generate short-term predictions for clock offset and drift (Experiment-14).

        EXPERIMENT-14: Now supports batch forecasting when drift_history is provided.
        TimesFM will predict BOTH offset and drift in parallel.

        Args:
            offset_history: Historical offset values
            drift_history: Historical drift values (μs/s) for batch forecasting (Experiment-14)
            covariates: Optional exogenous variables (system metrics)

        Returns:
            PredictionResult with offset and drift predictions, or None if prediction fails
        """
        if not self.short_term_model or not self.config['short_term']['enabled']:
            return None
        
        start_time = time.time()
        
        try:
            short_config = self.config['short_term']
            
            # Preprocess data
            processed_history = self._preprocess_data(offset_history)
            
            # Limit context length
            context_length = short_config.get('context_length', 100)
            if len(processed_history) > context_length:
                processed_history = processed_history[-context_length:]

            # Check minimum data requirements
            if len(processed_history) < 10:  # Minimum required points
                logger.warning("Insufficient data for short-term prediction")
                return None

            # Generate prediction
            horizon = short_config.get('prediction_horizon', 5)
            
            # EXPERIMENT-14: Check if drift_history provided for batch forecasting
            if drift_history is not None:
                logger.info(f"[BATCH_FORECAST] Experiment-14: Batch forecasting enabled")
                logger.info(f"[BATCH_FORECAST]   Offset history: {len(processed_history)} points")
                logger.info(f"[BATCH_FORECAST]   Drift history: {len(drift_history)} points")

                # Preprocess drift history
                processed_drift = self._preprocess_data(drift_history)
                if len(processed_drift) > context_length:
                    processed_drift = processed_drift[-context_length:]

                # EXPERIMENT-14: TimesFM batch forecasting - pass list as first positional argument "context"
                logger.info(f"[BATCH_FORECAST]   Calling TimesFM with batch inputs (list of 2 series: offset + drift)")

                result = self.short_term_model.forecast(
                    [processed_history, processed_drift],  # LIST for batch forecasting (context parameter)
                    horizon,
                    freq=self.frequency_info.freq_value
                )
                logger.info(f"[BATCH_FORECAST]   Batch forecast complete")

                # EXPERIMENT-14: Unpack batch results
                # Result shape: predictions (2, horizon), quantiles (2, horizon, 10)
                # Index 0 = offset, Index 1 = drift
                if isinstance(result.predictions, np.ndarray) and result.predictions.shape[0] == 2:
                    offset_predictions = result.predictions[0, :]
                    drift_predictions = result.predictions[1, :]
                    logger.info(f"[BATCH_FORECAST]   Unpacked offset predictions: {len(offset_predictions)}")
                    logger.info(f"[BATCH_FORECAST]   Unpacked drift predictions: {len(drift_predictions)}")

                    # Unpack quantiles if available
                    offset_quantiles = None
                    drift_quantiles = None
                    if result.quantiles is not None:
                        # Quantiles shape: (2, horizon, 10)
                        offset_quantiles = result.quantiles[0, :, :]  # (horizon, 10)
                        drift_quantiles = result.quantiles[1, :, :]   # (horizon, 10)
                        logger.info(f"[BATCH_FORECAST]   Unpacked quantiles for offset and drift")

                    # Create modified result with unpacked values
                    class BatchResult:
                        def __init__(self, offset_pred, drift_pred, offset_quant, drift_quant, metadata):
                            self.predictions = offset_pred
                            self.drift_predictions = drift_pred
                            self.quantiles = offset_quant
                            self.drift_quantiles = drift_quant
                            self.metadata = metadata

                    result = BatchResult(offset_predictions, drift_predictions,
                                       offset_quantiles, drift_quantiles, result.metadata)
                else:
                    logger.warning(f"[BATCH_FORECAST] Unexpected result shape: {result.predictions.shape}")
            elif covariates and self.config.get('covariates', {}).get('enabled', False):
                # Use covariates if available and enabled
                covariates_input = self._prepare_covariates_input(
                    processed_history, covariates, horizon
                )
                # Get use_covariates setting from config
                use_covariates = short_config.get('use_covariates', False)
                result = self.short_term_model.forecast_with_covariates(
                    covariates_input,
                    horizon,
                    frequency=self.frequency_info,
                    use_covariates=use_covariates  # Pass through to model
                )
            else:
                # Standard univariate prediction
                # TimesFM expects 2D input: shape (batch_size, sequence_length)
                # Reshape 1D history (length,) to 2D (1, length)
                if processed_history.ndim == 1:
                    processed_history = processed_history.reshape(1, -1)

                result = self.short_term_model.forecast(
                    processed_history,
                    horizon,
                    freq=self.frequency_info.freq_value
                )
            
            # Calculate uncertainty (if quantiles available)
            uncertainty = self._calculate_uncertainty(result.quantiles)

            # EXPERIMENT-14: Calculate drift uncertainty if drift quantiles available
            drift_uncertainty = None
            if hasattr(result, 'drift_quantiles') and result.drift_quantiles is not None:
                drift_uncertainty = self._calculate_uncertainty(result.drift_quantiles)
                logger.info(f"[BATCH_FORECAST]   Drift uncertainty calculated: {drift_uncertainty[0] if len(drift_uncertainty) > 0 else 'N/A'}")

            confidence = self._calculate_confidence(uncertainty, short_config.get('max_uncertainty', 0.1))

            # Create prediction result
            prediction = PredictionResult(
                predictions=result.predictions,
                uncertainty=uncertainty,
                quantiles=result.quantiles,
                drift_predictions=result.drift_predictions if hasattr(result, 'drift_predictions') else None,
                drift_uncertainty=drift_uncertainty,
                drift_quantiles=result.drift_quantiles if hasattr(result, 'drift_quantiles') else None,
                confidence=confidence,
                model_type=ModelType.SHORT_TERM,
                timestamp=time.time(),
                inference_time=time.time() - start_time,
                metadata=result.metadata
            )
            
            # Update performance stats
            self.performance_stats['short_term_inferences'] += 1
            self.performance_stats['total_inference_time'] += prediction.inference_time
            
            # Log prediction if enabled
            if self.config.get('logging', {}).get('log_predictions', False):
                logger.debug(f"Short-term prediction: {prediction.predictions[0]:.6f}s, "
                           f"confidence: {confidence:.3f}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Short-term prediction failed: {e}")
            return None
    
    def predict_long_term(self,
                         offset_history: np.ndarray,
                         drift_history: Optional[np.ndarray] = None,  # EXPERIMENT-14: Batch forecasting
                         covariates: Optional[Dict[str, np.ndarray]] = None) -> Optional[PredictionResult]:
        """
        Generate long-term predictions for clock offset and drift (Experiment-14).

        EXPERIMENT-14: Now supports batch forecasting when drift_history is provided.
        TimesFM will predict BOTH offset and drift in parallel.

        Args:
            offset_history: Historical offset values
            drift_history: Historical drift values (μs/s) for batch forecasting (Experiment-14)
            covariates: Optional exogenous variables

        Returns:
            PredictionResult with offset and drift predictions, or None if prediction fails
        """
        if not self.long_term_model or not self.config['long_term']['enabled']:
            return None
        
        start_time = time.time()
        
        try:
            long_config = self.config['long_term']
            
            # Preprocess data
            processed_history = self._preprocess_data(offset_history)
            
            # Use longer context for long-term model
            context_length = long_config.get('context_length', 300)
            if len(processed_history) > context_length:
                processed_history = processed_history[-context_length:]

            # Check minimum data requirements
            if len(processed_history) < 60:  # Need more data for long-term
                logger.warning("Insufficient data for long-term prediction")
                return None

            # Generate prediction
            horizon = long_config.get('prediction_horizon', 60)

            # EXPERIMENT-14: Check if drift_history provided for batch forecasting
            if drift_history is not None:
                logger.info(f"[BATCH_FORECAST_LT] Experiment-14: Long-term batch forecasting enabled")
                logger.info(f"[BATCH_FORECAST_LT]   Offset history: {len(processed_history)} points")
                logger.info(f"[BATCH_FORECAST_LT]   Drift history: {len(drift_history)} points")

                # Preprocess drift history
                processed_drift = self._preprocess_data(drift_history)
                if len(processed_drift) > context_length:
                    processed_drift = processed_drift[-context_length:]

                # EXPERIMENT-14: TimesFM batch forecasting - pass list as first positional argument "context"
                logger.info(f"[BATCH_FORECAST_LT]   Calling TimesFM with batch inputs (list of 2 series: offset + drift)")

                result = self.long_term_model.forecast(
                    [processed_history, processed_drift],  # LIST for batch forecasting (context parameter)
                    horizon,
                    freq=self.frequency_info.freq_value
                )
                logger.info(f"[BATCH_FORECAST_LT]   Batch forecast complete")

                # EXPERIMENT-14: Unpack batch results
                # Result shape: predictions (2, horizon), quantiles (2, horizon, 10)
                # Index 0 = offset, Index 1 = drift
                if isinstance(result.predictions, np.ndarray) and result.predictions.shape[0] == 2:
                    offset_predictions = result.predictions[0, :]
                    drift_predictions = result.predictions[1, :]
                    logger.info(f"[BATCH_FORECAST_LT]   Unpacked offset predictions: {len(offset_predictions)}")
                    logger.info(f"[BATCH_FORECAST_LT]   Unpacked drift predictions: {len(drift_predictions)}")

                    # Unpack quantiles if available
                    offset_quantiles = None
                    drift_quantiles = None
                    if result.quantiles is not None:
                        # Quantiles shape: (2, horizon, 10)
                        offset_quantiles = result.quantiles[0, :, :]  # (horizon, 10)
                        drift_quantiles = result.quantiles[1, :, :]   # (horizon, 10)
                        logger.info(f"[BATCH_FORECAST_LT]   Unpacked quantiles for offset and drift")

                    # Create modified result with unpacked values
                    class BatchResult:
                        def __init__(self, offset_pred, drift_pred, offset_quant, drift_quant, metadata):
                            self.predictions = offset_pred
                            self.drift_predictions = drift_pred
                            self.quantiles = offset_quant
                            self.drift_quantiles = drift_quant
                            self.metadata = metadata

                    result = BatchResult(offset_predictions, drift_predictions,
                                       offset_quantiles, drift_quantiles, result.metadata)
                else:
                    logger.warning(f"[BATCH_FORECAST_LT] Unexpected result shape: {result.predictions.shape}")
            elif covariates and self.config.get('covariates', {}).get('enabled', False):
                # Use covariates if available and enabled
                covariates_input = self._prepare_covariates_input(
                    processed_history, covariates, horizon
                )
                # Get use_covariates setting from config
                use_covariates = long_config.get('use_covariates', False)
                result = self.long_term_model.forecast_with_covariates(
                    covariates_input,
                    horizon,
                    frequency=self.frequency_info,
                    use_covariates=use_covariates  # Pass through to model
                )
            else:
                # Standard univariate prediction
                # TimesFM expects 2D input: shape (batch_size, sequence_length)
                # Reshape 1D history (length,) to 2D (1, length)
                if processed_history.ndim == 1:
                    processed_history = processed_history.reshape(1, -1)

                result = self.long_term_model.forecast(
                    processed_history,
                    horizon,
                    freq=self.frequency_info.freq_value
                )
            
            # Calculate uncertainty from quantiles
            uncertainty = self._calculate_uncertainty(result.quantiles)

            # EXPERIMENT-14: Calculate drift uncertainty if drift quantiles available
            drift_uncertainty = None
            if hasattr(result, 'drift_quantiles') and result.drift_quantiles is not None:
                drift_uncertainty = self._calculate_uncertainty(result.drift_quantiles)
                logger.info(f"[BATCH_FORECAST_LT]   Drift uncertainty calculated: {drift_uncertainty[0] if len(drift_uncertainty) > 0 else 'N/A'}")

            confidence = 1.0  # Long-term model assumed to be high confidence

            # Create prediction result
            prediction = PredictionResult(
                predictions=result.predictions,
                uncertainty=uncertainty,
                quantiles=result.quantiles,
                drift_predictions=result.drift_predictions if hasattr(result, 'drift_predictions') else None,
                drift_uncertainty=drift_uncertainty,
                drift_quantiles=result.drift_quantiles if hasattr(result, 'drift_quantiles') else None,
                confidence=confidence,
                model_type=ModelType.LONG_TERM,
                timestamp=time.time(),
                inference_time=time.time() - start_time,
                metadata=result.metadata
            )
            
            # Update performance stats
            self.performance_stats['long_term_inferences'] += 1
            self.performance_stats['total_inference_time'] += prediction.inference_time
            
            # Log prediction if enabled
            if self.config.get('logging', {}).get('log_predictions', False):
                logger.debug(f"Long-term prediction: {prediction.predictions[0]:.6f}s, "
                           f"horizon: {horizon}s")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Long-term prediction failed: {e}")
            return None
    
    def predict_fused(self,
                     offset_history: np.ndarray,
                     covariates: Optional[Dict[str, np.ndarray]] = None,
                     target_timestamp: Optional[float] = None) -> Optional[FusedPrediction]:
        """
        Generate fused predictions combining short-term and long-term models.
        
        Args:
            offset_history: Historical offset values
            covariates: Optional exogenous variables
            target_timestamp: Specific timestamp to predict (uses first prediction if None)
            
        Returns:
            FusedPrediction or None if fusion fails
        """
        if not self.config['fusion']['enabled']:
            return None
        
        start_time = time.time()
        
        try:
            # Get predictions from both models
            short_pred = self.predict_short_term(offset_history, covariates)
            long_pred = self.predict_long_term(offset_history, covariates)
            
            # Need at least one prediction
            if not short_pred and not long_pred:
                logger.warning("No predictions available for fusion")
                return None
            
            # Use fusion if both available, otherwise return single prediction
            if short_pred and long_pred:
                fused_result = self._fuse_predictions(short_pred, long_pred, target_timestamp)
            elif short_pred:
                # Only short-term available
                fused_result = FusedPrediction(
                    prediction=short_pred.predictions[0],
                    uncertainty=short_pred.uncertainty[0] if short_pred.uncertainty is not None else 0.0,
                    weights={'short_term': 1.0, 'long_term': 0.0},
                    source_predictions={'short_term': short_pred},
                    timestamp=time.time(),
                    metadata={'fusion_method': 'short_term_only'}
                )
            else:
                # Only long-term available
                fused_result = FusedPrediction(
                    prediction=long_pred.predictions[0],
                    uncertainty=long_pred.uncertainty[0] if long_pred.uncertainty is not None else 0.0,
                    weights={'short_term': 0.0, 'long_term': 1.0},
                    source_predictions={'long_term': long_pred},
                    timestamp=time.time(),
                    metadata={'fusion_method': 'long_term_only'}
                )
            
            # Update performance stats
            self.performance_stats['fusion_operations'] += 1
            
            # Log fusion if enabled
            if self.config.get('logging', {}).get('log_fusion_weights', False):
                logger.debug(f"Fusion weights: {fused_result.weights}")
            
            return fused_result
            
        except Exception as e:
            logger.error(f"Prediction fusion failed: {e}")
            return None
    
    def _fuse_predictions(self,
                         short_pred: PredictionResult,
                         long_pred: PredictionResult,
                         target_timestamp: Optional[float] = None) -> FusedPrediction:
        """
        Fuse short-term and long-term predictions using inverse variance weighting.
        
        Args:
            short_pred: Short-term prediction result
            long_pred: Long-term prediction result
            target_timestamp: Target timestamp for prediction
            
        Returns:
            FusedPrediction containing combined result
        """
        fusion_config = self.config['fusion']
        
        # Use first prediction if no specific timestamp requested
        short_value = short_pred.predictions[0]
        long_value = long_pred.predictions[0]
        
        # Calculate uncertainties (standard deviations)
        short_uncertainty = self._get_prediction_uncertainty(short_pred)
        long_uncertainty = self._get_prediction_uncertainty(long_pred)
        
        # Check if short-term model is too uncertain
        uncertainty_threshold = fusion_config['uncertainty_threshold']
        if short_uncertainty > uncertainty_threshold:
            logger.debug(f"Short-term uncertainty ({short_uncertainty:.6f}) exceeds threshold, "
                        f"increasing long-term weight")
        
        # Apply inverse variance weighting
        if fusion_config['method'] == 'inverse_variance' and short_uncertainty > 0 and long_uncertainty > 0:
            # Inverse variance weights
            short_variance = short_uncertainty ** 2
            long_variance = long_uncertainty ** 2
            
            total_precision = (1 / short_variance) + (1 / long_variance)
            short_weight = (1 / short_variance) / total_precision
            long_weight = (1 / long_variance) / total_precision
            
        else:
            # Fallback to configured weights
            short_weight = fusion_config['fallback_weights']['short_term']
            long_weight = fusion_config['fallback_weights']['long_term']
        
        # Apply temporal decay if configured
        decay = fusion_config.get('temporal_decay', 1.0)
        if decay < 1.0:
            age_factor = min(1.0, (time.time() - short_pred.timestamp) / 60.0)  # Age in minutes
            short_weight *= (decay ** age_factor)
            long_weight = 1.0 - short_weight  # Renormalize
        
        # Ensure weights sum to 1
        total_weight = short_weight + long_weight
        if total_weight > 0:
            short_weight /= total_weight
            long_weight /= total_weight
        
        # Compute fused prediction
        fused_value = short_weight * short_value + long_weight * long_value
        
        # Compute fused uncertainty
        fused_uncertainty = np.sqrt(
            (short_weight ** 2) * (short_uncertainty ** 2) +
            (long_weight ** 2) * (long_uncertainty ** 2)
        )
        
        return FusedPrediction(
            prediction=fused_value,
            uncertainty=fused_uncertainty,
            weights={
                'short_term': short_weight,
                'long_term': long_weight
            },
            source_predictions={
                'short_term': short_pred,
                'long_term': long_pred
            },
            timestamp=time.time(),
            metadata={
                'fusion_method': fusion_config['method'],
                'short_uncertainty': short_uncertainty,
                'long_uncertainty': long_uncertainty,
                'uncertainty_threshold': uncertainty_threshold
            }
        )
    
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data for predictions.

        Args:
            data: Input data to validate

        Returns:
            True if valid

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Input data must be numpy array, got {type(data)}")

        if data.size == 0:
            raise ValueError("Input data cannot be empty")

        if np.any(np.isnan(data)):
            raise ValueError("Input data contains NaN values")

        if np.any(np.isinf(data)):
            raise ValueError("Input data contains infinite values")

        return True

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess time series data.

        Args:
            data: Raw time series data

        Returns:
            Preprocessed data
        """
        processed = data.copy()

        preprocessing_config = self.config.get('preprocessing', {})

        # Remove outliers
        outlier_config = preprocessing_config.get('outlier_removal', {})
        if outlier_config.get('enabled', False):
            processed = remove_outliers(
                processed,
                method=outlier_config.get('method', 'iqr'),
                threshold=outlier_config.get('threshold', 2.0)
            )

        # Fill missing values
        missing_config = preprocessing_config.get('missing_value_handling', {})
        if missing_config.get('enabled', False):
            processed = fill_missing_values(
                processed,
                method=missing_config.get('method', 'interpolate')
            )

        return processed
    
    def _prepare_covariates_input(self,
                                 target: np.ndarray,
                                 covariates: Dict[str, np.ndarray],
                                 horizon: int) -> CovariatesInput:
        """
        Prepare covariates input for models that support exogenous variables.
        
        Args:
            target: Target time series
            covariates: Dictionary of covariate time series
            horizon: Prediction horizon
            
        Returns:
            CovariatesInput object
        """
        # Filter enabled covariates
        enabled_vars = self.config['covariates']['variables']
        filtered_covariates = {
            name: values for name, values in covariates.items()
            if name in enabled_vars
        }
        
        # Prepare future covariates if available
        future_vars = self.config['covariates'].get('future_variables', [])
        future_covariates = {
            name: values for name, values in covariates.items()
            if name in future_vars and len(values) >= len(target) + horizon
        }
        
        return self.factory.create_covariates_input(
            target=target,
            covariates=filtered_covariates,
            future_covariates=future_covariates if future_covariates else None,
            metadata={
                'timestamp': time.time(),
                'horizon': horizon,
                'context_length': len(target)
            }
        )
    
    def _calculate_uncertainty(self, quantiles: Optional[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
        """
        Calculate uncertainty from quantiles.
        
        Args:
            quantiles: Dictionary of quantile predictions
            
        Returns:
            Uncertainty estimates (standard deviations) or None
        """
        if not quantiles or '0.1' not in quantiles or '0.9' not in quantiles:
            return None
        
        # Estimate standard deviation from 10th and 90th percentiles
        # σ ≈ (Q90 - Q10) / 2.56 (for normal distribution)
        q10 = quantiles['0.1']
        q90 = quantiles['0.9']
        uncertainty = (q90 - q10) / 2.56
        
        return uncertainty
    
    def _calculate_confidence(self, uncertainty: Optional[np.ndarray], max_uncertainty: float) -> float:
        """
        Calculate confidence score based on uncertainty.
        
        Args:
            uncertainty: Uncertainty estimates
            max_uncertainty: Maximum acceptable uncertainty
            
        Returns:
            Confidence score between 0 and 1
        """
        if uncertainty is None:
            return 1.0
        
        # Use first prediction's uncertainty
        unc = uncertainty[0] if isinstance(uncertainty, np.ndarray) else uncertainty
        
        # Convert uncertainty to confidence (0 = no confidence, 1 = full confidence)
        confidence = max(0.0, 1.0 - (unc / max_uncertainty))
        return min(1.0, confidence)
    
    def _get_prediction_uncertainty(self, prediction: PredictionResult) -> float:
        """
        Extract uncertainty value from prediction result.
        
        Args:
            prediction: Prediction result
            
        Returns:
            Uncertainty value (standard deviation)
        """
        if prediction.uncertainty is not None:
            return prediction.uncertainty[0] if isinstance(prediction.uncertainty, np.ndarray) else prediction.uncertainty
        elif prediction.quantiles:
            uncertainty = self._calculate_uncertainty(prediction.quantiles)
            return uncertainty[0] if uncertainty is not None else 0.1  # Default uncertainty
        else:
            return 0.1  # Default uncertainty when none available
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the inference engine.
        
        Returns:
            Health status information
        """
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'models': {},
            'performance': self.performance_stats.copy(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        # Check model health
        if self.short_term_model:
            try:
                model_health = self.short_term_model.health_check()
                health['models']['short_term'] = model_health
            except Exception as e:
                health['models']['short_term'] = {'status': 'error', 'error': str(e)}
                health['status'] = 'degraded'
        
        if self.long_term_model:
            try:
                model_health = self.long_term_model.health_check()
                health['models']['long_term'] = model_health
            except Exception as e:
                health['models']['long_term'] = {'status': 'error', 'error': str(e)}
                health['status'] = 'degraded'
        
        return health
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Calculate derived metrics
        total_inferences = stats['short_term_inferences'] + stats['long_term_inferences']
        if total_inferences > 0:
            stats['average_inference_time'] = stats['total_inference_time'] / total_inferences
        else:
            stats['average_inference_time'] = 0.0
        
        # Add memory usage
        stats['current_memory_mb'] = psutil.Process().memory_info().rss / 1024 / 1024
        stats['max_memory_mb'] = self.config.get('performance', {}).get('max_memory_mb', 512)
        
        return stats
    
    def shutdown(self):
        """Shutdown the inference engine and cleanup resources."""
        logger.info("Shutting down ChronoTick Inference Engine")
        
        self.shutdown_event.set()
        
        # Unload models
        try:
            if self.short_term_model:
                model_name = self.config['short_term']['model_name']
                self.factory.unload_model(model_name)
                logger.info(f"Unloaded short-term model: {model_name}")
            
            if self.long_term_model:
                model_name = self.config['long_term']['model_name']
                self.factory.unload_model(model_name)
                logger.info(f"Unloaded long-term model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
        
        logger.info("ChronoTick Inference Engine shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize_models():
            raise RuntimeError("Failed to initialize models")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Convenience functions for common usage patterns
def create_inference_engine(config_path: str = "config.yaml") -> ChronoTickInferenceEngine:
    """
    Create and initialize a ChronoTick inference engine.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized ChronoTickInferenceEngine
    """
    engine = ChronoTickInferenceEngine(config_path)
    if not engine.initialize_models():
        raise RuntimeError("Failed to initialize inference engine")
    return engine


def quick_predict(offset_history: np.ndarray,
                 config_path: str = "config.yaml",
                 use_fusion: bool = True,
                 covariates: Optional[Dict[str, np.ndarray]] = None) -> Union[PredictionResult, FusedPrediction]:
    """
    Quick prediction function for simple use cases.
    
    Args:
        offset_history: Historical offset values
        config_path: Path to configuration file
        use_fusion: Whether to use model fusion
        covariates: Optional exogenous variables
        
    Returns:
        Prediction result
    """
    with create_inference_engine(config_path) as engine:
        if use_fusion:
            return engine.predict_fused(offset_history, covariates)
        else:
            # Try short-term first, fallback to long-term
            result = engine.predict_short_term(offset_history, covariates)
            if result is None:
                result = engine.predict_long_term(offset_history, covariates)
            return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic clock offset data
    t = np.arange(3600)  # 1 hour of data
    true_offset = 0.001 * t + 0.01 * np.sin(2 * np.pi * t / 300)  # Drift + periodic
    noise = np.random.normal(0, 0.0001, len(t))
    offset_data = true_offset + noise
    
    # Example system metrics as covariates
    cpu_usage = 50 + 20 * np.sin(2 * np.pi * t / 600) + np.random.normal(0, 5, len(t))
    temperature = 65 + 10 * np.sin(2 * np.pi * t / 1200) + np.random.normal(0, 2, len(t))
    
    covariates = {
        'cpu_usage': cpu_usage,
        'temperature': temperature
    }
    
    print("Testing ChronoTick Inference Engine...")
    
    try:
        with create_inference_engine() as engine:
            print(f"Engine initialized successfully")
            print(f"Health check: {engine.health_check()['status']}")
            
            # Test short-term prediction
            short_result = engine.predict_short_term(offset_data[-300:], covariates)
            if short_result:
                print(f"Short-term prediction: {short_result.predictions[0]:.6f}s")
                print(f"Confidence: {short_result.confidence:.3f}")
                print(f"Inference time: {short_result.inference_time:.3f}s")
            
            # Test fused prediction
            fused_result = engine.predict_fused(offset_data[-600:], covariates)
            if fused_result:
                print(f"Fused prediction: {fused_result.prediction:.6f}s")
                print(f"Weights: {fused_result.weights}")
                print(f"Uncertainty: {fused_result.uncertainty:.6f}s")
            
            # Performance stats
            stats = engine.get_performance_stats()
            print(f"Performance stats: {stats}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()