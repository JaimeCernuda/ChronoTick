"""
TSFM Model Wrapper for ChronoTick Integration

Bridges ChronoTickInferenceEngine models to work with RealDataPipeline/PredictiveScheduler.
"""

import logging
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PredictionWithUncertainty:
    """
    Prediction format expected by PredictiveScheduler.

    Attributes:
        offset: Clock offset correction (seconds)
        drift: Drift rate (seconds per second)
        offset_uncertainty: Offset uncertainty bounds (seconds)
        drift_uncertainty: Drift uncertainty bounds (seconds/second)
        confidence: Prediction confidence [0,1]
        timestamp: Prediction timestamp
        quantiles: Optional prediction quantiles (e.g., {'0.1': val, '0.5': val, '0.9': val})
    """
    offset: float
    drift: float
    offset_uncertainty: float
    drift_uncertainty: float = 0.0
    confidence: float = 0.95
    timestamp: float = 0.0
    quantiles: Optional[Dict[str, float]] = None


class TSFMModelWrapper:
    """
    Wrapper to adapt ChronoTickInferenceEngine to PredictiveScheduler interface.

    PredictiveScheduler expects models with:
        predict_with_uncertainty(horizon: int) -> List[PredictionWithUncertainty]

    ChronoTickInferenceEngine provides:
        predict_short_term(offset_history, covariates) -> PredictionResult
        predict_long_term(offset_history, covariates) -> PredictionResult

    This wrapper bridges the two interfaces.
    """

    def __init__(self,
                 inference_engine,
                 model_type: str,  # 'short_term' or 'long_term'
                 dataset_manager,
                 system_metrics_collector=None):
        """
        Initialize TSFM model wrapper.

        Args:
            inference_engine: ChronoTickInferenceEngine instance
            model_type: 'short_term' or 'long_term'
            dataset_manager: DatasetManager for historical data
            system_metrics_collector: Optional SystemMetricsCollector for covariates
        """
        self.engine = inference_engine
        self.model_type = model_type
        self.dataset_manager = dataset_manager
        self.system_metrics = system_metrics_collector

        # Determine which prediction method to use
        if model_type == 'short_term':
            self.predict_method = self.engine.predict_short_term
        elif model_type == 'long_term':
            self.predict_method = self.engine.predict_long_term
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'short_term' or 'long_term'")

        logger.info(f"Initialized TSFMModelWrapper for {model_type} model")

    def predict_with_uncertainty(self, horizon: int) -> List[PredictionWithUncertainty]:
        """
        Generate predictions with uncertainty for PredictiveScheduler.

        This is the main interface method that PredictiveScheduler calls.

        Args:
            horizon: Number of future time steps to predict

        Returns:
            List of predictions, one per time step
        """
        logger.debug(f"TSFMModelWrapper.predict_with_uncertainty ENTRY: horizon={horizon}, model_type={self.model_type}")

        try:
            # Get historical offset data from dataset manager
            logger.debug(f"Getting offset history from dataset manager...")
            offset_history = self._get_offset_history()
            logger.debug(f"offset_history: {len(offset_history) if offset_history is not None else 0} data points")

            if offset_history is None or len(offset_history) < 10:
                error_msg = f"CRITICAL: Insufficient historical data for {self.model_type} prediction (got {len(offset_history) if offset_history is not None else 0} points, need at least 10)"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Get system metrics as covariates (if available)
            logger.debug(f"Getting covariates...")
            covariates = self._get_covariates(horizon)
            logger.debug(f"Covariates: {list(covariates.keys()) if covariates else None}")

            # Make prediction using ChronoTickInferenceEngine
            logger.debug(f"Calling predict_method with offset_history length={len(offset_history)}")
            prediction_result = self.predict_method(offset_history, covariates)
            logger.debug(f"predict_method returned: {prediction_result is not None}")

            if prediction_result is None:
                # Gracefully handle None returns (insufficient data for long-term model)
                # Return empty list so scheduler can continue with other models
                logger.warning(f"{self.model_type} model returned None (insufficient data) - returning empty predictions")
                return []

            # Convert PredictionResult to List[PredictionWithUncertainty]
            logger.debug(f"Converting prediction result to list...")
            predictions = self._convert_prediction_result(prediction_result, horizon)

            logger.debug(f"TSFMModelWrapper.predict_with_uncertainty EXIT: Generated {len(predictions)} predictions for {self.model_type}")
            return predictions

        except Exception as e:
            # Log error but return empty list to avoid blocking scheduler
            logger.error(f"ERROR in {self.model_type} predict_with_uncertainty: {e}")
            logger.warning(f"{self.model_type} model failed - returning empty predictions to keep system running")
            return []

    def _get_offset_history(self) -> Optional[np.ndarray]:
        """
        Get historical offset data from dataset manager.

        Returns:
            Array of historical offsets, or None if insufficient data
        """
        logger.debug(f"_get_offset_history called")
        try:
            # Get ALL measurements (NTP + predictions) - no time window
            # This enables autoregressive training on recent predictions
            recent_measurements = self.dataset_manager.get_recent_measurements(window_seconds=None)

            if not recent_measurements:
                logger.warning("No recent measurements available from dataset manager")
                return None

            # Extract offsets (measurements are tuples of (timestamp, offset))
            offsets = np.array([offset for timestamp, offset in recent_measurements])

            logger.debug(f"Retrieved {len(offsets)} historical offsets")
            return offsets

        except Exception as e:
            logger.error(f"Failed to get offset history: {e}")
            return None

    def _get_covariates(self, horizon: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Get system metrics as covariates for prediction.

        Args:
            horizon: Prediction horizon (needed for future covariate values)

        Returns:
            Dictionary of covariate arrays, or None if not available
        """
        if self.system_metrics is None:
            return None

        try:
            # Get recent system metrics (last 500 seconds)
            # Returns Dict[str, np.ndarray] already in the right format
            recent_metrics = self.system_metrics.get_recent_metrics(window_seconds=500)

            if not recent_metrics:
                return None

            # Extend each metric with horizon future values (use last value)
            covariates = {}

            for metric_name, metric_array in recent_metrics.items():
                if len(metric_array) > 0:
                    # Extend with last value for future predictions
                    future_values = np.full(horizon, metric_array[-1])
                    covariates[metric_name] = np.concatenate([metric_array, future_values])

            if covariates:
                logger.debug(f"Retrieved {len(covariates)} covariates: {list(covariates.keys())}")
                return covariates
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get covariates: {e}")
            return None

    def _convert_prediction_result(self,
                                   prediction_result,
                                   horizon: int) -> List[PredictionWithUncertainty]:
        """
        Convert ChronoTickInferenceEngine PredictionResult to PredictiveScheduler format.

        Args:
            prediction_result: PredictionResult from inference engine
            horizon: Number of predictions needed

        Returns:
            List of PredictionWithUncertainty objects
        """
        predictions = []
        num_predictions = min(len(prediction_result.predictions), horizon)

        for i in range(num_predictions):
            # Get offset prediction
            offset = float(prediction_result.predictions[i])

            # Calculate drift rate (approximate from consecutive predictions)
            if i < num_predictions - 1:
                drift = float(prediction_result.predictions[i + 1] - prediction_result.predictions[i])
            else:
                # For last prediction, use previous drift
                if i > 0:
                    drift = float(prediction_result.predictions[i] - prediction_result.predictions[i - 1])
                else:
                    drift = 0.0

            # Get uncertainty
            if prediction_result.uncertainty is not None and i < len(prediction_result.uncertainty):
                offset_uncertainty = float(prediction_result.uncertainty[i])
            else:
                # Default uncertainty if not available
                offset_uncertainty = 0.001  # 1ms default

            # Drift uncertainty (approximate as 10% of offset uncertainty)
            drift_uncertainty = offset_uncertainty * 0.1

            # Extract quantiles for this timestep (if available)
            quantiles_dict = None
            if prediction_result.quantiles is not None:
                quantiles_dict = {}
                for q_level, q_array in prediction_result.quantiles.items():
                    if i < len(q_array):
                        quantiles_dict[q_level] = float(q_array[i])

            predictions.append(PredictionWithUncertainty(
                offset=offset,
                drift=drift,
                offset_uncertainty=offset_uncertainty,
                drift_uncertainty=drift_uncertainty,
                confidence=prediction_result.confidence,  # Use confidence from engine (based on uncertainty)
                timestamp=prediction_result.timestamp + i,
                quantiles=quantiles_dict
            ))

        # If we need more predictions than were generated, extrapolate
        while len(predictions) < horizon:
            last_pred = predictions[-1]
            predictions.append(PredictionWithUncertainty(
                offset=last_pred.offset + last_pred.drift,
                drift=last_pred.drift,
                offset_uncertainty=last_pred.offset_uncertainty * 1.1,  # Increase uncertainty
                drift_uncertainty=last_pred.drift_uncertainty * 1.1,
                confidence=last_pred.confidence * 0.95,  # Decrease confidence
                timestamp=last_pred.timestamp + 1,
                quantiles=None  # No quantiles for extrapolated predictions
            ))

        return predictions

    def _generate_fallback_predictions(self, horizon: int) -> List[PredictionWithUncertainty]:
        """
        Generate simple fallback predictions when model fails.

        Args:
            horizon: Number of predictions needed

        Returns:
            List of fallback predictions (zero offset, zero drift)
        """
        logger.warning(f"Using fallback predictions for {self.model_type}")

        return [
            PredictionWithUncertainty(
                offset=0.0,
                drift=0.0,
                offset_uncertainty=0.01,  # 10ms uncertainty
                drift_uncertainty=0.001,  # 1ms/s drift uncertainty
                confidence=0.5,  # Low confidence for fallback
                timestamp=0.0
            )
            for _ in range(horizon)
        ]

    def get_model_info(self) -> Dict:
        """Get information about the wrapped model."""
        return {
            'model_type': self.model_type,
            'engine_available': self.engine is not None,
            'covariates_enabled': self.system_metrics is not None,
            'dataset_manager_available': self.dataset_manager is not None
        }


# Factory function for easy creation
def create_model_wrappers(inference_engine, dataset_manager, system_metrics=None):
    """
    Create both short-term and long-term model wrappers.

    Args:
        inference_engine: ChronoTickInferenceEngine instance
        dataset_manager: DatasetManager instance
        system_metrics: Optional SystemMetricsCollector instance

    Returns:
        Tuple of (short_term_wrapper, long_term_wrapper)
    """
    short_term = TSFMModelWrapper(
        inference_engine=inference_engine,
        model_type='short_term',
        dataset_manager=dataset_manager,
        system_metrics_collector=system_metrics
    )

    long_term = TSFMModelWrapper(
        inference_engine=inference_engine,
        model_type='long_term',
        dataset_manager=dataset_manager,
        system_metrics_collector=system_metrics
    )

    logger.info("Created both short-term and long-term TSFM model wrappers")
    return short_term, long_term
