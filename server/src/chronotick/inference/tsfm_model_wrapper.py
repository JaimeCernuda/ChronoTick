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
                 system_metrics_collector=None,
                 enable_multivariate=False):  # Phase 3C: Multivariate predictions (disabled by default for now)
        """
        Initialize TSFM model wrapper.

        Args:
            inference_engine: ChronoTickInferenceEngine instance
            model_type: 'short_term' or 'long_term'
            dataset_manager: DatasetManager for historical data
            system_metrics_collector: Optional SystemMetricsCollector for covariates
            enable_multivariate: If True, predict both offset and drift (Phase 3C)
        """
        self.engine = inference_engine
        self.model_type = model_type
        self.dataset_manager = dataset_manager
        self.system_metrics = system_metrics_collector
        self.enable_multivariate = enable_multivariate  # Phase 3C

        # Track normalization bias for denormalization
        self.normalization_bias = None
        # Phase 3C: Track current drift estimate
        self.current_drift = 0.0

        # Determine which prediction method to use
        if model_type == 'short_term':
            self.predict_method = self.engine.predict_short_term
        elif model_type == 'long_term':
            self.predict_method = self.engine.predict_long_term
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'short_term' or 'long_term'")

        logger.info(f"Initialized TSFMModelWrapper for {model_type} model (multivariate={enable_multivariate})")

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
        Get historical offset data from dataset manager with normalization.

        Phase 3C: Now optionally retrieves drift data for multivariate predictions.

        Returns:
            Array of historical data (offsets only or offsets+drifts), or None if insufficient data
        """
        logger.debug(f"_get_offset_history called (multivariate={self.enable_multivariate})")
        try:
            if self.enable_multivariate:
                # Phase 3C: Get measurements with drift for multivariate input
                recent_measurements, normalization_bias, current_drift = \
                    self.dataset_manager.get_recent_measurements_with_drift(
                        window_seconds=None,
                        normalize=True
                    )

                if not recent_measurements:
                    logger.warning("No recent measurements available from dataset manager")
                    return None

                # Store normalization bias and current drift
                self.normalization_bias = normalization_bias
                self.current_drift = current_drift

                if normalization_bias is not None:
                    logger.info(f"[NORMALIZATION] Multivariate model training on residuals "
                               f"(baseline: {normalization_bias*1000:.3f}ms, drift: {current_drift:.3f}μs/s)")

                # Extract offsets and drifts (measurements are tuples of (timestamp, offset, drift))
                offsets = np.array([offset for timestamp, offset, drift in recent_measurements])
                drifts_us_per_s = np.array([drift for timestamp, offset, drift in recent_measurements])

                # Convert drift from μs/s to s/s for consistency with offset units
                drifts_s_per_s = drifts_us_per_s * 1e-6

                # Stack as multivariate input: shape (2, sequence_length)
                # Row 0: offsets, Row 1: drift rates
                multivariate_history = np.stack([offsets, drifts_s_per_s], axis=0)

                logger.debug(f"Retrieved {len(offsets)} measurements with drift for multivariate input")
                logger.debug(f"Multivariate shape: {multivariate_history.shape} (2 variables × {len(offsets)} timesteps)")

                return multivariate_history

            else:
                # Original univariate approach
                recent_measurements, normalization_bias = self.dataset_manager.get_recent_measurements(
                    window_seconds=None,
                    normalize=True
                )

                if not recent_measurements:
                    logger.warning("No recent measurements available from dataset manager")
                    return None

                # Store normalization bias for denormalization later
                self.normalization_bias = normalization_bias
                if normalization_bias is not None:
                    logger.info(f"[NORMALIZATION] Model will train on residuals (baseline: {normalization_bias*1000:.3f}ms)")

                # Extract offsets (measurements are tuples of (timestamp, offset))
                offsets = np.array([offset for timestamp, offset in recent_measurements])

                logger.debug(f"Retrieved {len(offsets)} historical offsets (normalized)")
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

        Applies denormalization to convert normalized residuals back to absolute offsets.

        Args:
            prediction_result: PredictionResult from inference engine
            horizon: Number of predictions needed

        Returns:
            List of PredictionWithUncertainty objects
        """
        predictions = []
        num_predictions = min(len(prediction_result.predictions), horizon)

        for i in range(num_predictions):
            # Get offset prediction (normalized residual)
            offset_residual = float(prediction_result.predictions[i])

            # DENORMALIZATION: Add back the NTP baseline bias
            if self.normalization_bias is not None:
                offset = offset_residual + self.normalization_bias
                if i == 0:  # Log once per prediction batch
                    logger.info(f"[DENORMALIZATION] Converting residuals to absolute offsets (bias: {self.normalization_bias*1000:.3f}ms)")
                    logger.debug(f"  Example: residual={offset_residual*1000:.3f}ms → offset={offset*1000:.3f}ms")
            else:
                offset = offset_residual

            # Phase 3D: Use calculated drift rate from NTP measurements
            # Convert from μs/s to s/s for consistency
            drift = self.current_drift * 1e-6  # Convert μs/s → s/s

            # Fallback: approximate from consecutive predictions if no drift available
            if drift == 0.0 and i < num_predictions - 1:
                drift = float(prediction_result.predictions[i + 1] - prediction_result.predictions[i])
            elif drift == 0.0 and i > 0:
                drift = float(prediction_result.predictions[i] - prediction_result.predictions[i - 1])

            # Get uncertainty
            if prediction_result.uncertainty is not None and i < len(prediction_result.uncertainty):
                offset_uncertainty = float(prediction_result.uncertainty[i])
            else:
                # Default uncertainty if not available
                offset_uncertainty = 0.001  # 1ms default

            # Drift uncertainty (approximate as 10% of offset uncertainty)
            drift_uncertainty = offset_uncertainty * 0.1

            # Extract quantiles for this timestep (if available)
            # DENORMALIZATION: Quantiles also need bias adjustment
            quantiles_dict = None
            if prediction_result.quantiles is not None:
                quantiles_dict = {}
                for q_level, q_array in prediction_result.quantiles.items():
                    if i < len(q_array):
                        quantile_residual = float(q_array[i])
                        # Denormalize quantiles too
                        if self.normalization_bias is not None:
                            quantiles_dict[q_level] = quantile_residual + self.normalization_bias
                        else:
                            quantiles_dict[q_level] = quantile_residual

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
