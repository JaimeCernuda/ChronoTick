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
        offset_uncertainty: Calibrated offset uncertainty bounds (seconds)
        drift_uncertainty: Drift uncertainty bounds (seconds/second)
        confidence: Prediction confidence [0,1]
        timestamp: Prediction timestamp
        quantiles: Optional prediction quantiles (e.g., {'0.1': val, '0.5': val, '0.9': val})
        raw_offset_uncertainty: Raw uncertainty from TimesFM before calibration (seconds)
        calibration_multiplier: Platform-specific calibration multiplier applied to uncertainties
    """
    offset: float
    drift: float
    offset_uncertainty: float
    drift_uncertainty: float = 0.0
    confidence: float = 0.95
    timestamp: float = 0.0
    quantiles: Optional[Dict[str, float]] = None
    raw_offset_uncertainty: Optional[float] = None
    calibration_multiplier: Optional[float] = None


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
            # EXPERIMENT-14: Get BOTH offset and drift histories for batch forecasting
            logger.debug(f"[BATCH_FORECAST] Getting batch inputs (offset + drift)...")
            offset_history, drift_history = self._get_batch_inputs()
            logger.debug(f"[BATCH_FORECAST] Offset history: {len(offset_history) if offset_history is not None else 0} data points")
            logger.debug(f"[BATCH_FORECAST] Drift history: {len(drift_history) if drift_history is not None else 0} data points")

            if offset_history is None or len(offset_history) < 10:
                error_msg = f"CRITICAL: Insufficient historical data for {self.model_type} prediction (got {len(offset_history) if offset_history is not None else 0} points, need at least 10)"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Log batch forecasting status
            logger.info(f"[BATCH_FORECAST] Batch inputs ready for {self.model_type} model:")
            logger.info(f"[BATCH_FORECAST]   Offset history: {len(offset_history)} points")
            logger.info(f"[BATCH_FORECAST]   Drift history: {len(drift_history) if drift_history is not None else 0} points")

            # Get system metrics as covariates (if available)
            logger.debug(f"Getting covariates...")
            covariates = self._get_covariates(horizon)
            logger.debug(f"Covariates: {list(covariates.keys()) if covariates else None}")

            # EXPERIMENT-14: Call predict_method with BOTH offset and drift
            logger.debug(f"[BATCH_FORECAST] Calling predict_method with batch inputs...")
            prediction_result = self.predict_method(offset_history, drift_history, covariates)
            logger.debug(f"[BATCH_FORECAST] predict_method returned: {prediction_result is not None}")

            if prediction_result is None:
                # Gracefully handle None returns (insufficient data for long-term model)
                # Return empty list so scheduler can continue with other models
                logger.warning(f"{self.model_type} model returned None (insufficient data) - returning empty predictions")
                return []

            # EXPERIMENT-14: Log batch forecasting results
            logger.info(f"[BATCH_FORECAST] Results received for {self.model_type} model:")
            logger.info(f"[BATCH_FORECAST]   Offset predictions: {len(prediction_result.predictions) if prediction_result.predictions is not None else 0}")
            logger.info(f"[BATCH_FORECAST]   Drift predictions: {len(prediction_result.drift_predictions) if prediction_result.drift_predictions is not None else 0}")
            logger.info(f"[BATCH_FORECAST]   Offset uncertainty: {len(prediction_result.uncertainty) if prediction_result.uncertainty is not None else 0}")
            logger.info(f"[BATCH_FORECAST]   Drift uncertainty: {len(prediction_result.drift_uncertainty) if prediction_result.drift_uncertainty is not None else 0}")

            # Log quantile/uncertainty info from prediction_result
            logger.info(f"[UNCERTAINTY_CHECK] {self.model_type} prediction_result:")
            logger.info(f"[UNCERTAINTY_CHECK]   - has uncertainty attr: {hasattr(prediction_result, 'uncertainty')}")
            logger.info(f"[UNCERTAINTY_CHECK]   - uncertainty is None: {prediction_result.uncertainty is None if hasattr(prediction_result, 'uncertainty') else 'N/A'}")
            logger.info(f"[UNCERTAINTY_CHECK]   - has quantiles attr: {hasattr(prediction_result, 'quantiles')}")
            logger.info(f"[UNCERTAINTY_CHECK]   - quantiles is None: {prediction_result.quantiles is None if hasattr(prediction_result, 'quantiles') else 'N/A'}")
            if hasattr(prediction_result, 'quantiles') and prediction_result.quantiles is not None:
                logger.info(f"[UNCERTAINTY_CHECK]   - quantiles keys: {list(prediction_result.quantiles.keys())}")

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

    def _get_batch_inputs(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get historical offset and drift data for batch forecasting (Experiment-14).

        EXPERIMENT-14: Returns BOTH offset and drift as separate 1D arrays for TimesFM batch forecasting.

        Returns:
            Tuple of (offset_history, drift_history), or (None, None) if insufficient data
        """
        logger.debug(f"_get_batch_inputs called for Experiment-14 batch forecasting")
        try:
            # Get measurements with drift from dataset manager
            recent_measurements, normalization_bias, current_drift = \
                self.dataset_manager.get_recent_measurements_with_drift(
                    window_seconds=None,
                    normalize=True
                )

            if not recent_measurements:
                logger.warning("No recent measurements available from dataset manager")
                return None, None

            # Store normalization bias and current drift
            self.normalization_bias = normalization_bias
            self.current_drift = current_drift

            if normalization_bias is not None:
                logger.info(f"[BATCH_INPUTS] Batch forecasting on residuals "
                           f"(baseline: {normalization_bias*1000:.3f}ms, drift: {current_drift:.3f}μs/s)")

            # Extract offsets and drifts as separate 1D arrays
            offsets = np.array([offset for timestamp, offset, drift in recent_measurements])
            drifts = np.array([drift for timestamp, offset, drift in recent_measurements])

            logger.info(f"[BATCH_INPUTS] Retrieved {len(offsets)} measurements")
            logger.info(f"[BATCH_INPUTS]   Offset range: [{offsets.min()*1000:.3f}, {offsets.max()*1000:.3f}] ms")
            logger.info(f"[BATCH_INPUTS]   Drift range: [{drifts.min():.3f}, {drifts.max():.3f}] μs/s")

            return offsets, drifts

        except Exception as e:
            logger.error(f"Failed to get batch inputs: {e}")
            return None, None

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

            # EXPERIMENT-14: Extract drift prediction from batch forecasting
            # NO FALLBACKS - TimesFM must provide drift predictions or system fails
            if prediction_result.drift_predictions is None or i >= len(prediction_result.drift_predictions):
                error_msg = f"[BATCH_CONVERT] ❌ CRITICAL: No drift predictions from TimesFM batch forecasting (index {i})"
                logger.error(error_msg)
                logger.error("[BATCH_CONVERT] System requires drift predictions - NO FALLBACKS allowed for research integrity")
                raise RuntimeError(error_msg)

            # Use predicted drift from TimesFM batch forecasting (ONLY source)
            drift_us_per_s = float(prediction_result.drift_predictions[i])
            drift_source = "timesfm_predicted"

            if i == 0:  # Log first prediction only
                logger.info(f"[BATCH_CONVERT] ✅ Using TimesFM drift predictions from batch forecasting")
                logger.info(f"[BATCH_CONVERT]   Source: {drift_source} (NO FALLBACKS - research mode)")

            # Convert drift from μs/s to s/s for PredictionWithUncertainty
            drift_s_per_s = drift_us_per_s * 1e-6

            # Log drift usage for verification (first prediction only)
            if i == 0:
                logger.info(f"[BATCH_CONVERT] Drift extraction:")
                logger.info(f"[BATCH_CONVERT]   Value: {drift_us_per_s:.3f} μs/s ({drift_s_per_s*1e6:.3f} μs/s as s/s)")
                logger.info(f"[BATCH_CONVERT]   Source: {drift_source}")

            # Get uncertainty - CRITICAL FOR UNCERTAINTY QUANTIFICATION
            raw_offset_uncertainty = None
            if prediction_result.uncertainty is not None and i < len(prediction_result.uncertainty):
                raw_offset_uncertainty = float(prediction_result.uncertainty[i])
                if i == 0:  # Log first prediction only
                    logger.info(f"[UNCERTAINTY] ✅ Model provided raw uncertainty: {raw_offset_uncertainty*1000:.3f}ms")
            else:
                # CRITICAL ERROR: Model must provide uncertainty for proper uncertainty quantification
                # Silent fallback was causing all uncertainties to be 1.0ms (see UNCERTAINTY_BUG_DEEPER_ROOT_CAUSE.md)
                if i == 0:  # Log once per prediction call
                    logger.error(f"[UNCERTAINTY] ❌ CRITICAL: {self.model_type} model returned NO UNCERTAINTY!")
                    logger.error(f"[UNCERTAINTY] prediction_result.uncertainty = {prediction_result.uncertainty}")
                    logger.error(f"[UNCERTAINTY] This indicates quantile prediction is not enabled in the model")
                    logger.error(f"[UNCERTAINTY] See UNCERTAINTY_BUG_DEEPER_ROOT_CAUSE.md for fix instructions")
                    logger.error(f"[UNCERTAINTY] Falling back to 1.0ms (THIS IS WRONG AND MUST BE FIXED!)")
                raw_offset_uncertainty = 0.001  # 1ms fallback - THIS SHOULD NOT HAPPEN

            # UNCERTAINTY CALIBRATION: Apply platform-specific multiplier
            calibration_multiplier = self.dataset_manager.get_calibration_multiplier()
            offset_uncertainty = raw_offset_uncertainty * calibration_multiplier
            if i == 0:  # Log calibration once per prediction batch
                is_calibrated = self.dataset_manager.is_uncertainty_calibrated()
                if is_calibrated:
                    logger.info(f"[CALIBRATION] ✅ Applying calibration multiplier: {calibration_multiplier:.2f}x")
                    logger.info(f"[CALIBRATION]   Raw: {raw_offset_uncertainty*1000:.3f}ms → Calibrated: {offset_uncertainty*1000:.3f}ms")
                else:
                    logger.info(f"[CALIBRATION] ⏳ Warmup phase - no calibration applied (multiplier=1.0)")

            # EXPERIMENT-14: Extract drift uncertainty from batch forecasting
            if prediction_result.drift_uncertainty is not None and i < len(prediction_result.drift_uncertainty):
                drift_uncertainty = float(prediction_result.drift_uncertainty[i])
                if i == 0:  # Log first prediction only
                    logger.info(f"[BATCH_CONVERT] ✅ Model provided drift uncertainty: {drift_uncertainty*1e6:.3f} μs/s")
            else:
                # Fallback: approximate as 10% of calibrated offset uncertainty
                drift_uncertainty = offset_uncertainty * 0.1
                if i == 0:  # Log warning once
                    logger.warning(f"[BATCH_CONVERT] ⚠ No drift uncertainty from model - using calibrated_offset*0.1 fallback")

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
                drift=drift_s_per_s,
                offset_uncertainty=offset_uncertainty,  # Calibrated uncertainty
                drift_uncertainty=drift_uncertainty,
                confidence=prediction_result.confidence,  # Use confidence from engine (based on uncertainty)
                timestamp=prediction_result.timestamp + i,
                quantiles=quantiles_dict,
                raw_offset_uncertainty=raw_offset_uncertainty,  # Raw uncertainty from TimesFM
                calibration_multiplier=calibration_multiplier  # Platform-specific multiplier
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
