# Batch Forecasting Implementation Changes

## File 1: server/src/chronotick/inference/engine.py

### Change 1.1: Add drift fields to PredictionResult (line 41)

```python
@dataclass
class PredictionResult:
    """Container for prediction results with batch support for offset and drift."""
    predictions: np.ndarray  # Offset predictions
    uncertainty: Optional[np.ndarray] = None  # Offset uncertainty
    quantiles: Optional[Dict[str, np.ndarray]] = None  # Offset quantiles

    # EXPERIMENT-14: Drift prediction support
    drift_predictions: Optional[np.ndarray] = None  # Drift predictions (μs/s)
    drift_uncertainty: Optional[np.ndarray] = None  # Drift uncertainty
    drift_quantiles: Optional[Dict[str, np.ndarray]] = None  # Drift quantiles

    confidence: float = 1.0
    model_type: ModelType = ModelType.SHORT_TERM
    timestamp: float = 0.0
    inference_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
```

### Change 1.2: Modify `predict_short_term()` to accept batch inputs (line 197)

**Key changes:**
- Accept `drift_history` parameter
- Stack [offset_history, drift_history] if drift provided
- Unpack batch outputs
- Populate drift fields in PredictionResult

```python
def predict_short_term(self,
                      offset_history: np.ndarray,
                      drift_history: Optional[np.ndarray] = None,  # EXPERIMENT-14
                      covariates: Optional[Dict[str, np.ndarray]] = None) -> Optional[PredictionResult]:
```

**Implementation:**
- Check if drift_history is provided
- If yes: Stack as (2, length) for batch forecasting
- Call model.forecast() with batch input
- Unpack outputs: offset_pred = result[0, :], drift_pred = result[1, :]
- Extract quantiles for both offset and drift
- Populate all fields in PredictionResult

### Change 1.3: Apply same changes to `predict_long_term()` (line 291)

Same pattern as short_term.

## File 2: server/src/chronotick/inference/tsfm_model_wrapper.py

### Change 2.1: Modify `_get_offset_history()` to also return drift history (line 152)

Rename to `_get_batch_inputs()` and return tuple of (offset_history, drift_history)

```python
def _get_batch_inputs(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get historical offset and drift data for batch forecasting.

    EXPERIMENT-14: Returns BOTH offset and drift as separate 1D arrays.

    Returns:
        Tuple of (offset_history, drift_history), or (None, None) if insufficient data
    """
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
```

### Change 2.2: Modify `predict_with_uncertainty()` to use batch inputs (line 89)

```python
def predict_with_uncertainty(self, horizon: int) -> List[PredictionWithUncertainty]:
    """Generate predictions with batch forecasting of offset and drift."""

    try:
        # EXPERIMENT-14: Get BOTH offset and drift histories
        logger.debug(f"[BATCH_FORECAST] Getting batch inputs...")
        offset_history, drift_history = self._get_batch_inputs()

        if offset_history is None or len(offset_history) < 10:
            error_msg = f"Insufficient data for {self.model_type} prediction"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"[BATCH_FORECAST] Batch inputs ready:")
        logger.info(f"[BATCH_FORECAST]   Offset history: {len(offset_history)} points")
        logger.info(f"[BATCH_FORECAST]   Drift history: {len(drift_history)} points")

        # Get covariates
        covariates = self._get_covariates(horizon)

        # EXPERIMENT-14: Call predict_method with BOTH offset and drift
        logger.debug(f"[BATCH_FORECAST] Calling predict_method with batch inputs...")
        prediction_result = self.predict_method(offset_history, drift_history, covariates)

        if prediction_result is None:
            logger.warning(f"{self.model_type} model returned None")
            return []

        # Log batch forecast results
        logger.info(f"[BATCH_FORECAST] Results received:")
        logger.info(f"[BATCH_FORECAST]   Offset predictions: {len(prediction_result.predictions)}")
        logger.info(f"[BATCH_FORECAST]   Drift predictions: {len(prediction_result.drift_predictions) if prediction_result.drift_predictions is not None else 0}")

        # Convert to PredictionWithUncertainty format
        predictions = self._convert_prediction_result(prediction_result, horizon)

        return predictions

    except Exception as e:
        logger.error(f"ERROR in {self.model_type} predict_with_uncertainty: {e}")
        return []
```

### Change 2.3: Modify `_convert_prediction_result()` to extract drift (line 271)

```python
def _convert_prediction_result(self,
                               prediction_result,
                               horizon: int) -> List[PredictionWithUncertainty]:
    """Convert PredictionResult to PredictiveScheduler format with drift support."""

    predictions = []
    num_predictions = min(len(prediction_result.predictions), horizon)

    for i in range(num_predictions):
        # Get offset prediction
        offset_residual = float(prediction_result.predictions[i])

        # Denormalization
        if self.normalization_bias is not None:
            offset = offset_residual + self.normalization_bias
        else:
            offset = offset_residual

        # EXPERIMENT-14: Extract drift prediction
        if prediction_result.drift_predictions is not None:
            drift_us_per_s = float(prediction_result.drift_predictions[i])
        else:
            drift_us_per_s = self.current_drift  # Fallback to current drift estimate

        # Convert drift from μs/s to s/s for PredictionWithUncertainty
        drift_s_per_s = drift_us_per_s * 1e-6

        # Extract uncertainties
        offset_unc = float(prediction_result.uncertainty[i]) if prediction_result.uncertainty is not None else abs(offset) * 0.1
        drift_unc = float(prediction_result.drift_uncertainty[i]) if prediction_result.drift_uncertainty is not None else abs(drift_s_per_s) * 0.1

        # Extract quantiles if available
        quantiles = None
        if prediction_result.quantiles:
            quantiles = {k: float(v[i]) for k, v in prediction_result.quantiles.items()}

        # Create prediction
        pred = PredictionWithUncertainty(
            offset=offset,
            drift=drift_s_per_s,
            offset_uncertainty=offset_unc,
            drift_uncertainty=drift_unc,
            confidence=prediction_result.confidence,
            timestamp=prediction_result.timestamp + i,
            quantiles=quantiles
        )
        predictions.append(pred)

        if i == 0:  # Log first prediction
            logger.info(f"[BATCH_CONVERT] Step {i+1}: offset={offset*1000:.3f}ms, drift={drift_us_per_s:.3f}μs/s")

    return predictions
```

## Testing Plan

1. Run local 5-minute test
2. Check logs for batch forecasting markers
3. Verify drift predictions are non-zero and reasonable
4. Confirm dataset stores drift values
5. Check no errors in the prediction loop

## Next Steps After Testing

1. Implement Fix3 and Fix4 correction formulas
2. Replace offset*0.1 with proper TimesFM quantiles
3. Update CSV output schema
4. Deploy to homelab/ARES for long-term testing
