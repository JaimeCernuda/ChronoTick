# Experiment-14 Batch Forecasting - Implementation Status

**Date**: 2025-10-25
**Status**: PARTIAL IMPLEMENTATION COMPLETE - NEEDS COMPLETION

## ✅ COMPLETED CHANGES

### 1. Data Structures (`engine.py`)
- ✓ Added `drift_predictions`, `drift_uncertainty`, `drift_quantiles` to `PredictionResult` (lines 41-56)
- ✓ Updated `predict_short_term()` signature to accept `drift_history` parameter (lines 203-220)
- ✓ Implemented batch forecasting logic with logging (lines 245-265)

### 2. Test Script
- ✓ Created `tsfm/test_batch_forecasting.py`
- ✓ Verified TimesFM batch forecasting works: (2, horizon) outputs

## 🔴 CRITICAL REMAINING WORK

### Task 1: Unpack Batch Results in `engine.py`

**Location**: After line 265 in `predict_short_term()`

Need to add AFTER `result = self.short_term_model.forecast(...)`:

```python
# EXPERIMENT-14: Unpack batch results
if drift_history is not None:
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
```

**Then update PredictionResult creation** (around line 295):

```python
# Calculate uncertainty (if quantiles available)
uncertainty = self._calculate_uncertainty(result.quantiles)
drift_uncertainty = None
if drift_history is not None and hasattr(result, 'drift_quantiles') and result.drift_quantiles is not None:
    drift_uncertainty = self._calculate_uncertainty(result.drift_quantiles)
    logger.info(f"[BATCH_FORECAST]   Drift uncertainty calculated")

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
```

### Task 2: Apply Same Changes to `predict_long_term()`

Copy the exact same batch forecasting logic to `predict_long_term()` method (starts around line 320).

### Task 3: Modify `tsfm_model_wrapper.py`

**File**: `server/src/chronotick/inference/tsfm_model_wrapper.py`

#### 3a. Rename `_get_offset_history()` → `_get_batch_inputs()` (line 152)

```python
def _get_batch_inputs(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get offset and drift histories for batch forecasting (Experiment-14)."""
    try:
        recent_measurements, normalization_bias, current_drift = \
            self.dataset_manager.get_recent_measurements_with_drift(
                window_seconds=None,
                normalize=True
            )

        if not recent_measurements:
            return None, None

        self.normalization_bias = normalization_bias
        self.current_drift = current_drift

        # Extract as separate 1D arrays
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

#### 3b. Update `predict_with_uncertainty()` (line 89)

Replace `offset_history = self._get_offset_history()` with:

```python
offset_history, drift_history = self._get_batch_inputs()
```

Then pass both to prediction:

```python
prediction_result = self.predict_method(offset_history, drift_history, covariates)
```

#### 3c. Update `_convert_prediction_result()` (line 271)

Extract drift from prediction_result:

```python
# EXPERIMENT-14: Extract drift prediction
drift_us_per_s = self.current_drift  # Default fallback
if prediction_result.drift_predictions is not None:
    drift_us_per_s = float(prediction_result.drift_predictions[i])
    logger.debug(f"[BATCH_CONVERT] Step {i+1}: using predicted drift {drift_us_per_s:.3f}μs/s")

drift_s_per_s = drift_us_per_s * 1e-6

# Extract drift uncertainty
drift_unc = abs(drift_s_per_s) * 0.1  # Fallback
if prediction_result.drift_uncertainty is not None:
    drift_unc = float(prediction_result.drift_uncertainty[i])
```

## ⚡ QUICK COMPLETION SCRIPT

Due to interconnected nature, easiest approach:

1. Complete Task 1 (unpack batch results in engine.py)
2. Apply to both predict_short_term() and predict_long_term()
3. Update tsfm_model_wrapper.py as specified
4. Test with simple Python import to catch syntax errors
5. Run 5-minute local test

## 🧪 TESTING CHECKLIST

After implementation:
- [ ] Python import test: `python -c "from chronotick.inference.engine import ChronoTickInferenceEngine"`
- [ ] Check logs show `[BATCH_FORECAST]` markers
- [ ] Verify drift_predictions populated in PredictionResult
- [ ] Confirm dataset stores drift values
- [ ] Run 5-minute warmup + prediction cycle

## 📝 ESTIMATED TIME

- Remaining code changes: 30-45 minutes
- Testing + debugging: 15-30 minutes
- **Total: ~1 hour to working implementation**

## Next Immediate Action

Complete Task 1 by adding batch result unpacking logic after line 265 in `engine.py`.
