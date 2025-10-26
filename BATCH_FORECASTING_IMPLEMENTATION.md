# Batch Forecasting Implementation for Experiment-14

## Implementation Summary

### Key Changes

1. **TSFMModelWrapper** (`tsfm_model_wrapper.py`):
   - Rename `_get_offset_history()` to `_get_batch_inputs()`
   - Return tuple of `(offset_history, drift_history)` as separate 1D arrays
   - Modify `predict_with_uncertainty()` to stack inputs as `[offset_history, drift_history]`
   - Modify `_convert_prediction_result()` to unpack batch outputs

2. **ChronoTickInferenceEngine** (`engine.py`):
   - Modify `predict_short_term()` and `predict_long_term()` to handle batch inputs
   - Stack offset and drift as (2, length) array for TimesFM
   - Unpack batch outputs into separate offset and drift predictions
   - Return both in PredictionResult

3. **PredictionResult** (data structures):
   - Add `drift_predictions` field
   - Add `drift_quantiles` field
   - Keep existing `predictions` (offset) and `quantiles` (offset)

### Expected Batch Output Shapes

- Point forecast: `(2, horizon)` where:
  - `point_forecast[0, :]` = offset predictions
  - `point_forecast[1, :]` = drift predictions

- Quantile forecast: `(2, horizon, 10)` where:
  - `quantile_forecast[0, :, :]` = offset quantiles
  - `quantile_forecast[1, :, :]` = drift quantiles

### Logging Strategy

Add extensive logging at each step:
1. **Input collection**: Log offset and drift history sizes
2. **Batch formation**: Log stacked input shape
3. **Model prediction**: Log output shapes
4. **Unpacking**: Log extracted offset and drift arrays
5. **Dataset storage**: Log what's being stored

## Testing Plan

1. Run local 5-minute test
2. Check logs for batch forecasting flow
3. Verify dataset contains drift predictions
4. Confirm no errors in the loop

## Status

Created: 2025-10-25
Implementation: IN PROGRESS
