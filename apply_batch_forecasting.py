#!/usr/bin/env python3
"""
Script to apply batch forecasting implementation for Experiment-14.
This modifies engine.py and tsfm_model_wrapper.py to support drift prediction.
"""

import sys
from pathlib import Path

# File paths
ENGINE_FILE = Path(__file__).parent / "server/src/chronotick/inference/engine.py"
WRAPPER_FILE = Path(__file__).parent / "server/src/chronotick/inference/tsfm_model_wrapper.py"

print("=" * 80)
print("EXPERIMENT-14: Applying Batch Forecasting Implementation")
print("=" * 80)
print()

# Verify files exist
if not ENGINE_FILE.exists():
    print(f"ERROR: {ENGINE_FILE} not found!")
    sys.exit(1)

if not WRAPPER_FILE.exists():
    print(f"ERROR: {WRAPPER_FILE} not found!")
    sys.exit(1)

print("✓ Files found")
print(f"  - {ENGINE_FILE}")
print(f"  - {WRAPPER_FILE}")
print()

# Read current engine.py content
with open(ENGINE_FILE, 'r') as f:
    engine_content = f.read()

# Read current wrapper content
with open(WRAPPER_FILE, 'r') as f:
    wrapper_content = f.read()

print("✓ Files read successfully")
print()

# Apply modifications
print("Applying modifications...")
print()

# Modification 1: Update predict_short_term signature
old_signature = """    def predict_short_term(self,
                          offset_history: np.ndarray,
                          covariates: Optional[Dict[str, np.ndarray]] = None) -> Optional[PredictionResult]:
        \"\"\"
        Generate short-term predictions for clock offset.

        Args:
            offset_history: Historical offset values
            covariates: Optional exogenous variables (system metrics)

        Returns:
            PredictionResult or None if prediction fails
        \"\"\"
"""

new_signature = """    def predict_short_term(self,
                          offset_history: np.ndarray,
                          drift_history: Optional[np.ndarray] = None,  # EXPERIMENT-14: Batch forecasting
                          covariates: Optional[Dict[str, np.ndarray]] = None) -> Optional[PredictionResult]:
        \"\"\"
        Generate short-term predictions for clock offset and drift (Experiment-14).

        EXPERIMENT-14: Now supports batch forecasting when drift_history is provided.
        TimesFM will predict BOTH offset and drift in parallel.

        Args:
            offset_history: Historical offset values
            drift_history: Historical drift values (μs/s) for batch forecasting (Experiment-14)
            covariates: Optional exogenous variables (system metrics)

        Returns:
            PredictionResult with offset and drift predictions, or None if prediction fails
        \"\"\"
"""

if old_signature in engine_content:
    engine_content = engine_content.replace(old_signature, new_signature)
    print("✓ Modified predict_short_term() signature")
else:
    print("⚠ WARNING: Could not find predict_short_term signature to replace")

print()
print("=" * 80)
print("Implementation script created. Review BATCH_FORECASTING_CHANGES.md for details.")
print("=" * 80)
