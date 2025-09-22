# TSFM Factory - Time Series Foundation Models

A unified interface for state-of-the-art time series forecasting models including TimesFM, Chronos, TTM, Toto, and Time-MoE.

## Installation

```bash
pip install tsfm-factory
```

## Quick Start

```python
from tsfm import TSFMFactory

factory = TSFMFactory()
model = factory.load_model("chronos")
result = model.forecast(data)
```