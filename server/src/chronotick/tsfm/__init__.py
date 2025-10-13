"""
TSFM: Time Series Foundation Models Library
A production-ready library for time series forecasting using state-of-the-art foundation models.
"""

__version__ = "1.0.0"
__author__ = "TSFM Development Team"

from .factory import TSFMFactory
from .base import (
    BaseTimeSeriesModel, 
    ForecastOutput, 
    ModelStatus,
    MultivariateInput,
    CovariatesInput,
    FrequencyInfo
)

__all__ = [
    "TSFMFactory",
    "BaseTimeSeriesModel", 
    "ForecastOutput",
    "ModelStatus",
    "MultivariateInput",
    "CovariatesInput", 
    "FrequencyInfo",
    "__version__"
]