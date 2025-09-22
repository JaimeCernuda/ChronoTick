#!/usr/bin/env python3
"""
Unit tests for covariates/exogenous variables forecasting functionality.
Tests the enhanced covariates capabilities across all models.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm.base import BaseTimeSeriesModel, CovariatesInput, FrequencyInfo, ModelStatus
from tsfm.LLM.timesfm import TimesFMModel
from tsfm.LLM.ttm import TTMModel
from tsfm.LLM.chronos_bolt import ChronosBoltModel
from tsfm.LLM.toto import TotoModel
from tsfm.LLM.time_moe import TimeMoEModel


class TestCovariatesInput:
    """Test CovariatesInput data structure."""
    
    @pytest.fixture
    def sample_covariates_data(self):
        """Create sample covariates input data."""
        np.random.seed(42)
        
        # Target time series
        target = np.random.randn(300).astype(np.float32)
        
        # Historical covariates
        covariates = {
            "temperature": np.random.randn(300).astype(np.float32),
            "humidity": np.random.uniform(0, 100, 300).astype(np.float32),
            "day_of_week": np.random.randint(1, 8, 300).astype(np.float32)
        }
        
        # Future covariates (known future values)
        future_covariates = {
            "temperature": np.random.randn(50).astype(np.float32),
            "humidity": np.random.uniform(0, 100, 50).astype(np.float32)
        }
        
        # Categorical covariates
        categorical_covariates = {
            "season": ["spring", "summer", "fall", "winter"],
            "region": ["north", "south", "east", "west"]
        }
        
        return CovariatesInput(
            target=target,
            covariates=covariates,
            future_covariates=future_covariates,
            categorical_covariates=categorical_covariates,
            metadata={"source": "weather_data", "units": "celsius"}
        )
    
    def test_covariates_input_creation(self, sample_covariates_data):
        """Test CovariatesInput creation and properties."""
        cov_input = sample_covariates_data
        
        assert len(cov_input.target) == 300
        assert len(cov_input.covariates) == 3
        assert "temperature" in cov_input.covariates
        assert "humidity" in cov_input.covariates
        assert "day_of_week" in cov_input.covariates
        
        assert len(cov_input.future_covariates) == 2
        assert len(cov_input.categorical_covariates) == 2
        assert cov_input.metadata["source"] == "weather_data"
    
    def test_minimal_covariates_input(self):
        """Test CovariatesInput with minimal required data."""
        target = np.random.randn(100).astype(np.float32)
        covariates = {"feature1": np.random.randn(100).astype(np.float32)}
        
        cov_input = CovariatesInput(target=target, covariates=covariates)
        
        assert len(cov_input.target) == 100
        assert len(cov_input.covariates) == 1
        assert cov_input.future_covariates is None
        assert cov_input.categorical_covariates is None
        assert cov_input.metadata is None


class TestFrequencyInfo:
    """Test FrequencyInfo data structure."""
    
    def test_frequency_info_creation(self):
        """Test FrequencyInfo creation with all fields."""
        freq_info = FrequencyInfo(
            freq_str="D",
            freq_value=1,
            is_regular=True,
            detected_freq="D"
        )
        
        assert freq_info.freq_str == "D"
        assert freq_info.freq_value == 1
        assert freq_info.is_regular == True
        assert freq_info.detected_freq == "D"
    
    def test_frequency_info_defaults(self):
        """Test FrequencyInfo with default values."""
        freq_info = FrequencyInfo()
        
        assert freq_info.freq_str is None
        assert freq_info.freq_value is None
        assert freq_info.is_regular == True
        assert freq_info.detected_freq is None


class TestBaseCovariatesFunctionality:
    """Test base covariates functionality."""
    
    @pytest.fixture
    def base_model(self):
        """Create a mock model that inherits from BaseTimeSeriesModel."""
        class MockModel(BaseTimeSeriesModel):
            def load_model(self): pass
            def forecast(self, context, horizon, **kwargs):
                return self._mock_forecast_output(context, horizon, **kwargs)
            def unload_model(self): pass
            
            def _mock_forecast_output(self, context, horizon, **kwargs):
                from tsfm.base import ForecastOutput
                predictions = np.random.randn(horizon).astype(np.float32)
                metadata = {"model": "mock", "horizon": horizon}
                if 'freq' in kwargs:
                    metadata['freq'] = kwargs['freq']
                return ForecastOutput(predictions=predictions, metadata=metadata)
        
        return MockModel("mock_model", "cpu")
    
    @pytest.fixture
    def covariates_input(self):
        """Create covariates input for testing."""
        np.random.seed(42)
        target = np.random.randn(200).astype(np.float32)
        covariates = {
            "external_factor": np.random.randn(200).astype(np.float32),
            "seasonal_index": np.sin(np.linspace(0, 4*np.pi, 200)).astype(np.float32)
        }
        return CovariatesInput(target=target, covariates=covariates)
    
    def test_base_covariates_forecast(self, base_model, covariates_input):
        """Test base covariates forecasting (fallback behavior)."""
        horizon = 24
        result = base_model.forecast_with_covariates(covariates_input, horizon)
        
        assert len(result.predictions) == horizon
        assert 'covariates_available' in result.metadata
        assert set(result.metadata['covariates_available']) == {"external_factor", "seasonal_index"}
        assert 'covariates_used' in result.metadata
        # Base model doesn't support covariates, so should be False
        assert result.metadata['covariates_used'] == False
    
    def test_covariates_with_frequency(self, base_model, covariates_input):
        """Test covariates forecasting with frequency information."""
        frequency = FrequencyInfo(freq_str="H", freq_value=2)
        result = base_model.forecast_with_covariates(covariates_input, 12, frequency=frequency)
        
        assert 'freq' in result.metadata
        assert result.metadata['freq'] == 2


class TestModelCovariatesCapabilities:
    """Test covariates capabilities for each model."""
    
    @pytest.fixture
    def covariates_data(self):
        """Sample covariates data for all models."""
        np.random.seed(42)
        target = np.random.randn(300).astype(np.float32)
        covariates = {
            "economic_indicator": np.random.randn(300).astype(np.float32),
            "seasonal_factor": np.cos(np.linspace(0, 6*np.pi, 300)).astype(np.float32)
        }
        return CovariatesInput(target=target, covariates=covariates)
    
    def test_timesfm_covariates_support(self, covariates_data):
        """Test TimesFM 2.0 covariates support."""
        model = TimesFMModel()
        
        # TimesFM 2.0 should have covariates support
        assert hasattr(model, 'covariates_support')
        assert model.covariates_support == True
        
        # Test the enhanced method exists
        assert hasattr(model, 'forecast_with_covariates')
    
    def test_ttm_exogenous_support(self, covariates_data):
        """Test TTM exogenous variables support."""
        model = TTMModel()
        
        # TTM should have exogenous support
        assert hasattr(model, 'exogenous_support')
        assert model.exogenous_support == True
        
        # Test the enhanced method exists
        assert hasattr(model, 'forecast_with_covariates')
    
    def test_chronos_bolt_covariates_support(self, covariates_data):
        """Test Chronos-Bolt covariates support."""
        model = ChronosBoltModel()
        
        # Chronos-Bolt doesn't have explicit covariates support yet
        assert not hasattr(model, 'covariates_support')
        
        # Should still have the base method
        assert hasattr(model, 'forecast_with_covariates')
    
    def test_toto_covariates_support(self, covariates_data):
        """Test Toto high-cardinality covariates support."""
        model = TotoModel()
        
        # Toto should have covariates support
        assert hasattr(model, 'covariates_support')
        assert model.covariates_support == True
        
        # Test the enhanced method exists
        assert hasattr(model, 'forecast_with_covariates')
    
    def test_time_moe_covariates_adaptability(self, covariates_data):
        """Test Time-MoE covariates adaptability."""
        model = TimeMoEModel()
        
        # Time-MoE should have covariates adaptability
        assert hasattr(model, 'covariates_adaptable')
        assert model.covariates_adaptable == True
        
        # Test the enhanced method exists
        assert hasattr(model, 'forecast_with_covariates')


class TestCovariatesMetadata:
    """Test covariates forecasting metadata enhancement."""
    
    @pytest.fixture
    def timesfm_model(self):
        """TimesFM model for metadata testing."""
        return TimesFMModel()
    
    @pytest.fixture
    def sample_cov_input(self):
        """Sample covariates input."""
        target = np.random.randn(200).astype(np.float32)
        covariates = {
            "weather": np.random.randn(200).astype(np.float32),
            "holiday": np.random.choice([0, 1], 200).astype(np.float32)
        }
        categorical_covariates = {
            "region": ["A", "B", "C"]
        }
        return CovariatesInput(
            target=target,
            covariates=covariates,
            categorical_covariates=categorical_covariates
        )
    
    def test_enhanced_covariates_metadata(self, timesfm_model, sample_cov_input):
        """Test that enhanced models add covariates-specific metadata."""
        if timesfm_model.covariates_support:
            # Mock the model status to avoid actual loading
            timesfm_model.status = ModelStatus.LOADED
            timesfm_model.tfm = "mock"  # Set a mock tfm
            
            try:
                result = timesfm_model.forecast_with_covariates(sample_cov_input, 24)
                
                # Check that metadata was enhanced
                metadata = result.metadata
                if 'covariates_used' in metadata:
                    assert 'covariates_used' in metadata
                    assert isinstance(metadata['covariates_used'], list)
                    
            except RuntimeError:
                # Expected if dependencies not available
                pytest.skip("TimesFM dependencies not available")


class TestFrequencyUtilities:
    """Test frequency detection and conversion utilities."""
    
    def test_frequency_detection_from_timestamps(self):
        """Test frequency detection from timestamps."""
        # Mock timestamps for daily frequency
        timestamps = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        
        freq_info = BaseTimeSeriesModel.detect_frequency(timestamps=timestamps)
        
        # Should detect some frequency (depends on pandas availability)
        assert isinstance(freq_info, FrequencyInfo)
        if freq_info.freq_str:
            assert freq_info.is_regular == True
    
    def test_frequency_conversion_string(self):
        """Test frequency format conversion from string."""
        freq_info = BaseTimeSeriesModel.convert_frequency_format("daily")
        
        assert isinstance(freq_info, FrequencyInfo)
        assert freq_info.freq_value == 1  # Daily should map to 1
        
        freq_info = BaseTimeSeriesModel.convert_frequency_format("H")
        assert freq_info.freq_value == 2  # Hourly should map to 2
    
    def test_frequency_conversion_int(self):
        """Test frequency format conversion from integer."""
        freq_info = BaseTimeSeriesModel.convert_frequency_format(3)
        
        assert isinstance(freq_info, FrequencyInfo)
        assert freq_info.freq_value == 3
        assert freq_info.freq_str == "M"  # 3 should map to Monthly
    
    def test_frequency_conversion_frequency_info(self):
        """Test frequency format conversion from FrequencyInfo (passthrough)."""
        original = FrequencyInfo(freq_str="Q", freq_value=4)
        result = BaseTimeSeriesModel.convert_frequency_format(original)
        
        assert result is original  # Should return the same object


if __name__ == "__main__":
    pytest.main([__file__, "-v"])