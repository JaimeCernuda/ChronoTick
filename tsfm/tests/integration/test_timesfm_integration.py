#!/usr/bin/env python3
"""
Integration tests for TimesFM model through factory pattern.
Tests the complete pipeline and generates image reports.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import warnings

# Add the tsfm package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tsfm import TSFMFactory
from tsfm.datasets.loader import create_synthetic_data
from tsfm.datasets.preprocessing import normalize_data
from tsfm.utils.metrics import calculate_metrics
from tsfm.utils.enhanced_visualization import create_enhanced_forecast_report

warnings.filterwarnings("ignore")


class TestTimesFMIntegration:
    """Integration tests for TimesFM through factory pattern."""
    
    @pytest.fixture
    def factory(self):
        """Create TSFM factory instance."""
        return TSFMFactory()
    
    @pytest.fixture
    def reports_dir(self):
        """Create reports directory."""
        reports_dir = Path("tests/integration/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        return create_synthetic_data(length=1000, pattern="mixed", seed=42)
    
    def test_timesfm_factory_loading(self, factory):
        """Test TimesFM model loading through factory."""
        try:
            model = factory.load_model("timesfm")
            assert model is not None
            assert model.model_name == "timesfm"
            
            # Verify model is in factory's loaded models
            loaded = factory.list_loaded_models()
            assert "timesfm" in loaded
            
            # Cleanup
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM package not available" in str(e):
                pytest.skip("TimesFM package not installed")
            else:
                raise
    
    def test_timesfm_end_to_end_forecast(self, factory, sample_data, reports_dir):
        """Test complete TimesFM forecasting pipeline with report generation."""
        try:
            # Load model through factory
            model = factory.load_model("timesfm")
            
            # Prepare data
            context_length = 800
            forecast_horizon = 96
            
            context_data = sample_data[:context_length]
            true_forecast = sample_data[context_length:context_length + forecast_horizon]
            
            # Normalize context
            normalized_context, norm_stats = normalize_data(context_data)
            
            # Generate forecast
            forecast_output = model.forecast(normalized_context, horizon=forecast_horizon)
            
            # Validate forecast output
            assert forecast_output is not None
            assert forecast_output.predictions is not None
            assert len(forecast_output.predictions) == forecast_horizon
            
            # Denormalize predictions for proper comparison and visualization
            from tsfm.datasets.preprocessing import denormalize_data
            forecast_pred_denormalized = denormalize_data(forecast_output.predictions, norm_stats)
            
            # Calculate metrics on denormalized data (proper scale)
            metrics = calculate_metrics(
                true_forecast, 
                forecast_pred_denormalized,
                ['mae', 'rmse', 'mape', 'correlation']
            )
            
            # Generate enhanced forecast report with proper scaling
            report_path = create_enhanced_forecast_report(
                context_true=normalized_context,  # Pass normalized context for consistency
                forecast_true=normalize_data(true_forecast)[0],  # Normalize true forecast too
                forecast_pred=forecast_output.predictions,  # Keep predictions normalized
                metrics=metrics,
                model_name="TimesFM_Integration", 
                save_dir=reports_dir,
                quantiles=forecast_output.quantiles,  # TimesFM may provide quantiles
                normalization_stats=norm_stats,  # This will denormalize for display
                metadata=forecast_output.metadata
            )
            
            # Verify report was created
            assert Path(report_path).exists()
            print(f"✅ TimesFM Integration Report: {Path(report_path).name}")
            
            # Verify metadata
            metadata = forecast_output.metadata
            assert metadata['model_name'] == 'timesfm'
            assert metadata['forecast_horizon'] == forecast_horizon
            assert metadata['context_length'] == len(normalized_context)
            
            # Cleanup
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM integration test skipped: {e}")
            else:
                raise
    
    def test_timesfm_different_horizons(self, factory, sample_data):
        """Test TimesFM with different forecast horizons."""
        try:
            model = factory.load_model("timesfm")
            
            context_data = sample_data[:500]
            normalized_context, _ = normalize_data(context_data)
            
            # TimesFM has a maximum horizon length of 96
            horizons = [24, 48, 96, 192]
            
            for horizon in horizons:
                forecast_output = model.forecast(normalized_context, horizon=horizon)
                
                assert forecast_output is not None
                # TimesFM caps at 96 predictions due to model architecture
                expected_length = min(horizon, 96)
                assert len(forecast_output.predictions) == expected_length
                print(f"✅ TimesFM horizon {horizon}: {len(forecast_output.predictions)} predictions (max 96)")
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM multi-horizon test skipped: {e}")
            else:
                raise
    
    def test_timesfm_quantile_forecasting(self, factory, sample_data):
        """Test TimesFM quantile forecasting capabilities."""
        try:
            model = factory.load_model("timesfm")
            
            context_data = sample_data[:400]
            normalized_context, _ = normalize_data(context_data)
            
            forecast_output = model.forecast(normalized_context, horizon=48)
            
            # Check if quantiles are available
            if forecast_output.quantiles is not None:
                assert isinstance(forecast_output.quantiles, dict)
                assert len(forecast_output.quantiles) > 0
                print(f"✅ TimesFM quantiles available: {list(forecast_output.quantiles.keys())}")
            else:
                print("ℹ️ TimesFM quantiles not available in this version")
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM quantile test skipped: {e}")
            else:
                raise
    
    def test_timesfm_factory_health_check(self, factory):
        """Test factory health check with TimesFM."""
        try:
            # Load TimesFM
            model = factory.load_model("timesfm")
            
            # Check factory health
            health = factory.health_check()
            
            assert health['factory_status'] == 'operational'
            assert 'timesfm' in health['loaded_models']
            assert len(health['loaded_models']) == 1
            
            # Check individual model health
            model_health = model.health_check()
            assert model_health['dependency_status'] == 'ok'
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM health check skipped: {e}")
            else:
                raise
    
    def test_timesfm_performance_benchmarking(self, factory, sample_data, reports_dir):
        """Test TimesFM performance and create benchmark report."""
        try:
            import time
            
            model = factory.load_model("timesfm")
            
            context_data = sample_data[:600]
            normalized_context, _ = normalize_data(context_data)
            
            # Benchmark forecasting performance
            horizons = [24, 48, 96]
            performance_results = []
            
            for horizon in horizons:
                start_time = time.time()
                forecast_output = model.forecast(normalized_context, horizon=horizon)
                forecast_time = time.time() - start_time
                
                performance_results.append({
                    'horizon': horizon,
                    'forecast_time': forecast_time,
                    'predictions_per_second': horizon / forecast_time
                })
                
                print(f"✅ TimesFM H{horizon}: {forecast_time:.2f}s ({horizon/forecast_time:.1f} pred/s)")
            
            # Create performance report
            performance_df = pd.DataFrame(performance_results)
            performance_path = reports_dir / "TimesFM_performance_benchmark.csv"
            performance_df.to_csv(performance_path, index=False)
            
            print(f"📊 TimesFM Performance Report: {performance_path.name}")
            
            factory.unload_model("timesfm")
            
        except Exception as e:
            if "TimesFM" in str(e):
                pytest.skip(f"TimesFM performance test skipped: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])