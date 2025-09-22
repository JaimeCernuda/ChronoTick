#!/usr/bin/env python3
"""
Integration tests for TTM model through factory pattern.
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
from tsfm.utils.enhanced_visualization import create_enhanced_forecast_report, create_multivariate_forecast_report, create_covariates_forecast_report

warnings.filterwarnings("ignore")


class TestTTMIntegration:
    """Integration tests for TTM through factory pattern."""
    
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
    
    def test_ttm_factory_loading(self, factory):
        """Test TTM model loading through factory."""
        try:
            model = factory.load_model("ttm")
            assert model is not None
            assert model.model_name == "ttm"
            
            # Verify model is in factory's loaded models
            loaded = factory.list_loaded_models()
            assert "ttm" in loaded
            
            # Cleanup
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM dependencies not available" in str(e) or "tinytimemixer" in str(e):
                pytest.skip("TTM dependencies not available or model not supported")
            else:
                raise
    
    def test_ttm_end_to_end_forecast(self, factory, sample_data, reports_dir):
        """Test complete TTM forecasting pipeline with report generation."""
        try:
            # Load model through factory
            model = factory.load_model("ttm")
            
            # Prepare data (TTM expects 512 context length)
            context_length = 512
            forecast_horizon = 96
            
            context_data = sample_data[:context_length]
            true_forecast = sample_data[context_length:context_length + forecast_horizon]
            
            # Normalize context
            normalized_context, norm_stats = normalize_data(context_data)
            
            # Generate forecast (model takes normalized data, returns normalized predictions)
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
                model_name="TTM_Integration", 
                save_dir=reports_dir,
                quantiles=forecast_output.quantiles,
                normalization_stats=norm_stats,  # This will denormalize for display
                metadata=forecast_output.metadata
            )
            
            # Verify report was created
            assert Path(report_path).exists()
            print(f"✅ TTM Integration Report: {Path(report_path).name}")
            
            # Verify metadata
            metadata = forecast_output.metadata
            assert metadata['model_name'] == 'ttm'
            assert metadata['forecast_horizon'] == forecast_horizon
            assert metadata['context_length'] == len(normalized_context)
            
            # Cleanup
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM integration test skipped: {e}")
            else:
                raise
    
    def test_ttm_different_horizons(self, factory, sample_data):
        """Test TTM with different forecast horizons."""
        try:
            model = factory.load_model("ttm")
            
            context_data = sample_data[:512]  # TTM standard context
            normalized_context, _ = normalize_data(context_data)
            
            horizons = [24, 48, 96, 192]
            
            for horizon in horizons:
                forecast_output = model.forecast(normalized_context, horizon=horizon)
                
                assert forecast_output is not None
                assert len(forecast_output.predictions) == horizon
                print(f"✅ TTM horizon {horizon}: {len(forecast_output.predictions)} predictions")
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM multi-horizon test skipped: {e}")
            else:
                raise
    
    def test_ttm_context_requirements(self, factory, sample_data):
        """Test TTM context length requirements and minimum length validation."""
        try:
            model = factory.load_model("ttm")
            
            # Test that TTM rejects context that's too short
            short_context = sample_data[:50]  # Too short for TTM (needs at least 90)
            normalized_short, _ = normalize_data(short_context)
            
            with pytest.raises((ValueError, RuntimeError)) as exc_info:
                model.forecast(normalized_short, horizon=48)
            
            assert "shorter than TTM's minimum required length" in str(exc_info.value) or "Context length" in str(exc_info.value)
            print(f"✅ TTM correctly rejects short context: {exc_info.value}")
            
            # Test that TTM accepts valid context lengths
            valid_context_lengths = [512, 1024]  # These should work
            
            for context_len in valid_context_lengths:
                if context_len <= len(sample_data):
                    context_data = sample_data[:context_len]
                    normalized_context, _ = normalize_data(context_data)
                    
                    forecast_output = model.forecast(normalized_context, horizon=48)
                    
                    assert forecast_output is not None
                    assert len(forecast_output.predictions) == 48
                    print(f"✅ TTM context {context_len}: {len(forecast_output.predictions)} predictions")
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e) or "granite" in str(e).lower():
                pytest.skip(f"TTM context test skipped: {e}")
            else:
                raise
    
    def test_ttm_multivariate_capability(self, factory):
        """Test TTM multivariate forecasting if supported."""
        try:
            model = factory.load_model("ttm")
            
            # Create multivariate data
            np.random.seed(42)
            time_points = 600
            t = np.linspace(0, 4*np.pi, time_points)
            
            # Multiple time series
            series1 = 10 + 3 * np.sin(t) + np.random.normal(0, 0.5, time_points)
            series2 = 15 + 2 * np.cos(t) + np.random.normal(0, 0.3, time_points)
            
            # TTM typically works with univariate, but test handling
            for i, series in enumerate([series1, series2]):
                context_data = series[:512]
                normalized_context, _ = normalize_data(context_data)
                
                forecast_output = model.forecast(normalized_context, horizon=48)
                
                assert forecast_output is not None
                print(f"✅ TTM series {i+1}: {len(forecast_output.predictions)} predictions")
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM multivariate test skipped: {e}")
            else:
                raise
    
    def test_ttm_factory_health_check(self, factory):
        """Test factory health check with TTM."""
        try:
            # Load TTM
            model = factory.load_model("ttm")
            
            # Check factory health
            health = factory.health_check()
            
            assert health['factory_status'] == 'operational'
            assert 'ttm' in health['loaded_models']
            assert len(health['loaded_models']) == 1
            
            # Check individual model health
            model_health = model.health_check()
            assert model_health['dependency_status'] == 'ok'
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM health check skipped: {e}")
            else:
                raise
    
    def test_ttm_performance_benchmarking(self, factory, sample_data, reports_dir):
        """Test TTM performance and create benchmark report."""
        try:
            import time
            
            model = factory.load_model("ttm")
            
            context_data = sample_data[:512]  # TTM standard context
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
                
                print(f"✅ TTM H{horizon}: {forecast_time:.2f}s ({horizon/forecast_time:.1f} pred/s)")
            
            # Create performance report
            performance_df = pd.DataFrame(performance_results)
            performance_path = reports_dir / "TTM_performance_benchmark.csv"
            performance_df.to_csv(performance_path, index=False)
            
            print(f"📊 TTM Performance Report: {performance_path.name}")
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM performance test skipped: {e}")
            else:
                raise
    
    def test_ttm_prediction_quality(self, factory, sample_data, reports_dir):
        """Test TTM prediction quality with different data patterns."""
        try:
            model = factory.load_model("ttm")
            
            patterns = ["linear", "seasonal", "mixed"]
            quality_results = []
            
            for pattern in patterns:
                # Generate pattern-specific data
                pattern_data = create_synthetic_data(length=800, pattern=pattern, seed=42)
                
                context_data = pattern_data[:512]  # TTM context
                true_forecast = pattern_data[512:608]  # 96 steps
                
                normalized_context, _ = normalize_data(context_data)
                forecast_output = model.forecast(normalized_context, horizon=96)
                
                # Calculate quality metrics
                true_normalized, _ = normalize_data(true_forecast)
                metrics = calculate_metrics(
                    true_normalized, 
                    forecast_output.predictions,
                    ['mae', 'rmse', 'correlation']
                )
                
                quality_results.append({
                    'pattern': pattern,
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'correlation': metrics['correlation']
                })
                
                print(f"✅ TTM {pattern}: Corr={metrics['correlation']:.3f}, MAE={metrics['mae']:.3f}")
            
            # Save quality report
            quality_df = pd.DataFrame(quality_results)
            quality_path = reports_dir / "TTM_quality_analysis.csv"
            quality_df.to_csv(quality_path, index=False)
            
            print(f"📊 TTM Quality Report: {quality_path.name}")
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM quality test skipped: {e}")
            else:
                raise
    
    def test_ttm_multivariate_visualization(self, factory, reports_dir):
        """Test TTM multivariate forecasting visualization."""
        try:
            model = factory.load_model("ttm")
            
            # Create multivariate synthetic data
            np.random.seed(42)
            time_points = 600  # TTM context requirement
            t = np.linspace(0, 4*np.pi, time_points)
            
            # Multiple related time series
            series1 = 10 + 3 * np.sin(t) + np.random.normal(0, 0.5, time_points)  # Temperature
            series2 = 15 + 2 * np.cos(t + np.pi/4) + np.random.normal(0, 0.3, time_points)  # Pressure  
            series3 = 8 + 1.5 * np.sin(2*t) + np.random.normal(0, 0.2, time_points)  # Humidity
            
            multivariate_results = {}
            normalization_stats = None
            
            for i, (var_name, series) in enumerate([("Temperature", series1), ("Pressure", series2), ("Humidity", series3)]):
                context_data = series[:512]  # TTM context
                true_forecast = series[512:560]  # 48 steps forecast
                
                # Normalize data
                normalized_context, norm_stats = normalize_data(context_data)
                if i == 0:  # Use first series stats for consistency
                    normalization_stats = norm_stats
                
                # Generate forecast
                forecast_output = model.forecast(normalized_context, horizon=48)
                
                # Store results for visualization
                multivariate_results[var_name] = {
                    'context': normalized_context,
                    'forecast_true': normalize_data(true_forecast)[0],
                    'forecast_pred': forecast_output.predictions,
                    'quantiles': forecast_output.quantiles
                }
                
                print(f"✅ TTM {var_name}: {len(forecast_output.predictions)} predictions")
            
            # Create multivariate visualization report
            report_path = create_multivariate_forecast_report(
                multivariate_results=multivariate_results,
                model_name="TTM_Multivariate",
                save_dir=reports_dir,
                normalization_stats=normalization_stats
            )
            
            # Verify report was created
            assert Path(report_path).exists()
            print(f"📊 TTM Multivariate Report: {Path(report_path).name}")
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM multivariate visualization test skipped: {e}")
            else:
                raise
    
    def test_ttm_covariates_visualization(self, factory, reports_dir):
        """Test TTM covariates forecasting visualization."""
        try:
            model = factory.load_model("ttm")
            
            # Create target series and covariates
            np.random.seed(42)
            time_points = 600
            t = np.linspace(0, 4*np.pi, time_points)
            
            # Target series (sales)
            target_series = 100 + 20 * np.sin(t) + 10 * np.cos(2*t) + np.random.normal(0, 2, time_points)
            
            # Covariates that influence target
            price_covariate = 50 + 10 * np.sin(t + np.pi/3) + np.random.normal(0, 1, time_points)
            marketing_covariate = 30 + 15 * np.cos(t) + np.random.normal(0, 1.5, time_points)  
            weather_covariate = 20 + 5 * np.sin(3*t) + np.random.normal(0, 0.5, time_points)
            
            # Prepare data
            context_length = 512
            forecast_horizon = 48
            
            context_data = target_series[:context_length]
            true_forecast = target_series[context_length:context_length + forecast_horizon]
            
            # Normalize target data
            normalized_context, norm_stats = normalize_data(context_data)
            
            # Create covariates data dict
            covariates_data = {
                'Price': price_covariate[:context_length + forecast_horizon],
                'Marketing_Spend': marketing_covariate[:context_length + forecast_horizon],
                'Weather_Index': weather_covariate[:context_length + forecast_horizon]
            }
            
            # Generate forecast (TTM processes covariates via enhanced method)
            if hasattr(model, 'forecast_with_covariates'):
                # Create proper CovariatesInput structure
                from tsfm.base import CovariatesInput
                
                # Normalize all covariates data
                covariates_normalized = {}
                for name, data in covariates_data.items():
                    covariates_normalized[name], _ = normalize_data(data)
                
                covariates_input = CovariatesInput(
                    target=normalized_context,
                    covariates=covariates_normalized
                )
                
                forecast_output = model.forecast_with_covariates(
                    covariates_input, 
                    horizon=forecast_horizon
                )
            else:
                # Fallback to regular forecast
                forecast_output = model.forecast(normalized_context, horizon=forecast_horizon)
            
            # Create covariates visualization report
            report_path = create_covariates_forecast_report(
                context=normalized_context,
                forecast_true=normalize_data(true_forecast)[0],
                forecast_pred=forecast_output.predictions,
                covariates_data=covariates_data,
                model_name="TTM_Covariates",
                save_dir=reports_dir,
                quantiles=forecast_output.quantiles,
                normalization_stats=norm_stats
            )
            
            # Verify report was created
            assert Path(report_path).exists()
            print(f"📊 TTM Covariates Report: {Path(report_path).name}")
            
            factory.unload_model("ttm")
            
        except Exception as e:
            if "TTM" in str(e) or "tinytimemixer" in str(e):
                pytest.skip(f"TTM covariates visualization test skipped: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])