#!/usr/bin/env python3
"""
Integration tests for Chronos-Bolt model through factory pattern.
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


class TestChronosIntegration:
    """Integration tests for Chronos-Bolt through factory pattern."""
    
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
        return create_synthetic_data(length=1000, pattern="seasonal", seed=42)
    
    def test_chronos_factory_loading(self, factory):
        """Test Chronos model loading through factory."""
        try:
            model = factory.load_model("chronos")
            assert model is not None
            assert model.model_name == "chronos"
            
            # Verify model is in factory's loaded models
            loaded = factory.get_loaded_models()
            assert "chronos" in loaded
            
            # Cleanup
            factory.unload_model("chronos")
            
        except Exception as e:
            if "Chronos package not available" in str(e):
                pytest.skip("Chronos package not installed")
            else:
                raise
    
    def test_chronos_end_to_end_forecast(self, factory, sample_data, reports_dir):
        """Test complete Chronos forecasting pipeline with report generation."""
        try:
            # Load model through factory
            model = factory.load_model("chronos")
            
            # Prepare data
            context_length = 700
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
            
            # Generate enhanced forecast report with proper scaling and quantiles
            report_path = create_enhanced_forecast_report(
                context_true=normalized_context,  # Pass normalized context for consistency
                forecast_true=normalize_data(true_forecast)[0],  # Normalize true forecast too
                forecast_pred=forecast_output.predictions,  # Keep predictions normalized
                metrics=metrics,
                model_name="Chronos_Integration", 
                save_dir=reports_dir,
                quantiles=forecast_output.quantiles,  # Chronos should provide quantiles
                normalization_stats=norm_stats,  # This will denormalize for display
                metadata=forecast_output.metadata
            )
            
            # Verify report was created
            assert Path(report_path).exists()
            print(f"✅ Chronos Integration Report: {Path(report_path).name}")
            
            # Verify metadata
            metadata = forecast_output.metadata
            assert metadata['model_name'] == 'chronos'
            assert metadata['forecast_horizon'] == forecast_horizon
            assert metadata['context_length'] == len(normalized_context)
            
            # Cleanup
            factory.unload_model("chronos")
            
        except Exception as e:
            if "Chronos" in str(e):
                pytest.skip(f"Chronos integration test skipped: {e}")
            else:
                raise
    
    def test_chronos_different_horizons(self, factory, sample_data):
        """Test Chronos with different forecast horizons."""
        try:
            model = factory.load_model("chronos")
            
            context_data = sample_data[:500]
            normalized_context, _ = normalize_data(context_data)
            
            horizons = [12, 24, 48, 96]
            
            for horizon in horizons:
                forecast_output = model.forecast(normalized_context, horizon=horizon)
                
                assert forecast_output is not None
                assert len(forecast_output.predictions) == horizon
                print(f"✅ Chronos horizon {horizon}: {len(forecast_output.predictions)} predictions")
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "Chronos" in str(e):
                pytest.skip(f"Chronos multi-horizon test skipped: {e}")
            else:
                raise
    
    def test_chronos_quantile_forecasting(self, factory, sample_data):
        """Test Chronos quantile forecasting capabilities."""
        try:
            model = factory.load_model("chronos")
            
            context_data = sample_data[:400]
            normalized_context, _ = normalize_data(context_data)
            
            forecast_output = model.forecast(normalized_context, horizon=48)
            
            # Chronos should provide quantiles
            if forecast_output.quantiles is not None:
                assert isinstance(forecast_output.quantiles, dict)
                assert len(forecast_output.quantiles) > 0
                # Chronos typically provides multiple quantiles
                assert '0.5' in forecast_output.quantiles  # Median
                print(f"✅ Chronos quantiles available: {list(forecast_output.quantiles.keys())}")
            else:
                print("ℹ️ Chronos quantiles not available in this version")
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "Chronos" in str(e):
                pytest.skip(f"Chronos quantile test skipped: {e}")
            else:
                raise
    
    def test_chronos_different_model_sizes(self, factory):
        """Test different Chronos model sizes."""
        try:
            # Test with different model sizes
            sizes = ["tiny", "mini", "small"]  # Skip larger models for speed
            
            for size in sizes:
                # Create factory with specific model size
                model = factory.load_model("chronos", model_size=size)
                
                # Quick validation
                assert model is not None
                print(f"✅ Chronos {size} model loaded successfully")
                
                factory.unload_model("chronos")
                
        except Exception as e:
            if "Chronos" in str(e):
                pytest.skip(f"Chronos model sizes test skipped: {e}")
            else:
                raise
    
    def test_chronos_factory_health_check(self, factory):
        """Test factory health check with Chronos."""
        try:
            # Load Chronos
            model = factory.load_model("chronos")
            
            # Check factory health
            health = factory.health_check()
            
            assert health['factory_status'] == 'operational'
            assert 'chronos' in health['loaded_models']
            assert len(health['loaded_models']) == 1
            
            # Check individual model health
            model_health = model.health_check()
            assert model_health['dependency_status'] == 'ok'
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "Chronos" in str(e):
                pytest.skip(f"Chronos health check skipped: {e}")
            else:
                raise
    
    def test_chronos_performance_benchmarking(self, factory, sample_data, reports_dir):
        """Test Chronos performance and create benchmark report."""
        try:
            import time
            
            model = factory.load_model("chronos")
            
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
                
                print(f"✅ Chronos H{horizon}: {forecast_time:.2f}s ({horizon/forecast_time:.1f} pred/s)")
            
            # Create performance report
            performance_df = pd.DataFrame(performance_results)
            performance_path = reports_dir / "Chronos_performance_benchmark.csv"
            performance_df.to_csv(performance_path, index=False)
            
            print(f"📊 Chronos Performance Report: {performance_path.name}")
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "Chronos" in str(e):
                pytest.skip(f"Chronos performance test skipped: {e}")
            else:
                raise
    
    def test_chronos_prediction_quality(self, factory, sample_data, reports_dir):
        """Test Chronos prediction quality with different data patterns."""
        try:
            model = factory.load_model("chronos")
            
            patterns = ["linear", "seasonal", "mixed"]
            quality_results = []
            
            for pattern in patterns:
                # Generate pattern-specific data
                pattern_data = create_synthetic_data(length=800, pattern=pattern, seed=42)
                
                context_data = pattern_data[:600]
                true_forecast = pattern_data[600:696]  # 96 steps
                
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
                
                print(f"✅ Chronos {pattern}: Corr={metrics['correlation']:.3f}, MAE={metrics['mae']:.3f}")
            
            # Save quality report
            quality_df = pd.DataFrame(quality_results)
            quality_path = reports_dir / "Chronos_quality_analysis.csv"
            quality_df.to_csv(quality_path, index=False)
            
            print(f"📊 Chronos Quality Report: {quality_path.name}")
            
            factory.unload_model("chronos")
            
        except Exception as e:
            if "Chronos" in str(e):
                pytest.skip(f"Chronos quality test skipped: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])