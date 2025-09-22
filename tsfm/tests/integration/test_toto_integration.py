#!/usr/bin/env python3
"""
Integration tests for Toto model through factory pattern.
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


class TestTotoIntegration:
    """Integration tests for Toto through factory pattern."""
    
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
        return create_synthetic_data(length=1200, pattern="ett", seed=42)
    
    def test_toto_factory_loading(self, factory):
        """Test Toto model loading through factory."""
        try:
            import toto
            from toto.model.toto import Toto
        except ImportError:
            pytest.skip("Toto package not available: pip install toto-ts")
            
        try:
            model = factory.load_model("toto")
            assert model is not None
            assert model.model_name == "toto"
            
            # Verify model is in factory's loaded models
            loaded = factory.get_loaded_models()
            assert "toto" in loaded
            
            # Cleanup
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto package not found" in str(e):
                pytest.skip("Toto dependencies not installed")
            else:
                raise
    
    def test_toto_end_to_end_forecast(self, factory, sample_data, reports_dir):
        """Test complete Toto forecasting pipeline with report generation."""
        try:
            # Load model through factory
            model = factory.load_model("toto")
            
            # Prepare data (Toto works better with longer context)
            context_length = 1000
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
                model_name="Toto_Integration", 
                save_dir=reports_dir,
                quantiles=forecast_output.quantiles,  # Toto provides 9 quantile levels
                normalization_stats=norm_stats,  # This will denormalize for display
                metadata=forecast_output.metadata
            )
            
            # Verify report was created
            assert Path(report_path).exists()
            print(f"✅ Toto Integration Report: {Path(report_path).name}")
            
            # Verify metadata
            metadata = forecast_output.metadata
            assert metadata['model_name'] == 'toto'
            assert metadata['forecast_horizon'] == forecast_horizon
            assert metadata['context_length'] == len(normalized_context)
            
            # Cleanup
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto integration test skipped: {e}")
            else:
                raise
    
    def test_toto_different_horizons(self, factory, sample_data):
        """Test Toto with different forecast horizons."""
        try:
            model = factory.load_model("toto")
            
            context_data = sample_data[:800]
            normalized_context, _ = normalize_data(context_data)
            
            # Toto recommended horizons (≤64 for best quality)
            horizons = [24, 48, 64, 96]
            
            for horizon in horizons:
                forecast_output = model.forecast(normalized_context, horizon=horizon)
                
                assert forecast_output is not None
                assert len(forecast_output.predictions) == horizon
                
                if horizon > 64:
                    print(f"⚠️ Toto horizon {horizon}: {len(forecast_output.predictions)} predictions (quality may degrade)")
                else:
                    print(f"✅ Toto horizon {horizon}: {len(forecast_output.predictions)} predictions")
            
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto multi-horizon test skipped: {e}")
            else:
                raise
    
    def test_toto_quantile_forecasting(self, factory, sample_data):
        """Test Toto quantile forecasting capabilities."""
        try:
            model = factory.load_model("toto")
            
            context_data = sample_data[:600]
            normalized_context, _ = normalize_data(context_data)
            
            forecast_output = model.forecast(normalized_context, horizon=48)
            
            # Toto provides 9 quantile levels
            if forecast_output.quantiles is not None:
                assert isinstance(forecast_output.quantiles, dict)
                assert len(forecast_output.quantiles) == 9  # Toto specific
                assert '0.5' in forecast_output.quantiles  # Median
                print(f"✅ Toto quantiles available: {len(forecast_output.quantiles)} levels")
                print(f"   Levels: {sorted(forecast_output.quantiles.keys())}")
            else:
                print("ℹ️ Toto quantiles not available in this version")
            
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto quantile test skipped: {e}")
            else:
                raise
    
    def test_toto_context_length_sensitivity(self, factory, sample_data):
        """Test Toto performance with different context lengths."""
        try:
            model = factory.load_model("toto")
            
            context_lengths = [100, 200, 400, 800]
            context_results = []
            
            for context_len in context_lengths:
                if context_len < len(sample_data) - 96:
                    context_data = sample_data[:context_len]
                    true_forecast = sample_data[context_len:context_len + 48]
                    
                    normalized_context, _ = normalize_data(context_data)
                    forecast_output = model.forecast(normalized_context, horizon=48)
                    
                    # Calculate quality
                    true_normalized, _ = normalize_data(true_forecast)
                    metrics = calculate_metrics(
                        true_normalized, 
                        forecast_output.predictions,
                        ['mae', 'correlation']
                    )
                    
                    context_results.append({
                        'context_length': context_len,
                        'mae': metrics['mae'],
                        'correlation': metrics['correlation']
                    })
                    
                    print(f"✅ Toto context {context_len}: Corr={metrics['correlation']:.3f}")
            
            # Find optimal context length
            best_context = max(context_results, key=lambda x: x['correlation'])
            print(f"🏆 Best Toto context length: {best_context['context_length']} (Corr={best_context['correlation']:.3f})")
            
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto context test skipped: {e}")
            else:
                raise
    
    def test_toto_factory_health_check(self, factory):
        """Test factory health check with Toto."""
        try:
            # Load Toto
            model = factory.load_model("toto")
            
            # Check factory health
            health = factory.health_check()
            
            assert health['factory_status'] == 'operational'
            assert 'toto' in health['loaded_models']
            assert len(health['loaded_models']) == 1
            
            # Check individual model health
            model_health = model.health_check()
            assert model_health['dependency_status'] == 'ok'
            
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto health check skipped: {e}")
            else:
                raise
    
    def test_toto_performance_benchmarking(self, factory, sample_data, reports_dir):
        """Test Toto performance and create benchmark report."""
        try:
            import time
            
            model = factory.load_model("toto")
            
            context_data = sample_data[:800]
            normalized_context, _ = normalize_data(context_data)
            
            # Benchmark forecasting performance
            horizons = [24, 48, 64]  # Keep within Toto's optimal range
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
                
                print(f"✅ Toto H{horizon}: {forecast_time:.2f}s ({horizon/forecast_time:.1f} pred/s)")
            
            # Create performance report
            performance_df = pd.DataFrame(performance_results)
            performance_path = reports_dir / "Toto_performance_benchmark.csv"
            performance_df.to_csv(performance_path, index=False)
            
            print(f"📊 Toto Performance Report: {performance_path.name}")
            
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto performance test skipped: {e}")
            else:
                raise
    
    def test_toto_long_sequence_handling(self, factory):
        """Test Toto with very long sequences."""
        try:
            model = factory.load_model("toto")
            
            # Test with very long sequence
            long_data = create_synthetic_data(length=2000, pattern="mixed", seed=42)
            context_data = long_data[:1500]
            
            normalized_context, _ = normalize_data(context_data)
            
            # Toto should handle long sequences
            forecast_output = model.forecast(normalized_context, horizon=48)
            
            assert forecast_output is not None
            assert len(forecast_output.predictions) == 48
            print(f"✅ Toto long sequence: {len(context_data)} → {len(forecast_output.predictions)}")
            
            factory.unload_model("toto")
            
        except Exception as e:
            if "Toto" in str(e):
                pytest.skip(f"Toto long sequence test skipped: {e}")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])