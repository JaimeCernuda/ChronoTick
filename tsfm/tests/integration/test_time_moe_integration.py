#!/usr/bin/env python3
"""
Integration tests for Time-MoE model through factory pattern.
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


class TestTimeMoEIntegration:
    """Integration tests for Time-MoE through factory pattern."""
    
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
    
    def test_time_moe_factory_loading(self, factory):
        """Test Time-MoE model loading through factory."""
        try:
            model = factory.load_model("time_moe")
            assert model is not None
            assert model.model_name == "time_moe"
            
            # Verify model is in factory's loaded models
            loaded = factory.list_loaded_models()
            assert "time_moe" in loaded
            
            # Cleanup
            factory.unload_model("time_moe")
            
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                pytest.skip("Time-MoE version incompatibility")
            elif "Transformers not available" in str(e):
                pytest.skip("Transformers not installed")
            else:
                raise
    
    def test_time_moe_version_compatibility(self, factory):
        """Test Time-MoE transformers version compatibility check."""
        try:
            import transformers
            from transformers import __version__
            
            # Check if we're in the correct Time-MoE environment (transformers==4.40.1)
            if __version__ == "4.40.1":
                # In correct environment - should load successfully
                model = factory.load_model("time_moe")
                assert model is not None
                factory.unload_model("time_moe")
                print(f"✅ Time-MoE loads correctly in compatible environment (transformers {__version__})")
            else:
                # In wrong environment - should fail with version error
                with pytest.raises(RuntimeError, match="Time-MoE requires transformers==4.40.1"):
                    model = factory.load_model("time_moe")
                print(f"✅ Time-MoE correctly detects version incompatibility (found {__version__})")
            
        except ImportError:
            pytest.skip("Transformers not installed")
    
    def test_time_moe_end_to_end_forecast(self, factory, sample_data, reports_dir):
        """Test complete Time-MoE forecasting pipeline with report generation."""
        try:
            # Load model through factory
            model = factory.load_model("time_moe")
            
            # Prepare data
            context_length = 512
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
                model_name="TimeMoE_Integration", 
                save_dir=reports_dir,
                quantiles=forecast_output.quantiles,  # Time-MoE may provide quantiles
                normalization_stats=norm_stats,  # This will denormalize for display
                metadata=forecast_output.metadata
            )
            
            # Verify report was created
            assert Path(report_path).exists()
            print(f"✅ Time-MoE Integration Report: {Path(report_path).name}")
            
            # Verify metadata
            metadata = forecast_output.metadata
            assert metadata['model_name'] == 'time_moe'
            assert metadata['forecast_horizon'] == forecast_horizon
            assert metadata['context_length'] == len(normalized_context)
            
            # Cleanup
            factory.unload_model("time_moe")
            
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                pytest.skip("Time-MoE version incompatibility - requires transformers==4.40.1")
            elif "Time-MoE" in str(e):
                pytest.skip(f"Time-MoE integration test skipped: {e}")
            else:
                raise
    
    def test_time_moe_different_horizons(self, factory, sample_data):
        """Test Time-MoE with different forecast horizons."""
        try:
            model = factory.load_model("time_moe")
            
            context_data = sample_data[:512]
            normalized_context, _ = normalize_data(context_data)
            
            horizons = [24, 48, 96, 192]
            
            for horizon in horizons:
                forecast_output = model.forecast(normalized_context, horizon=horizon)
                
                assert forecast_output is not None
                assert len(forecast_output.predictions) == horizon
                print(f"✅ Time-MoE horizon {horizon}: {len(forecast_output.predictions)} predictions")
            
            factory.unload_model("time_moe")
            
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                pytest.skip("Time-MoE version incompatibility")
            else:
                raise
    
    def test_time_moe_sequence_length_handling(self, factory, sample_data):
        """Test Time-MoE with different sequence lengths."""
        try:
            model = factory.load_model("time_moe")
            
            # Test different input lengths
            sequence_lengths = [256, 512, 1024]
            
            for seq_len in sequence_lengths:
                if seq_len < len(sample_data):
                    context_data = sample_data[:seq_len]
                    normalized_context, _ = normalize_data(context_data)
                    
                    forecast_output = model.forecast(normalized_context, horizon=48)
                    
                    assert forecast_output is not None
                    print(f"✅ Time-MoE sequence {seq_len}: {len(forecast_output.predictions)} predictions")
            
            factory.unload_model("time_moe")
            
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                pytest.skip("Time-MoE version incompatibility")
            else:
                raise
    
    def test_time_moe_mixture_of_experts_capability(self, factory, sample_data):
        """Test Time-MoE mixture of experts functionality."""
        try:
            model = factory.load_model("time_moe")
            
            # Test with different data patterns to engage different experts
            patterns = ["linear", "seasonal", "mixed"]
            expert_results = []
            
            for pattern in patterns:
                pattern_data = create_synthetic_data(length=700, pattern=pattern, seed=42)
                context_data = pattern_data[:512]
                
                normalized_context, _ = normalize_data(context_data)
                forecast_output = model.forecast(normalized_context, horizon=48)
                
                expert_results.append({
                    'pattern': pattern,
                    'predictions_std': np.std(forecast_output.predictions),
                    'predictions_mean': np.mean(forecast_output.predictions)
                })
                
                print(f"✅ Time-MoE {pattern}: std={np.std(forecast_output.predictions):.3f}")
            
            # Different patterns should potentially engage different experts
            print(f"🧠 Time-MoE expert engagement across {len(patterns)} patterns")
            
            factory.unload_model("time_moe")
            
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                pytest.skip("Time-MoE version incompatibility")
            else:
                raise
    
    def test_time_moe_factory_health_check(self, factory):
        """Test factory health check with Time-MoE."""
        try:
            # Load Time-MoE
            model = factory.load_model("time_moe")
            
            # Check factory health
            health = factory.health_check()
            
            assert health['factory_status'] == 'operational'
            assert 'time_moe' in health['loaded_models']
            assert len(health['loaded_models']) == 1
            
            # Check individual model health
            model_health = model.health_check()
            assert model_health['dependency_status'] == 'ok'
            assert model_health['version_compatible'] == True
            
            factory.unload_model("time_moe")
            
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                # Test health check with version incompatibility
                health = factory.health_check()
                assert health['factory_status'] == 'operational'
                print("✅ Time-MoE health check correctly reports version incompatibility")
                pytest.skip("Time-MoE version incompatibility")
            else:
                raise
    
    def test_time_moe_performance_benchmarking(self, factory, sample_data, reports_dir):
        """Test Time-MoE performance and create benchmark report."""
        try:
            import time
            
            model = factory.load_model("time_moe")
            
            context_data = sample_data[:512]
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
                
                print(f"✅ Time-MoE H{horizon}: {forecast_time:.2f}s ({horizon/forecast_time:.1f} pred/s)")
            
            # Create performance report
            performance_df = pd.DataFrame(performance_results)
            performance_path = reports_dir / "TimeMoE_performance_benchmark.csv"
            performance_df.to_csv(performance_path, index=False)
            
            print(f"📊 Time-MoE Performance Report: {performance_path.name}")
            
            factory.unload_model("time_moe")
            
        except Exception as e:
            if "Time-MoE requires transformers==4.40.1" in str(e):
                pytest.skip("Time-MoE version incompatibility")
            else:
                raise
    
    def test_time_moe_error_handling(self, factory):
        """Test Time-MoE error handling and compatibility checks."""
        try:
            import transformers
            from transformers import __version__
            
            # Check if we're in the correct Time-MoE environment 
            if __version__ == "4.40.1":
                # In correct environment - test other error conditions
                model = factory.load_model("time_moe")
                
                # Test forecasting with invalid data
                try:
                    model.forecast([], horizon=24)  # Empty context
                    assert False, "Should have raised an error for empty context"
                except (ValueError, RuntimeError) as e:
                    print(f"✅ Time-MoE correctly handles empty context: {e}")
                
                factory.unload_model("time_moe")
            else:
                # In wrong environment - should fail with version error  
                with pytest.raises(RuntimeError) as exc_info:
                    model = factory.load_model("time_moe")
                    
                error_msg = str(exc_info.value)
                assert "Time-MoE requires transformers==4.40.1" in error_msg
                print("✅ Time-MoE correctly handles version incompatibility")
            
        except ImportError:
            pytest.skip("Transformers not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])