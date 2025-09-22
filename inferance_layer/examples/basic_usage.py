#!/usr/bin/env python3
"""
ChronoTick Inference Layer - Basic Usage Examples

This script demonstrates the core functionality of the ChronoTick inference engine.
"""

import sys
import numpy as np
import time
from pathlib import Path

# Add the chronotick_inference package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference import (
    ChronoTickInferenceEngine,
    ClockDataGenerator,
    SystemMetricsCollector,
    PredictionVisualizer,
    create_inference_engine,
    quick_predict
)

def demo_basic_prediction():
    """Demonstrate basic clock offset prediction."""
    print("=== Basic Clock Offset Prediction Demo ===\n")
    
    # Generate synthetic clock data
    generator = ClockDataGenerator(seed=42)
    offset_data, system_metrics = generator.generate_realistic_scenario(
        scenario="server_load", 
        duration=1800  # 30 minutes
    )
    
    print(f"Generated {len(offset_data)} offset measurements")
    print(f"Offset range: {offset_data.min()*1e6:.3f} to {offset_data.max()*1e6:.3f} μs")
    print(f"System metrics: {list(system_metrics.keys())}\n")
    
    try:
        # Create inference engine
        config_path = Path(__file__).parent.parent / "chronotick_inference" / "config.yaml"
        
        with create_inference_engine(str(config_path)) as engine:
            print("✓ Inference engine initialized successfully")
            
            # Check health
            health = engine.health_check()
            print(f"✓ Engine status: {health['status']}")
            print(f"✓ Memory usage: {health['memory_usage_mb']:.1f} MB\n")
            
            # Short-term prediction
            print("--- Short-term Prediction (next 5 seconds) ---")
            context = offset_data[-300:]  # Last 5 minutes
            short_result = engine.predict_short_term(context, system_metrics)
            
            if short_result:
                print(f"✓ Prediction: {short_result.predictions[0]*1e6:.3f} μs")
                print(f"✓ Confidence: {short_result.confidence:.3f}")
                print(f"✓ Inference time: {short_result.inference_time:.3f}s")
                if short_result.quantiles:
                    q10 = short_result.quantiles['0.1'][0] * 1e6
                    q90 = short_result.quantiles['0.9'][0] * 1e6
                    print(f"✓ 80% confidence interval: [{q10:.3f}, {q90:.3f}] μs")
                print()
            
            # Long-term prediction
            print("--- Long-term Prediction (next 5 minutes) ---")
            long_context = offset_data[-1800:]  # Last 30 minutes
            long_result = engine.predict_long_term(long_context, system_metrics)
            
            if long_result:
                print(f"✓ Prediction horizon: {len(long_result.predictions)} steps")
                print(f"✓ First prediction: {long_result.predictions[0]*1e6:.3f} μs")
                print(f"✓ Last prediction: {long_result.predictions[-1]*1e6:.3f} μs")
                print(f"✓ Inference time: {long_result.inference_time:.3f}s")
                print()
            
            # Fused prediction
            print("--- Fused Prediction (optimal combination) ---")
            fused_result = engine.predict_fused(offset_data[-600:], system_metrics)
            
            if fused_result:
                print(f"✓ Fused prediction: {fused_result.prediction*1e6:.3f} μs")
                print(f"✓ Uncertainty: ±{fused_result.uncertainty*1e6:.3f} μs")
                print(f"✓ Short-term weight: {fused_result.weights['short_term']:.3f}")
                print(f"✓ Long-term weight: {fused_result.weights['long_term']:.3f}")
                print()
            
            # Performance statistics
            stats = engine.get_performance_stats()
            print("--- Performance Statistics ---")
            print(f"✓ Short-term inferences: {stats['short_term_inferences']}")
            print(f"✓ Long-term inferences: {stats['long_term_inferences']}")
            print(f"✓ Fusion operations: {stats['fusion_operations']}")
            print(f"✓ Average inference time: {stats.get('average_inference_time', 0):.3f}s")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demo_quick_predict():
    """Demonstrate the quick_predict convenience function."""
    print("\n=== Quick Predict Demo ===\n")
    
    # Generate test data
    generator = ClockDataGenerator(seed=123)
    offset_data, system_metrics = generator.generate_realistic_scenario(
        scenario="thermal_cycle",
        duration=600  # 10 minutes
    )
    
    print(f"Generated {len(offset_data)} measurements for thermal cycle scenario")
    
    try:
        config_path = Path(__file__).parent.parent / "chronotick_inference" / "config.yaml"
        
        # Quick prediction without covariates
        print("--- Quick prediction (no covariates) ---")
        result = quick_predict(
            offset_history=offset_data,
            config_path=str(config_path),
            use_fusion=False
        )
        
        if result:
            print(f"✓ Quick prediction: {result.predictions[0]*1e6:.3f} μs")
            print(f"✓ Model type: {result.model_type.value}")
        
        # Quick prediction with covariates
        print("\n--- Quick prediction (with covariates) ---")
        result_with_cov = quick_predict(
            offset_history=offset_data,
            config_path=str(config_path),
            use_fusion=True,
            covariates=system_metrics
        )
        
        if result_with_cov:
            print(f"✓ Fused prediction: {result_with_cov.prediction*1e6:.3f} μs")
            print(f"✓ Used covariates: {list(system_metrics.keys())[:3]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_real_time_metrics():
    """Demonstrate real-time system metrics collection."""
    print("\n=== Real-time Metrics Collection Demo ===\n")
    
    try:
        # Start metrics collection
        collector = SystemMetricsCollector(collection_interval=0.5)
        print("Starting real-time metrics collection...")
        collector.start_collection()
        
        # Let it collect for a few seconds
        for i in range(6):
            time.sleep(1)
            summary = collector.get_metrics_summary()
            if summary['status'] == 'collecting':
                print(f"Sample {i+1}: CPU={summary['cpu_usage']['current']:.1f}%, "
                      f"Memory={summary['memory_usage']['current']:.1f}%, "
                      f"Samples={summary['total_samples']}")
        
        # Get recent metrics
        print("\n--- Recent Metrics (last 5 seconds) ---")
        recent_metrics = collector.get_recent_metrics(window_seconds=5)
        for metric_name, values in recent_metrics.items():
            print(f"✓ {metric_name}: {len(values)} samples, "
                  f"avg={np.mean(values):.2f}, "
                  f"std={np.std(values):.2f}")
        
        # Stop collection
        collector.stop_collection()
        print("\n✓ Metrics collection stopped")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_visualization():
    """Demonstrate prediction visualization."""
    print("\n=== Visualization Demo ===\n")
    
    # Generate test data with known pattern
    t = np.arange(300)  # 5 minutes
    actual_offsets = 1e-4 * np.sin(2 * np.pi * t / 100) + 1e-5 * t
    predicted_offsets = actual_offsets + np.random.normal(0, 1e-5, len(t))
    uncertainties = np.full_like(actual_offsets, 2e-5)
    
    print(f"Visualizing {len(actual_offsets)} predictions...")
    
    try:
        visualizer = PredictionVisualizer()
        
        # Create plot (text-based if matplotlib not available)
        plot_result = visualizer.plot_predictions(
            timestamps=t,
            actual_offsets=actual_offsets,
            predictions=predicted_offsets,
            uncertainties=uncertainties,
            title="ChronoTick Demo Predictions"
        )
        
        print("--- Prediction Plot ---")
        print(plot_result)
        
        # Create performance report
        predictions_list = [
            {'prediction': pred, 'model_type': 'short_term'} 
            for pred in predicted_offsets
        ]
        
        report = visualizer.create_performance_report(
            predictions=predictions_list,
            actual_values=actual_offsets.tolist()
        )
        
        print("\n--- Performance Report ---")
        print(report)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_different_scenarios():
    """Demonstrate different clock behavior scenarios."""
    print("\n=== Different Scenarios Demo ===\n")
    
    generator = ClockDataGenerator(seed=456)
    scenarios = ["server_load", "thermal_cycle", "network_spike"]
    
    for scenario in scenarios:
        print(f"--- {scenario.replace('_', ' ').title()} Scenario ---")
        
        try:
            offset_data, metrics = generator.generate_realistic_scenario(scenario, duration=300)
            
            print(f"✓ Generated {len(offset_data)} samples")
            print(f"✓ Offset range: {offset_data.min()*1e6:.3f} to {offset_data.max()*1e6:.3f} μs")
            print(f"✓ Offset std: {np.std(offset_data)*1e6:.3f} μs")
            
            # Show key metric characteristics
            cpu_avg = np.mean(metrics['cpu_usage'])
            temp_avg = np.mean(metrics['temperature'])
            print(f"✓ Avg CPU usage: {cpu_avg:.1f}%")
            print(f"✓ Avg temperature: {temp_avg:.1f}°C")
            print()
            
        except Exception as e:
            print(f"❌ Error generating {scenario}: {e}")


def main():
    """Run all demos."""
    print("ChronoTick Inference Layer - Interface Demonstration")
    print("=" * 60)
    
    try:
        # Basic prediction demo
        demo_basic_prediction()
        
        # Quick predict demo
        demo_quick_predict()
        
        # Real-time metrics demo
        demo_real_time_metrics()
        
        # Visualization demo
        demo_visualization()
        
        # Different scenarios demo
        demo_different_scenarios()
        
        print("\n" + "=" * 60)
        print("✓ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Modify config.yaml to tune model parameters")
        print("2. Try different TSFM model environments")  
        print("3. Integrate with real clock measurements")
        print("4. Run the test suite: pytest tests/")
        
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()