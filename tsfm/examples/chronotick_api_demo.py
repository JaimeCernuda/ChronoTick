#!/usr/bin/env python3
"""
ChronoTick API Demonstration

Shows how to use the chronotick.time() API as a drop-in replacement
for time.time() with ML-based clock drift correction.
"""

import time
import sys
from pathlib import Path

# Add the chronotick package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chronotick


def demo_basic_usage():
    """Demonstrate basic chronotick.time() usage."""
    print("=== Basic ChronoTick Usage ===")
    print()
    
    # Start ChronoTick (auto-selects optimal configuration)
    print("Starting ChronoTick...")
    success = chronotick.start()
    
    if success:
        print("✓ ChronoTick started successfully")
        
        # Wait for daemon to initialize and make predictions
        print("Waiting for inference engine to initialize...")
        time.sleep(3)
        
        print("\n--- Time Comparison ---")
        for i in range(5):
            # Get both standard and corrected time
            standard_time = time.time()
            corrected_time = chronotick.time()
            
            offset_correction = corrected_time - standard_time
            
            print(f"Sample {i+1}:")
            print(f"  Standard time.time(): {standard_time:.6f}")
            print(f"  ChronoTick.time():    {corrected_time:.6f}")
            print(f"  Offset correction:    {offset_correction*1e6:+.3f}μs")
            print()
            
            time.sleep(1)
        
        # Stop ChronoTick
        chronotick.stop()
        print("✓ ChronoTick stopped")
    else:
        print("✗ Failed to start ChronoTick")


def demo_detailed_usage():
    """Demonstrate detailed timestamp information."""
    print("\n=== Detailed ChronoTick Usage ===")
    print()
    
    # Start with specific CPU affinity
    print("Starting ChronoTick with CPU affinity...")
    success = chronotick.start(cpu_affinity=[1])
    
    if success:
        print("✓ ChronoTick started")
        
        # Wait for initialization
        time.sleep(3)
        
        print("\n--- Detailed Timestamp Information ---")
        for i in range(3):
            # Get detailed corrected time
            ct = chronotick.time_detailed()
            
            print(f"Request {i+1}:")
            print(f"  Raw timestamp:     {ct.raw_timestamp:.6f}")
            print(f"  Corrected timestamp: {ct.timestamp:.6f}")
            print(f"  Offset correction: {ct.offset_correction*1e6:+.3f}μs")
            
            if ct.uncertainty:
                print(f"  Uncertainty:       ±{ct.uncertainty*1e6:.3f}μs")
            
            if ct.confidence:
                print(f"  Confidence:        {ct.confidence:.3f}")
            
            if ct.lower_bound and ct.upper_bound:
                print(f"  95% bounds:        [{ct.lower_bound:.6f}, {ct.upper_bound:.6f}]")
                bound_range = (ct.upper_bound - ct.lower_bound) * 1e6
                print(f"  Bound range:       ±{bound_range/2:.3f}μs")
            
            print()
            time.sleep(2)
        
        chronotick.stop()
        print("✓ ChronoTick stopped")
    else:
        print("✗ Failed to start ChronoTick")


def demo_status_monitoring():
    """Demonstrate status and performance monitoring."""
    print("\n=== Status and Performance Monitoring ===")
    print()
    
    chronotick.start()
    
    # Make some time requests
    print("Making time requests...")
    for _ in range(10):
        _ = chronotick.time()
        time.sleep(0.5)
    
    # Get status
    status = chronotick.status()
    
    print("\n--- ChronoTick Status ---")
    print(f"✓ Started: {status['started']}")
    print(f"✓ Config: {Path(status['config_path']).name if status['config_path'] else 'auto'}")
    print(f"✓ CPU affinity: {status['cpu_affinity']}")
    print()
    
    print("--- API Statistics ---")
    print(f"✓ Total calls: {status['total_calls']}")
    print(f"✓ Successful calls: {status['successful_calls']}")
    print(f"✓ Fallback calls: {status['fallback_calls']}")
    print(f"✓ Success rate: {status['success_rate']:.1%}")
    print()
    
    if 'daemon_running' in status:
        print("--- Daemon Statistics ---")
        print(f"✓ Daemon running: {status['daemon_running']}")
        print(f"✓ Daemon PID: {status['daemon_pid']}")
        print(f"✓ Uptime: {status['daemon_uptime']:.1f}s")
        print(f"✓ Memory usage: {status['daemon_memory_mb']:.1f}MB")
        print(f"✓ Daemon requests: {status['daemon_requests']}")
        print(f"✓ Daemon success rate: {status['daemon_success_rate']:.1%}")
        if status.get('avg_inference_time_ms'):
            print(f"✓ Avg inference time: {status['avg_inference_time_ms']:.1f}ms")
    
    if status.get('last_known_offset_us'):
        print(f"\n--- Current Correction ---")
        print(f"✓ Last known offset: {status['last_known_offset_us']:+.3f}μs")
        if status.get('cache_age'):
            print(f"✓ Cache age: {status['cache_age']:.1f}s")
    
    chronotick.stop()


def demo_context_manager():
    """Demonstrate context manager usage."""
    print("\n=== Context Manager Usage ===")
    print()
    
    # Use ChronoTick as a context manager
    with chronotick.ChronoTick() as ct:
        print("Starting ChronoTick context...")
        
        if ct.start():
            print("✓ ChronoTick started in context")
            
            # Wait for initialization
            time.sleep(2)
            
            # Use the context manager instance
            for i in range(3):
                corrected_time = ct.time()
                detailed = ct.time_detailed()
                
                print(f"Context call {i+1}:")
                print(f"  Corrected time: {corrected_time:.6f}")
                if detailed.uncertainty:
                    print(f"  Uncertainty: ±{detailed.uncertainty*1e6:.3f}μs")
                
                time.sleep(1)
        
        print("✓ ChronoTick will stop automatically on context exit")
    
    print("✓ Context exited, ChronoTick stopped")


def demo_drop_in_replacement():
    """Demonstrate as drop-in replacement for time.time()."""
    print("\n=== Drop-in Replacement Demo ===")
    print()
    
    print("Starting application that uses time.time()...")
    chronotick.start()
    
    # Wait for initialization
    time.sleep(2)
    
    # Simulate application code that would normally use time.time()
    def application_function():
        """Simulated application function."""
        start_time = chronotick.time()  # Drop-in replacement
        
        # Simulate some work
        work_duration = 0.5
        time.sleep(work_duration)
        
        end_time = chronotick.time()  # Drop-in replacement
        
        measured_duration = end_time - start_time
        error = abs(measured_duration - work_duration)
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'measured_duration': measured_duration,
            'expected_duration': work_duration,
            'error': error
        }
    
    print("Running application function with ChronoTick timestamps...")
    
    results = []
    for i in range(3):
        result = application_function()
        results.append(result)
        
        print(f"Run {i+1}:")
        print(f"  Duration measured: {result['measured_duration']:.6f}s")
        print(f"  Duration expected: {result['expected_duration']:.6f}s")
        print(f"  Timing error: {result['error']*1000:+.3f}ms")
        print()
    
    # Compare with standard time.time()
    print("Comparing with standard time.time()...")
    
    def standard_application_function():
        """Same function using standard time.time()."""
        start_time = time.time()  # Standard library
        
        work_duration = 0.5
        time.sleep(work_duration)
        
        end_time = time.time()  # Standard library
        
        measured_duration = end_time - start_time
        error = abs(measured_duration - work_duration)
        
        return {
            'measured_duration': measured_duration,
            'error': error
        }
    
    std_result = standard_application_function()
    chronotick_avg_error = sum(r['error'] for r in results) / len(results)
    
    print("--- Timing Accuracy Comparison ---")
    print(f"✓ Standard time.time() error: {std_result['error']*1000:+.3f}ms")
    print(f"✓ ChronoTick average error: {chronotick_avg_error*1000:+.3f}ms")
    
    improvement = ((std_result['error'] - chronotick_avg_error) / std_result['error']) * 100
    print(f"✓ Accuracy improvement: {improvement:+.1f}%")
    
    chronotick.stop()


def main():
    """Run all demonstrations."""
    print("ChronoTick API Demonstration")
    print("=" * 60)
    
    try:
        demo_basic_usage()
        demo_detailed_usage()
        demo_status_monitoring()
        demo_context_manager()
        demo_drop_in_replacement()
        
        print("\n" + "=" * 60)
        print("✓ All demonstrations completed successfully!")
        print()
        print("Key takeaways:")
        print("1. chronotick.time() is a drop-in replacement for time.time()")
        print("2. Provides automatic ML-based clock drift correction")
        print("3. Includes uncertainty bounds and confidence metrics")
        print("4. Optimizes automatically for your hardware")
        print("5. Handles fallbacks gracefully when daemon unavailable")
        
    except KeyboardInterrupt:
        print("\n\n❌ Demonstration interrupted by user")
        chronotick.stop()
    except Exception as e:
        print(f"\n\n❌ Demonstration failed: {e}")
        chronotick.stop()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()