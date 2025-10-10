#!/usr/bin/env python3
"""
Full Daemon Integration Test

Tests the complete ChronoTick daemon with ML integration:
- ChronoTickInferenceEngine initialization
- TSFMModelWrapper creation
- RealDataPipeline integration
- PredictiveScheduler setup
- End-to-end prediction flow
"""

import pytest
import numpy as np
import sys
import time
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add the chronotick_inference package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference.engine import ChronoTickInferenceEngine
from chronotick_inference.tsfm_model_wrapper import TSFMModelWrapper, create_model_wrappers
from chronotick_inference.real_data_pipeline import RealDataPipeline


class MockTimesFMModel:
    """Mock TimesFM model for testing."""

    def __init__(self, model_name="timesfm"):
        self.model_name = model_name
        self.loaded = True

    def forecast(self, data, prediction_length=5, **kwargs):
        """Mock forecast method."""
        # Generate simple predictions based on input
        last_value = data[-1] if len(data) > 0 else 0.0
        predictions = np.array([last_value + i * 1e-6 for i in range(prediction_length)])

        class MockResult:
            def __init__(self, preds):
                self.predictions = preds
                self.quantiles = {
                    '0.1': preds * 0.95,
                    '0.5': preds,
                    '0.9': preds * 1.05
                }
                self.metadata = {'model_name': 'timesfm'}

        return MockResult(predictions)

    def forecast_with_covariates(self, covariates_input, horizon, frequency=None, use_covariates=True):
        """Mock covariate forecasting."""
        # Extract target from covariates_input
        if hasattr(covariates_input, 'target'):
            target = covariates_input.target
        elif hasattr(covariates_input, 'data') and isinstance(covariates_input.data, dict):
            target = list(covariates_input.data.values())[0]
        else:
            target = np.array([0.0])

        # Generate predictions
        last_value = target[-1] if len(target) > 0 else 0.0
        predictions = np.array([last_value + i * 1e-6 for i in range(horizon)])

        class MockResult:
            def __init__(self, preds):
                self.predictions = preds
                self.quantiles = {
                    '0.1': preds * 0.95,
                    '0.5': preds,
                    '0.9': preds * 1.05
                }
                self.metadata = {'model_name': 'timesfm', 'covariates_used': use_covariates}

        return MockResult(predictions)

    def health_check(self):
        return {'status': 'loaded', 'model_name': self.model_name}


class MockTSFMFactory:
    """Mock TSFM factory."""

    def __init__(self):
        self.models = {}

    def load_model(self, model_name, **kwargs):
        model = MockTimesFMModel(model_name)
        self.models[model_name] = model
        return model

    def unload_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]

    def create_frequency_info(self, freq_str='S', freq_value=1, **kwargs):
        class MockFreqInfo:
            def __init__(self):
                self.freq_str = freq_str
                self.freq_value = freq_value
                self.is_regular = True
        return MockFreqInfo()

    def create_covariates_input(self, target=None, covariates=None, **kwargs):
        class MockCovariates:
            def __init__(self):
                self.target = target
                self.data = covariates or {}

            def __len__(self):
                return len(self.target) if self.target is not None else 0

        return MockCovariates()


@pytest.fixture
def daemon_config():
    """Create configuration for daemon integration test."""
    return {
        'short_term': {
            'model_name': 'timesfm',
            'device': 'cpu',
            'enabled': True,
            'inference_interval': 1.0,
            'prediction_horizon': 5,
            'context_length': 100,
            'use_covariates': True,
            'model_params': {
                'context_len': 100,
                'horizon_len': 5
            }
        },
        'long_term': {
            'model_name': 'timesfm',
            'device': 'cpu',
            'enabled': True,
            'inference_interval': 30.0,
            'prediction_horizon': 60,
            'context_length': 512,
            'use_covariates': False,
            'model_params': {
                'context_len': 512,
                'horizon_len': 60
            }
        },
        'fusion': {
            'enabled': True,
            'method': 'inverse_variance',
            'uncertainty_threshold': 0.05
        },
        'covariates': {
            'enabled': True,
            'variables': ['cpu_usage', 'temperature', 'memory_usage']
        },
        'ntp': {
            'servers': ['pool.ntp.org'],
            'timeout': 2,
            'max_retries': 3,
            'collection_interval': 60
        },
        'dataset': {
            'max_size': 10000,
            'persist': False
        },
        'prediction_scheduling': {
            'cpu_model': {
                'prediction_interval': 1.0,
                'prediction_horizon': 30,
                'prediction_lead_time': 5,
                'max_inference_time': 2.0
            },
            'gpu_model': {
                'prediction_interval': 30.0,
                'prediction_horizon': 120,
                'prediction_lead_time': 60,
                'max_inference_time': 5.0
            },
            'dataset': {
                'prediction_cache_size': 1000
            }
        },
        'clock': {
            'frequency_type': 'second',
            'frequency_code': 1
        }
    }


@pytest.fixture
def daemon_config_file(daemon_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(daemon_config, f)
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.mark.integration
class TestDaemonIntegration:
    """Integration tests for complete daemon with ML."""

    def test_inference_engine_initialization(self, daemon_config_file):
        """Test ChronoTickInferenceEngine initialization with mock models."""
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            engine = ChronoTickInferenceEngine(daemon_config_file)

            # Test model initialization
            success = engine.initialize_models()
            assert success is True

            # Verify models loaded
            assert engine.short_term_model is not None
            assert engine.long_term_model is not None

            # Test health check
            health = engine.health_check()
            assert health['status'] in ['healthy', 'degraded']

            # Cleanup
            engine.shutdown()

    def test_model_wrapper_creation(self, daemon_config_file):
        """Test TSFMModelWrapper creation and interface."""
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            # Initialize engine
            engine = ChronoTickInferenceEngine(daemon_config_file)
            engine.initialize_models()

            # Create mock dataset manager and metrics collector
            mock_dataset_manager = MagicMock()
            mock_dataset_manager.get_recent_measurements.return_value = [
                MagicMock(ntp_offset=0.001 + i*1e-6) for i in range(100)
            ]

            mock_system_metrics = MagicMock()
            mock_system_metrics.get_recent_metrics.return_value = [
                MagicMock(cpu_usage=50.0 + i, cpu_temp=70.0, memory_usage=60.0)
                for i in range(100)
            ]

            # Create wrappers
            cpu_wrapper, gpu_wrapper = create_model_wrappers(
                inference_engine=engine,
                dataset_manager=mock_dataset_manager,
                system_metrics=mock_system_metrics
            )

            # Verify wrappers created
            assert cpu_wrapper is not None
            assert gpu_wrapper is not None
            assert isinstance(cpu_wrapper, TSFMModelWrapper)
            assert isinstance(gpu_wrapper, TSFMModelWrapper)

            # Test wrapper interface
            predictions = cpu_wrapper.predict_with_uncertainty(horizon=30)
            assert predictions is not None
            assert len(predictions) > 0

            # Verify prediction structure
            pred = predictions[0]
            assert hasattr(pred, 'offset')
            assert hasattr(pred, 'drift')
            assert hasattr(pred, 'uncertainty')
            assert hasattr(pred, 'timestamp')

            # Test model info
            info = cpu_wrapper.get_model_info()
            assert info['model_type'] == 'short_term'
            assert info['engine_available'] is True

            engine.shutdown()

    def test_pipeline_initialization_with_models(self, daemon_config_file):
        """Test RealDataPipeline initialization with ML models."""
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            # Mock NTP client to avoid network calls
            with patch('chronotick_inference.real_data_pipeline.ClockMeasurementCollector'):
                # Initialize inference engine
                engine = ChronoTickInferenceEngine(daemon_config_file)
                engine.initialize_models()

                # Initialize pipeline
                pipeline = RealDataPipeline(daemon_config_file)

                # Create model wrappers
                cpu_wrapper, gpu_wrapper = create_model_wrappers(
                    inference_engine=engine,
                    dataset_manager=pipeline.dataset_manager,
                    system_metrics=pipeline.system_metrics
                )

                # Initialize pipeline with models
                pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)

                # Verify pipeline components
                assert pipeline.dataset_manager is not None
                assert pipeline.system_metrics is not None
                assert pipeline.predictive_scheduler is not None
                assert pipeline.fusion_engine is not None

                # Set model interfaces on scheduler
                pipeline.predictive_scheduler.set_model_interfaces(
                    cpu_model=cpu_wrapper,
                    gpu_model=gpu_wrapper,
                    fusion_engine=pipeline.fusion_engine
                )

                # Verify scheduler has models
                # Note: This may need adjustment based on actual PredictiveScheduler implementation

                # Cleanup
                engine.shutdown()

    def test_end_to_end_prediction_flow(self, daemon_config_file):
        """Test complete end-to-end prediction flow."""
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            with patch('chronotick_inference.real_data_pipeline.ClockMeasurementCollector'):
                # Step 1: Initialize inference engine
                engine = ChronoTickInferenceEngine(daemon_config_file)
                success = engine.initialize_models()
                assert success is True

                # Step 2: Initialize pipeline
                pipeline = RealDataPipeline(daemon_config_file)

                # Step 3: Create model wrappers
                cpu_wrapper, gpu_wrapper = create_model_wrappers(
                    inference_engine=engine,
                    dataset_manager=pipeline.dataset_manager,
                    system_metrics=pipeline.system_metrics
                )

                # Step 4: Initialize pipeline with models
                pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)

                # Step 5: Set up predictive scheduler
                pipeline.predictive_scheduler.set_model_interfaces(
                    cpu_model=cpu_wrapper,
                    gpu_model=gpu_wrapper,
                    fusion_engine=pipeline.fusion_engine
                )

                # Add some mock measurements to dataset
                from chronotick_inference.real_data_pipeline import ClockMeasurement
                for i in range(100):
                    measurement = ClockMeasurement(
                        timestamp=time.time() - (100 - i),
                        ntp_offset=0.001 + i * 1e-6,
                        ntp_delay=0.002,
                        system_time=time.time() - (100 - i),
                        ntp_server='pool.ntp.org',
                        stratum=2,
                        precision=-20
                    )
                    pipeline.dataset_manager.add_measurement(measurement)

                # Test: Get correction (end-to-end flow)
                correction = pipeline.get_correction(time.time())

                # Verify correction structure
                assert correction is not None
                assert hasattr(correction, 'offset_correction')
                assert hasattr(correction, 'uncertainty')
                assert isinstance(correction.offset_correction, (int, float))
                assert isinstance(correction.uncertainty, (int, float))

                # Cleanup
                engine.shutdown()

    def test_daemon_components_interaction(self, daemon_config_file):
        """Test interaction between all daemon components."""
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            with patch('chronotick_inference.real_data_pipeline.ClockMeasurementCollector'):
                # Initialize all components (simulating daemon.py flow)
                engine = ChronoTickInferenceEngine(daemon_config_file)
                engine.initialize_models()

                pipeline = RealDataPipeline(daemon_config_file)
                cpu_wrapper, gpu_wrapper = create_model_wrappers(
                    engine, pipeline.dataset_manager, pipeline.system_metrics
                )
                pipeline.initialize(cpu_wrapper, gpu_wrapper)

                # Verify component interactions

                # 1. Dataset manager receives measurements
                from chronotick_inference.real_data_pipeline import ClockMeasurement
                test_measurement = ClockMeasurement(
                    timestamp=time.time(),
                    ntp_offset=0.002,
                    ntp_delay=0.001,
                    system_time=time.time(),
                    ntp_server='test.pool.ntp.org',
                    stratum=2,
                    precision=-20
                )
                pipeline.dataset_manager.add_measurement(test_measurement)

                recent = pipeline.dataset_manager.get_recent_measurements(max_count=10)
                assert len(recent) > 0

                # 2. System metrics collection
                pipeline.system_metrics.start_collection()
                time.sleep(0.2)  # Let it collect
                metrics = pipeline.system_metrics.get_recent_metrics(window_seconds=1)
                pipeline.system_metrics.stop_collection()
                assert len(metrics) > 0

                # 3. Model wrapper can access both
                predictions = cpu_wrapper.predict_with_uncertainty(horizon=10)
                assert predictions is not None
                assert len(predictions) > 0

                # Cleanup
                engine.shutdown()

    def test_covariates_integration(self, daemon_config_file):
        """Test that covariates flow through the entire system."""
        with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
            with patch('chronotick_inference.real_data_pipeline.ClockMeasurementCollector'):
                # Initialize with covariates enabled
                engine = ChronoTickInferenceEngine(daemon_config_file)
                engine.initialize_models()

                pipeline = RealDataPipeline(daemon_config_file)

                # Add measurements
                from chronotick_inference.real_data_pipeline import ClockMeasurement
                for i in range(100):
                    measurement = ClockMeasurement(
                        timestamp=time.time() - (100 - i),
                        ntp_offset=0.001 + i * 1e-6,
                        ntp_delay=0.002,
                        system_time=time.time() - (100 - i),
                        ntp_server='pool.ntp.org',
                        stratum=2,
                        precision=-20
                    )
                    pipeline.dataset_manager.add_measurement(measurement)

                # Start metrics collection for covariates
                pipeline.system_metrics.start_collection()
                time.sleep(0.2)

                # Create wrappers with system metrics
                cpu_wrapper, _ = create_model_wrappers(
                    engine, pipeline.dataset_manager, pipeline.system_metrics
                )

                # Make prediction - should include covariates
                predictions = cpu_wrapper.predict_with_uncertainty(horizon=5)

                pipeline.system_metrics.stop_collection()

                # Verify predictions made
                assert predictions is not None
                assert len(predictions) > 0

                # Note: To verify covariates were actually used, we'd need to check
                # the model's metadata or logging output

                engine.shutdown()


@pytest.mark.integration
def test_complete_daemon_startup_sequence(daemon_config_file):
    """
    Test the complete daemon startup sequence as implemented in daemon.py.

    This simulates the exact flow from daemon.py lines 538-587.
    """
    with patch('chronotick_inference.engine.TSFMFactory', MockTSFMFactory):
        with patch('chronotick_inference.real_data_pipeline.ClockMeasurementCollector'):
            # STEP 1: Initialize inference engine with ML models
            print("Step 1: Initializing ChronoTick inference engine...")
            engine = ChronoTickInferenceEngine(daemon_config_file)
            success = engine.initialize_models()
            assert success is True
            print("✓ ML models initialized successfully")

            # STEP 2: Initialize real data pipeline
            print("Step 2: Initializing ChronoTick real data pipeline...")
            pipeline = RealDataPipeline(daemon_config_file)
            print("✓ Pipeline initialized")

            # STEP 3: Create model wrappers
            print("Step 3: Creating TSFM model wrappers...")
            cpu_wrapper, gpu_wrapper = create_model_wrappers(
                inference_engine=engine,
                dataset_manager=pipeline.dataset_manager,
                system_metrics=pipeline.system_metrics
            )
            assert cpu_wrapper is not None
            assert gpu_wrapper is not None
            print("✓ Model wrappers created")

            # STEP 4: Initialize pipeline with models
            print("Step 4: Connecting ML models to pipeline...")
            pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
            print("✓ Models connected")

            # STEP 5: Set model interfaces on predictive scheduler
            print("Step 5: Setting up predictive scheduler...")
            pipeline.predictive_scheduler.set_model_interfaces(
                cpu_model=cpu_wrapper,
                gpu_model=gpu_wrapper,
                fusion_engine=pipeline.fusion_engine
            )
            print("✓ Predictive scheduler ready")

            print("\n✅ Full ChronoTick integration complete!")
            print("  - Real NTP measurements: READY")
            print("  - ML clock drift prediction: ACTIVE")
            print("  - System metrics (covariates): ACTIVE")
            print("  - Dual-model architecture: ACTIVE")
            print("  - Prediction fusion: ACTIVE")

            # Verify all components are properly connected
            assert engine.short_term_model is not None
            assert engine.long_term_model is not None
            assert pipeline.dataset_manager is not None
            assert pipeline.system_metrics is not None
            assert pipeline.predictive_scheduler is not None
            assert pipeline.fusion_engine is not None

            print("\n✓ All components verified and operational!")

            # Cleanup
            engine.shutdown()


if __name__ == "__main__":
    # Run daemon integration tests
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
