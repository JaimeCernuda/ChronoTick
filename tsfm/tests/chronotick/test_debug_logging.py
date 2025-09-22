#!/usr/bin/env python3
"""
Tests for ChronoTick Debug Logging System

Verifies that comprehensive debug logging works correctly for function calls,
model I/O, IPC communication, and system diagnostics.
"""

import pytest
import logging
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
import asyncio
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chronotick_inference.mcp_server import debug_trace, ChronoTickMCPServer
from chronotick_inference.real_data_pipeline import debug_trace_pipeline, CorrectionWithBounds


class TestDebugLogging:
    """Test debug logging functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary log file
        self.log_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False)
        self.log_file.close()
        
        # Setup debug logger
        self.debug_logger = logging.getLogger("test.debug")
        self.debug_logger.setLevel(logging.DEBUG)
        
        # Create file handler
        self.handler = logging.FileHandler(self.log_file.name)
        self.handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)
        
        self.debug_logger.addHandler(self.handler)
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.debug_logger.removeHandler(self.handler)
        self.handler.close()
        Path(self.log_file.name).unlink(missing_ok=True)
    
    def test_debug_trace_decorator_sync(self):
        """Test debug trace decorator with synchronous function"""
        
        @debug_trace(include_args=True, include_result=True, include_timing=True)
        def test_function(x, y=10):
            time.sleep(0.01)  # Small delay for timing test
            return x + y
        
        # Mock the debug logger to capture output
        with patch('chronotick_inference.mcp_server.debug_logger') as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            result = test_function(5, y=15)
            
            # Verify result
            assert result == 20
            
            # Verify logging calls
            assert mock_logger.debug.call_count == 2  # Entry and exit
            
            # Check entry log
            entry_call = mock_logger.debug.call_args_list[0]
            entry_msg = entry_call[0][0]
            assert "ENTRY:" in entry_msg
            assert "test_function" in entry_msg
            
            # Check exit log
            exit_call = mock_logger.debug.call_args_list[1]
            exit_msg = exit_call[0][0]
            assert "EXIT:" in exit_msg
            assert "execution_time_ms" in exit_msg
    
    @pytest.mark.asyncio
    async def test_debug_trace_decorator_async(self):
        """Test debug trace decorator with async function"""
        
        @debug_trace(include_args=True, include_result=True, include_timing=True)
        async def async_test_function(data):
            await asyncio.sleep(0.01)
            return {"processed": data}
        
        with patch('chronotick_inference.mcp_server.debug_logger') as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            result = await async_test_function("test_data")
            
            # Verify result
            assert result == {"processed": "test_data"}
            
            # Verify logging calls
            assert mock_logger.debug.call_count == 2
    
    def test_debug_trace_pipeline_decorator(self):
        """Test pipeline-specific debug tracing"""
        
        @debug_trace_pipeline(include_args=True, include_result=True, include_timing=True)
        def pipeline_function(data_array, config=None):
            import numpy as np
            arr = np.array(data_array)
            return CorrectionWithBounds(
                offset_correction=0.000025,
                drift_rate=0.000001,
                offset_uncertainty=0.000005,
                drift_uncertainty=0.0000001,
                prediction_time=time.time(),
                valid_until=time.time() + 300,
                confidence=0.85,
                source="test"
            )
        
        with patch('chronotick_inference.real_data_pipeline.debug_logger') as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            result = pipeline_function([1, 2, 3, 4, 5], config={"test": True})
            
            # Verify result
            assert isinstance(result, CorrectionWithBounds)
            assert result.confidence == 0.85
            
            # Verify logging calls
            assert mock_logger.debug.call_count == 2
            
            # Check that numpy array was logged properly
            entry_call = mock_logger.debug.call_args_list[0]
            entry_msg = entry_call[0][0]
            assert "PIPELINE_ENTRY:" in entry_msg
    
    def test_debug_logging_argument_serialization(self):
        """Test safe serialization of complex arguments"""
        
        @debug_trace()
        def complex_function(simple_arg, complex_obj, large_list):
            return "success"
        
        class ComplexObject:
            def __init__(self):
                self.data = {"key": "value"}
        
        complex_obj = ComplexObject()
        large_list = list(range(1000))
        
        with patch('chronotick_inference.mcp_server.debug_logger') as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            result = complex_function("test", complex_obj, large_list)
            
            # Verify function succeeded
            assert result == "success"
            
            # Check that complex objects were handled safely
            entry_call = mock_logger.debug.call_args_list[0]
            entry_msg = entry_call[0][0]
            
            # Parse the JSON from the log message
            json_start = entry_msg.find('{')
            json_data = entry_msg[json_start:]
            log_data = json.loads(json_data)
            
            # Verify safe serialization
            assert log_data["args"][0] == "'test'"  # Simple string
            assert "ComplexObject object" in log_data["args"][1]  # Complex object
            assert "list" in log_data["args"][2]  # Large list
    
    def test_debug_logging_disabled_performance(self):
        """Test that debug logging has minimal impact when disabled"""
        
        @debug_trace()
        def performance_test_function(x):
            return x * 2
        
        with patch('chronotick_inference.mcp_server.debug_logger') as mock_logger:
            mock_logger.isEnabledFor.return_value = False  # Disabled
            
            start_time = time.time()
            for i in range(1000):
                result = performance_test_function(i)
            end_time = time.time()
            
            # Verify no logging calls were made
            assert mock_logger.debug.call_count == 0
            
            # Verify performance (should be very fast)
            execution_time = end_time - start_time
            assert execution_time < 0.1  # Should be much faster than 100ms
    
    def test_debug_logging_error_handling(self):
        """Test debug logging with function errors"""
        
        @debug_trace()
        def error_function():
            raise ValueError("Test error")
        
        with patch('chronotick_inference.mcp_server.debug_logger') as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            with pytest.raises(ValueError, match="Test error"):
                error_function()
            
            # Verify error was logged
            assert mock_logger.debug.call_count == 2  # Entry and error
            
            error_call = mock_logger.debug.call_args_list[1]
            error_msg = error_call[0][0]
            assert "ERROR:" in error_msg
            assert "ValueError" in error_msg
    
    def test_debug_logging_result_truncation(self):
        """Test that large results are properly truncated"""
        
        @debug_trace(include_result=True)
        def large_result_function():
            return {"large_data": "x" * 2000}  # Large result
        
        with patch('chronotick_inference.mcp_server.debug_logger') as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            
            result = large_result_function()
            
            # Verify function succeeded
            assert len(result["large_data"]) == 2000
            
            # Check that result was truncated in logs
            exit_call = mock_logger.debug.call_args_list[1]
            exit_msg = exit_call[0][0]
            
            assert "truncated" in exit_msg


class TestDebugLoggingIntegration:
    """Integration tests for debug logging system"""
    
    @pytest.mark.asyncio
    async def test_mcp_server_debug_integration(self):
        """Test debug logging integration with MCP server methods"""
        
        # Create test config
        config = {
            'clock_measurement': {
                'ntp': {
                    'servers': ['pool.ntp.org'],
                    'timeout_seconds': 2.0,
                    'max_acceptable_uncertainty': 0.050
                },
                'timing': {
                    'warm_up': {'duration_seconds': 5},
                    'normal_operation': {'measurement_interval': 10.0}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            server = ChronoTickMCPServer(config_path)
            
            # Mock daemon communication
            with patch.object(server, '_is_daemon_ready', return_value=False):
                with patch('chronotick_inference.mcp_server.debug_logger') as mock_logger:
                    mock_logger.isEnabledFor.return_value = True
                    
                    # This should trigger debug logging and raise an error
                    with pytest.raises(RuntimeError, match="ChronoTick daemon not ready"):
                        await server._handle_get_time({})
                    
                    # Verify debug logging occurred
                    assert mock_logger.debug.call_count >= 2  # Entry and error
        
        finally:
            Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])