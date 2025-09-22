#!/usr/bin/env python3
"""
Tests for ChronoTick MCP Server

Comprehensive tests for the Model Context Protocol server that provides
high-precision time services to AI agents.
"""

import pytest
import asyncio
import json
import time
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import multiprocessing as mp
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chronotick_inference.mcp_server import (
    ChronoTickMCPServer, TimeResponse, DaemonStatus, create_mcp_server
)
from chronotick_inference.real_data_pipeline import CorrectionWithBounds


@pytest.fixture
def test_mcp_config():
    """Create test configuration for MCP server"""
    config = {
        'clock_measurement': {
            'ntp': {
                'servers': ['pool.ntp.org', 'time.google.com'],
                'timeout_seconds': 2.0,
                'max_acceptable_uncertainty': 0.050,
                'min_stratum': 2
            },
            'timing': {
                'warm_up': {
                    'duration_seconds': 5,  # Short for testing
                    'measurement_interval': 1.0
                },
                'normal_operation': {
                    'measurement_interval': 10.0
                }
            }
        },
        'prediction_scheduling': {
            'cpu_model': {
                'prediction_interval': 30.0,
                'prediction_horizon': 10,
                'prediction_lead_time': 2.0,
                'max_inference_time': 5.0
            },
            'gpu_model': {
                'prediction_interval': 60.0,
                'prediction_horizon': 20,
                'prediction_lead_time': 5.0,
                'max_inference_time': 10.0
            },
            'dataset': {
                'prediction_cache_size': 50
            }
        },
        'short_term': {
            'model_name': 'chronos',
            'device': 'cpu',
            'enabled': True
        },
        'long_term': {
            'model_name': 'chronos',
            'device': 'cpu',
            'enabled': True
        },
        'fusion': {
            'enabled': True,
            'method': 'inverse_variance'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name


@pytest.fixture
def mock_correction():
    """Create mock correction data"""
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


class TestChronoTickMCPServer:
    """Test ChronoTick MCP Server functionality"""
    
    def test_server_initialization(self, test_mcp_config):
        """Test MCP server initializes correctly"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        assert server.config_path == test_mcp_config
        assert server.server is not None
        assert server.total_requests == 0
        assert server.successful_requests == 0
        assert server.request_latencies == []
    
    def test_default_config_detection(self):
        """Test default configuration detection"""
        # Should find a configuration file or raise FileNotFoundError
        try:
            server = ChronoTickMCPServer()
            assert server.config_path is not None
        except FileNotFoundError:
            # This is expected if no default config exists
            pass
    
    def test_tool_listing(self, test_mcp_config):
        """Test MCP tool listing"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Test that server has been initialized with tool handlers
        assert server.server is not None
        # The MCP server should have capabilities
        assert hasattr(server.server, 'get_capabilities')
    
    @pytest.mark.asyncio
    async def test_get_time_tool_structure(self, test_mcp_config):
        """Test get_time tool structure and validation"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Mock daemon communication
        with patch.object(server, '_is_daemon_ready', return_value=True):
            with patch.object(server, '_get_correction_from_daemon') as mock_get_correction:
                with patch.object(server, '_get_daemon_status_from_daemon') as mock_get_status:
                    
                    # Setup mocks
                    mock_correction = CorrectionWithBounds(
                        offset_correction=0.000020,
                        drift_rate=0.000001,
                        offset_uncertainty=0.000005,
                        drift_uncertainty=0.0000001,
                        prediction_time=time.time(),
                        valid_until=time.time() + 100,
                        confidence=0.85,
                        source="test"
                    )
                    mock_get_correction.return_value = mock_correction
                    
                    mock_status = DaemonStatus(
                        status="ready",
                        warmup_progress=1.0,
                        warmup_remaining_seconds=0.0,
                        total_corrections=10,
                        success_rate=1.0,
                        average_latency_ms=15.0,
                        memory_usage_mb=128.0,
                        cpu_affinity=[0, 1],
                        uptime_seconds=300.0,
                        last_error=None
                    )
                    mock_get_status.return_value = mock_status
                    
                    # Test get_time
                    result = await server._handle_get_time({"include_stats": False})
                    
                    # Validate response structure
                    assert isinstance(result, dict)
                    required_fields = [
                        'corrected_time', 'system_time', 'offset_correction',
                        'drift_rate', 'offset_uncertainty', 'drift_uncertainty',
                        'time_uncertainty', 'confidence', 'source',
                        'prediction_time', 'valid_until', 'daemon_status', 'call_latency_ms'
                    ]
                    
                    for field in required_fields:
                        assert field in result, f"Missing field: {field}"
                    
                    # Validate data types
                    assert isinstance(result['corrected_time'], float)
                    assert isinstance(result['system_time'], float)
                    assert isinstance(result['offset_correction'], float)
                    assert isinstance(result['drift_rate'], float)
                    assert isinstance(result['offset_uncertainty'], float)
                    assert isinstance(result['drift_uncertainty'], float)
                    assert isinstance(result['time_uncertainty'], float)
                    assert isinstance(result['confidence'], float)
                    assert isinstance(result['source'], str)
                    assert isinstance(result['call_latency_ms'], float)
                    
                    # Validate ranges
                    assert 0 <= result['confidence'] <= 1
                    assert result['offset_uncertainty'] >= 0
                    assert result['drift_uncertainty'] >= 0
                    assert result['time_uncertainty'] >= 0
                    assert result['call_latency_ms'] >= 0
    
    @pytest.mark.asyncio
    async def test_daemon_not_ready_error(self, test_mcp_config):
        """Test error handling when daemon is not ready"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        with patch.object(server, '_is_daemon_ready', return_value=False):
            with pytest.raises(RuntimeError, match="ChronoTick daemon not ready"):
                await server._handle_get_time({})
    
    @pytest.mark.asyncio
    async def test_daemon_status_tool(self, test_mcp_config):
        """Test daemon status tool"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        mock_status = DaemonStatus(
            status="ready",
            warmup_progress=1.0,
            warmup_remaining_seconds=0.0,
            total_corrections=25,
            success_rate=0.96,
            average_latency_ms=12.5,
            memory_usage_mb=150.0,
            cpu_affinity=[0, 1, 2],
            uptime_seconds=1800.0,
            last_error=None
        )
        
        with patch.object(server, '_get_daemon_status_from_daemon', return_value=mock_status):
            result = await server._handle_get_daemon_status()
            
            # Validate status structure
            assert result['status'] == 'ready'
            assert result['total_corrections'] == 25
            assert result['success_rate'] == 0.96
            assert result['average_latency_ms'] == 12.5
            assert result['memory_usage_mb'] == 150.0
            assert result['cpu_affinity'] == [0, 1, 2]
            assert result['uptime_seconds'] == 1800.0
            
            # Check MCP server stats are included
            assert 'mcp_server' in result
            mcp_stats = result['mcp_server']
            assert 'uptime_seconds' in mcp_stats
            assert 'total_requests' in mcp_stats
            assert 'successful_requests' in mcp_stats
            assert 'success_rate' in mcp_stats
    
    @pytest.mark.asyncio
    async def test_future_uncertainty_projection(self, test_mcp_config):
        """Test time with future uncertainty projection"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Mock the get_time response
        base_response = {
            'corrected_time': 1000.000000,
            'system_time': 1000.000000,
            'offset_correction': 0.000020,
            'drift_rate': 0.000001,
            'offset_uncertainty': 0.000005,
            'drift_uncertainty': 0.0000001,
            'time_uncertainty': 0.000005,
            'confidence': 0.85,
            'source': 'test',
            'prediction_time': 999.5,
            'valid_until': 1100.0,
            'daemon_status': 'ready',
            'call_latency_ms': 15.0
        }
        
        mock_correction = CorrectionWithBounds(
            offset_correction=0.000020,
            drift_rate=0.000001,
            offset_uncertainty=0.000005,
            drift_uncertainty=0.0000001,
            prediction_time=999.5,
            valid_until=1100.0,
            confidence=0.85,
            source="test"
        )
        
        with patch.object(server, '_handle_get_time', return_value=base_response):
            with patch.object(server, '_get_correction_from_daemon', return_value=mock_correction):
                
                result = await server._handle_get_time_with_future_uncertainty({"future_seconds": 100.0})
                
                # Should include future projection fields
                assert 'future_timestamp' in result
                assert 'future_seconds' in result
                assert 'future_uncertainty' in result
                assert 'uncertainty_increase' in result
                
                assert result['future_seconds'] == 100.0
                assert result['future_timestamp'] == 1100.000000  # corrected_time + future_seconds
                assert result['future_uncertainty'] > result['time_uncertainty']  # Should increase
    
    def test_ipc_communication_mock(self, test_mcp_config):
        """Test IPC communication with mocked queues"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Mock the queues
        mock_request_queue = Mock()
        mock_response_queue = Mock()
        
        server.request_queue = mock_request_queue
        server.response_queue = mock_response_queue
        
        # Mock response data
        correction_data = {
            "type": "correction",
            "data": {
                "offset_correction": 0.000025,
                "drift_rate": 0.000001,
                "offset_uncertainty": 0.000005,
                "drift_uncertainty": 0.0000001,
                "prediction_time": time.time(),
                "valid_until": time.time() + 300,
                "confidence": 0.85,
                "source": "test"
            }
        }
        
        mock_response_queue.get.return_value = correction_data
        
        # Test IPC communication
        async def test_ipc():
            correction = await server._get_correction_from_daemon()
            
            # Verify request was sent
            mock_request_queue.put.assert_called_once()
            call_args = mock_request_queue.put.call_args[0][0]
            assert call_args["type"] == "get_time"
            assert "timestamp" in call_args
            
            # Verify response was processed
            mock_response_queue.get.assert_called_once_with(timeout=0.1)
            
            # Verify correction object
            assert correction is not None
            assert correction.offset_correction == 0.000025
            assert correction.source == "test"
        
        asyncio.run(test_ipc())
    
    def test_daemon_ready_check(self, test_mcp_config):
        """Test daemon ready check logic"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # No daemon process
        assert not server._is_daemon_ready()
        
        # Mock daemon process
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        server.daemon_process = mock_process
        
        # Mock status queue
        mock_status_queue = Mock()
        mock_status_queue.empty.return_value = False
        mock_status_queue.get_nowait.return_value = {"status": "ready"}
        server.status_queue = mock_status_queue
        
        assert server._is_daemon_ready()
        
        # Test non-ready status
        mock_status_queue.get_nowait.return_value = {"status": "warmup"}
        assert not server._is_daemon_ready()
    
    @pytest.mark.asyncio
    async def test_detailed_stats(self, test_mcp_config):
        """Test detailed statistics collection"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Add some mock request latencies
        server.request_latencies = [10.0, 15.0, 12.0, 18.0, 9.0]
        server.total_requests = 10
        server.successful_requests = 8
        
        mock_daemon_status = DaemonStatus(
            status="ready",
            warmup_progress=1.0,
            warmup_remaining_seconds=0.0,
            total_corrections=50,
            success_rate=0.95,
            average_latency_ms=13.2,
            memory_usage_mb=200.0,
            cpu_affinity=[0, 1],
            uptime_seconds=3600.0,
            last_error=None
        )
        
        with patch.object(server, '_get_daemon_status_from_daemon', return_value=mock_daemon_status):
            stats = await server._get_detailed_stats()
            
            # Validate structure
            assert 'daemon' in stats
            assert 'mcp_server' in stats
            
            # Validate daemon stats
            daemon_stats = stats['daemon']
            assert daemon_stats['status'] == 'ready'
            assert daemon_stats['total_corrections'] == 50
            
            # Validate MCP server stats
            mcp_stats = stats['mcp_server']
            assert mcp_stats['total_requests'] == 10
            assert mcp_stats['successful_requests'] == 8
            assert mcp_stats['success_rate'] == 0.8
            
            # Validate request latency stats
            latency_stats = mcp_stats['request_latencies']
            assert latency_stats['count'] == 5
            assert latency_stats['average_ms'] == 12.8  # (10+15+12+18+9)/5
            assert latency_stats['min_ms'] == 9.0
            assert latency_stats['max_ms'] == 18.0


class TestMCPServerIntegration:
    """Integration tests for MCP server with real components"""
    
    @pytest.mark.asyncio
    async def test_server_lifecycle_mock(self, test_mcp_config):
        """Test server lifecycle with mocked daemon"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Mock the daemon process lifecycle
        with patch.object(server, '_run_daemon_process') as mock_daemon:
            with patch('multiprocessing.Process') as mock_process_class:
                mock_process = Mock()
                mock_process.start.return_value = None
                mock_process.is_alive.return_value = True
                mock_process_class.return_value = mock_process
                
                # Mock successful daemon startup
                async def mock_wait_ready():
                    pass
                
                with patch.object(server, '_wait_for_daemon_ready', side_effect=mock_wait_ready):
                    await server.start_daemon()
                    
                    assert server.daemon_process is not None
                    assert server.request_queue is not None
                    assert server.response_queue is not None
                    assert server.status_queue is not None
                    
                    # Test stop
                    await server.stop_daemon()
    
    def test_create_mcp_server_function(self, test_mcp_config):
        """Test MCP server creation function"""
        server = create_mcp_server(test_mcp_config)
        
        assert isinstance(server, ChronoTickMCPServer)
        assert server.config_path == test_mcp_config
    
    def test_create_mcp_server_default_config(self):
        """Test MCP server creation with default config"""
        try:
            server = create_mcp_server()
            assert isinstance(server, ChronoTickMCPServer)
        except FileNotFoundError:
            # Expected if no default config exists
            pass


class TestMCPServerPerformance:
    """Performance tests for MCP server"""
    
    @pytest.mark.asyncio
    async def test_request_latency_tracking(self, test_mcp_config):
        """Test request latency tracking and statistics"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Simulate multiple requests with latencies
        server.request_latencies = [10.5, 12.3, 9.8, 15.2, 11.7]
        server.total_requests = 5
        server.successful_requests = 5
        
        stats = await server._get_detailed_stats()
        
        mcp_stats = stats['mcp_server']
        latency_stats = mcp_stats['request_latencies']
        
        assert latency_stats['count'] == 5
        assert latency_stats['min_ms'] == 9.8
        assert latency_stats['max_ms'] == 15.2
        assert abs(latency_stats['average_ms'] - 11.9) < 0.1  # (10.5+12.3+9.8+15.2+11.7)/5
    
    @pytest.mark.asyncio
    async def test_latency_history_management(self, test_mcp_config):
        """Test that latency history is properly managed"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Add more than 1000 latencies
        server.request_latencies = list(range(1200))
        server.total_requests = 1200
        server.successful_requests = 1200
        
        # Test the latency management logic directly
        # This should trigger history trimming since we have > 1000 latencies
        server._track_request_latency(15.5)  # Add one more latency to trigger trimming
        
        # History should be trimmed to 500 most recent
        assert len(server.request_latencies) <= 501  # 500 + the new request


class TestMCPErrorHandling:
    """Test error handling in MCP server"""
    
    @pytest.mark.asyncio
    async def test_ipc_timeout_handling(self, test_mcp_config):
        """Test IPC timeout handling"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # Mock timeout
        mock_request_queue = Mock()
        mock_response_queue = Mock()
        mock_response_queue.get.side_effect = mp.TimeoutError()
        
        server.request_queue = mock_request_queue
        server.response_queue = mock_response_queue
        
        correction = await server._get_correction_from_daemon()
        assert correction is None
    
    @pytest.mark.asyncio
    async def test_daemon_communication_failure(self, test_mcp_config):
        """Test daemon communication failure handling"""
        server = ChronoTickMCPServer(test_mcp_config)
        
        # No queues set up
        assert server.request_queue is None
        assert server.response_queue is None
        
        correction = await server._get_correction_from_daemon()
        assert correction is None
        
        status = await server._get_daemon_status_from_daemon()
        assert status is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])