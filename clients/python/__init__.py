"""
ChronoTick Python Client Library

High-performance shared memory client for accessing ChronoTick time corrections.

Example usage:
    from chronotick_client import ChronoTickClient

    client = ChronoTickClient()
    offset, drift, uncertainty = client.get_time()
    corrected_time = time.time() + offset
"""

from .client import ChronoTickClient
from .shm_config import ChronoTickSHMConfig

__all__ = ['ChronoTickClient', 'ChronoTickSHMConfig']
