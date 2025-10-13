#!/usr/bin/env python3
"""
Distributed Synchronization Example

This example demonstrates how to use ChronoTick for precise coordination
between multiple distributed processes or nodes.

Use case: Coordinating distributed database writes, synchronized data collection,
multi-agent AI coordination, or any scenario requiring sub-millisecond timing precision.

Run:
    # Terminal 1 (Node A):
    python examples/distributed_sync_example.py --node-id node-a

    # Terminal 2 (Node B):
    python examples/distributed_sync_example.py --node-id node-b

    # Terminal 3 (Node C):
    python examples/distributed_sync_example.py --node-id node-c

Requirements:
    - ChronoTick daemon must be running on each node: chronotick-daemon
    - All nodes must have network connectivity (for coordination messages)
"""

import argparse
import time
import json
import socket
from typing import Optional, List, Tuple
from chronotick_shm import ChronoTickClient


# ============================================================================
# Distributed Synchronization Coordinator
# ============================================================================

class DistributedSyncCoordinator:
    """
    Coordinates synchronized actions across multiple distributed nodes.

    Uses ChronoTick's corrected time to ensure all nodes execute actions
    at the same absolute time, compensating for clock drift and NTP latency.
    """

    def __init__(self, node_id: str):
        """
        Initialize coordinator for this node.

        Args:
            node_id: Unique identifier for this node (e.g., "node-a", "server-001")
        """
        self.node_id = node_id
        self.client = ChronoTickClient()

        # Verify daemon is ready
        if not self.client.is_daemon_ready():
            raise RuntimeError(
                f"ChronoTick daemon not ready on {node_id}. "
                f"Start with: chronotick-daemon"
            )

    def schedule_synchronized_action(
        self,
        delay_seconds: float,
        tolerance_ms: float = 1.0
    ) -> Tuple[float, float]:
        """
        Schedule a synchronized action to execute across all nodes.

        Args:
            delay_seconds: How many seconds from now to execute (e.g., 10.0 for 10 seconds)
            tolerance_ms: Acceptable timing variance in milliseconds (default: 1.0ms)

        Returns:
            Tuple of (target_time, uncertainty_at_target)

        Example:
            >>> target_time, uncertainty = coordinator.schedule_synchronized_action(5.0)
            >>> print(f"Action scheduled for: {target_time}")
            >>> print(f"Expected uncertainty: ±{uncertainty*1000:.3f}ms")
        """
        # Get current corrected time
        time_info = self.client.get_time()

        # Calculate target time
        target_time = time_info.corrected_timestamp + delay_seconds

        # Project uncertainty to target time
        future_time = self.client.get_future_time(delay_seconds)
        target_uncertainty = future_time.uncertainty_seconds

        print(f"[{self.node_id}] Action scheduled:")
        print(f"  Current time:       {time_info.corrected_timestamp:.6f}")
        print(f"  Target time:        {target_time:.6f}")
        print(f"  Delay:              {delay_seconds:.1f} seconds")
        print(f"  Current uncertainty: ±{time_info.uncertainty_seconds * 1000:.3f}ms")
        print(f"  Target uncertainty:  ±{target_uncertainty * 1000:.3f}ms")
        print(f"  Tolerance:          ±{tolerance_ms:.3f}ms")

        # Check if uncertainty is acceptable
        if target_uncertainty * 1000 > tolerance_ms * 2:
            print(f"  ⚠️  WARNING: Target uncertainty ({target_uncertainty*1000:.3f}ms) "
                  f"exceeds tolerance ({tolerance_ms:.3f}ms)")
            print(f"      Consider shorter delay or looser tolerance")

        return target_time, target_uncertainty

    def wait_for_sync_point(
        self,
        target_time: float,
        tolerance_ms: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Wait until target time is reached, then execute action.

        Args:
            target_time: Target corrected timestamp to wait for
            tolerance_ms: Acceptable early arrival tolerance (default: 1.0ms)

        Returns:
            Tuple of (actual_execution_time, timing_error, uncertainty)

        Example:
            >>> target_time = 1697125240.0  # From coordination protocol
            >>> exec_time, error, uncertainty = coordinator.wait_for_sync_point(target_time)
            >>> print(f"Timing error: {error*1000:.3f}ms")
        """
        print(f"\n[{self.node_id}] Waiting for synchronization point...")

        # Wait until target time (high-precision)
        self.client.wait_until(target_time, tolerance_ms=tolerance_ms)

        # Get actual execution time
        execution_time_info = self.client.get_time()
        actual_time = execution_time_info.corrected_timestamp
        uncertainty = execution_time_info.uncertainty_seconds

        # Calculate timing error
        timing_error = actual_time - target_time

        print(f"[{self.node_id}] Synchronization point reached!")
        print(f"  Target time:    {target_time:.6f}")
        print(f"  Actual time:    {actual_time:.6f}")
        print(f"  Timing error:   {timing_error * 1000:+.3f}ms")
        print(f"  Uncertainty:    ±{uncertainty * 1000:.3f}ms")
        print(f"  Status:         {'✓ SYNCHRONIZED' if abs(timing_error) < uncertainty else '⚠ WARNING: Outside uncertainty bounds'}")

        return actual_time, timing_error, uncertainty

    def verify_synchronization_quality(
        self,
        execution_times: List[Tuple[str, float]],
        target_time: float
    ):
        """
        Verify synchronization quality across multiple nodes.

        Args:
            execution_times: List of (node_id, execution_time) tuples
            target_time: Original target time

        Example:
            >>> coordinator.verify_synchronization_quality([
            ...     ("node-a", 1697125240.001234),
            ...     ("node-b", 1697125240.001456),
            ...     ("node-c", 1697125240.001789)
            ... ], target_time=1697125240.0)
        """
        print(f"\n{'='*70}")
        print(f"Synchronization Quality Report")
        print(f"{'='*70}")

        # Calculate statistics
        errors = [exec_time - target_time for _, exec_time in execution_times]
        max_skew = max(execution_times, key=lambda x: x[1])[1] - min(execution_times, key=lambda x: x[1])[1]
        avg_error = sum(errors) / len(errors)
        max_error = max(abs(e) for e in errors)

        print(f"\nNodes:           {len(execution_times)}")
        print(f"Target time:     {target_time:.6f}")
        print(f"Average error:   {avg_error * 1000:+.3f}ms")
        print(f"Max error:       {max_error * 1000:.3f}ms")
        print(f"Max skew:        {max_skew * 1000:.3f}ms")

        print(f"\nPer-node results:")
        for node_id, exec_time in sorted(execution_times):
            error = exec_time - target_time
            print(f"  {node_id:12s}: {exec_time:.6f}  (error: {error*1000:+.3f}ms)")

        # Quality assessment
        print(f"\nQuality Assessment:")
        if max_skew < 0.001:  # <1ms
            print(f"  ✓ EXCELLENT: All nodes within 1ms")
        elif max_skew < 0.005:  # <5ms
            print(f"  ✓ GOOD: All nodes within 5ms")
        elif max_skew < 0.010:  # <10ms
            print(f"  ⚠ ACCEPTABLE: All nodes within 10ms")
        else:
            print(f"  ✗ POOR: Nodes exceed 10ms variance")

        print(f"{'='*70}\n")


# ============================================================================
# Demonstration Scenarios
# ============================================================================

def scenario_1_simple_sync(node_id: str):
    """
    Scenario 1: Simple synchronized action across nodes.

    All nodes schedule and execute an action at the same time.
    """
    print("\n" + "="*70)
    print("Scenario 1: Simple Synchronized Action")
    print("="*70)

    coordinator = DistributedSyncCoordinator(node_id)

    # Schedule action 5 seconds from now
    target_time, uncertainty = coordinator.schedule_synchronized_action(
        delay_seconds=5.0,
        tolerance_ms=1.0
    )

    # Simulate coordination protocol (in real scenario, nodes would exchange target_time)
    print(f"\n[{node_id}] Broadcasting target time to other nodes...")
    print(f"[{node_id}] Target: {target_time:.6f}")

    # Wait for synchronization point
    actual_time, error, final_uncertainty = coordinator.wait_for_sync_point(
        target_time,
        tolerance_ms=1.0
    )

    # Execute synchronized action
    print(f"\n[{node_id}] 🎯 EXECUTING SYNCHRONIZED ACTION")
    perform_synchronized_action(node_id)

    # Display summary
    print(f"\n[{node_id}] Action completed:")
    print(f"  Timing error:  {error * 1000:+.3f}ms")
    print(f"  Uncertainty:   ±{final_uncertainty * 1000:.3f}ms")
    print(f"  Quality:       {'✓ Excellent' if abs(error) < 0.001 else '✓ Good' if abs(error) < 0.005 else '⚠ Acceptable'}")


def scenario_2_periodic_sync(node_id: str):
    """
    Scenario 2: Periodic synchronized actions.

    Nodes execute actions at regular intervals with tight synchronization.
    """
    print("\n" + "="*70)
    print("Scenario 2: Periodic Synchronized Actions")
    print("="*70)

    coordinator = DistributedSyncCoordinator(node_id)

    # Get initial time
    time_info = coordinator.client.get_time()
    base_time = time_info.corrected_timestamp

    # Schedule 5 actions at 2-second intervals
    num_actions = 5
    interval = 2.0

    print(f"\n[{node_id}] Scheduling {num_actions} actions at {interval}s intervals...")

    execution_records = []

    for i in range(num_actions):
        target_time = base_time + (i + 1) * interval

        print(f"\n[{node_id}] --- Action {i+1}/{num_actions} ---")
        print(f"[{node_id}] Target: {target_time:.6f}")

        # Wait for sync point
        actual_time, error, uncertainty = coordinator.wait_for_sync_point(
            target_time,
            tolerance_ms=1.0
        )

        # Execute action
        print(f"[{node_id}] 🎯 Executing action {i+1}")
        perform_synchronized_action(node_id, action_id=i+1)

        # Record execution
        execution_records.append({
            "action_id": i+1,
            "target_time": target_time,
            "actual_time": actual_time,
            "error_ms": error * 1000,
            "uncertainty_ms": uncertainty * 1000
        })

    # Summary report
    print(f"\n[{node_id}] Periodic Sync Summary:")
    print(f"{'='*70}")
    for record in execution_records:
        print(f"  Action {record['action_id']}: "
              f"error={record['error_ms']:+.3f}ms, "
              f"uncertainty=±{record['uncertainty_ms']:.3f}ms")


def scenario_3_coordinated_write(node_id: str):
    """
    Scenario 3: Coordinated distributed write.

    Simulates a distributed database write where all replicas must commit
    at the same timestamp for consistency.
    """
    print("\n" + "="*70)
    print("Scenario 3: Coordinated Distributed Write")
    print("="*70)

    coordinator = DistributedSyncCoordinator(node_id)

    # Prepare transaction
    transaction_id = f"tx-{int(time.time())}"
    print(f"\n[{node_id}] Preparing transaction: {transaction_id}")

    # Schedule write commit
    print(f"[{node_id}] Scheduling commit in 3 seconds...")
    target_time, uncertainty = coordinator.schedule_synchronized_action(
        delay_seconds=3.0,
        tolerance_ms=0.5  # Tighter tolerance for writes
    )

    # Verify uncertainty is acceptable for writes
    if uncertainty > 0.002:  # >2ms
        print(f"[{node_id}] ⚠️  WARNING: Uncertainty too high for coordinated write")
        print(f"[{node_id}]     Consider waiting for better NTP sync")

    # Wait for commit point
    print(f"\n[{node_id}] Waiting for commit point...")
    actual_time, error, final_uncertainty = coordinator.wait_for_sync_point(
        target_time,
        tolerance_ms=0.5
    )

    # Execute write
    print(f"\n[{node_id}] 💾 COMMITTING WRITE AT {actual_time:.6f}")
    perform_distributed_write(node_id, transaction_id, actual_time)

    # Verify success
    print(f"\n[{node_id}] Write committed successfully")
    print(f"[{node_id}] Transaction: {transaction_id}")
    print(f"[{node_id}] Commit time: {actual_time:.6f}")
    print(f"[{node_id}] Timing error: {error * 1000:+.3f}ms")


# ============================================================================
# Action Implementations (Simulated)
# ============================================================================

def perform_synchronized_action(node_id: str, action_id: Optional[int] = None):
    """Simulate executing a synchronized action."""
    action_str = f" #{action_id}" if action_id else ""
    print(f"[{node_id}] ⚡ Action{action_str} executed!")
    # In real scenario: perform actual work here


def perform_distributed_write(node_id: str, transaction_id: str, commit_time: float):
    """Simulate a distributed database write."""
    print(f"[{node_id}] Writing transaction {transaction_id}")
    print(f"[{node_id}] Commit timestamp: {commit_time}")
    # In real scenario: write to database with timestamp


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ChronoTick Distributed Synchronization Example"
    )
    parser.add_argument(
        "--node-id",
        type=str,
        required=True,
        help="Unique node identifier (e.g., 'node-a', 'server-001')"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["simple", "periodic", "write", "all"],
        default="simple",
        help="Which scenario to run (default: simple)"
    )

    args = parser.parse_args()

    print("="*70)
    print(f"ChronoTick Distributed Synchronization Demo")
    print(f"Node ID: {args.node_id}")
    print(f"Hostname: {socket.gethostname()}")
    print("="*70)

    try:
        if args.scenario == "simple" or args.scenario == "all":
            scenario_1_simple_sync(args.node_id)

        if args.scenario == "periodic" or args.scenario == "all":
            if args.scenario == "all":
                time.sleep(2)  # Brief pause between scenarios
            scenario_2_periodic_sync(args.node_id)

        if args.scenario == "write" or args.scenario == "all":
            if args.scenario == "all":
                time.sleep(2)  # Brief pause between scenarios
            scenario_3_coordinated_write(args.node_id)

        print("\n" + "="*70)
        print("✅ All scenarios completed successfully!")
        print("="*70)

    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure ChronoTick daemon is running:")
        print("  chronotick-daemon")
        return 1

    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted by user")
        return 130

    return 0


if __name__ == "__main__":
    exit(main())
