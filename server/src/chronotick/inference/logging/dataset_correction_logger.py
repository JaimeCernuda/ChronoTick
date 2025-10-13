#!/usr/bin/env python3
"""
Dataset Correction Logger

Asynchronously logs dataset corrections to show the effect of NTP corrections
on stored historical data. This is separate from what's returned to clients.

Usage:
    logger = DatasetCorrectionLogger("corrections.csv", enabled=True)
    logger.log_correction_event(dataset_manager, ntp_measurement, method, interval_start, interval_end, error)
    logger.close()
"""

import csv
import threading
import queue
import time
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DatasetCorrectionLogger:
    """
    Asynchronously logs dataset corrections to avoid impacting real-time performance.

    Captures:
    - Predictions returned to clients (what they received)
    - Dataset state after corrections (what ML model sees)
    - The delta between them (effect of corrections)
    """

    def __init__(self, csv_path: str, enabled: bool = True, max_queue_size: int = 1000):
        """
        Initialize correction logger.

        Args:
            csv_path: Path to CSV file for logging
            enabled: If False, logging is disabled (no performance impact)
            max_queue_size: Maximum number of queued log entries
        """
        self.enabled = enabled
        if not enabled:
            return

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Async queue for logging
        self.log_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()

        # Start background logging thread
        self.log_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.log_thread.start()

        logger.info(f"Dataset correction logger started: {csv_path}")

    def _logging_worker(self):
        """Background thread that writes logs asynchronously"""
        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'ntp_event_timestamp', 'ntp_offset_ms', 'ntp_uncertainty_ms',
                    'correction_method', 'error_ms',
                    'interval_start', 'interval_end', 'interval_duration_s',
                    'measurement_timestamp', 'time_since_interval_start_s',
                    'offset_before_correction_ms', 'offset_after_correction_ms',
                    'correction_delta_ms', 'was_corrected'
                ])
                f.flush()

                while not self.stop_event.is_set():
                    try:
                        # Get log entry with timeout
                        entry = self.log_queue.get(timeout=0.1)
                        writer.writerow(entry)
                        f.flush()
                        self.log_queue.task_done()
                    except queue.Empty:
                        continue

                # Drain remaining queue
                while not self.log_queue.empty():
                    try:
                        entry = self.log_queue.get_nowait()
                        writer.writerow(entry)
                        self.log_queue.task_done()
                    except queue.Empty:
                        break
                f.flush()

        except Exception as e:
            logger.error(f"Dataset correction logger error: {e}")

    def log_correction_event(self, dataset_manager, ntp_measurement, method: str,
                            interval_start: float, interval_end: float, error: float):
        """
        Log a correction event by capturing dataset state.

        This captures the BEFORE and AFTER state of the dataset to show
        how corrections modify stored historical data.

        Args:
            dataset_manager: DatasetManager instance
            ntp_measurement: NTPMeasurement that triggered the correction
            method: Correction method used ('none', 'linear', 'drift_aware', 'advanced')
            interval_start: Start of correction interval
            interval_end: End of correction interval (NTP timestamp)
            error: Measured error at NTP time (NTP_truth - Prediction)
        """
        if not self.enabled:
            return

        try:
            # Snapshot dataset state AFTER correction
            interval_duration = interval_end - interval_start

            for timestamp in range(int(interval_start), int(interval_end) + 1):
                measurement = dataset_manager.get_measurement_at_time(timestamp)

                if measurement:
                    offset_after = measurement['offset']
                    was_corrected = measurement.get('corrected', False)
                    time_since_start = timestamp - interval_start

                    # Calculate what it was BEFORE based on correction method
                    # This is a reverse calculation - not perfect but gives estimate
                    offset_before = offset_after  # Default (for 'none' or uncorrected)

                    if was_corrected and method != 'none':
                        # Reverse-calculate the correction delta
                        if method == 'linear':
                            # Linear: correction(t) = (t - start) / (end - start) × error
                            alpha = time_since_start / interval_duration if interval_duration > 0 else 0
                            correction_delta = alpha * error
                            offset_before = offset_after - correction_delta

                        elif method == 'drift_aware':
                            # Drift-aware: more complex, use approximate reversal
                            # Would need to store original values for exact reversal
                            # For now, approximate based on linear
                            alpha = time_since_start / interval_duration if interval_duration > 0 else 0
                            correction_delta = alpha * error
                            offset_before = offset_after - correction_delta

                        elif method == 'advanced':
                            # Advanced: variance-weighted, approximate reversal
                            # Exact reversal would need original uncertainty calculations
                            alpha = time_since_start / interval_duration if interval_duration > 0 else 0
                            correction_delta = alpha * error
                            offset_before = offset_after - correction_delta

                    correction_delta_ms = (offset_after - offset_before) * 1000

                    # Queue the log entry (non-blocking)
                    entry = [
                        ntp_measurement.timestamp,
                        ntp_measurement.offset * 1000,
                        ntp_measurement.uncertainty * 1000,
                        method,
                        error * 1000,
                        interval_start,
                        interval_end,
                        interval_duration,
                        timestamp,
                        time_since_start,
                        offset_before * 1000,
                        offset_after * 1000,
                        correction_delta_ms,
                        was_corrected
                    ]

                    try:
                        self.log_queue.put_nowait(entry)
                    except queue.Full:
                        logger.warning("Correction log queue full, dropping entry")

        except Exception as e:
            logger.error(f"Error logging correction event: {e}")

    def close(self):
        """Stop the logging thread and flush remaining entries"""
        if not self.enabled:
            return

        self.stop_event.set()

        # Wait for queue to be processed
        self.log_queue.join()

        # Wait for thread to finish
        if self.log_thread.is_alive():
            self.log_thread.join(timeout=5.0)

        logger.info("Dataset correction logger stopped")


# For better visualization, also log client-returned predictions
class ClientPredictionLogger:
    """
    Logs predictions RETURNED to clients (separate from dataset state).

    This is the "solid line" in the visualization - what users actually received.
    """

    def __init__(self, csv_path: str, enabled: bool = True):
        """
        Initialize client prediction logger.

        Args:
            csv_path: Path to CSV file
            enabled: If False, logging is disabled
        """
        self.enabled = enabled
        if not enabled:
            return

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Header
        self.csv_writer.writerow([
            'timestamp', 'offset_correction_ms', 'drift_rate_us_per_s',
            'offset_uncertainty_ms', 'confidence', 'source'
        ])
        self.csv_file.flush()

        logger.info(f"Client prediction logger started: {csv_path}")

    def log_prediction(self, timestamp: float, offset_correction: float, drift_rate: float,
                      offset_uncertainty: float, confidence: float, source: str):
        """
        Log a prediction that was returned to the client.

        Args:
            timestamp: Time when prediction was made
            offset_correction: Offset correction in seconds
            drift_rate: Drift rate in seconds/second
            offset_uncertainty: Uncertainty in seconds
            confidence: Confidence [0, 1]
            source: Prediction source
        """
        if not self.enabled:
            return

        try:
            self.csv_writer.writerow([
                timestamp,
                offset_correction * 1000,
                drift_rate * 1e6,
                offset_uncertainty * 1000,
                confidence,
                source
            ])
            self.csv_file.flush()
        except Exception as e:
            logger.error(f"Error logging client prediction: {e}")

    def close(self):
        """Close the log file"""
        if not self.enabled:
            return

        self.csv_file.close()
        logger.info("Client prediction logger stopped")
