# ChronoTick Evaluation 1: Correctness & Accuracy

This directory contains the complete implementation for Evaluation 1 of the ChronoTick system, focusing on correctness and accuracy validation with real-world datasets.

## Overview

Evaluation 1 validates ChronoTick's predictive synchronization accuracy against ground truth timing data across three environments:
- **Local**: AMD GPU machine (thermal drift patterns)
- **ARES**: HPC cluster nodes (network jitter, load variations)
- **Chameleon**: Cloud GPU nodes (virtualization effects)

## Directory Structure

```
eval1/
├── dataset_collection/          # Data collection scripts
│   ├── collect_local.py         # Local AMD GPU machine
│   ├── collect_ares.py          # ARES HPC cluster
│   ├── collect_chameleon.py     # Chameleon Cloud
│   ├── collect_config.yaml      # Collection configuration
│   └── utils.py                 # Shared utilities
├── datasets/                    # Collected and processed data
│   ├── raw/                     # Raw measurement files
│   ├── processed/               # Processed datasets
│   └── ground_truth/            # Ground truth references
├── metrics/                     # Analysis modules
│   ├── accuracy_metrics.py      # MAE, RMSE calculations
│   ├── uncertainty_metrics.py   # Uncertainty calibration
├── experiments/                 # Experimental protocols
│   ├── replay_dataset.py        # Historical data replay
│   └── model_comparison.py      # Foundation model comparison
├── analysis/                    # Results and reports
├── deployment/                  # Deployment scripts
└── README.md                    # This file
```

## Quick Start

### 1. Test Basic Functionality

```bash
cd eval1/
python3 test_basic_collection.py
```

This verifies NTP connectivity, system metrics, and data collection work correctly.

### 2. Local Machine Collection

```bash
# Short test (1 hour)
cd dataset_collection/
python3 collect_local.py --duration 1 --output ../datasets/raw/local_test

# Full collection (7 days)
python3 collect_local.py --duration 168 --output ../datasets/raw/local_full
```

### 3. ARES HPC Collection

**Prerequisites:**
- Direct node access on ARES cluster
- Adequate storage space on node or shared filesystem

```bash
# SSH to ARES node
ssh your_username@ares-login.domain

# Transfer evaluation code
scp -r eval1/ ares-node:/path/to/chronotick/eval/

# On ARES node:
cd /path/to/chronotick/eval/eval1/dataset_collection/
python3 collect_ares.py --duration 168 --output ../datasets/raw/ares_node1
```

### 4. Chameleon Cloud Collection

**Prerequisites:**
- Active Chameleon Cloud allocation
- GPU-enabled instance if testing thermal effects

```bash
# SSH to Chameleon instance
ssh cc@your-instance.chameleoncloud.org

# Install dependencies and transfer code
# On Chameleon instance:
cd /home/cc/chronotick/eval/eval1/dataset_collection/
python3 collect_chameleon.py --duration 168 --output ../datasets/raw/chameleon_gpu1
```

## Data Collection Details

### Configuration

Edit `dataset_collection/collect_config.yaml` to customize:

```yaml
collection:
  duration_hours: 168        # 7 days
  sampling_interval_seconds: 1.0
  backup_interval_hours: 6

ntp:
  servers:
    - "pool.ntp.org"
    - "time.google.com"
    - "time.cloudflare.com"
  timeout_seconds: 2.0
  max_acceptable_uncertainty: 0.010  # 10ms
```

### Output Format

All collectors generate CSV files with the following columns:

- `timestamp`: Unix timestamp
- `node_id`: Unique node identifier
- `clock_offset`: NTP-measured offset (seconds)
- `drift_rate`: Calculated drift rate (seconds/second)
- `ntp_delay`: NTP round-trip delay (seconds)
- `ntp_stratum`: NTP server stratum level
- `ntp_server`: NTP server used
- `cpu_temp`: CPU temperature (°C)
- `gpu_temp`: GPU temperature (°C, if available)
- `cpu_freq`: CPU frequency (MHz)
- `cpu_load`: CPU load percentage
- `memory_usage`: Memory usage percentage
- `network_latency`: Network latency (ms)
- `ground_truth_offset`: Reference time offset (seconds)
- `measurement_uncertainty`: Measurement uncertainty (seconds)
- `source_type`: Data source identifier
- `quality_flags`: Quality control flags

### Data Quality

Quality control is applied automatically:
- NTP measurements with >10ms uncertainty are flagged
- Clock offsets >10 seconds are flagged
- High drift rates (>1ms/s) are flagged
- Environment-specific issues (GPU thermal throttling, network problems) are flagged

## Metrics and Analysis

### Accuracy Metrics

```bash
cd metrics/
python3 accuracy_metrics.py \
  --dataset ../datasets/raw/local_test.csv.gz \
  --predictions ../experiments/predictions.npy \
  --output ../analysis/accuracy_results/
```

Calculates:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Percentile errors (50th, 95th, 99th)
- Error distribution analysis
- Improvement vs NTP baseline

### Uncertainty Metrics

```bash
python3 uncertainty_metrics.py \
  --dataset ../datasets/raw/local_test.csv.gz \
  --predictions ../experiments/predictions.npy \
  --uncertainties ../experiments/uncertainties.npy \
  --output ../analysis/uncertainty_results/
```

Calculates:
- Coverage probability (95% CI should contain true value 95% of time)
- Uncertainty calibration metrics
- Uncertainty-error correlation
- Uncertainty reduction vs baseline

### Dataset Replay Experiments

```bash
cd experiments/
python3 replay_dataset.py \
  --dataset ../datasets/raw/local_test.csv.gz \
  --output ../analysis/replay_results/ \
  --model chronos \
  --warmup 300
```

Tests different foundation models:
- `chronos`: Amazon Chronos-Bolt
- `timesfm`: Google TimesFM 2.0
- `ttm`: IBM Tiny Time Mixer

## Expected Results

Based on the ChronoTick paper claims:

### Accuracy Targets
- **MAE**: <10μs after 180s warmup
- **95th percentile error**: <50μs
- **Improvement vs NTP**: 86% uncertainty reduction

### Coverage Targets
- **95% confidence intervals**: Should contain true value 95% ± 2% of time
- **Uncertainty correlation**: >0.7 correlation between predicted uncertainty and actual error

### Performance Targets
- **Response time**: <1ms for cached predictions
- **Warmup time**: <180s to achieve target accuracy
- **Memory usage**: <200MB per model

## Troubleshooting

### Collection Issues

**"Permission denied" for temperature sensors:**
```bash
# Check available thermal zones
ls -la /sys/class/thermal/
# May need to run with elevated privileges for some sensors
```

**NTP timeouts:**
```bash
# Test NTP connectivity
python3 -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(5)
sock.sendto(b'\\x1b' + b'\\x00'*47, ('pool.ntp.org', 123))
print('NTP accessible')
"
```

**Storage space issues:**
```bash
# Monitor file sizes
du -h datasets/raw/
# Configure smaller max_file_size_mb in config
```

### Analysis Issues

**Missing dependencies:**
```bash
# Install required packages
pip install --user numpy pandas matplotlib seaborn scipy PyYAML psutil requests
# Or use system packages:
sudo apt install python3-numpy python3-pandas python3-matplotlib python3-seaborn python3-scipy python3-yaml python3-psutil python3-requests
```

**Memory issues with large datasets:**
```bash
# Process datasets in chunks
python3 -c "
import pandas as pd
chunk_size = 10000
for chunk in pd.read_csv('large_dataset.csv.gz', chunksize=chunk_size):
    # Process chunk
    pass
"
```

## Advanced Usage

### Multi-Node ARES Collection

```bash
# Create node list
echo "ares-compute-{001..010}" > nodes.txt

# Deploy to all nodes
for node in $(cat nodes.txt); do
  scp -r eval1/ $node:/tmp/chronotick_eval/
  ssh $node "cd /tmp/chronotick_eval/eval1/dataset_collection && \
    python3 collect_ares.py --duration 168 --node-id $node \
    --output /shared/storage/datasets/raw/$node/" &
done
wait
```

### Custom Ground Truth

To use ChronoTick itself as ground truth (for relative accuracy testing):

```yaml
# In collect_config.yaml
ground_truth:
  method: "chronotick"  # Instead of "ntp_ensemble"
```

### Continuous Collection

```bash
# Start collection with automatic restart
while true; do
  python3 collect_local.py --duration 168 --output ../datasets/raw/local_continuous_$(date +%Y%m%d)
  sleep 60  # 1 minute break between collection periods
done
```

## Results Integration

All results are designed to integrate with the paper's evaluation section:

- **CSV outputs** for statistical analysis in R/Python
- **JSON reports** for automated metric extraction
- **PNG plots** for paper figures
- **Numpy arrays** for further processing

The evaluation framework supports the paper's claims about ChronoTick achieving microsecond-precision coordination with 86% uncertainty reduction compared to reactive NTP approaches.