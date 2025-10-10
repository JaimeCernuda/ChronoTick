# ChronoTick Evaluation 1 - Deployment Instructions

This document provides detailed deployment instructions for running Evaluation 1 across all three target environments: local machine, ARES HPC cluster, and Chameleon Cloud.

## Prerequisites

### All Environments
- Python 3.8+ with standard library
- Network connectivity for NTP queries
- Sufficient storage space (≥1GB for 7-day collection)
- Basic system monitoring access (CPU, memory)

### Local Machine (AMD GPU)
- AMD GPU with thermal monitoring (optional)
- Administrative access for temperature sensors (optional)
- Stable power and cooling for 7-day runs

### ARES HPC Cluster
- Active ARES allocation and login credentials
- Direct node access (not just login nodes)
- Access to shared storage or node-local storage
- InfiniBand and parallel filesystem access (optional)

### Chameleon Cloud
- Active Chameleon Cloud project
- GPU-enabled instance for thermal drift testing
- OpenStack CLI access (optional)
- Instance with ≥4GB RAM, ≥20GB storage

## Installation

### Method 1: Minimal Dependencies (Recommended)

The evaluation system is designed to work with Python standard library only for basic functionality:

```bash
# Clone/copy the evaluation code
git clone https://github.com/JaimeCernuda/ChronoTick.git
cd ChronoTick/eval/eval1/

# Test basic functionality (no extra dependencies)
python3 test_basic_collection.py
```

### Method 2: Full Dependencies (For Analysis)

If you need the full analysis capabilities:

```bash
# Using pip (may require virtual environment)
pip install -r requirements.txt

# Using system packages (Ubuntu/Debian)
sudo apt update
sudo apt install python3-numpy python3-pandas python3-matplotlib \
                 python3-seaborn python3-scipy python3-yaml \
                 python3-psutil python3-requests

# Using conda
conda install numpy pandas matplotlib seaborn scipy pyyaml psutil requests
```

## Deployment Guide

### Local Machine Deployment

#### 1. Preparation

```bash
# Create deployment directory
mkdir -p ~/chronotick_eval
cd ~/chronotick_eval

# Copy evaluation code
cp -r /path/to/ChronoTick/eval/eval1 .

# Test system
cd eval1/
python3 test_basic_collection.py
```

#### 2. Configuration

Edit `dataset_collection/collect_config.yaml`:

```yaml
environments:
  local:
    node_id: "local-amd-$(hostname)"
    gpu_monitoring: true
    thermal_stress_test: false  # Set true for GPU load testing

collection:
  duration_hours: 168  # 7 days
  sampling_interval_seconds: 1.0
  backup_interval_hours: 6
```

#### 3. Execution

```bash
# Short test run (1 hour)
cd dataset_collection/
python3 collect_local.py --duration 1 --output ../datasets/raw/local_test

# Production run (7 days) - use screen/tmux for persistence
screen -S chronotick_eval
python3 collect_local.py --duration 168 --output ../datasets/raw/local_full
# Ctrl+A, D to detach
```

#### 4. Monitoring

```bash
# Check progress
screen -r chronotick_eval

# Monitor output files
ls -lh datasets/raw/local_*/
tail -f datasets/raw/local_*/progress_backup_*.json

# Monitor system resources
htop
```

### ARES HPC Cluster Deployment

#### 1. Access and Transfer

```bash
# SSH to ARES login node
ssh your_username@ares-login.ncsa.illinois.edu

# Create working directory
mkdir -p /scratch/users/$USER/chronotick_eval
cd /scratch/users/$USER/chronotick_eval

# Transfer evaluation code (from local machine)
scp -r ChronoTick/eval/eval1 $USER@ares-login.ncsa.illinois.edu:/scratch/users/$USER/chronotick_eval/
```

#### 2. Node Selection and Access

```bash
# Check available compute nodes
sinfo -p normal
salloc -p normal -N 1 -t 168:00:00  # Request 7-day allocation

# Once allocated, SSH to compute node
ssh ares-compute-XXX

# Navigate to evaluation directory
cd /scratch/users/$USER/chronotick_eval/eval1
```

#### 3. HPC-Specific Configuration

Edit `dataset_collection/collect_config.yaml`:

```yaml
environments:
  ares:
    node_id_prefix: "ares-compute"
    pfs_monitoring: true
    infiniband_monitoring: true

collection:
  duration_hours: 168
  max_file_size_mb: 50  # Smaller files for parallel filesystem
  backup_interval_hours: 4  # More frequent backups
```

#### 4. Execution

```bash
# Test basic functionality
python3 test_basic_collection.py

# Start collection
cd dataset_collection/
nohup python3 collect_ares.py --duration 168 \
  --output /scratch/users/$USER/chronotick_eval/datasets/raw/ares_$(hostname) \
  --log-file ../logs/collection_$(hostname).log &

# Monitor progress
tail -f ../logs/collection_$(hostname).log
```

#### 5. Multi-Node Collection (Optional)

```bash
# Create SLURM batch script
cat > chronotick_multi_node.slurm << 'EOF'
#!/bin/bash
#SBATCH -J chronotick_eval
#SBATCH -p normal
#SBATCH -N 10
#SBATCH -t 168:00:00
#SBATCH -o chronotick_%j.out
#SBATCH -e chronotick_%j.err

# Run on each allocated node
srun --ntasks-per-node=1 bash -c '
cd /scratch/users/$USER/chronotick_eval/eval1/dataset_collection
python3 collect_ares.py --duration 168 \
  --output ../datasets/raw/ares_$(hostname) \
  --node-id ares-$(hostname) &
wait
'
EOF

# Submit job
sbatch chronotick_multi_node.slurm
```

### Chameleon Cloud Deployment

#### 1. Instance Creation

```bash
# Using OpenStack CLI (if available)
openstack server create --flavor m1.large --image CC-Ubuntu20.04 \
  --key-name your-key --network sharednet1 chronotick-eval-1

# Or use Chameleon web interface:
# 1. Login to Chameleon Dashboard
# 2. Launch Instance > m1.large or GPU flavor
# 3. Select Ubuntu 20.04+ image
# 4. Add your SSH key
```

#### 2. Instance Access and Setup

```bash
# SSH to instance
ssh cc@your-instance.chameleoncloud.org

# Update system
sudo apt update
sudo apt install python3-full git

# Create working directory
mkdir -p ~/chronotick_eval
cd ~/chronotick_eval

# Transfer evaluation code
scp -r your-local-machine:ChronoTick/eval/eval1 .
```

#### 3. Cloud-Specific Configuration

Edit `dataset_collection/collect_config.yaml`:

```yaml
environments:
  chameleon:
    node_id_prefix: "chameleon-gpu"
    gpu_monitoring: true
    openstack_metadata: true

system_metrics:
  network_latency: true  # Important for cloud environments
  disk_io: true         # Monitor cloud storage performance
```

#### 4. Execution

```bash
cd eval1/

# Test cloud-specific features
python3 test_basic_collection.py

# Start collection with cloud optimizations
cd dataset_collection/
screen -S chronotick_collection
python3 collect_chameleon.py --duration 168 \
  --output ../datasets/raw/chameleon_gpu1 \
  --log-level INFO
# Ctrl+A, D to detach
```

#### 5. GPU Workload Testing (Optional)

```bash
# Install CUDA/GPU tools if testing thermal effects
sudo apt install nvidia-utils-470  # or appropriate version

# Start GPU stress test
nvidia-smi -lgc 1200  # Lock GPU clock for consistent thermal load

# Monitor GPU temperature during collection
watch -n 1 nvidia-smi
```

## Data Collection Management

### Starting Collections

```bash
# Local machine
python3 collect_local.py --duration 168 --output ../datasets/raw/local_$(date +%Y%m%d)

# ARES HPC
python3 collect_ares.py --duration 168 --output ../datasets/raw/ares_$(hostname)_$(date +%Y%m%d)

# Chameleon Cloud
python3 collect_chameleon.py --duration 168 --output ../datasets/raw/chameleon_$(hostname)_$(date +%Y%m%d)
```

### Monitoring Progress

```bash
# Check collection status
ls -lh datasets/raw/*/

# View recent measurements
zcat datasets/raw/*/chronotick_eval1_*.csv.gz | tail -10

# Monitor backup files
cat datasets/raw/*/progress_backup_*.json | jq .
```

### Stopping Collections

```bash
# Graceful stop (Ctrl+C)
# Collections automatically save progress and cleanup

# Force stop if needed
pkill -f collect_local.py
pkill -f collect_ares.py
pkill -f collect_chameleon.py
```

## Data Transfer and Aggregation

### From ARES to Local

```bash
# Compress data for transfer
cd /scratch/users/$USER/chronotick_eval/eval1/datasets/raw/
tar -czf ares_collection_$(date +%Y%m%d).tar.gz ares_*/

# Transfer to local machine
scp ares_collection_*.tar.gz your-local:~/chronotick_results/

# On local machine
cd ~/chronotick_results/
tar -xzf ares_collection_*.tar.gz
```

### From Chameleon to Local

```bash
# On Chameleon instance
cd ~/chronotick_eval/eval1/datasets/raw/
tar -czf chameleon_collection_$(date +%Y%m%d).tar.gz chameleon_*/

# Transfer to local machine
scp chameleon_collection_*.tar.gz your-local:~/chronotick_results/
```

### Data Aggregation

```bash
# Combine all datasets
cd ~/chronotick_results/
mkdir -p eval1_combined/datasets/raw/

# Copy all collections
cp -r local_*/ eval1_combined/datasets/raw/
cp -r ares_*/ eval1_combined/datasets/raw/
cp -r chameleon_*/ eval1_combined/datasets/raw/

# Create metadata
echo "Collection completed: $(date)" > eval1_combined/collection_metadata.txt
echo "Environments: local, ares, chameleon" >> eval1_combined/collection_metadata.txt
du -sh eval1_combined/datasets/raw/*/ >> eval1_combined/collection_metadata.txt
```

## Troubleshooting

### Network Issues

```bash
# Test NTP connectivity
for server in pool.ntp.org time.google.com time.cloudflare.com; do
  echo "Testing $server..."
  timeout 5 python3 -c "
import socket, struct, time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2)
packet = b'\\x1b' + b'\\x00'*47
sock.sendto(packet, ('$server', 123))
response = sock.recv(1024)
print('Success:', len(response), 'bytes')
"
done
```

### Storage Issues

```bash
# Check available space
df -h

# Monitor collection size
du -sh datasets/raw/*/

# Clean up if needed
find datasets/raw/ -name "*.csv.gz" -mtime +7 -exec ls -lh {} \;
```

### Permission Issues

```bash
# Check file permissions
ls -la dataset_collection/collect_*.py

# Make executable if needed
chmod +x dataset_collection/collect_*.py

# Check system access
python3 -c "
import os
print('Thermal zones:', os.listdir('/sys/class/thermal/') if os.path.exists('/sys/class/thermal/') else 'None')
print('GPU devices:', os.listdir('/dev/') if any('nvidia' in f for f in os.listdir('/dev/')) else 'None')
"
```

### Performance Issues

```bash
# Monitor system resources during collection
htop

# Check collection performance
tail -f logs/collection_*.log | grep "Progress:"

# Adjust collection interval if needed (in config)
sampling_interval_seconds: 2.0  # Reduce from 1.0 if CPU limited
```

## Validation

### Post-Collection Validation

```bash
# Check data integrity
cd datasets/raw/
for dir in */; do
  echo "Checking $dir..."
  count=$(zcat $dir/*.csv.gz | wc -l)
  echo "  Total measurements: $count"

  # Should be ~604,800 for 7 days at 1Hz (plus header)
  if [ $count -gt 600000 ]; then
    echo "  ✓ Complete collection"
  else
    echo "  ⚠ Partial collection"
  fi
done
```

### Data Quality Checks

```bash
# Run quality analysis
python3 -c "
import pandas as pd
import sys

for dataset in sys.argv[1:]:
    print(f'Analyzing {dataset}...')
    df = pd.read_csv(dataset, compression='gzip' if dataset.endswith('.gz') else None)

    print(f'  Total measurements: {len(df)}')
    print(f'  Time span: {(df.timestamp.max() - df.timestamp.min()) / 3600:.1f} hours')
    print(f'  Success rate: {len(df[df.quality_flags.str.len() == 0]) / len(df) * 100:.1f}%')
    print(f'  Median offset: {df.clock_offset.median() * 1e6:.1f} μs')
    print()
" datasets/raw/*/chronotick_eval1_*.csv.gz
```

This completes the comprehensive deployment instructions for ChronoTick Evaluation 1 across all target environments.