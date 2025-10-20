# ARES NTP Proxy Configuration

## Problem Statement

ARES compute nodes (ares-comp-11, ares-comp-12) **cannot resolve DNS names** for public NTP servers. This causes ChronoTick validation to completely fail with:
```
⚠️  NTP query failed: [Errno -3] Temporary failure in name resolution
```

## Network Topology

### ARES Head Node (ares.ares.local)
- **Public IP**: 216.47.152.168/24 (eno1)
- **Internal Networks**:
  - 172.20.1.1/16 (eno2) - **Primary network for compute nodes**
  - 172.25.1.1/16 (ens1np0) - Secondary network
  - 172.29.1.1/16 (eno2)
  - 172.30.1.1/16 (eno2)

### ARES Compute Nodes
**ares-comp-11**:
- 172.20.101.11/16 (eno1) - **Primary network**
- 172.25.101.11/16 (enp47s0np0)

**ares-comp-12**:
- 172.20.101.12/16 (eno1) - **Primary network**
- 172.25.101.12/16 (enp47s0np0)

## NTP Proxy Setup

### Proxy Location
- **Running on**: ARES head node (ares.ares.local)
- **Process**: `python3 ntp_proxy.py --config ntp_proxy_config.yaml`
- **PID file**: `/tmp/ntp_proxy.pid`
- **Log file**: `/tmp/ntp_proxy.log`
- **Port**: UDP 123 (NTP standard port)

### Verified Connectivity

✅ **VERIFIED** on 2025-10-19 using `/mnt/common/jcernudagarcia/test_ntp_connectivity.py`:

**From ares-comp-11**:
- NTP proxy at **172.20.1.1:123** is reachable
- Round-trip time: **0.58 ms**
- Valid NTP response received

**From ares-comp-12**:
- NTP proxy at **172.20.1.1:123** is reachable
- Round-trip time: **0.52 ms**
- Valid NTP response received

## Configuration for ARES Compute Nodes

### ⚠️ CRITICAL: Always use IP address, NEVER use DNS names

**WRONG** (DNS names - will fail on compute nodes):
```yaml
clock_measurement:
  ntp:
    servers:
    - pool.ntp.org
    - time.google.com
    - time.cloudflare.com
    - time.nist.gov
```

**CORRECT** (IP address - works on compute nodes):
```yaml
clock_measurement:
  ntp:
    servers:
    - 172.20.1.1  # NTP proxy on ARES head node
```

### Example Config Files for ARES

When running ChronoTick on ARES compute nodes, use configs that specify the proxy IP:

**`configs/config_ares.yaml`** (or similar):
```yaml
clock_measurement:
  ntp:
    max_acceptable_uncertainty: 0.1
    max_stratum: 16
    measurement_mode: advanced
    min_stratum: 1
    servers:
    - 172.20.1.1  # ARES head node NTP proxy
    timeout_seconds: 2.0
    outlier_window_size: 20
    outlier_sigma_threshold: 3.0
  timing:
    normal_operation:
      measurement_interval: 120
    warm_up:
      duration_seconds: 60
      measurement_interval: 1
```

## Verification Checklist

Before claiming ARES tests are working, **ALWAYS** verify:

1. ✅ NTP proxy is running on head node:
   ```bash
   ssh ares "ps aux | grep ntp_proxy | grep -v grep"
   ssh ares "ss -tulpn | grep :123"
   ```

2. ✅ Compute nodes can reach the proxy:
   ```bash
   ssh ares "ssh ares-comp-11 'python3 /mnt/common/jcernudagarcia/test_ntp_connectivity.py 172.20.1.1'"
   ssh ares "ssh ares-comp-12 'python3 /mnt/common/jcernudagarcia/test_ntp_connectivity.py 172.20.1.1'"
   ```

3. ✅ ChronoTick configs use IP address (not DNS):
   ```bash
   grep -A 10 "servers:" configs/config_ares*.yaml
   # Should show: - 172.20.1.1
   # Should NOT show: - pool.ntp.org or other DNS names
   ```

4. ✅ Check logs for actual NTP measurements (not just process running):
   ```bash
   ssh ares "ssh ares-comp-11 'tail -100 /tmp/ares_comp11_*.log | grep \"NTP measurement\"'"
   # Should show successful measurements, NOT DNS errors
   ```

## Common Mistakes to Avoid

### ❌ Mistake #1: Using DNS names instead of IP
**Symptom**: `[Errno -3] Temporary failure in name resolution`
**Fix**: Change `pool.ntp.org` → `172.20.1.1` in config

### ❌ Mistake #2: Claiming tests work without checking logs
**Symptom**: 8-hour test with 0 NTP measurements, 692 total samples
**Fix**: Always check logs for `NTP measurement` entries, not just process PID

### ❌ Mistake #3: Assuming /tmp files are shared
**Symptom**: "No such file or directory" when running scripts on compute nodes
**Fix**: Use `/mnt/common/jcernudagarcia/` for shared files across nodes

### ❌ Mistake #4: Testing from head node instead of compute nodes
**Symptom**: Tests pass on head node but fail on compute nodes
**Fix**: Always test from actual compute nodes where ChronoTick will run

## Testing Script

The connectivity test script is available at:
- **Location**: `/mnt/common/jcernudagarcia/test_ntp_connectivity.py`
- **Usage**: `python3 test_ntp_connectivity.py <proxy_ip>`
- **Example**: `python3 test_ntp_connectivity.py 172.20.1.1`

## History

This has failed **3 times** previously due to:
1. Using DNS names instead of IP addresses in configs
2. Not verifying actual NTP measurements in logs
3. Claiming tests work when only checking if process exists

This documentation was created on **2025-10-19** after the third failure.

## Related Files

- NTP proxy script: `/mnt/common/jcernudagarcia/ChronoTick/ntp_proxy.py`
- NTP proxy config: `/mnt/common/jcernudagarcia/ChronoTick/ntp_proxy_config.yaml`
- Connectivity test: `/mnt/common/jcernudagarcia/test_ntp_connectivity.py`
- ARES config examples: `server/configs/config_ares*.yaml`
