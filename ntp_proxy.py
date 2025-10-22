#!/usr/bin/env python3
"""
Lightweight NTP UDP Proxy for ARES Cluster

Forwards NTP requests from compute nodes (without internet access)
to external NTP servers through the head node.

Features:
- Multi-port support: Each port forwards to a specific NTP server
- Concurrent clients: Supports multiple compute nodes simultaneously
- Lightweight: <10 MB memory, minimal CPU usage
- No sudo required: Runs on high ports (8123+)
- Maintains causality: External NTP servers are ground truth

Usage:
    # Single server mode
    python ntp_proxy.py --listen-port 8123 --ntp-server pool.ntp.org

    # Multi-server mode with config file
    python ntp_proxy.py --config ntp_proxy_config.yaml
"""

import socket
import threading
import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime


def proxy_worker(listen_port: int, ntp_server: str, ntp_port: int = 123, verbose: bool = False):
    """
    Single proxy worker thread that forwards NTP requests

    Args:
        listen_port: Port to listen on (e.g., 8123)
        ntp_server: External NTP server to forward to
        ntp_port: NTP server port (default: 123)
        verbose: Print forwarding activity
    """
    # Create listening socket (bind to all interfaces)
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(('0.0.0.0', listen_port))

    # Create forwarding socket
    forward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    forward_sock.settimeout(5.0)  # 5 second timeout for NTP responses

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] ✓ Proxy port {listen_port} -> {ntp_server}:{ntp_port}")

    request_count = 0
    error_count = 0

    while True:
        try:
            # Receive NTP request from compute node
            data, client_addr = listen_sock.recvfrom(1024)

            if len(data) < 48:  # NTP packets are 48 bytes minimum
                if verbose:
                    print(f"[{listen_port}] ⚠️  Invalid packet size from {client_addr[0]}")
                continue

            # Forward to external NTP server
            forward_sock.sendto(data, (ntp_server, ntp_port))

            # Receive response from NTP server
            response, server_addr = forward_sock.recvfrom(1024)

            # Send response back to compute node
            listen_sock.sendto(response, client_addr)

            request_count += 1

            if verbose:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] [{listen_port}] {client_addr[0]} -> {ntp_server} (#{request_count})")

        except socket.timeout:
            error_count += 1
            if verbose:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] [{listen_port}] ⚠️  Timeout from {ntp_server} (#{error_count})")

        except Exception as e:
            error_count += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{listen_port}] ✗ Error: {e}")


def run_single_proxy(listen_port: int, ntp_server: str, ntp_port: int = 123, verbose: bool = False):
    """Run single proxy worker (blocking)"""
    print(f"\n{'='*60}")
    print(f"NTP UDP Proxy - Single Server Mode")
    print(f"{'='*60}")
    print(f"Listen:  0.0.0.0:{listen_port}")
    print(f"Forward: {ntp_server}:{ntp_port}")
    print(f"{'='*60}\n")

    try:
        proxy_worker(listen_port, ntp_server, ntp_port, verbose)
    except KeyboardInterrupt:
        print("\n✓ Proxy stopped")
        sys.exit(0)


def run_multi_proxy(config_file: str, verbose: bool = False):
    """Run multiple proxy workers from config file"""
    config_path = Path(config_file)

    if not config_path.exists():
        print(f"✗ Config file not found: {config_file}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if 'proxy_mappings' not in config:
        print(f"✗ Invalid config: missing 'proxy_mappings'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"NTP UDP Proxy - Multi-Server Mode")
    print(f"{'='*60}")
    print(f"Config: {config_file}")
    print(f"Servers: {len(config['proxy_mappings'])}")
    print(f"{'='*60}\n")

    threads = []
    for mapping in config['proxy_mappings']:
        listen_port = mapping['listen_port']
        ntp_server = mapping['ntp_server']
        ntp_port = mapping.get('ntp_port', 123)

        t = threading.Thread(
            target=proxy_worker,
            args=(listen_port, ntp_server, ntp_port, verbose),
            daemon=True,
            name=f"NTP-Proxy-{listen_port}"
        )
        t.start()
        threads.append(t)

    print(f"\n✓ Running {len(threads)} proxy workers")
    print(f"  Press Ctrl+C to stop\n")

    # Wait for Ctrl+C
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n✓ Proxy stopped")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight NTP UDP Proxy for ARES Cluster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single server mode
  python ntp_proxy.py --listen-port 8123 --ntp-server pool.ntp.org

  # Multi-server mode with config file
  python ntp_proxy.py --config ntp_proxy_config.yaml

  # Verbose mode
  python ntp_proxy.py --config ntp_proxy_config.yaml --verbose
"""
    )

    # Single server mode
    parser.add_argument('--listen-port', type=int,
                       help='Port to listen on (single server mode)')
    parser.add_argument('--ntp-server', type=str,
                       help='NTP server to forward to (single server mode)')
    parser.add_argument('--ntp-port', type=int, default=123,
                       help='NTP server port (default: 123)')

    # Multi-server mode
    parser.add_argument('--config', type=str,
                       help='Config file with multiple server mappings (YAML)')

    # Common options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print forwarding activity')

    args = parser.parse_args()

    # Validate arguments
    if args.config:
        # Multi-server mode
        run_multi_proxy(args.config, args.verbose)
    elif args.listen_port and args.ntp_server:
        # Single server mode
        run_single_proxy(args.listen_port, args.ntp_server, args.ntp_port, args.verbose)
    else:
        parser.print_help()
        print("\n✗ Error: Must specify either --config OR (--listen-port AND --ntp-server)")
        sys.exit(1)


if __name__ == '__main__':
    main()
