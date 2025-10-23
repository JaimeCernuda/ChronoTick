#!/usr/bin/env python3
"""Test NTP connectivity to proxy server"""
import socket
import struct
import sys
import time

def test_ntp_connection(server_ip, port=123, timeout=5):
    """Test NTP connection to a server"""
    try:
        print(f"\n{'='*60}")
        print(f"Testing NTP connectivity to {server_ip}:{port}")
        print(f"{'='*60}")

        # Create NTP request packet (48 bytes, mode 3 = client)
        ntp_packet = b'\x1b' + 47 * b'\0'

        # Create UDP socket
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(timeout)

        # Send request
        print(f"[1] Sending NTP request to {server_ip}:{port}...")
        send_time = time.time()
        client.sendto(ntp_packet, (server_ip, port))
        print(f"✓ NTP request sent successfully")

        # Receive response
        print(f"[2] Waiting for NTP response (timeout: {timeout}s)...")
        try:
            response, address = client.recvfrom(1024)
            recv_time = time.time()
            rtt_ms = (recv_time - send_time) * 1000

            print(f"✓ Received {len(response)} bytes from {address}")
            print(f"✓ Round-trip time: {rtt_ms:.2f} ms")

            if len(response) >= 48:
                # Parse NTP response (simplified)
                unpacked = struct.unpack('!12I', response[0:48])
                print(f"✓ Valid NTP response received")
                print(f"\n{'='*60}")
                print(f"SUCCESS: NTP proxy at {server_ip}:{port} is reachable!")
                print(f"{'='*60}\n")
                return True
            else:
                print(f"⚠️  Response too short ({len(response)} bytes)")
                return False

        except socket.timeout:
            print(f"✗ TIMEOUT: No response received within {timeout}s")
            print(f"⚠️  This means the proxy is NOT reachable from this node")
            return False

    except Exception as e:
        print(f"✗ ERROR: {e}")
        print(f"⚠️  Failed to connect to NTP proxy")
        return False
    finally:
        client.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_ntp_connectivity.py <proxy_ip> [port]")
        print("  port defaults to 123 if not specified")
        sys.exit(1)

    proxy_ip = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 123

    # Get local hostname/IP
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "unknown"

    print(f"\nLocal machine: {hostname} ({local_ip})")

    # Test connectivity
    success = test_ntp_connection(proxy_ip, port)

    sys.exit(0 if success else 1)
