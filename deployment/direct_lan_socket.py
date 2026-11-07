#!/usr/bin/env python3
"""OpenSSH ProxyCommand that binds its TCP socket to an explicit LAN interface.

No fallback route: a missing/unusable interface fails rather than using a VPN.
Usage: ssh -o 'ProxyCommand=python3 THIS_FILE --interface WLAN %h %p' HOST
"""
import argparse
import os
import select
import socket
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', required=True)
    parser.add_argument('host')
    parser.add_argument('port', type=int)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, args.interface.encode()+b'\0')
        sock.settimeout(10)
        sock.connect((args.host, args.port))
        sock.settimeout(None)
        while True:
            ready, _, _ = select.select([sock, 0], [], [])
            if sock in ready:
                data = sock.recv(65536)
                if not data:
                    return
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            if 0 in ready:
                data = os.read(0, 65536)
                if not data:
                    return
                sock.sendall(data)


if __name__ == '__main__':
    main()
