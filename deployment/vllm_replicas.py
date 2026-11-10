"""Run one independent vLLM server per visible GPU as a single supervised service unit.

The platform starts this as one owned unit and sets CUDA_VISIBLE_DEVICES from the
service's GPU list; replica i serves on the i-th port of --ports with the i-th GPU.
If any replica exits, the others are stopped and the unit fails, so the platform never
reports a partially running planner pool as ready.

    python deployment/vllm_replicas.py --ports 18110,18111,18112 -- vllm serve MODEL ...
"""
import argparse
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    ports = args.ports.split(",")
    if not command or "--port" in command or len(gpus) != len(ports):
        raise SystemExit("Give a command without --port and exactly one port per visible GPU")
    replicas = [
        subprocess.Popen(command + ["--port", port], env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        for gpu, port in zip(gpus, ports)
    ]
    while all(r.poll() is None for r in replicas):
        time.sleep(2)
    for r in replicas:
        if r.poll() is None:
            r.terminate()
    for r in replicas:
        r.wait()
    sys.exit(1)


if __name__ == "__main__":
    main()
