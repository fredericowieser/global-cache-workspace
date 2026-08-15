from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("producer", "resumer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    producer_pid = None
    if args.role == "resumer":
        producer = json.loads(sys.stdin.buffer.read())
        producer_pid = int(producer["pid"])
        if producer_pid == os.getpid():
            raise RuntimeError("resumer reused producer PID")
    payload = {
        "role": args.role,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "producer_pid": producer_pid,
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
