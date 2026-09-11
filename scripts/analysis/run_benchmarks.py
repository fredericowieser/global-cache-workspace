"""Compatibility entrypoint for ``gcw eval gsm8k-reproduction``."""

from __future__ import annotations

import sys

from gcw.cli import main

if __name__ == "__main__":
    main(["eval", "gsm8k-reproduction", *sys.argv[1:]])
