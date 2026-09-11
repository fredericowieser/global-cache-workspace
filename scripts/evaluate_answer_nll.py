"""Compatibility entrypoint for ``gcw eval answer-nll``."""

from __future__ import annotations

import sys

from gcw.cli import main

if __name__ == "__main__":
    main(["eval", "answer-nll", *sys.argv[1:]])
