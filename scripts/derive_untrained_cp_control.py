from __future__ import annotations

import argparse
import json
from pathlib import Path

from gcw.opsd_controls import derive_untrained_processor_control


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a seed-identical FP32 untrained cache-processor control checkpoint."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--tokenizer-source", required=True)
    parser.add_argument("--repository", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    report = derive_untrained_processor_control(
        args.source,
        args.destination,
        seed=args.seed,
        source_revision=args.source_revision,
        tokenizer_source=args.tokenizer_source,
        repository=args.repository,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
