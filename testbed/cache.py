"""``python -m testbed.cache report <path/to/simuser_cache.sqlite>``: C4 hits, misses and simulator hours
against the ceiling (Workbook Ch. 8, command table)."""
from __future__ import annotations

import json
import sys

from .simuser_cache import report


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "report":
        print(__doc__)
        return 2
    print(json.dumps(report(argv[1]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
