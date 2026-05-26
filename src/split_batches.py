"""Read tiles.geojson + emit the matrix specification for GH Actions.

The bake workflow distributes work across N parallel GH Actions runners
via `strategy.matrix.batch_idx: [0, 1, ..., N-1]`. This script computes
N from total-limit / batch-size and writes it (plus the release tag) as
JSON to stdout in a shape that the prep job can pipe into
GITHUB_OUTPUT.

Usage:
  python3 src/split_batches.py --limit 1000 --batch-size 50
  → prints {"batches": [0, 1, 2, ..., 19], "n_tiles": 1000, "n_batches": 20}
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default="data/tiles.geojson",
                   help="GeoJSON of all candidate tiles (sorted by importance)")
    p.add_argument("--limit", type=int, default=1000,
                   help="Top-N tiles to bake")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Tiles per matrix job (smaller=more parallelism)")
    p.add_argument("--admin1", default="",
                   help="Comma-separated admin1 codes to include (e.g. "
                        "'WLS,SCT'). Empty = all admin1 codes. Filter "
                        "applied before --limit so 'top-N within W+S' works.")
    args = p.parse_args()

    tiles = json.loads(Path(args.tiles).read_text())["features"]
    if args.admin1:
        wanted = {c.strip().upper() for c in args.admin1.split(",") if c.strip()}
        tiles = [t for t in tiles
                 if (t.get("properties") or {}).get("admin1", "") in wanted]
    n_tiles = min(len(tiles), args.limit)
    n_batches = (n_tiles + args.batch_size - 1) // args.batch_size

    if n_batches > 256:
        print(f"!! {n_batches} batches exceeds GH Actions matrix cap (256). "
              f"Increase --batch-size.", file=sys.stderr)
        return 1

    out = {
        "batches": list(range(n_batches)),
        "n_tiles": n_tiles,
        "n_batches": n_batches,
        "batch_size": args.batch_size,
    }
    print(json.dumps(out, separators=(',', ':')))
    return 0


if __name__ == "__main__":
    sys.exit(main())
