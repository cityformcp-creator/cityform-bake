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
import subprocess
import sys
from pathlib import Path


def _list_baked_tile_ids(release_tags: list[str]) -> set[str]:
    """Pull asset names from each release via `gh release view` and
    extract the tile_ids that already have at least a GLB uploaded."""
    out: set[str] = set()
    for tag in release_tags:
        try:
            r = subprocess.run(
                ["gh", "release", "view", tag, "--json", "assets"],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"  (skip-baked: could not read {tag}: {exc.stderr[:200]})",
                  file=sys.stderr)
            continue
        for a in json.loads(r.stdout).get("assets", []):
            name = a.get("name", "")
            if name.endswith("__city.glb"):
                out.add(name.split("__", 1)[0])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default="data/tiles.geojson",
                   help="GeoJSON of all candidate tiles (sorted by importance)")
    p.add_argument("--limit", type=int, default=1000,
                   help="Top-N tiles to bake")
    p.add_argument("--offset", type=int, default=0,
                   help="Skip the first N candidates (after admin1/coverage/"
                        "skip-baked filters) before taking --limit. Lets you "
                        "shard a big run across multiple workflow invocations "
                        "without re-baking the front of the list.")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Tiles per matrix job (smaller=more parallelism)")
    p.add_argument("--admin1", default="",
                   help="Comma-separated admin1 codes to include (e.g. "
                        "'WLS,SCT'). Empty = all admin1 codes. Filter "
                        "applied before --limit so 'top-N within W+S' works.")
    p.add_argument("--coverage-index", default="",
                   help="Path to coverage_index.json (from build_coverage_index.py). "
                        "When set, drops tiles whose tile_id is not covered. Skip "
                        "tiles still occupy a batch slot, so always pre-filter rather "
                        "than skip-at-bake when the goal is N successful tiles.")
    p.add_argument("--skip-baked-tags", default="",
                   help="Comma-separated release tags whose tiles are already baked. "
                        "Their tile_ids are dropped from the candidate list. Used to "
                        "stack bakes without re-doing existing tiles.")
    args = p.parse_args()

    tiles = json.loads(Path(args.tiles).read_text())["features"]
    if args.admin1:
        wanted = {c.strip().upper() for c in args.admin1.split(",") if c.strip()}
        tiles = [t for t in tiles
                 if (t.get("properties") or {}).get("admin1", "") in wanted]
    if args.coverage_index:
        idx = json.loads(Path(args.coverage_index).read_text())
        coverage = idx.get("tile_coverage", {})
        before = len(tiles)
        tiles = [t for t in tiles
                 if coverage.get(t["properties"]["tile_id"], {}).get("covered")]
        print(f"  coverage filter: {before} → {len(tiles)} "
              f"(dropped {before - len(tiles)} not-covered)", file=sys.stderr)
    if args.skip_baked_tags:
        tags = [t.strip() for t in args.skip_baked_tags.split(",") if t.strip()]
        already = _list_baked_tile_ids(tags)
        before = len(tiles)
        tiles = [t for t in tiles
                 if t["properties"]["tile_id"] not in already]
        print(f"  skip-baked filter ({len(tags)} tags, {len(already)} ids): "
              f"{before} → {len(tiles)}", file=sys.stderr)
    if args.offset:
        before = len(tiles)
        tiles = tiles[args.offset:]
        print(f"  offset {args.offset}: {before} → {len(tiles)}", file=sys.stderr)
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
