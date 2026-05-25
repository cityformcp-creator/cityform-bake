"""Build manifest.json from the assets uploaded to a release.

Runs after all `bake` matrix jobs complete (in the `manifest` job of
the workflow). Walks the release's assets via `gh release view --json
assets`, extracts each baked tile_id from the asset names (which use
the `<tile_id>__<file>` convention), and writes a single manifest.json
that the storefront picker fetches:

  {
    "release_tag": "bake-2026-05-25-…",
    "generated_at": "2026-05-25T20:15:00Z",
    "tile_count": 982,
    "base_url": "https://github.com/<user>/cityform-bake/releases/latest/download",
    "tiles": {
      "SK3587": {
        "place": "Sheffield",
        "lat": 53.380, "lng": -1.464,
        "stl": "SK3587__city.stl",
        "glb": "SK3587__city.glb",
        "preview": "SK3587__preview.png",
        "meta": "SK3587__meta.json"
      },
      …
    }
  }

The picker fetches manifest.json once on mount and lazy-loads per-tile
assets via base_url + tile.glb / tile.preview as needed.
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def list_release_assets(release_tag: str, repo: str) -> list[dict]:
    """Run `gh release view --json assets` to get the asset list."""
    args = ["gh", "release", "view", release_tag, "--repo", repo, "--json", "assets"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh release view failed: {result.stderr[:500]}")
    return json.loads(result.stdout).get("assets", [])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default="data/tiles.geojson",
                   help="Source of canonical tile metadata (place, centre coords)")
    p.add_argument("--release-tag", required=True)
    p.add_argument("--repo", required=True,
                   help="owner/repo (e.g. cityformcp-creator/cityform-bake)")
    p.add_argument("--out", default="manifest.json")
    args = p.parse_args()

    # Index tile metadata by tile_id so we can enrich the manifest entries.
    fc = json.loads(Path(args.tiles).read_text())
    tile_meta_by_id = {
        f["properties"]["tile_id"]: f["properties"]
        for f in fc["features"]
    }

    assets = list_release_assets(args.release_tag, args.repo)
    print(f"[manifest] release {args.release_tag!r} has {len(assets)} assets",
          file=sys.stderr)

    # Bucket assets by tile_id (parsed from `<tile_id>__<file>` naming).
    by_tile: dict[str, dict[str, str]] = defaultdict(dict)
    for a in assets:
        name = a.get("name", "")
        if "__" not in name:
            continue
        tile_id, fname = name.split("__", 1)
        if fname == "city.stl":     by_tile[tile_id]["stl"] = name
        elif fname == "city.glb":   by_tile[tile_id]["glb"] = name
        elif fname == "preview.png":by_tile[tile_id]["preview"] = name
        elif fname == "meta.json":  by_tile[tile_id]["meta"] = name

    # Only keep tiles with at least a GLB (the picker needs it).
    complete = {tid: files for tid, files in by_tile.items() if "glb" in files}
    print(f"[manifest] {len(complete)} tiles complete (have GLB), "
          f"{len(by_tile) - len(complete)} partial", file=sys.stderr)

    tiles: dict[str, dict] = {}
    for tile_id, files in sorted(complete.items()):
        meta = tile_meta_by_id.get(tile_id, {})
        tiles[tile_id] = {
            "place": meta.get("place", ""),
            "place_type": meta.get("place_type", ""),
            "lat": meta.get("centre_lat"),
            "lng": meta.get("centre_lng"),
            **files,
        }

    base_url = f"https://github.com/{args.repo}/releases/latest/download"
    manifest = {
        "release_tag": args.release_tag,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tile_count": len(tiles),
        "base_url": base_url,
        "tiles": tiles,
    }

    Path(args.out).write_text(json.dumps(manifest, separators=(',', ':')))
    print(f"[manifest] wrote {args.out} ({len(tiles)} tiles, "
          f"{Path(args.out).stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
