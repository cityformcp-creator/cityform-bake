"""Build manifest.json by walking all bake-* releases.

Runs after all `bake` matrix jobs complete. Walks every release with a
`bake-*` tag prefix (newest first), extracts each baked tile_id from
the `<tile_id>__<file>` asset names, and writes a unified manifest.json
referencing per-tile absolute URLs. Most-recent release wins for tile
collisions — re-baking SK3587 overrides the older entry.

Schema (v2 — multi-release):

  {
    "schema_version": 2,
    "release_tag": "bake-2026-05-26-…",          # the release this manifest uploads to
    "generated_at": "…",
    "tile_count": 982,
    "base_url": "https://github.com/<user>/cityform-bake/releases/download",
    "tiles": {
      "SK3587": {
        "place": "Sheffield",
        "lat": 53.380, "lng": -1.464,
        "release_tag": "bake-2026-05-26-OLD",     # which release holds this tile's assets
        "stl": "SK3587__city.stl",
        "glb": "SK3587__city.glb"
      },
      "NT2773": { "release_tag": "bake-2026-05-26-NEW", … },
      …
    }
  }

Picker URL construction: `${base_url}/${tile.release_tag}/${tile.glb}`.
Falls back to v1 single-release behaviour if `--no-multi-release` is
passed (kept for the legacy bake workflow until that's retired).
"""

from __future__ import annotations
import argparse
import json
import re
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


def list_bake_releases(repo: str) -> list[str]:
    """List all release tags matching the `bake-*` prefix, newest first.

    `gh release list` returns rows; the tag is the 3rd tab-separated
    column. Limit to 50 most recent — keeps the manifest build under a
    few seconds even if the project accumulates many bakes over time.
    """
    args = ["gh", "release", "list", "--repo", repo, "--limit", "50"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh release list failed: {result.stderr[:500]}")
    tags: list[str] = []
    for line in result.stdout.splitlines():
        cols = line.split("\t")
        if len(cols) >= 3 and cols[2].startswith("bake-"):
            tags.append(cols[2])
    # `gh release list` orders by created-at desc — preserve that.
    return tags


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default="data/tiles.geojson",
                   help="Source of canonical tile metadata (place, centre coords)")
    p.add_argument("--release-tag", required=True,
                   help="Tag of the release this manifest uploads to (also "
                        "used as schema's `release_tag` field)")
    p.add_argument("--repo", required=True,
                   help="owner/repo (e.g. cityformcp-creator/cityform-bake)")
    p.add_argument("--out", default="manifest.json")
    p.add_argument("--no-multi-release", action="store_true",
                   help="Disable multi-release walk; only include tiles from "
                        "--release-tag (legacy v1 behaviour). Default: walk "
                        "every bake-* release and merge — needed so a new "
                        "bake doesn't orphan previously-baked tiles.")
    args = p.parse_args()

    # Index tile metadata by tile_id so we can enrich the manifest entries.
    fc = json.loads(Path(args.tiles).read_text())
    tile_meta_by_id = {
        f["properties"]["tile_id"]: f["properties"]
        for f in fc["features"]
    }

    if args.no_multi_release:
        release_tags = [args.release_tag]
    else:
        release_tags = list_bake_releases(args.repo)
        # Ensure the target release is in the list even if `gh release list`
        # hasn't surfaced it yet (race with the create-release step).
        if args.release_tag not in release_tags:
            release_tags.insert(0, args.release_tag)
    print(f"[manifest] walking {len(release_tags)} bake release(s): "
          f"{release_tags[:5]}{'...' if len(release_tags) > 5 else ''}",
          file=sys.stderr)

    # tile_id → (release_tag, {stl: …, glb: …})
    # Iterate releases newest-first; first occurrence wins so re-bakes
    # override older versions.
    by_tile: dict[str, tuple[str, dict[str, str]]] = {}
    for tag in release_tags:
        try:
            assets = list_release_assets(tag, args.repo)
        except RuntimeError as exc:
            print(f"[manifest] skipping {tag}: {exc}", file=sys.stderr)
            continue
        print(f"[manifest]   {tag}: {len(assets)} assets", file=sys.stderr)
        for a in assets:
            name = a.get("name", "")
            if "__" not in name:
                continue
            tile_id, fname = name.split("__", 1)
            if tile_id in by_tile:
                continue   # newer release already claimed this tile
            files: dict[str, str] = {}
            if fname == "city.stl":     files["stl"] = name
            elif fname == "city.glb":   files["glb"] = name
            elif fname == "preview.png":files["preview"] = name
            elif fname == "meta.json":  files["meta"] = name
            else:
                continue
            # We may have seen another file for this tile in the same
            # release iteration — merge file-kinds within the release.
            existing = by_tile.get(tile_id)
            if existing and existing[0] == tag:
                existing[1].update(files)
            else:
                by_tile[tile_id] = (tag, files)

    # Only keep tiles with at least a GLB (the picker needs it for the
    # 3D viewer).
    complete = {tid: (tag, files) for tid, (tag, files) in by_tile.items()
                if "glb" in files}
    print(f"[manifest] {len(complete)} tiles complete (have GLB), "
          f"{len(by_tile) - len(complete)} partial", file=sys.stderr)

    tiles: dict[str, dict] = {}
    skipped_no_meta = 0
    for tile_id, (release_tag, files) in sorted(complete.items()):
        meta = tile_meta_by_id.get(tile_id, {})
        lat, lng = meta.get("centre_lat"), meta.get("centre_lng")
        # Defensive: tiles present in older releases but no longer in the
        # current tiles.geojson (e.g. dropped when admin1 filter tightened
        # to ENG+WLS+SCT only) get null lat/lng. Leaflet crashes when it
        # tries to project a null LatLng → whole picker fails to render.
        # Drop these orphans from the manifest entirely.
        if lat is None or lng is None:
            skipped_no_meta += 1
            continue
        tiles[tile_id] = {
            "place": meta.get("place", ""),
            "place_type": meta.get("place_type", ""),
            "admin1": meta.get("admin1", ""),
            "lat": lat,
            "lng": lng,
            "release_tag": release_tag,
            **files,
        }

    # base_url is the per-release-download prefix; picker constructs
    # the per-tile URL as `${base_url}/${tile.release_tag}/${tile.glb}`.
    base_url = f"https://github.com/{args.repo}/releases/download"
    manifest = {
        "schema_version": 2,
        "release_tag": args.release_tag,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tile_count": len(tiles),
        "base_url": base_url,
        "tiles": tiles,
    }

    Path(args.out).write_text(json.dumps(manifest, separators=(',', ':')))
    print(f"[manifest] wrote {args.out} ({len(tiles)} tiles, "
          f"{Path(args.out).stat().st_size / 1024:.1f} KB)", file=sys.stderr)
    if skipped_no_meta:
        print(f"[manifest] dropped {skipped_no_meta} tiles with null lat/lng "
              f"(orphans from older releases not in current tiles.geojson)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
