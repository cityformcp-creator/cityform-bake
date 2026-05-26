"""Bake ONE batch of tiles + upload outputs to the release.

Used by the bake matrix job in .github/workflows/bake-all.yml. Each
matrix runner gets a `--batch-idx` and processes tiles[idx*size : (idx+1)*size]
of the top-LIMIT tiles by importance.

For each tile:
  1. Run bake_tile.bake() → produces city.stl, city.glb, preview.png, meta.json
  2. Upload all 4 assets to the release via `gh release upload`,
     using <tile_id>__<file> naming so all assets are flat in the release.
  3. On failure, log and continue to the next tile (fail-fast=false).

After the batch finishes, prints a one-line JSON summary on stdout so
the workflow log is parseable.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Make sibling bake_tile.py importable as a module so we don't fork+exec.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_tile    # noqa: E402


def _gh_release_upload(release_tag: str, files: list[Path],
                       clobber: bool = True) -> None:
    """Upload assets to the named release via `gh release upload`.

    `--clobber` overwrites if an asset with the same name already exists
    (useful when re-running a failed batch). Requires GH_TOKEN env var
    with `repo` scope, which `actions/checkout` sets up automatically
    on GH Actions."""
    args = ["gh", "release", "upload", release_tag]
    args.extend(str(f) for f in files)
    if clobber:
        args.append("--clobber")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gh release upload failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout[:500]}\n"
            f"  stderr: {result.stderr[:500]}"
        )


def _flatten_outputs(tile_id: str, src_dir: Path, dst_dir: Path) -> list[Path]:
    """Rename baked outputs from <src_dir>/{city.stl,city.glb,…} to
    <dst_dir>/<tile_id>__<file> so they sit flat in the release.

    We DELIBERATELY skip uploading meta.json and preview.png — their
    data lives in the aggregated manifest.json (built by build_manifest.py
    in the manifest job). Skipping them halves our asset-per-tile count
    from 4 → 2, doubling how many tiles fit under GitHub's 1000-asset-
    per-release cap. See task #15 for proper sharding."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "city.stl": f"{tile_id}__city.stl",
        "city.glb": f"{tile_id}__city.glb",
        # meta.json + preview.png intentionally omitted — see docstring.
    }
    out: list[Path] = []
    for src_name, dst_name in mapping.items():
        src = src_dir / src_name
        if not src.exists():
            continue
        dst = dst_dir / dst_name
        shutil.copy2(src, dst)
        out.append(dst)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default="data/tiles.geojson")
    p.add_argument("--limit", type=int, required=True,
                   help="Top-N tiles to bake total (across all batches)")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--batch-idx", type=int, required=True,
                   help="0-indexed batch number to process")
    p.add_argument("--release-tag", required=True,
                   help="Release tag to upload outputs to (must already exist)")
    p.add_argument("--out-root", default="./bake-output",
                   help="Per-tile output dirs go under here (gitignored)")
    p.add_argument("--admin1", default="",
                   help="Comma-separated admin1 codes to include. MUST match "
                        "what split_batches.py was invoked with — otherwise "
                        "the batch_idx slicing won't line up.")
    args = p.parse_args()

    fc = json.loads(Path(args.tiles).read_text())
    features = fc["features"]
    if args.admin1:
        wanted = {c.strip().upper() for c in args.admin1.split(",") if c.strip()}
        features = [t for t in features
                    if (t.get("properties") or {}).get("admin1", "") in wanted]
    all_features = features[:args.limit]
    start = args.batch_idx * args.batch_size
    end = min(start + args.batch_size, len(all_features))
    batch_features = all_features[start:end]

    if not batch_features:
        print(f"[batch {args.batch_idx}] empty (start={start} >= total={len(all_features)})",
              file=sys.stderr)
        return 0

    print(f"[batch {args.batch_idx}] {len(batch_features)} tiles "
          f"(indices {start}..{end-1}, release={args.release_tag})",
          file=sys.stderr)

    out_root = Path(args.out_root)
    flat_dir = out_root / "flat"
    successes: list[str] = []
    failures: list[dict] = []
    t_batch_start = time.time()

    for i, feat in enumerate(batch_features):
        props = feat["properties"]
        tile_id = props["tile_id"]
        lat = props["centre_lat"]
        lng = props["centre_lng"]
        place = props["place"]

        tile_dir = out_root / tile_id
        try:
            bake_tile.bake(
                tile_id=tile_id, centre_lat=lat, centre_lng=lng,
                out_dir=tile_dir, place_name=place,
            )
            # Flatten + upload + remove from disk to keep runner storage low.
            assets = _flatten_outputs(tile_id, tile_dir, flat_dir)
            if not assets:
                raise RuntimeError("bake reported success but no output files found")
            _gh_release_upload(args.release_tag, assets)
            successes.append(tile_id)
            # Free disk — flatten dir is uploaded, source dir keeps cache for
            # subsequent runs but we don't need the per-tile output.
            shutil.rmtree(tile_dir, ignore_errors=True)
            for a in assets:
                a.unlink(missing_ok=True)
            print(f"  ✓ {tile_id} ({i+1}/{len(batch_features)})", file=sys.stderr)
        except Exception as exc:    # noqa: BLE001
            failures.append({"tile_id": tile_id, "error": str(exc)[:200]})
            print(f"  ✗ {tile_id} ({i+1}/{len(batch_features)}): {exc}",
                  file=sys.stderr)

    elapsed = round(time.time() - t_batch_start, 1)
    summary = {
        "batch_idx": args.batch_idx,
        "n_attempted": len(batch_features),
        "n_succeeded": len(successes),
        "n_failed": len(failures),
        "elapsed_s": elapsed,
        "release_tag": args.release_tag,
        "failures": failures[:10],
    }
    print(json.dumps(summary, separators=(',', ':')))
    # Exit non-zero only if EVERY tile in the batch failed — otherwise
    # we let the run continue and the manifest job will skip missing ones.
    return 2 if successes == [] and failures else 0


if __name__ == "__main__":
    sys.exit(main())
