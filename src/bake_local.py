"""Bake tiles on the local Mac in parallel — no GitHub required.

Useful when GH Actions / the upload account is unavailable but you still
want to advance the tile library. Outputs land in --out-root as flat
files:

    <out-root>/<tile_id>__city.glb
    <out-root>/_done.json    # resume state, list of completed tile_ids
    <out-root>/_failed.json  # list of {tile_id, error} for failed bakes

When the GitHub account is back, run sync_local_to_release.py to upload
the entire directory into a new release.

Resumable: re-running with the same --out-root skips tiles already in
_done.json. Crashes mid-bake are fine — the next run resumes.

Performance: bake is CPU+RAM heavy (rasterio + trimesh + pyfqmr).
Set --workers to roughly NUM_CORES/2 to leave headroom; on a 2025 M-class
Mac, --workers=4 sustains ~4-5 tiles/min for England (1 m tiles) and
~1-2 tiles/min for Scotland (50 cm phase tiles). 1000 tiles ≈ overnight.

Filters mirror split_batches.py / bake_batch.py exactly:
  --admin1         comma-list of ENG / WLS / SCT codes
  --coverage-index path to data/coverage_index.json
  --skip-baked-tags  comma-list of release tags to dedupe against
  --offset / --limit  slice into the post-filter candidate list
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# bake_tile.py sits next to us — borrow its bake() function directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_tile   # noqa: E402


def _load_done(out_root: Path) -> set[str]:
    p = out_root / "_done.json"
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except json.JSONDecodeError:
        return set()


def _save_done(out_root: Path, done: set[str]) -> None:
    (out_root / "_done.json").write_text(
        json.dumps(sorted(done), separators=(",", ":"))
    )


def _append_failed(out_root: Path, tile_id: str, err: str) -> None:
    p = out_root / "_failed.json"
    existing = []
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except json.JSONDecodeError:
            existing = []
    existing.append({"tile_id": tile_id, "error": err[:400],
                     "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    p.write_text(json.dumps(existing, indent=2))


def _list_baked_from_releases(tags: list[str]) -> set[str]:
    """gh-CLI helper, same shape as split_batches._list_baked_tile_ids
    but defined locally so this script is standalone."""
    out: set[str] = set()
    for tag in tags:
        try:
            r = subprocess.run(["gh", "release", "view", tag, "--json", "assets"],
                               capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            continue
        for a in json.loads(r.stdout).get("assets", []):
            name = a.get("name", "")
            if name.endswith("__city.glb"):
                out.add(name.split("__", 1)[0])
    return out


def _select_candidates(args) -> list[dict]:
    fc = json.loads(Path(args.tiles).read_text())
    features = fc["features"]

    if args.admin1:
        wanted = {c.strip().upper() for c in args.admin1.split(",") if c.strip()}
        features = [f for f in features
                    if (f.get("properties") or {}).get("admin1", "") in wanted]

    if args.coverage_index:
        idx = json.loads(Path(args.coverage_index).read_text())
        cov = idx.get("tile_coverage", {})
        features = [f for f in features
                    if cov.get(f["properties"]["tile_id"], {}).get("covered")]

    if args.skip_baked_tags:
        tags = [t.strip() for t in args.skip_baked_tags.split(",") if t.strip()]
        if tags:
            already = _list_baked_from_releases(tags)
            features = [f for f in features
                        if f["properties"]["tile_id"] not in already]

    if args.skip_baked_manifest:
        # Accept either a URL or a local path. Anonymous fetch works even
        # when the gh CLI is unauthenticated — the public manifest lives
        # under releases/latest/download/.
        src = args.skip_baked_manifest
        if src.startswith(("http://", "https://")):
            # Use requests (already a bake dep) so we get certifi's CA
            # bundle — system Python's urllib trips on SSL otherwise.
            import requests
            r = requests.get(src, timeout=30, allow_redirects=True)
            r.raise_for_status()
            m = r.json()
        else:
            m = json.loads(Path(src).read_text())
        already_m = set((m.get("tiles") or {}).keys())
        before = len(features)
        features = [f for f in features
                    if f["properties"]["tile_id"] not in already_m]
        print(f"[skip-baked-manifest] skipped {len(already_m)} already-baked, "
              f"{before} → {len(features)} candidates remain", file=sys.stderr)

    if args.offset:
        features = features[args.offset:]
    if args.limit:
        features = features[:args.limit]

    return features


def _bake_one(args_tuple: tuple) -> dict:
    """Worker-process entry. Pulls one tile through bake_tile.bake().

    Returns a status dict the main process accumulates."""
    feat, out_root_str, cache_dir_str = args_tuple
    props = feat["properties"]
    tile_id = props["tile_id"]
    out_root = Path(out_root_str)
    cache_dir = Path(cache_dir_str)

    tmp_dir = out_root / f"_tmp_{tile_id}_{os.getpid()}"
    t0 = time.time()
    try:
        bake_tile.bake(
            tile_id=tile_id,
            centre_lat=props["centre_lat"],
            centre_lng=props["centre_lng"],
            out_dir=tmp_dir,
            place_name=props.get("place", ""),
            cache_dir=cache_dir,
            skip_png=True,    # picker doesn't need preview PNG
            admin1=props.get("admin1") or None,
        )
        glb_src = tmp_dir / "city.glb"
        if not glb_src.exists():
            raise RuntimeError("bake completed but no city.glb produced")
        # Validity gate: a real bake produces 1-5 MB. <500 KB means the
        # input LIDAR was mostly NaN (typically a Scottish coastal /
        # waterside cell where most pixels are sea or loch) and the
        # mesh build collapsed to an effectively-empty trimesh. Don't
        # ship these — re-classify as failure so _done.json stays clean
        # and they get re-attempted on a future run (e.g. after we add
        # water-aware nodata handling to tier3_with_water).
        DEGEN_GLB_THRESHOLD_BYTES = 500_000
        src_size = glb_src.stat().st_size
        if src_size < DEGEN_GLB_THRESHOLD_BYTES:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                f"degenerate output ({src_size} bytes) — likely all-NaN "
                f"DTM/DSM (water-dominated cell)"
            )
        glb_dst = out_root / f"{tile_id}__city.glb"
        shutil.copy2(glb_src, glb_dst)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"tile_id": tile_id, "ok": True,
                "elapsed_s": round(time.time() - t0, 1),
                "glb_mb": round(glb_dst.stat().st_size / 1e6, 2)}
    except Exception as exc:    # noqa: BLE001
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"tile_id": tile_id, "ok": False,
                "elapsed_s": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}"[:400]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default="data/tiles.geojson")
    p.add_argument("--coverage-index", default="data/coverage_index.json")
    p.add_argument("--admin1", default="")
    p.add_argument("--skip-baked-tags", default="",
                   help="Comma-separated release tags to dedupe against (skips "
                        "tile_ids already uploaded to those releases). Calls "
                        "`gh release view` — won't work while the account is "
                        "suspended; use --skip-baked-manifest in that case.")
    p.add_argument("--skip-baked-manifest", default="",
                   help="URL (http[s]://…) or local path to a manifest.json. "
                        "All tile_ids present in the manifest are skipped. "
                        "Works even when the gh CLI is unauthenticated — the "
                        "public manifest is anonymously fetchable.")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-root", default="./local-bake-output",
                   help="Where GLBs accumulate (one __city.glb per tile)")
    p.add_argument("--cache-dir", default="./.lidar-cache",
                   help="LIDAR + OSM cache shared across workers (saves bandwidth)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="Parallel bake workers (default: CPU/2)")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Persist _done.json after every N completions")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    done = _load_done(out_root)
    print(f"[local-bake] resume state: {len(done)} tiles already done",
          file=sys.stderr)

    candidates = _select_candidates(args)
    todo = [f for f in candidates
            if f["properties"]["tile_id"] not in done]
    print(f"[local-bake] {len(candidates)} candidates after filters, "
          f"{len(todo)} not-yet-baked", file=sys.stderr)
    if not todo:
        print("[local-bake] nothing to do", file=sys.stderr)
        return 0

    print(f"[local-bake] starting pool with {args.workers} workers", file=sys.stderr)
    work = [(f, str(out_root), str(cache_dir)) for f in todo]

    # SIGINT handler — flush done.json before exiting so the next run
    # picks up cleanly after a Ctrl-C.
    interrupted = {"flag": False}
    def _on_sigint(signum, frame):
        interrupted["flag"] = True
        print("\n[local-bake] interrupted — finishing current tiles, then exit",
              file=sys.stderr)
    signal.signal(signal.SIGINT, _on_sigint)

    t_batch_start = time.time()
    n_ok = 0
    n_fail = 0
    with mp.Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_bake_one, work)):
            tid = result["tile_id"]
            if result["ok"]:
                done.add(tid)
                n_ok += 1
                print(f"  ✓ {tid:<8} ({i+1}/{len(work)}) "
                      f"{result['elapsed_s']}s, {result['glb_mb']} MB",
                      file=sys.stderr)
            else:
                n_fail += 1
                _append_failed(out_root, tid, result["error"])
                print(f"  ✗ {tid:<8} ({i+1}/{len(work)}) "
                      f"{result['elapsed_s']}s — {result['error']}",
                      file=sys.stderr)
            if (i + 1) % args.checkpoint_every == 0:
                _save_done(out_root, done)
            if interrupted["flag"]:
                pool.terminate()
                break

    _save_done(out_root, done)
    elapsed = round(time.time() - t_batch_start, 1)
    print(f"\n[local-bake] done in {elapsed}s — "
          f"ok={n_ok}, failed={n_fail}, total_done={len(done)}",
          file=sys.stderr)
    return 0 if n_ok > 0 or not todo else 2


if __name__ == "__main__":
    sys.exit(main())
