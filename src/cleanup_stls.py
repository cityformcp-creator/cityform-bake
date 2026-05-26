"""Delete all `*.stl` release assets across every `bake-*` release.

Storage policy: GLBs are kept (the picker needs them for the 3D viewer);
STLs are dropped because fulfilment re-bakes from the saved cart coords
when an order ships. Each STL averages ~200 MB vs ~3 MB for the GLB, so
dropping them recovers ~99% of release storage.

Idempotent. Dry-run by default — pass --apply to actually delete.

Usage:
  python3 src/cleanup_stls.py            # preview what would be deleted
  python3 src/cleanup_stls.py --apply    # actually delete

Requires gh CLI authenticated with `repo` scope.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def _run_gh(args: list[str], check: bool = True) -> str:
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {r.stderr[:400]}")
    return r.stdout


def list_bake_releases() -> list[str]:
    out = _run_gh(["release", "list", "--limit", "50"])
    tags: list[str] = []
    for line in out.splitlines():
        cols = line.split("\t")
        if len(cols) >= 3 and cols[2].startswith("bake-"):
            tags.append(cols[2])
    return tags


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true",
                   help="Actually delete (default is dry-run preview).")
    p.add_argument("--only-tag", default="",
                   help="Limit to one release tag (debug).")
    args = p.parse_args()

    tags = [args.only_tag] if args.only_tag else list_bake_releases()
    print(f"[cleanup-stls] {'APPLYING' if args.apply else 'DRY-RUN'}: "
          f"{len(tags)} release(s)")

    total_bytes = 0
    total_files = 0
    for tag in tags:
        r = _run_gh(["release", "view", tag, "--json", "assets"], check=False)
        if not r.strip():
            continue
        assets = json.loads(r).get("assets", [])
        stls = [a for a in assets if a.get("name", "").endswith(".stl")]
        if not stls:
            print(f"  {tag}: no STLs")
            continue
        size_gb = sum(a.get("size", 0) for a in stls) / 1e9
        total_bytes += sum(a.get("size", 0) for a in stls)
        total_files += len(stls)
        print(f"  {tag}: {len(stls):>4} STL assets, {size_gb:.1f} GB")
        if not args.apply:
            continue
        # Delete in series. `gh release delete-asset` doesn't accept multiple
        # names in one call; one-asset-per-call keeps it simple and recoverable
        # if the script is interrupted halfway.
        for i, a in enumerate(stls):
            name = a["name"]
            _run_gh(["release", "delete-asset", tag, name, "--yes"], check=False)
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(stls)} deleted from {tag}…")
        time.sleep(0.5)   # tiny breather between releases

    print()
    print(f"[cleanup-stls] {'DELETED' if args.apply else 'WOULD DELETE'}: "
          f"{total_files} files, {total_bytes/1e9:.1f} GB")
    if not args.apply:
        print("[cleanup-stls] re-run with --apply to actually delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
