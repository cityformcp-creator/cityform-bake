"""Generate compact glb preview meshes for the in-browser 3D viewer.

Cityform STLs run 100 MB – 1 GB at full resolution, far too heavy for the
library page's three.js modal. This module derives a ~1–3 MB Draco-less glb
preview per entry by:

  1. Snapping vertices to a 100 µm grid + welding duplicates (closes most
     float-quantisation seams from heightmap meshing).
  2. Pre-decimating to ~600k faces with pyfqmr aggressiveness=6,
     preserve_border=False. This shrinks the working set enough for
     pymeshfix to repair without running out of RAM, and exposes the
     remaining real holes for repair.
  3. Sealing remaining holes with pymeshfix (Marco Attene robust repair).
  4. Final pyfqmr pass to the target face count with aggressiveness=4,
     preserve_border=True — the input is watertight by this point so
     preserve_border gives crisp silhouettes without bloating face count.
  5. Recomputing consistent vertex normals + exporting indexed binary glb.

The result is watertight, decimation-clean, and ~50–500× smaller than the
source STL. The preview glb is purely for browser display — the source
STL on disk is never touched, and slicer/printer workflows continue to
use it as the source of truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyfqmr
import pymeshfix
import trimesh

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "cache" / "library"
DEFAULT_TARGET = 150_000
PRE_DECIMATE_TARGET = 600_000
SNAP_GRID_MM = 0.1  # 100 µm — well below any visible feature


def _count_open_borders(faces: np.ndarray) -> int:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    es = np.sort(edges, axis=1)
    _, counts = np.unique(es, axis=0, return_counts=True)
    return int((counts == 1).sum())


def generate_preview_glb(
    stl_path: Path,
    glb_path: Path,
    target_faces: int = DEFAULT_TARGET,
    verbose: bool = False,
) -> dict:
    """Run the snap → pre-decimate → repair → decimate → export pipeline.

    Returns a dict of stage timings + face counts. Writes glb to disk.
    """
    stats: dict = {"stl_mb": stl_path.stat().st_size / 1e6}

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    t = time.monotonic()
    mesh = trimesh.load(str(stl_path), force="mesh")
    stats["src_faces"] = len(mesh.faces)
    stats["src_borders"] = _count_open_borders(np.asarray(mesh.faces))
    stats["t_load"] = time.monotonic() - t
    log(f"  loaded  {stats['src_faces']:>10,} faces  borders={stats['src_borders']:>8,}  ({stats['t_load']:.1f}s)")

    # Stage 1: snap + weld
    t = time.monotonic()
    mesh.vertices = np.round(mesh.vertices / SNAP_GRID_MM) * SNAP_GRID_MM
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    stats["t_snap"] = time.monotonic() - t

    # Stage 2: pre-decimate (only if the mesh is big enough to bother)
    t = time.monotonic()
    if len(mesh.faces) > PRE_DECIMATE_TARGET * 1.2:
        simp = pyfqmr.Simplify()
        simp.setMesh(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
        )
        simp.simplify_mesh(
            target_count=PRE_DECIMATE_TARGET,
            aggressiveness=6,
            preserve_border=False,
            verbose=False,
        )
        v, f, _ = simp.getMesh()
    else:
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
    stats["t_predeci"] = time.monotonic() - t
    log(f"  pre-dec {len(f):>10,} faces  borders={_count_open_borders(f):>8,}  ({stats['t_predeci']:.1f}s)")

    # Stage 3: pymeshfix repair (skip when already watertight — saves
    # 30–90 s on terrain/landmark meshes that came out clean).
    # Safety net: pymeshfix's internal degeneracy/intersection removal can
    # nuke heightmap-derived city meshes from 600k down to ~100 faces.
    # If repair drops more than half the geometry, discard it and use the
    # unrepaired pre-decimated mesh (the viewer renders DoubleSide so
    # remaining open borders show as fully-shaded surfaces, not holes).
    t = time.monotonic()
    pre_repair_faces = len(f)
    stats["repair_applied"] = False
    if _count_open_borders(f) > 0:
        try:
            fixer = pymeshfix.MeshFix(v.astype(np.float64), f.astype(np.int32))
            fixer.repair(joincomp=False, remove_smallest_components=False)
            v2 = np.asarray(fixer.points)
            f2 = np.asarray(fixer.faces)
            if len(f2) >= pre_repair_faces * 0.5:
                v, f = v2, f2
                stats["repair_applied"] = True
            else:
                log(f"  repair  rejected (kept {len(f2):,}/{pre_repair_faces:,} faces — would destroy mesh)")
        except Exception as exc:    # noqa: BLE001
            log(f"  repair  failed ({exc}); keeping unrepaired mesh")
    stats["t_repair"] = time.monotonic() - t
    stats["mid_faces"] = len(f)
    stats["mid_borders"] = _count_open_borders(f)
    log(f"  repair  {stats['mid_faces']:>10,} faces  borders={stats['mid_borders']:>8,}  ({stats['t_repair']:.1f}s, applied={stats['repair_applied']})")

    # Stage 4: final decimation
    t = time.monotonic()
    if len(f) > target_faces:
        simp = pyfqmr.Simplify()
        simp.setMesh(v.astype(np.float64), f.astype(np.int32))
        simp.simplify_mesh(
            target_count=target_faces,
            aggressiveness=4,
            preserve_border=True,
            verbose=False,
        )
        v, f, _ = simp.getMesh()
    stats["t_finaldec"] = time.monotonic() - t

    # Stage 5: export
    t = time.monotonic()
    out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    out.fix_normals()
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(out.export(file_type="glb"))
    stats["t_export"] = time.monotonic() - t
    stats["out_faces"] = len(f)
    stats["glb_mb"] = glb_path.stat().st_size / 1e6
    stats["ratio"] = stats["stl_mb"] / max(stats["glb_mb"], 0.001)
    stats["watertight"] = bool(out.is_watertight)
    log(f"  out     {stats['out_faces']:>10,} faces  → {stats['glb_mb']:.2f} MB  ({stats['ratio']:.0f}× smaller)")
    return stats


def _atomic_write_manifest(manifest_path: Path, manifest: list) -> None:
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(manifest_path)


def batch_all(
    target_faces: int = DEFAULT_TARGET,
    force: bool = False,
    only_printed: bool = False,
) -> None:
    """Run the pipeline across every manifest entry that has an STL on disk.

    Saves the manifest every entry so a crash doesn't lose progress.
    Skips entries whose glb is newer than the source STL unless force=True.
    """
    manifest_path = LIBRARY_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    candidates = []
    for entry in manifest:
        if only_printed and not entry.get("printed"):
            continue
        if not entry.get("stl_path"):
            continue
        stl = LIBRARY_DIR / entry["stl_path"]
        if not stl.exists():
            continue
        candidates.append((entry, stl))

    total = len(candidates)
    print(f"Processing {total} entries (target {target_faces:,} faces)\n", flush=True)

    t_batch = time.monotonic()
    done = 0
    skipped = 0
    failed = 0
    for i, (entry, stl) in enumerate(candidates, 1):
        safe_dir = entry["safe_dir"]
        glb = LIBRARY_DIR / safe_dir / f"{safe_dir}_preview.glb"

        if not force and glb.exists() and glb.stat().st_mtime >= stl.stat().st_mtime:
            print(f"[{i:>3}/{total}] {entry['name']:<45} skip (up to date, {glb.stat().st_size/1e6:.2f} MB)", flush=True)
            entry["preview_glb_path"] = f"{safe_dir}/{safe_dir}_preview.glb"
            skipped += 1
            continue

        t0 = time.monotonic()
        try:
            stats = generate_preview_glb(stl, glb, target_faces=target_faces, verbose=False)
            entry["preview_glb_path"] = f"{safe_dir}/{safe_dir}_preview.glb"
            done += 1
            dt = time.monotonic() - t0
            print(
                f"[{i:>3}/{total}] {entry['name']:<45} "
                f"{stats['stl_mb']:>6.0f}MB → {stats['glb_mb']:>5.2f}MB  "
                f"({stats['out_faces']:>6,} tris, {stats['ratio']:>3.0f}× smaller, {dt:>5.1f}s)",
                flush=True,
            )
        except Exception as exc:    # noqa: BLE001
            failed += 1
            print(f"[{i:>3}/{total}] {entry['name']:<45} FAILED: {exc}", flush=True)
            continue

        # Persist progress every entry — atomic via .tmp swap
        _atomic_write_manifest(manifest_path, manifest)

    dt_total = time.monotonic() - t_batch
    print(
        f"\nDone in {dt_total/60:.1f} min. "
        f"generated={done}  skipped={skipped}  failed={failed}",
        flush=True,
    )


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--all", action="store_true", help="batch over the manifest")
    p.add_argument("--printed", action="store_true", help="restrict --all to printed entries")
    p.add_argument("--force", action="store_true", help="re-generate even if glb is up to date")
    p.add_argument("--target", type=int, default=DEFAULT_TARGET, help=f"target face count (default {DEFAULT_TARGET:,})")
    p.add_argument("safe_dir", nargs="?", help="single entry to process by safe_dir")
    args = p.parse_args()

    if args.all:
        batch_all(target_faces=args.target, force=args.force, only_printed=args.printed)
        return

    if not args.safe_dir:
        p.error("provide --all or a safe_dir to process")

    manifest = json.loads((LIBRARY_DIR / "manifest.json").read_text())
    entry = next((e for e in manifest if e["safe_dir"] == args.safe_dir), None)
    if entry is None:
        sys.exit(f"no entry with safe_dir={args.safe_dir!r}")
    stl = LIBRARY_DIR / entry["stl_path"]
    glb = LIBRARY_DIR / args.safe_dir / f"{args.safe_dir}_preview.glb"
    stats = generate_preview_glb(stl, glb, target_faces=args.target, verbose=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
