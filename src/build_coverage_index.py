"""Build a coverage index mapping each candidate 1 km tile to whether
LIDAR data is actually available for it.

Sources of truth (matches pipeline/sources.py routing):

  ENG  →  Environment Agency LIDAR (assumed covered — composite is
          ~99% of England; rare upland gaps caught at bake time).
  WLS  →  Welsh Government 2020-23 WFS catalogue
          (geonode:welsh_government_lidar_tile_catalogue_2020_2023).
          Each catalogue feature = one 1 km tile with confirmed data.
  SCT  →  AWS S3 bucket s3://srsp-open-data (eu-west-2).
          Walks lidar/<phase>/dtm/27700/gridded/ for every phase,
          parses the OS grid ref from each filename, then checks if a
          candidate tile (or its 5 km / 10 km parent) is covered.

Output: data/coverage_index.json — a per-tile-id boolean covered flag,
plus the resolution + dataset that backs each covered tile.

Run once after a release-quality bake to refresh; the source datasets
update only when new captures are flown (months apart).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

import requests


# Match the table in pipeline/scotland.py — kept in sync because this
# index drives which tiles the bake even attempts. If we add a new
# Scottish dataset to scotland.py, mirror it here.
SRSP_DATASETS = [
    ("national-lidar-programme", "1km"),
    ("phase-6", "5km"),
    ("phase-5", "5km"),
    ("phase-4", "5km"),
    ("phase-3", "5km"),
    ("phase-1", "10km"),
    ("phase-2", "10km"),
]
S3_BASE = "https://srsp-open-data.s3.eu-west-2.amazonaws.com"
NRW_WFS_BASE = "https://datamap.gov.wales/geoserver/ows"
NRW_LAYER = "geonode:welsh_government_lidar_tile_catalogue_2020_2023"


# OS National Grid letter table — mirrors select_tiles.py + scotland.py.
_NG_LETTERS = [
    ["SV", "SW", "SX", "SY", "SZ", "TV", "TW"],
    ["SQ", "SR", "SS", "ST", "SU", "TQ", "TR"],
    ["SL", "SM", "SN", "SO", "SP", "TL", "TM"],
    ["SF", "SG", "SH", "SJ", "SK", "TF", "TG"],
    ["SA", "SB", "SC", "SD", "SE", "TA", "TB"],
    ["NV", "NW", "NX", "NY", "NZ", "OV", "OW"],
    ["NQ", "NR", "NS", "NT", "NU", "OQ", "OR"],
    ["NL", "NM", "NN", "NO", "NP", "OL", "OM"],
    ["NF", "NG", "NH", "NJ", "NK", "OF", "OG"],
    ["NA", "NB", "NC", "ND", "NE", "OA", "OB"],
    ["HV", "HW", "HX", "HY", "HZ", "JV", "JW"],
    ["HQ", "HR", "HS", "HT", "HU", "JQ", "JR"],
    ["HL", "HM", "HN", "HO", "HP", "JL", "JM"],
]


def _parents_of(tile_id: str) -> tuple[str, str, str]:
    """For a 1km grid ref like 'NT2773', return (1km, 5km_quadrant, 10km)."""
    letters = tile_id[:2]
    e_km = int(tile_id[2:4])
    n_km = int(tile_id[4:6])
    # 5km quadrant: e_km / n_km 0-9 within the 10km tile (which is e_km//10, n_km//10)
    e_10 = e_km // 10
    n_10 = n_km // 10
    e_within = e_km - e_10 * 10
    n_within = n_km - n_10 * 10
    ns = "N" if n_within >= 5 else "S"
    ew = "E" if e_within >= 5 else "W"
    quad = f"{letters}{e_10}{n_10}{ns}{ew}"
    ten = f"{letters}{e_10}{n_10}"
    return (tile_id, quad, ten)


# ── Scotland: list S3 bucket contents ────────────────────────────────────

def _list_s3_keys(prefix: str) -> Iterator[str]:
    """Paginated ListObjectsV2 over the public srsp-open-data bucket."""
    token = ""
    while True:
        params = f"?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            params += f"&continuation-token={token}"
        resp = requests.get(f"{S3_BASE}/{params}", timeout=60)
        resp.raise_for_status()
        tree = ET.fromstring(resp.text)
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for c in tree.findall("s:Contents", ns):
            k = c.find("s:Key", ns)
            if k is not None and k.text:
                yield k.text
        truncated = tree.findtext("s:IsTruncated", "false", ns)
        if truncated.lower() != "true":
            return
        next_token = tree.findtext("s:NextContinuationToken", "", ns)
        if not next_token:
            return
        token = requests.utils.quote(next_token, safe="")


def build_scotland_index() -> dict[str, set[str]]:
    """Return {dataset_kind: set_of_grid_refs_covered}.

    Iterates every phase's DTM listing once. Refs in the returned sets
    are the OS grid identifiers as they appear in the filename (1 km,
    5 km-quadrant, or 10 km depending on the phase).
    """
    print("[scotland] walking S3 bucket s3://srsp-open-data …", file=sys.stderr)
    by_kind: dict[str, set[str]] = {"1km": set(), "5km": set(), "10km": set()}
    for prefix, kind in SRSP_DATASETS:
        t0 = time.time()
        before = len(by_kind[kind])
        s3_prefix = f"lidar/{prefix}/dtm/27700/gridded/"
        for key in _list_s3_keys(s3_prefix):
            # Filename like 'lidar/phase-5/dtm/27700/gridded/NT27NE_50CM_DTM_PHASE5.tif'
            fname = key.rsplit("/", 1)[-1]
            ref = fname.split("_", 1)[0]   # 'NT27NE' or 'NT2773' or 'NT27'
            by_kind[kind].add(ref)
        added = len(by_kind[kind]) - before
        print(f"  {prefix:<30} +{added:>5} {kind} cells "
              f"({time.time()-t0:.1f}s)", file=sys.stderr)
    for kind, s in by_kind.items():
        print(f"[scotland] {kind:<5} total: {len(s):,} cells", file=sys.stderr)
    return by_kind


def scotland_covers(tile_id: str, sct_index: dict[str, set[str]]) -> bool:
    """True if the 1 km tile is covered by any Scottish dataset."""
    one_km, quad_5km, ten_km = _parents_of(tile_id)
    return (one_km in sct_index["1km"]
            or quad_5km in sct_index["5km"]
            or ten_km   in sct_index["10km"])


# ── Wales: download the full WFS catalogue once ──────────────────────────

def build_wales_index() -> set[str]:
    """Return the set of OS 4-digit grid refs (e.g. 'ST1876') covered by
    the Welsh Government 2020-23 LiDAR catalogue.

    Single WFS request without a bbox returns the whole catalogue — a
    few thousand 1 km tiles. ~5-10 MB JSON, ~10 s to fetch."""
    print(f"[wales] downloading full {NRW_LAYER} catalogue…", file=sys.stderr)
    t0 = time.time()
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": NRW_LAYER,
        "srsName": "EPSG:4326",
        "outputFormat": "application/json",
    }
    resp = requests.get(NRW_WFS_BASE, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    feats = data.get("features", []) or []
    refs: set[str] = set()
    for f in feats:
        ref = (f.get("properties") or {}).get("british_gr") or ""
        ref = ref.strip()
        if ref:
            refs.add(ref)
    print(f"[wales] {len(refs):,} covered 1 km tiles ({time.time()-t0:.1f}s)",
          file=sys.stderr)
    return refs


def wales_covers(tile_id: str, wls_index: set[str]) -> bool:
    return tile_id in wls_index


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiles", default="data/tiles.geojson",
                   help="Input candidate tile list")
    p.add_argument("--out", default="data/coverage_index.json")
    p.add_argument("--skip-scotland", action="store_true",
                   help="Skip S3 walk (use a cached index from a previous run)")
    p.add_argument("--skip-wales", action="store_true",
                   help="Skip WFS download (use a cached index)")
    args = p.parse_args()

    out_path = Path(args.out)
    cached: dict = {}
    if out_path.exists():
        try:
            cached = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            cached = {}

    if args.skip_scotland and "scotland_by_kind" in cached:
        sct_index = {k: set(v) for k, v in cached["scotland_by_kind"].items()}
        print("[scotland] using cached index from previous run", file=sys.stderr)
    else:
        sct_index = build_scotland_index()

    if args.skip_wales and "wales" in cached:
        wls_index = set(cached["wales"])
        print("[wales] using cached index from previous run", file=sys.stderr)
    else:
        wls_index = build_wales_index()

    # Now annotate each candidate tile.
    tiles = json.loads(Path(args.tiles).read_text())
    by_tile: dict[str, dict] = {}
    by_admin1_count = {"ENG": [0, 0], "WLS": [0, 0], "SCT": [0, 0]}   # [covered, total]
    for f in tiles["features"]:
        props = f["properties"]
        tid = props["tile_id"]
        admin1 = props.get("admin1", "")
        if admin1 not in by_admin1_count:
            continue
        by_admin1_count[admin1][1] += 1
        if admin1 == "ENG":
            covered = True            # EA composite ~99% — accept all
            backend = "ea"
        elif admin1 == "WLS":
            covered = wales_covers(tid, wls_index)
            backend = "nrw"
        else:   # SCT
            covered = scotland_covers(tid, sct_index)
            backend = "srsp"
        if covered:
            by_admin1_count[admin1][0] += 1
        by_tile[tid] = {"covered": covered, "backend": backend, "admin1": admin1}

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scotland_by_kind": {k: sorted(v) for k, v in sct_index.items()},
        "wales": sorted(wls_index),
        "tile_coverage": by_tile,
        "summary": {
            a: {"covered": c, "total": t,
                "coverage_pct": round(100.0 * c / t, 1) if t else 0.0}
            for a, (c, t) in by_admin1_count.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, separators=(",", ":")))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[done] wrote {out_path} ({size_mb:.1f} MB)", file=sys.stderr)
    for a, s in summary["summary"].items():
        print(f"  {a}: {s['covered']:>6,} / {s['total']:>6,}  "
              f"({s['coverage_pct']}%)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
