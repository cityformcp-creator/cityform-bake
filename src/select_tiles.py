"""Pick the set of 1 km × 1 km GB tiles worth pre-baking.

Heuristic: every Geonames-named populated place in GB → the 1 km OS
National Grid cell containing its centre. Deduplicated by cell, ranked
by feature-code importance so "Sheffield" wins over a hamlet that
happens to be inside the same cell.

Why Geonames (not Overpass): Overpass mirrors choke on country-wide
queries (connections drop, 504s). Geonames ships the entire GB
toponym set as a single 15 MB zip with explicit licence, no API key,
no rate limit — exactly what a one-time bake-list build wants.
Source: http://download.geonames.org/export/dump/GB.zip

Why 1 km OS grid (not WGS84 cells): tile_id is a stable string
("SK3587") that's compact, geographically meaningful, and aligns to the
EA LIDAR tile boundaries the bake will fetch — so we don't waste cache
on overlapping tiles.

Output: GeoJSON FeatureCollection at the path given by --out, one
Feature per tile with properties:
  tile_id   "SK3587"
  centre    [lng, lat] of the OS grid cell centre
  place     "Sheffield"
  place_type "PPLA" (city) | "PPL" (town) | "PPLX" (suburb) | etc.
  population int (from Geonames if available, else 0)
  bng_easting, bng_northing
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests
from pyproj import Transformer


# Geonames dump for GB. ~15 MB zip, contains GB.txt (tab-separated).
# Schema reference: https://download.geonames.org/export/dump/readme.txt
GEONAMES_URL = "https://download.geonames.org/export/dump/GB.zip"

# Geonames feature codes we treat as bakeable. P.* are populated places.
# Ranked by importance — when two places fall in the same 1 km grid
# cell we keep the higher-ranked one. Codes:
#   PPLC = capital, PPLA = first-order admin centre, PPLA2 = second,
#   PPL = generic populated place, PPLS = section of populated place,
#   PPLX = subdivision (suburb), PPLL = locality, PPLF = farm village,
#   PPLH = historical (skip), PPLR = religious (skip).
FEATURE_RANK = {
    "PPLC":  10,
    "PPLA":  9,
    "PPLA2": 8,
    "PPLA3": 7,
    "PPL":   6,
    "PPLX":  5,
    "PPLS":  4,
    "PPLL":  3,
    "PPLF":  2,
    # Anything else (PPLH historical, PPLR religious, PPLW destroyed,
    # PPLCH historical capital, etc.) excluded from the bake list.
}


# WGS84 → OS National Grid (EPSG:27700). always_xy keeps lng/lat order.
_WGS84_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
_BNG_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


# OS National Grid 100 km square letters. Index = (east_idx, north_idx)
# where each idx is 0..6 (covers GB). Standard reference.
# See https://digimap.edina.ac.uk/help/our-maps-and-data/bng/
_NG_LETTERS = [
    ["SV", "SW", "SX", "SY", "SZ", "TV", "TW"],   # north_idx 0 (S row)
    ["SQ", "SR", "SS", "ST", "SU", "TQ", "TR"],   # 1
    ["SL", "SM", "SN", "SO", "SP", "TL", "TM"],   # 2
    ["SF", "SG", "SH", "SJ", "SK", "TF", "TG"],   # 3
    ["SA", "SB", "SC", "SD", "SE", "TA", "TB"],   # 4
    ["NV", "NW", "NX", "NY", "NZ", "OV", "OW"],   # 5
    ["NQ", "NR", "NS", "NT", "NU", "OQ", "OR"],   # 6
    ["NL", "NM", "NN", "NO", "NP", "OL", "OM"],   # 7
    ["NF", "NG", "NH", "NJ", "NK", "OF", "OG"],   # 8
    ["NA", "NB", "NC", "ND", "NE", "OA", "OB"],   # 9
    ["HV", "HW", "HX", "HY", "HZ", "JV", "JW"],   # 10 (Shetland)
    ["HQ", "HR", "HS", "HT", "HU", "JQ", "JR"],   # 11
    ["HL", "HM", "HN", "HO", "HP", "JL", "JM"],   # 12
]


def bng_to_grid_ref(easting: float, northing: float) -> str | None:
    """Convert BNG (easting, northing) in metres → 4-digit 1km grid ref
    like 'SK3587' (i.e. SK 35 87 with implicit km units)."""
    e_idx = int(easting // 100_000)
    n_idx = int(northing // 100_000)
    if not (0 <= e_idx < 7 and 0 <= n_idx < len(_NG_LETTERS)):
        return None
    letters = _NG_LETTERS[n_idx][e_idx]
    e_km = int((easting % 100_000) // 1000)
    n_km = int((northing % 100_000) // 1000)
    return f"{letters}{e_km:02d}{n_km:02d}"


def grid_ref_to_bng(ref: str) -> tuple[float, float] | None:
    """Inverse of bng_to_grid_ref. 'SK3587' → BNG centre (e, n) of that
    1km cell (i.e. lower-left + 500 m so the point is the cell middle)."""
    if len(ref) != 6:
        return None
    letters = ref[:2].upper()
    # Find letters in the grid
    for n_idx, row in enumerate(_NG_LETTERS):
        for e_idx, letter in enumerate(row):
            if letter == letters:
                base_e = e_idx * 100_000
                base_n = n_idx * 100_000
                try:
                    e_km = int(ref[2:4])
                    n_km = int(ref[4:6])
                except ValueError:
                    return None
                return (base_e + e_km * 1000 + 500.0,
                        base_n + n_km * 1000 + 500.0)
    return None


def fetch_geonames_places() -> list[dict[str, Any]]:
    """Download Geonames GB.zip and parse out populated-place records.
    Returns a list of dicts with keys: name, lat, lng, feature_code,
    population."""
    headers = {"User-Agent": "cityform-bake/0.1 (https://cityform.co.uk)"}
    print(f"[geonames] downloading {GEONAMES_URL}…", file=sys.stderr)
    t0 = time.time()
    resp = requests.get(GEONAMES_URL, headers=headers, timeout=300)
    resp.raise_for_status()
    print(f"[geonames] got {len(resp.content)//1024} KB in {time.time()-t0:.1f}s",
          file=sys.stderr)

    # The zip contains GB.txt and readme.txt. We want GB.txt.
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("GB.txt") as f:
            raw = f.read().decode("utf-8")

    # Schema (tab-separated, no header):
    #  0 geonameid  1 name  2 asciiname  3 alternatenames
    #  4 latitude   5 longitude
    #  6 feature_class  7 feature_code
    #  8 country_code  9 cc2  10 admin1  11 admin2  12 admin3  13 admin4
    # 14 population  15 elevation  16 dem  17 timezone  18 mod_date
    places: list[dict[str, Any]] = []
    skipped_class = 0
    skipped_code = 0
    for line in raw.splitlines():
        cols = line.split("\t")
        if len(cols) < 15:
            continue
        if cols[6] != "P":     # populated places only
            skipped_class += 1
            continue
        code = cols[7]
        if code not in FEATURE_RANK:
            skipped_code += 1
            continue
        try:
            lat = float(cols[4])
            lng = float(cols[5])
            pop = int(cols[14] or 0)
        except ValueError:
            continue
        places.append({
            "name": cols[1],
            "lat": lat,
            "lng": lng,
            "feature_code": code,
            "population": pop,
        })
    print(f"[geonames] {len(places)} populated places "
          f"(skipped: {skipped_class} non-P class, {skipped_code} excluded codes)",
          file=sys.stderr)
    return places


def build_tiles(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group places into 1 km OS grid cells. Keep the highest-ranked
    place per cell."""
    # tile_id → { place metadata }
    by_cell: dict[str, dict[str, Any]] = {}
    skipped_outside = 0

    for place in places:
        try:
            easting, northing = _WGS84_TO_BNG.transform(place["lng"], place["lat"])
        except Exception:
            continue
        tile_id = bng_to_grid_ref(easting, northing)
        if tile_id is None:
            skipped_outside += 1
            continue

        rank = FEATURE_RANK.get(place["feature_code"], 0)
        candidate = {
            "tile_id": tile_id,
            "name": place["name"],
            "place_type": place["feature_code"],
            "rank": rank,
            "population": place["population"],
        }

        existing = by_cell.get(tile_id)
        if existing is None:
            by_cell[tile_id] = candidate
            continue
        # Tiebreaker: higher place rank, then higher population.
        if (candidate["rank"], candidate["population"]) > (existing["rank"], existing["population"]):
            by_cell[tile_id] = candidate

    print(
        f"[group] {len(by_cell)} unique tiles (skipped: {skipped_outside} outside BNG)",
        file=sys.stderr,
    )
    return list(by_cell.values())


def to_geojson(tiles: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a GeoJSON FeatureCollection. Each feature is a Point at
    the BNG cell centre (in WGS84) with all our metadata as properties."""
    features = []
    for t in tiles:
        bng = grid_ref_to_bng(t["tile_id"])
        if bng is None:
            continue
        easting, northing = bng
        centre_lng, centre_lat = _BNG_TO_WGS84.transform(easting, northing)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [centre_lng, centre_lat]},
            "properties": {
                "tile_id": t["tile_id"],
                "place": t["name"],
                "place_type": t["place_type"],
                "population": t["population"],
                "centre_lat": round(centre_lat, 6),
                "centre_lng": round(centre_lng, 6),
                "bng_easting": round(easting, 1),
                "bng_northing": round(northing, 1),
            },
        })
    # Sort: rank-desc, then population-desc, then tile_id alpha. Makes
    # the file diff-friendly and the bake order pick "obviously
    # interesting" tiles first when truncated.
    features.sort(key=lambda f: (
        -FEATURE_RANK.get(f["properties"]["place_type"], 0),
        -f["properties"]["population"],
        f["properties"]["tile_id"],
    ))
    return {"type": "FeatureCollection", "features": features}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", required=True, help="Output GeoJSON path")
    p.add_argument("--cache", default=str(Path(__file__).parent.parent / "data" / "_geonames_cache.json"),
                   help="Cache the parsed Geonames places here so re-runs skip the download")
    p.add_argument("--refresh", action="store_true",
                   help="Ignore cache and re-download Geonames GB.zip")
    p.add_argument("--max-tiles", type=int, default=0,
                   help="If >0, truncate output to top-N tiles by rank (useful for sample bake)")
    args = p.parse_args()

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not args.refresh:
        print(f"[cache] loading {cache_path}", file=sys.stderr)
        places = json.loads(cache_path.read_text())
    else:
        places = fetch_geonames_places()
        cache_path.write_text(json.dumps(places, indent=2))
        print(f"[cache] saved to {cache_path}", file=sys.stderr)

    tiles = build_tiles(places)
    fc = to_geojson(tiles)

    if args.max_tiles > 0 and args.max_tiles < len(fc["features"]):
        fc["features"] = fc["features"][:args.max_tiles]
        print(f"[truncate] kept top {len(fc['features'])} tiles", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc, indent=2))
    print(f"[done] {len(fc['features'])} tiles → {out_path}", file=sys.stderr)

    # Quick distribution summary on stderr for sanity.
    by_type: dict[str, int] = {}
    for f in fc["features"]:
        by_type[f["properties"]["place_type"]] = by_type.get(f["properties"]["place_type"], 0) + 1
    for code in sorted(by_type, key=lambda c: -FEATURE_RANK.get(c, 0)):
        print(f"  {code:<6} {by_type[code]:>6}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
