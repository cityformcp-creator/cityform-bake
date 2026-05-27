"""
OSM Overpass fetcher.

Queries public Overpass instances for building footprints and water polygons
within a WGS84 bbox. Returns GeoJSON FeatureCollections.

Falls back across multiple Overpass mirrors so transient outages on one
mirror don't kill the pipeline.
"""

import hashlib
import json
from pathlib import Path
from typing import Iterable

import requests

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

BUILDING_QUERY = """[out:json][timeout:60];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out geom;"""

# Water query — covers both polygon water (natural=water, riverbank, dock,
# basin, reservoir) AND linestring waterways (river, stream, canal, drain).
# Most UK rivers below the Thames are tagged ONLY as waterway=river ways,
# never as natural=water polygons — without the linestring pull, the bake
# silently misses them and the print shows the river as elevated land.
# Linestring waterways are buffered to polygons downstream by
# osm_waterway_features_to_bng_polygons (uses the OSM width=* tag where
# present, sensible defaults per class otherwise).
WATER_QUERY = """[out:json][timeout:60];
(
  way["natural"="water"]({s},{w},{n},{e});
  way["waterway"="riverbank"]({s},{w},{n},{e});
  way["waterway"="dock"]({s},{w},{n},{e});
  way["waterway"="canal"]({s},{w},{n},{e});
  way["landuse"="basin"]({s},{w},{n},{e});
  way["landuse"="reservoir"]({s},{w},{n},{e});
  relation["natural"="water"]({s},{w},{n},{e});
  relation["waterway"="riverbank"]({s},{w},{n},{e});
  relation["waterway"="dock"]({s},{w},{n},{e});
  relation["landuse"="basin"]({s},{w},{n},{e});
  relation["landuse"="reservoir"]({s},{w},{n},{e});
  // Linestring waterways — buffered to polygons downstream. waterway=ditch
  // intentionally excluded (too small to read on a 1:11000 print and adds
  // noise from agricultural drainage).
  way["waterway"="river"]({s},{w},{n},{e});
  way["waterway"="stream"]({s},{w},{n},{e});
  way["waterway"="drain"]({s},{w},{n},{e});
);
out geom;"""

# Bridge query — captures any way tagged as a bridge plus the rarer
# `man_made=bridge` polygon. Most bridges are linestrings (the way carries
# a road/path over a feature) so the consumer is expected to buffer them
# into thin polygons before rasterising.
BRIDGE_QUERY = """[out:json][timeout:60];
(
  way["bridge"]({s},{w},{n},{e});
  way["man_made"="bridge"]({s},{w},{n},{e});
  relation["bridge"]({s},{w},{n},{e});
  relation["man_made"="bridge"]({s},{w},{n},{e});
);
out geom;"""

# Coastline query — `natural=coastline` is OSM's land/sea boundary, mapped
# as LineStrings (not polygons). The consumer needs to clip to bbox + close
# against the bbox boundary + classify each enclosed region by OSM's
# water-on-the-right convention to produce sea Polygons. Captures both
# the open coast and man-made breakwaters / piers.
COASTLINE_QUERY = """[out:json][timeout:60];
(
  way["natural"="coastline"]({s},{w},{n},{e});
  relation["natural"="coastline"]({s},{w},{n},{e});
);
out geom;"""

# Road query — engraved into the terrain top as shallow grooves to give the
# print a legible street grid. Excludes pedestrian-only ways (footway, path,
# cycleway, steps), service roads, and tracks — those would clutter the print
# without adding readability. The consumer rasterises these LineStrings,
# buffers them by ~2 m, and subtracts a fixed depth from terrain cells.
ROAD_QUERY = """[out:json][timeout:60];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street)$"]({s},{w},{n},{e});
);
out geom;"""

# Landmark query — narrow tall features that the LIDAR builder's morphological
# opening (iterations=2) tends to erase entirely. Church spires, monument
# towers, slim chimneys etc. are typically <5 m wide and get filtered out
# along with thin noise. The consumer rasterises these polygons and OR's
# them back into the final building mask, gated by `hag > 2.5` so we never
# lift ground that wasn't actually elevated. Captures common landmark tags
# from `building`, `man_made`, and `historic`.
LANDMARK_QUERY = """[out:json][timeout:60];
(
  way["building"~"^(church|cathedral|chapel|tower)$"]({s},{w},{n},{e});
  relation["building"~"^(church|cathedral|chapel|tower)$"]({s},{w},{n},{e});
  way["man_made"~"^(tower|spire|chimney|obelisk|lighthouse)$"]({s},{w},{n},{e});
  relation["man_made"~"^(tower|spire|chimney|obelisk|lighthouse)$"]({s},{w},{n},{e});
  way["historic"~"^(castle|monument|memorial)$"]({s},{w},{n},{e});
  relation["historic"~"^(castle|monument|memorial)$"]({s},{w},{n},{e});
);
out geom;"""


# Railway query — mainline rail, light rail, narrow gauge, and tram lines
# engraved into the terrain alongside roads. Railways are major orientation
# features (stations, viaducts, cuttings) and are typically wider corridors
# than streets. Excludes `subway` (underground — invisible on surface),
# `abandoned`, `disused`, and `construction`. The consumer buffers each
# LineString by ~2.5 m and rasterises like roads.
RAILWAY_QUERY = """[out:json][timeout:60];
(
  way["railway"~"^(rail|light_rail|narrow_gauge|tram)$"]({s},{w},{n},{e});
);
out geom;"""

# Park / green-space query — leisure parks, gardens, recreation grounds,
# and grass landuse. Only the polygon *boundary* is engraved (thin outline),
# not the filled area, so terrain inside parks is preserved. Small features
# (< 500 m²) are filtered by the consumer to avoid clutter on small prints.
PARK_QUERY = """[out:json][timeout:60];
(
  way["leisure"="park"]({s},{w},{n},{e});
  relation["leisure"="park"]({s},{w},{n},{e});
  way["leisure"="garden"]({s},{w},{n},{e});
  relation["leisure"="garden"]({s},{w},{n},{e});
  way["landuse"="grass"]({s},{w},{n},{e});
  relation["landuse"="grass"]({s},{w},{n},{e});
  way["landuse"="recreation_ground"]({s},{w},{n},{e});
  relation["landuse"="recreation_ground"]({s},{w},{n},{e});
);
out geom;"""

# OSM Simple 3D Buildings (S3DB) building:part polygons. Each part is a
# stacked sub-volume of a complex building tagged with `height`,
# `min_height`, and `roof:shape`. Used opt-in to override the LIDAR DSM
# at major landmarks (Shard, Walkie-Talkie, Cheesegrater, Gherkin) where
# volunteers have invested in proper 3D modelling. Outside London the
# data is sparse so the override is largely a no-op.
BUILDING_PART_QUERY = """[out:json][timeout:60];
(
  way["building:part"]({s},{w},{n},{e});
  relation["building:part"]({s},{w},{n},{e});
);
out geom;"""


def _post_overpass(query: str, mirrors: Iterable[str] = OVERPASS_MIRRORS) -> dict:
    """POST a query to Overpass with mirror failover."""
    last_err: Exception | None = None
    for mirror in mirrors:
        try:
            resp = requests.post(
                mirror, data={"data": query}, timeout=120,
                headers={"User-Agent": "cityform-tool/0.1"},
            )
            if resp.status_code == 200:
                return resp.json()
            last_err = RuntimeError(f"{mirror} returned HTTP {resp.status_code}")
        except (requests.RequestException, ValueError) as exc:
            last_err = exc
    raise RuntimeError(f"All Overpass mirrors failed. Last error: {last_err}")


def _osm_to_geojson(osm: dict) -> dict:
    """Minimal Overpass JSON → GeoJSON converter for ways and simple relations.

    For ways with closed geometry, emit a Polygon. For multipolygon relations,
    assemble outer/inner ring members into a Polygon (or MultiPolygon if there
    are multiple disjoint outers). Skips relations whose member rings can't be
    matched, which is rare for buildings/water in well-tagged urban OSM.
    """
    elements = osm.get("elements", [])
    nodes_by_id: dict[int, tuple[float, float]] = {}
    ways_by_id: dict[int, list[tuple[float, float]]] = {}
    for el in elements:
        if el.get("type") == "node":
            nodes_by_id[el["id"]] = (el["lon"], el["lat"])
        elif el.get("type") == "way" and "geometry" in el:
            coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
            ways_by_id[el["id"]] = coords

    features: list[dict] = []

    for el in elements:
        if el.get("type") == "way" and "geometry" in el:
            coords = [(g["lon"], g["lat"]) for g in el["geometry"]]
            if len(coords) < 2:
                continue
            # Closed ways → Polygon. Open ways (roads, bridges, canal centre
            # lines) → LineString. Previously this converter blindly closed
            # every way into a Polygon which produced self-intersecting
            # garbage for line features.
            is_closed = len(coords) >= 4 and coords[0] == coords[-1]
            if is_closed:
                features.append({
                    "type": "Feature",
                    "properties": el.get("tags", {}),
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                })
            else:
                features.append({
                    "type": "Feature",
                    "properties": el.get("tags", {}),
                    "geometry": {"type": "LineString", "coordinates": coords},
                })
        elif el.get("type") == "relation":
            outers: list[list[tuple[float, float]]] = []
            inners: list[list[tuple[float, float]]] = []
            for member in el.get("members", []):
                if member.get("type") != "way":
                    continue
                way_id = member.get("ref")
                geom = member.get("geometry")
                if geom:
                    coords = [(g["lon"], g["lat"]) for g in geom]
                else:
                    coords = ways_by_id.get(way_id, [])
                if len(coords) < 2:
                    continue
                if member.get("role") == "inner":
                    inners.append(coords)
                else:
                    outers.append(coords)
            # Stitch ways into closed rings (simple greedy matcher)
            outer_rings = _stitch_rings(outers)
            inner_rings = _stitch_rings(inners)
            if not outer_rings:
                continue
            polys = []
            for outer in outer_rings:
                if len(outer) < 4:
                    continue
                # Assign holes to whichever outer they're inside.
                # We do a simple bounding-box test — buildings rarely have
                # complex multi-outer holes that need exact geometric tests.
                holes = []
                for inn in inner_rings:
                    if _bbox_contains(outer, inn):
                        holes.append(inn)
                polys.append([outer] + holes)
            if not polys:
                continue
            if len(polys) == 1:
                geom = {"type": "Polygon", "coordinates": polys[0]}
            else:
                geom = {"type": "MultiPolygon", "coordinates": polys}
            features.append({
                "type": "Feature",
                "properties": el.get("tags", {}),
                "geometry": geom,
            })

    return {"type": "FeatureCollection", "features": features}


def _stitch_rings(ways: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Greedy ring stitcher — joins ways end-to-end into closed rings."""
    rings: list[list[tuple[float, float]]] = []
    pool = [list(w) for w in ways]
    while pool:
        cur = pool.pop(0)
        progress = True
        while progress and cur[0] != cur[-1]:
            progress = False
            for i, other in enumerate(pool):
                if cur[-1] == other[0]:
                    cur.extend(other[1:])
                    pool.pop(i)
                    progress = True
                    break
                if cur[-1] == other[-1]:
                    cur.extend(reversed(other[:-1]))
                    pool.pop(i)
                    progress = True
                    break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur)
    return rings


def _bbox_contains(outer: list[tuple[float, float]], inner: list[tuple[float, float]]) -> bool:
    ox = [p[0] for p in outer]; oy = [p[1] for p in outer]
    ix = [p[0] for p in inner]; iy = [p[1] for p in inner]
    return min(ox) <= min(ix) and max(ox) >= max(ix) and min(oy) <= min(iy) and max(oy) >= max(iy)


class OverpassFetcher:
    """Caches OSM responses by query+bbox hash so re-runs are instant."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cached(self, name: str, query: str) -> dict:
        h = hashlib.sha1(query.encode()).hexdigest()[:12]
        cache_path = self.cache_dir / f"osm_{name}_{h}.geojson"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        osm = _post_overpass(query)
        gj = _osm_to_geojson(osm)
        cache_path.write_text(json.dumps(gj))
        return gj

    def fetch_buildings(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = BUILDING_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("buildings", q)

    def fetch_places(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> list:
        """Named locality nodes for map labels (city/town/village/suburb/
        neighbourhood/quarter). _osm_to_geojson drops standalone nodes, so
        this parses node elements directly and caches the small label list."""
        q = (f'[out:json][timeout:30];'
             f'(node["place"~"^(city|town|village|suburb|neighbourhood|quarter)$"]'
             f'["name"]({lat_min},{lng_min},{lat_max},{lng_max}););out;')
        h = hashlib.sha1(q.encode()).hexdigest()[:12]
        cache_path = self.cache_dir / f"osm_places_{h}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        try:
            osm = _post_overpass(q)
        except Exception:
            return []
        rank = {"city": 0, "town": 1, "suburb": 2, "village": 3,
                "neighbourhood": 4, "quarter": 5}
        out = []
        for el in osm.get("elements", []):
            if el.get("type") != "node":
                continue
            tags = el.get("tags", {})
            name = tags.get("name")
            if not name:
                continue
            out.append({
                "text": name,
                "lat": el["lat"],
                "lng": el["lon"],
                "kind": "place",
                "rank": rank.get(tags.get("place", ""), 6),
            })
        out.sort(key=lambda p: p["rank"])
        cache_path.write_text(json.dumps(out))
        return out

    def fetch_water(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = WATER_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("water", q)

    def fetch_bridges(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = BRIDGE_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("bridges", q)

    def fetch_coastline(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = COASTLINE_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("coastline", q)

    def fetch_landmarks(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = LANDMARK_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("landmarks", q)

    def fetch_roads(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = ROAD_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("roads", q)

    def fetch_railways(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = RAILWAY_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("railways", q)

    def fetch_parks(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = PARK_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("parks", q)

    def fetch_building_parts(self, lat_min: float, lng_min: float, lat_max: float, lng_max: float) -> dict:
        q = BUILDING_PART_QUERY.format(s=lat_min, w=lng_min, n=lat_max, e=lng_max)
        return self._cached("building_parts", q)
