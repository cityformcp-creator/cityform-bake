# cityform-bake

Batch pre-render every populated 1 km × 1 km tile in Great Britain so the
Cityform storefront can offer a custom-area picker without running a
server per order.

Output of this repo goes to a public Cloudflare R2 bucket
(`cityform-tiles`), one folder per tile keyed by OS National Grid
reference (e.g. `SK35/SK3587/`). Each folder holds:

```
SK3587/
  meta.json     # centre lat/lng, OS grid, nearest place, build timestamp
  city.stl      # print-ready STL (build mesh + LIDAR terrain + buildings)
  city.glb      # Draco-compressed GLB for storefront 3D viewer (~600 KB)
  preview.png   # 512×512 top-down PNG thumbnail for picker hover state
```

A separate Cloudflare Worker handles the runtime piece (customer's
browser uploads the customised label SVG + cart properties), but that's
in `cityform-offline/shopify_migration/cloudflare-worker/`, not here.

## Build steps

```bash
# 1. Pick which tiles to bake (run once, commits tiles.geojson)
python3 src/select_tiles.py --out data/tiles.geojson

# 2. Bake one tile locally (smoke test)
python3 src/bake_tile.py --tile SK3587 --out bake-sample/

# 3. Full bake via GitHub Actions matrix
gh workflow run bake-all.yml
```

## Repo layout

```
cityform-bake/
├── src/
│   ├── select_tiles.py    # Overpass query → tiles.geojson
│   ├── bake_tile.py       # one tile → STL + GLB + PNG
│   └── upload_to_r2.py    # rclone-based R2 sync
├── data/
│   └── tiles.geojson      # ~30k populated GB tiles (committed)
├── bake-sample/           # local smoke-test outputs (gitignored)
└── .github/workflows/
    └── bake-all.yml       # matrix of N runners × M tiles each
```

## Dependencies

The bake driver shells out to `cityform-tool`'s `auto_tier3.py` for the
LIDAR fetch + STL build. You need the full `cityform-offline` repo
checked out alongside this one (or specify its path with `CITYFORM_TOOL`).
