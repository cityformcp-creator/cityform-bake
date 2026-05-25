# cityform-bake

Batch pre-render every populated 1 km × 1 km tile in Great Britain so the
Cityform storefront can offer a custom-area picker without running a
server per order.

Output of this repo is uploaded as **GitHub Release assets** on this
repo — no Cloudflare R2, no AWS S3, no card on file anywhere. Each
release tag covers one bake run, the `latest` tag always points at the
freshest output. The Cityform storefront's picker fetches the manifest
from `releases/latest/download/manifest.json` and lazy-loads per-tile
GLB/PNG assets from the same release.

Asset naming convention (flat per release):

```
SK3587__city.stl       (print-ready STL — only the fulfilment workflow reads this)
SK3587__city.glb       (storefront 3D viewer)
SK3587__preview.png    (picker hover thumbnail)
SK3587__meta.json      (per-tile metadata)
manifest.json          (aggregate list of every tile + its asset URLs)
```

The runtime piece (customer's browser → tile selection → /cart/add)
lives in `cityform-offline/shopify_migration/jsx-src/cf-picker.jsx` —
the picker reads from this repo's releases, not from anywhere else.

## Build steps

```bash
# 1. Pick which tiles to bake (run once, commits tiles.geojson)
python3 src/select_tiles.py --out data/tiles.geojson

# 2. Bake one tile locally (smoke test)
python3 src/bake_tile.py --tile-id SK3587 --lat 53.380 --lng -1.464 --out bake-sample/SK3587

# 3. Full bake via GitHub Actions matrix (after .github/workflows/bake-all.yml exists)
gh workflow run bake-all.yml -f tile_limit=1000
```

## Repo layout

```
cityform-bake/
├── src/
│   ├── select_tiles.py        # Geonames query → tiles.geojson
│   ├── bake_tile.py           # one tile → STL + GLB + PNG + meta
│   └── upload_to_release.py   # wraps `gh release upload`
├── data/
│   └── tiles.geojson          # ~38k populated GB tiles (committed)
├── bake-sample/               # local smoke-test outputs (gitignored)
└── .github/workflows/
    └── bake-all.yml           # matrix of N runners × M tiles each
```

## Dependencies

The bake driver shells out to `cityform-tool`'s `auto_tier3.py` for the
LIDAR fetch + STL build. The full `cityform-offline` repo needs to be
checked out alongside this one (or specify its path with `CITYFORM_TOOL`).
On GitHub Actions, the workflow does `git clone cityform-offline` first.
