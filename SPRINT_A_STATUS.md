# Sprint A status

**Done locally — 2026-05-25.** Ready for you to push to GitHub.

## What works

- ✅ `src/select_tiles.py` — Geonames → 37,919 GB tiles → `data/tiles.geojson`
- ✅ `src/bake_tile.py` — wraps cityform-tool pipeline, produces STL+GLB+PNG+meta per tile
- ✅ Smoke test: Sheffield (SK3587) baked in 37.3s, all outputs valid (see `bake-sample/SK3587/`)

## What you need to do (~20 min) before Sprint B

### 1. Create the public GitHub repo

```bash
cd ~/Downloads/cityform-bake
gh repo create cityform-bake --public --source . --push \
  --description "Pre-bake GB 1km tiles for Cityform's storefront picker"
```

### 2. Cloudflare R2 bucket

Cloudflare dashboard → R2 → Create bucket:
- Name: `cityform-tiles`
- Public access: ON
- CORS: allow `*.myshopify.com` and `cityform.co.uk`

Generate an R2 API token (Read + Write) and add as GitHub Secrets on the
new `cityform-bake` repo:
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_ACCOUNT_ID`
- `R2_BUCKET_NAME` = `cityform-tiles`

### 3. OS DataHub API key (for the picker UI, not the bake)

https://osdatahub.os.uk/ → sign up → create a project → grab the API
key. Add to the **cityform-offline** repo Wrangler env (not cityform-bake):

```bash
cd ~/Downloads/cityform-offline/shopify_migration/cloudflare-worker
npx wrangler secret put OS_DATAHUB_KEY
```

Then it's available to the picker JSX via the Worker.

## When all three are done

Tell next-Claude: **"All Sprint A setup done. Continue Sprint B."**

Next-Claude will:
- Write `.github/workflows/bake-all.yml` (the GH Actions matrix)
- Write `src/upload_to_r2.py` + a GitHub-Releases STL uploader
- Trigger the 1,000-tile bake
- Start Sprint B's storefront picker (`sections/cf-picker.liquid`, `jsx-src/cf-picker.jsx`)

## What's in the repo right now

```
cityform-bake/
├── README.md
├── SPRINT_A_STATUS.md            (this file)
├── src/
│   ├── select_tiles.py           # Geonames → tiles.geojson
│   └── bake_tile.py              # one tile → STL+GLB+PNG+meta
├── data/
│   ├── tiles.geojson             # 37,919 tiles (committed)
│   └── _geonames_cache.json      # raw GB dump (gitignored — re-generate via --refresh)
└── bake-sample/
    └── SK3587/                   # Sheffield smoke-test output (gitignored)
        ├── city.stl
        ├── city.glb
        ├── preview.png
        ├── wireframe.png
        └── meta.json
```

## Cost confirmation

For top-1,000 launch: **$0/month, forever.**

| Service | Usage | Free tier | Status |
|---|---|---|---|
| Cloudflare R2 | 3.2 GB (GLB+PNG+meta) | 10 GB | ✅ FREE |
| GitHub Releases | 50 GB (STLs) | unlimited on public repo | ✅ FREE |
| GitHub Actions | ~30 min/bake | unlimited on public repo | ✅ FREE |
| OS DataHub | <250k tx/mo | 250k/mo | ✅ FREE |
| Cloudflare Worker | (Phase 2 only) | 100k req/day | ✅ FREE |
| Shopify | (already paid) | — | ✅ existing |
