# lucky-crawler (Cloudflare Worker)

JS port of `src/crawler.py` that runs on a Cloudflare cron and commits new
draws to `data/*.jsonl` via the GitHub Contents API. The push matches the
`data/**` filter in `.github/workflows/daily.yml`, which rebuilds the dashboard.

## Setup

```bash
cd cloudflare/crawler
npm install
npm test                                  # parser vs the same fixture as tests/test_crawler.py
npx wrangler login
npx wrangler secret put CRAWL_KEY         # any random string; guards /probe and /run
npx wrangler secret put GITHUB_TOKEN      # fine-grained PAT: this repo only, Contents read/write
npx wrangler deploy
```

## Step 1 — check that vietlott.vn lets Cloudflare in

```
curl "https://lucky-crawler.<you>.workers.dev/probe?key=<CRAWL_KEY>"
```

- `"page 0: 6 rows"` for both games → it works; leave the cron on.
- `HTTP 403` or `non-JSON response` → vietlott.vn blocks Cloudflare's egress the
  same way it blocks GitHub runners. Disable the cron (`"crons": []`) and keep the PC.

Cron runs don't get Smart Placement, so after the probe passes, also check one
scheduled run in **Workers → lucky-crawler → Logs** (or trigger it from the
dashboard's *Triggers* tab).

## Routes

- `GET /probe?key=…` — fetch only, no writes; shows the colo and what parsed.
- `GET /run?key=…` — crawl and commit now (same as the cron).

Commits are made by a PAT, so unlike `GITHUB_TOKEN` commits they **do** trigger
`daily.yml`.
