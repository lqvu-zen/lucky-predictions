/**
 * Cloudflare Worker port of src/crawler.py.
 *
 * On a cron schedule it fetches the latest Power 6/55 and 6/45 draws from
 * vietlott.vn's ajaxpro endpoint, merges them into data/*.jsonl (dedupe by
 * draw id, same record format as the Python crawler) and commits the result
 * to GitHub through the Contents API. That push matches the `data/**` path
 * filter in .github/workflows/daily.yml, which rebuilds and deploys the
 * dashboard — so the PC no longer has to be on.
 *
 * HTTP routes (protected by CRAWL_KEY):
 *   GET /probe?key=...   fetch page 0 of both games, report what parsed (no writes)
 *   GET /run?key=...     run the full crawl+commit now (same as the cron)
 */

const BASE = "https://vietlott.vn/ajaxpro/Vietlott.PlugIn.WebParts.";

// Mirrors the Product dataclass in src/config.py — keep the two in sync.
const PRODUCTS = {
  power_655: {
    label: "Power 6/55",
    url: BASE + "Game655CompareWebPart,Vietlott.PlugIn.WebParts.ashx",
    key: "23bbd667",
    arrayRows: 5,
    referer: "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/winning-number-655",
    path: "data/power655.jsonl",
  },
  power_645: {
    label: "Power 6/45",
    url: BASE + "Game645CompareWebPart,Vietlott.PlugIn.WebParts.ashx",
    key: "8290fce2",
    arrayRows: 6,
    referer: "https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/winning-number-645",
    path: "data/power645.jsonl",
  },
};

const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0";

// ---------------------------------------------------------------- vietlott

/** Anti-bot cookie handshake: /ajaxpro/ sets a cookie via a JS snippet. */
async function getCookie() {
  try {
    const r = await fetch("https://vietlott.vn/ajaxpro/", { headers: { "User-Agent": UA } });
    const text = await r.text();
    const m = text.match(/document\.cookie="(.*?)"/);
    if (m) return m[1].split(";")[0];
  } catch (e) {
    console.log(`[crawl] cookie handshake failed: ${e}`);
  }
  return "";
}

function body(p, pageIndex) {
  return {
    ORenderInfo: {
      SiteId: "main.frontend.vi", SiteAlias: "main.vi", UserSessionId: "",
      SiteLang: "vi", IsPageDesign: false, SiteName: "Vietlott", System: 1,
    },
    Key: p.key,
    GameDrawId: "",
    ArrayNumbers: Array.from({ length: p.arrayRows }, () => Array(18).fill("")),
    CheckMulti: false,
    PageIndex: pageIndex,
  };
}

const stripTags = (s) => s.replace(/<[^>]*>/g, "").replace(/&nbsp;/g, " ").trim();

/** Vietnam wall-clock time as a naive ISO string, like Python's datetime.now(). */
function vnNow() {
  return new Date(Date.now() + 7 * 3600_000).toISOString().replace("Z", "000");
}

/** Port of crawler._parse_html — the results table fragment into records. */
export function parseHtml(html) {
  const rows = [];
  const trs = [...(html || "").matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)].map((m) => m[1]);
  trs.forEach((tr, i) => {
    if (i === 0) return; // header row
    const tds = [...tr.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/gi)].map((m) => m[1]);
    if (tds.length < 3) return;
    const d = stripTags(tds[0]).match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
    if (!d) return;
    const date = `${d[3]}-${d[2].padStart(2, "0")}-${d[1].padStart(2, "0")}`;
    const id = stripTags(tds[1]);
    const numbers = [...tds[2].matchAll(/<span\b[^>]*>([\s\S]*?)<\/span>/gi)]
      .map((m) => stripTags(m[1]))
      .filter((t) => t && t !== "|")
      .map((t) => parseInt(t, 10));
    if (!numbers.length || numbers.some(Number.isNaN)) return;
    rows.push({ date, id, result: numbers, process_time: vnNow() });
  });
  return rows;
}

async function fetchPages(p, from = 0, to = 1) {
  const cookie = await getCookie();
  const out = [];
  const log = [];
  for (let page = from; page <= to; page++) {
    const resp = await fetch(p.url, {
      method: "POST",
      body: JSON.stringify(body(p, page)),
      headers: {
        "User-Agent": UA,
        Accept: "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Content-Type": "text/plain; charset=utf-8",
        "X-AjaxPro-Method": "ServerSideDrawResult",
        "X-Requested-With": "XMLHttpRequest",
        Origin: "https://vietlott.vn",
        Referer: p.referer,
        ...(cookie ? { Cookie: cookie } : {}),
      },
    });
    const text = await resp.text();
    if (!resp.ok) {
      log.push(`page ${page}: HTTP ${resp.status} (${text.length} chars)`);
      throw Object.assign(new Error(`${p.label} HTTP ${resp.status}`), { log, sample: text.slice(0, 300) });
    }
    let html = "";
    try {
      html = JSON.parse(text)?.value?.HtmlContent ?? "";
    } catch {
      log.push(`page ${page}: non-JSON response (${text.length} chars) — likely blocked`);
      continue;
    }
    const rows = parseHtml(html);
    log.push(`page ${page}: ${rows.length} rows`);
    out.push(...rows);
  }
  return { rows: out, log, cookie: Boolean(cookie) };
}

// ------------------------------------------------------------------ github

const b64decode = (s) => new TextDecoder().decode(Uint8Array.from(atob(s.replace(/\n/g, "")), (c) => c.charCodeAt(0)));
function b64encode(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

function gh(env, path, init = {}) {
  return fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/contents/${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "User-Agent": "lucky-predictions-crawler",
      "X-GitHub-Api-Version": "2022-11-28",
      ...(init.headers || {}),
    },
  });
}

/** Serialize exactly like Python's json.dumps(r, ensure_ascii=False). */
const toLine = (r) =>
  `{"date": "${r.date}", "id": "${r.id}", "result": [${r.result.join(", ")}], "process_time": "${r.process_time}"}`;

async function mergeAndCommit(env, p, fetched) {
  const branch = env.GITHUB_BRANCH || "main";
  const get = await gh(env, `${p.path}?ref=${branch}`);
  if (!get.ok) throw new Error(`GitHub GET ${p.path}: ${get.status} ${await get.text()}`);
  const file = await get.json();
  const existingLines = b64decode(file.content).split("\n").filter((l) => l.trim());
  const existing = existingLines.map((l) => JSON.parse(l));

  const seen = new Set(existing.map((r) => r.id));
  const fresh = [];
  for (const r of fetched) if (!seen.has(r.id)) { fresh.push(r); seen.add(r.id); }
  if (!fresh.length) return { new: 0, total: existing.length };

  // Keep existing lines byte-for-byte; only new draws get serialized.
  const merged = existingLines.map((line, i) => ({ line, r: existing[i] }))
    .concat(fresh.map((r) => ({ line: toLine(r), r })));
  merged.sort((a, b) => (a.r.date + a.r.id < b.r.date + b.r.id ? -1 : a.r.date + a.r.id > b.r.date + b.r.id ? 1 : 0));
  const content = merged.map((m) => m.line).join("\n") + "\n";

  const ids = fresh.map((r) => r.id).join(", ");
  const put = await gh(env, p.path, {
    method: "PUT",
    body: JSON.stringify({
      message: `crawl: ${p.label} +${fresh.length} (${ids}) [cloudflare]`,
      content: b64encode(content),
      sha: file.sha,
      branch,
      committer: { name: "lucky-crawler", email: "lucky-crawler@users.noreply.github.com" },
    }),
  });
  if (!put.ok) throw new Error(`GitHub PUT ${p.path}: ${put.status} ${await put.text()}`);
  return { new: fresh.length, total: merged.length, ids };
}

// ----------------------------------------------------------------- handlers

async function crawlAll(env, { commit }) {
  const report = {};
  for (const [name, p] of Object.entries(PRODUCTS)) {
    try {
      const { rows, log, cookie } = await fetchPages(p, 0, Number(env.PAGES ?? 1));
      report[name] = { fetched: rows.length, cookie, log, latest: rows[0] ?? null };
      if (commit) Object.assign(report[name], await mergeAndCommit(env, p, rows));
    } catch (e) {
      report[name] = { error: String(e.message || e), log: e.log, sample: e.sample };
    }
  }
  console.log(JSON.stringify(report));
  return report;
}

export default {
  async scheduled(_event, env, ctx) {
    ctx.waitUntil(crawlAll(env, { commit: true }));
  },

  async fetch(req, env) {
    const url = new URL(req.url);
    if (!env.CRAWL_KEY || url.searchParams.get("key") !== env.CRAWL_KEY) {
      return new Response("not found", { status: 404 });
    }
    const colo = req.cf?.colo;
    if (url.pathname === "/probe") {
      return Response.json({ colo, ...(await crawlAll(env, { commit: false })) });
    }
    if (url.pathname === "/run") {
      return Response.json({ colo, ...(await crawlAll(env, { commit: true })) });
    }
    return new Response("not found", { status: 404 });
  },
};
