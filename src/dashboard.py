"""Generate a self-contained HTML dashboard from the lottery draw data.

Produces a single `reports/dashboard.html` with:
  - hero header + KPI cards (draw count, date range, latest draw)
  - an all-time frequency bar chart (Chart.js from CDN)
  - a full number heatmap (every number shaded by how often it's drawn)
  - hot / cold (last 30 draws) and most-overdue panels
  - next-draw suggested lines (one per strategy)
  - the 10 most recent draws

All data is embedded as JSON, so the file works by just opening it in a
browser — no server needed. Chart.js + Google Fonts load from CDNs (need
internet the first time it's opened; everything else is offline).

⚠️ For fun/learning only — lottery draws are random.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime

from analyze import hot_cold, load_draws, summary
from config import PRODUCTS, REPORTS_DIR, VN_TZ, get_product
from predict import suggest_all
import randomness
import bankroll
import jackpot
import ticket_ev


def _proper_summary(name: str, test_draws: int = 200):
    """Log-loss / Brier of the position grid (best effort — never break build)."""
    try:
        from ml.proper import evaluate
        return evaluate(name, test_draws=test_draws)
    except Exception:  # noqa: BLE001
        return None


def _overfit_summary(name: str):
    """In-sample vs out-of-sample by training size (best effort)."""
    try:
        from ml.genetic import sweep
        return sweep(name)
    except Exception:  # noqa: BLE001
        return None


def _residual_summary(name: str):
    """Observed grid vs theory, as a test (best effort)."""
    try:
        from ml.residual import summary
        return summary(name)
    except Exception:  # noqa: BLE001
        return None


def _ceiling_summary(name: str):
    """Best achievable score + per-model noise bands (best effort)."""
    try:
        from ml.ceiling import with_observed
        return with_observed(name)
    except Exception:  # noqa: BLE001
        return None

try:
    from ml.score import load_scorecard
except Exception:  # ml package optional
    load_scorecard = lambda: None  # noqa: E731

try:
    from ml import joint as _joint      # pure-Python, safe to import
except Exception:
    _joint = None


def _product_payload(name: str, scorecard: dict | None) -> dict:
    product = get_product(name)
    draws = load_draws(product)
    if not draws:
        return {"label": product.label, "draws": 0}

    ml = None
    if scorecard and name in scorecard.get("games", {}):
        ml = scorecard["games"][name]

    s = summary(name)
    hot30, cold30 = hot_cold(draws, product, 30)
    freq = s["frequency"]
    nums = list(range(product.min_value, product.max_value + 1))
    recent = [
        {"date": d["date"], "id": d["id"], "main": d["main"],
         "bonus": d["result"][-1] if len(d["result"]) > product.main_count else None}
        for d in reversed(draws[-10:])
    ]

    # "for fun" heuristic lines, seeded by the next draw date so they stay
    # locked for a given draw instead of reshuffling daily
    seed = int(product.next_draw_date().strftime("%Y%m%d"))
    predictions = {k: v[0] for k, v in suggest_all(name, tickets=1, seed=seed).items()}

    joint_data = None
    if _joint is not None:
        try:
            gt = _joint.grid_transposed(product, draws)
            ticket = _joint.predict_ticket(
                _joint.empirical_grid(product, draws, smoothing=0.5), product)
            mx = max((max(row) for row in gt), default=0) or 1.0
            joint_data = {"gridT": [[round(x, 5) for x in row] for row in gt],
                          "ticket": ticket, "max": mx}
        except Exception:
            joint_data = None
    return {
        "label": product.label,
        "range": {"min": product.min_value, "max": product.max_value},
        "draws": len(draws),
        "date_range": s["date_range"],
        "latest": recent[0],
        "freq_labels": nums,
        "freq_values": [freq[n] for n in nums],
        "most_common": s["most_common"],
        "hot": hot30,
        "cold": cold30,
        "overdue": s["most_overdue"],
        "recent": recent,
        "predictions": predictions,
        "next_draw": product.next_draw_date().isoformat(),
        "joint": joint_data,
        "randomness": randomness.summary(name),
        "bankroll": bankroll.simulate(name),
        "jackpot": jackpot.summary(name),
        "proper": _proper_summary(name),
        "ceiling": _ceiling_summary(name),
        "residual": _residual_summary(name),
        "ev": ticket_ev.summary(name),
        "overfit": _overfit_summary(name),
        "ml": ml,
    }


# Brand mark: a gold four-leaf clover badge (works as favicon + header logo)
LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<defs><linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#ffe89a"/><stop offset="1" stop-color="#f2a900"/>'
    '</linearGradient></defs>'
    '<rect x="3" y="3" width="58" height="58" rx="15" fill="#141b2b" stroke="#33406b"/>'
    '<path d="M32 37 C31 45 31 49 33 53" fill="none" stroke="#37e0a6" '
    'stroke-width="3" stroke-linecap="round"/>'
    '<g fill="url(#lg)">'
    '<circle cx="32" cy="21" r="8.6"/><circle cx="23" cy="30" r="8.6"/>'
    '<circle cx="41" cy="30" r="8.6"/><circle cx="32" cy="39" r="8.6"/></g>'
    '<circle cx="32" cy="30" r="3.2" fill="#141b2b"/>'
    '</svg>'
)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lucky Predictions Dashboard</title>
<link rel="icon" type="image/svg+xml" href="__FAVICON__">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0b0f1a; --bg2:#0e1524;
    --card:rgba(255,255,255,.045); --card-brd:rgba(255,255,255,.09);
    --ink:#eef2fb; --muted:#93a0bd; --faint:#5f6b85;
    --gold:#f7c948; --gold2:#ff9f1c;
    --hot:#ff5d6c; --cold:#4da6ff; --mint:#37e0a6; --violet:#9b6dff;
    --line:rgba(255,255,255,.08);
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{
    margin:0; color:var(--ink);
    font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif;
    background:
      radial-gradient(1200px 600px at 80% -10%, rgba(155,109,255,.18), transparent 60%),
      radial-gradient(1000px 500px at -10% 10%, rgba(77,166,255,.14), transparent 55%),
      linear-gradient(180deg,var(--bg),var(--bg2));
    background-attachment:fixed; min-height:100vh;
  }
  .wrap{max-width:1080px; margin:0 auto; padding:28px 22px 56px}
  h1,h2,h3,.mono{font-family:"Space Grotesk",Inter,sans-serif}

  /* hero */
  .hero{
    position:relative; overflow:hidden; border-radius:22px; padding:28px 28px 24px;
    background:linear-gradient(135deg, rgba(247,201,72,.16), rgba(155,109,255,.14) 55%, rgba(77,166,255,.12));
    border:1px solid var(--card-brd);
    box-shadow:0 20px 60px -30px rgba(0,0,0,.8);
  }
  .brand{display:flex; align-items:center; gap:16px}
  .logo{flex:0 0 auto; width:56px; height:56px; filter:drop-shadow(0 8px 16px rgba(247,201,72,.28))}
  .logo svg{width:100%; height:100%; display:block}
  .hero h1{margin:0; font-size:30px; letter-spacing:-.5px;
    background:linear-gradient(90deg,var(--gold),#fff 60%); -webkit-background-clip:text; background-clip:text; color:transparent}
  .hero .sub{color:var(--muted); font-size:13px; margin-top:6px}
  .pill{display:inline-flex; align-items:center; gap:7px; margin-top:14px; padding:7px 13px;
    font-size:12px; color:#ffe9a8; background:rgba(247,201,72,.12);
    border:1px solid rgba(247,201,72,.28); border-radius:999px}
  .pill .dot{width:7px;height:7px;border-radius:50%;background:var(--gold);box-shadow:0 0 10px var(--gold)}

  /* tabs */
  .tabs{display:flex; gap:10px; margin:22px 0 4px}
  .tab{padding:11px 20px; border-radius:14px; cursor:pointer; font-weight:600; font-size:14px;
    color:var(--muted); background:var(--card); border:1px solid var(--card-brd); transition:.18s}
  .tab:hover{color:var(--ink); transform:translateY(-1px)}
  .tab.active{color:#12151d;
    background:linear-gradient(135deg,var(--gold),var(--gold2)); border-color:transparent;
    box-shadow:0 10px 24px -10px rgba(247,201,72,.6)}

  .panel{display:none; animation:fade .35s ease}
  .panel.active{display:block}
  @keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}

  /* kpis */
  .kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin-top:18px}
  .kpi{background:var(--card); border:1px solid var(--card-brd); border-radius:16px; padding:16px 18px;
    backdrop-filter:blur(8px)}
  .kpi .l{font-size:11px; text-transform:uppercase; letter-spacing:.8px; color:var(--faint)}
  .kpi .n{font-family:"Space Grotesk"; font-size:24px; font-weight:700; margin-top:6px}
  .kpi .n.small{font-size:16px}

  /* cards grid */
  .grid{display:grid; grid-template-columns:repeat(12,1fr); gap:16px; margin-top:16px}
  .card{background:var(--card); border:1px solid var(--card-brd); border-radius:18px; padding:18px 20px;
    backdrop-filter:blur(8px)}
  .card h3{margin:0 0 14px; font-size:12px; text-transform:uppercase; letter-spacing:.9px; color:var(--muted);
    display:flex; align-items:center; gap:8px}
  .card h3 .ic{width:8px;height:8px;border-radius:50%}
  .col12{grid-column:span 12} .col8{grid-column:span 8} .col6{grid-column:span 6} .col4{grid-column:span 4}
  @media(max-width:820px){.col8,.col6,.col4{grid-column:span 12}}
  /* bounded box stops Chart.js from growing infinitely */
  .chartbox{position:relative; height:300px; width:100%}
  .chartbox canvas{position:absolute; inset:0; width:100%!important; height:100%!important}

  /* balls */
  .balls{display:flex; flex-wrap:wrap; gap:9px; margin-top:16px}
  .ball{width:44px;height:44px;border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-family:"Space Grotesk"; font-weight:700; font-size:16px; color:#0c1020;
    background:radial-gradient(circle at 32% 28%, #fff, #cdd6ea 70%, #aeb9d6);
    box-shadow:0 6px 14px -6px rgba(0,0,0,.7), inset 0 -3px 6px rgba(0,0,0,.18);
    animation:pop .4s backwards}
  .ball.bonus{color:#2a1c00; background:radial-gradient(circle at 32% 28%, #ffe9a6, var(--gold) 65%, var(--gold2))}
  .ball.sm{width:34px;height:34px;font-size:13px}
  @keyframes pop{from{opacity:0;transform:scale(.5)}to{opacity:1;transform:none}}

  /* heatmap */
  .heat{display:grid; grid-template-columns:repeat(11,1fr); gap:7px}
  @media(max-width:820px){.heat{grid-template-columns:repeat(9,1fr)}}
  .cell{aspect-ratio:1; border-radius:10px; display:flex; flex-direction:column; align-items:center; justify-content:center;
    font-size:13px; font-weight:600; color:#0b0f1a; position:relative; border:1px solid rgba(255,255,255,.08)}
  .cell small{font-size:9px; opacity:.7; font-weight:600}
  .legend{display:flex; align-items:center; gap:8px; margin-top:12px; font-size:11px; color:var(--muted)}
  .legend .bar{flex:1; height:8px; border-radius:6px;
    background:linear-gradient(90deg,#2b4a7a,#4da6ff,#37e0a6,#f7c948,#ff5d6c)}
  /* consensus numbers */
  .cnums{display:flex; flex-wrap:wrap; gap:6px; margin-top:4px}
  .cnum{display:inline-flex; flex-direction:column; align-items:center; justify-content:center;
    width:40px; height:42px; border-radius:10px; font-weight:700; color:#0b0f1a; font-size:14px;
    font-family:"Space Grotesk"}
  .cnum small{font-size:9px; opacity:.8; font-weight:600}
  /* joint number x position heatmap */
  .jrow{display:flex; align-items:center; gap:8px; margin-bottom:3px}
  .jlab{width:26px; text-align:right; font-size:11px; color:var(--muted); font-family:"Space Grotesk"}
  .jline{display:grid; flex:1; gap:1px}
  .jcell{height:16px; border-radius:2px}
  .jaxis{display:flex; justify-content:space-between; font-size:10px; color:var(--faint); margin:5px 0 0 34px}
  /* prediction history */
  .hscroll{max-height:420px; overflow-y:auto; margin:2px -4px 10px 0; padding-right:8px}
  .hscroll::-webkit-scrollbar{width:8px}
  .hscroll::-webkit-scrollbar-thumb{background:var(--card-brd); border-radius:8px}
  .hdraw{border:1px solid var(--card-brd); border-radius:12px; padding:12px 14px; margin-bottom:12px}
  .hhead{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:8px}
  .hdate{font-family:"Space Grotesk"; font-weight:700; font-size:13px}
  .tk{display:inline-flex; gap:3px}
  .tkb{display:inline-flex; align-items:center; justify-content:center; width:24px; height:22px;
    border-radius:5px; font-size:11px; font-weight:700; background:#20293b; color:var(--muted); border:1px solid var(--line)}
  .tkb.x{background:var(--mint); color:#08110c}          /* correct pos */
  .tkb.o{background:rgba(247,201,72,.22); color:var(--gold); border-color:rgba(247,201,72,.4)} /* right number */
  .htab{width:100%; border-collapse:collapse; font-size:12px}
  .htab td{padding:4px 6px; border-bottom:1px solid var(--line)}
  .htab td:first-child{color:var(--muted); white-space:nowrap}
  .sc{font-family:"Space Grotesk"; font-weight:700}

  /* lists */
  table{width:100%; border-collapse:collapse; font-size:13.5px}
  td,th{padding:8px 6px; text-align:left; border-bottom:1px solid var(--line)}
  tr:last-child td{border-bottom:none}
  th{color:var(--faint); font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.6px}
  .num{display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:9px;
    font-family:"Space Grotesk"; font-weight:700; font-size:13px}
  .num.hot{background:rgba(255,93,108,.16); color:#ff8a95; border:1px solid rgba(255,93,108,.3)}
  .num.cold{background:rgba(77,166,255,.14); color:#8cc4ff; border:1px solid rgba(77,166,255,.3)}
  .num.over{background:rgba(247,201,72,.14); color:#ffd970; border:1px solid rgba(247,201,72,.3)}
  .mini{height:6px; border-radius:4px; background:linear-gradient(90deg,var(--cold),var(--gold)); }

  /* predictions */
  .pred{display:flex; flex-direction:column; gap:12px}
  .pred .row{display:flex; align-items:center; gap:14px; flex-wrap:wrap}
  .pred .tag{width:84px; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; color:var(--muted)}
  .recent-nums{font-family:"Space Grotesk"; letter-spacing:1px; font-weight:600}
  .recent-nums .b{color:var(--gold)}

  footer{margin-top:30px; text-align:center; color:var(--faint); font-size:12px}
  a{color:var(--cold); text-decoration:none}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="brand">
      <span class="logo">__LOGO__</span>
      <div>
        <h1>Lucky Predictions</h1>
        <div class="sub">Power 6/55 &amp; 6/45 &middot; generated __GENERATED__</div>
      </div>
    </div>
    <div class="pill"><span class="dot"></span>Lottery draws are random — stats describe the past and can't predict the future. For fun only.</div>
  </div>

  <div class="tabs" id="tabs"></div>
  <div id="main"></div>

  <footer>Built with the lucky-predictions project · numbers are for entertainment, not advice.</footer>
</div>

<script>
const DATA = __DATA__;
const keys = Object.keys(DATA);
const charts = {};

const pad = n => String(n).padStart(2,'0');
function vnd(x){
  if(Math.abs(x) >= 1e9) return (x/1e9).toFixed(1)+'bn';
  if(Math.abs(x) >= 1e6) return (x/1e6).toFixed(1)+'m';
  return Math.round(x).toLocaleString();
}
function balls(main, bonus, sm){
  const c = sm ? ' sm' : '';
  let h = main.map((n,i)=>`<div class="ball${c}" style="animation-delay:${i*45}ms">${pad(n)}</div>`).join('');
  if(bonus!=null) h += `<div class="ball bonus${c}" style="animation-delay:${main.length*45}ms">${pad(bonus)}</div>`;
  return `<div class="balls">${h}</div>`;
}
// blue -> mint -> gold -> red ramp for a normalized value t in [0,1]
function heatColor(t){
  const stops = [[43,74,122],[77,166,255],[55,224,166],[247,201,72],[255,93,108]];
  const x = Math.max(0,Math.min(1,t))*(stops.length-1);
  const i = Math.floor(x), f = x-i;
  const a = stops[i], b = stops[Math.min(i+1,stops.length-1)];
  const c = a.map((v,k)=>Math.round(v+(b[k]-v)*f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function rowsTable(list, cls, unit){
  const max = Math.max(...list.map(x=>x[1]),1);
  return list.map(([n,v])=>`<tr>
    <td><span class="num ${cls}">${n}</span></td>
    <td>${v}${unit||''}</td>
    <td style="width:38%"><div class="mini" style="width:${Math.max(8,v/max*100)}%"></div></td>
  </tr>`).join('');
}
// small balls for a ticket, coloured vs the actual draw:
//  green = correct number at correct sorted position, amber = right number
//  wrong position, plain = miss
function ticketBalls(ticket, actual){
  const act = actual.slice().sort((a,b)=>a-b);
  const tik = ticket.slice().sort((a,b)=>a-b);
  const set = new Set(act);
  return `<span class="tk">` + tik.map((v,i)=>{
    const cls = (v===act[i]) ? 'x' : (set.has(v) ? 'o' : '');
    return `<span class="tkb ${cls}">${pad(v)}</span>`;
  }).join('') + `</span>`;
}

const tabs = document.getElementById('tabs');
const main = document.getElementById('main');

keys.forEach((k,idx)=>{
  const d = DATA[k];
  const btn = document.createElement('button');
  btn.className='tab'+(idx===0?' active':''); btn.textContent=d.label;
  btn.onclick=()=>activate(k); tabs.appendChild(btn);

  const p=document.createElement('div');
  p.className='panel'+(idx===0?' active':''); p.id='panel-'+k;

  if(!d.draws){ p.innerHTML=`<div class="card" style="margin-top:16px">No data yet for ${d.label}.</div>`; main.appendChild(p); return; }

  const avg = d.freq_values.reduce((a,b)=>a+b,0)/d.freq_values.length;
  const fmin = Math.min(...d.freq_values), fmax = Math.max(...d.freq_values);
  const heat = d.freq_labels.map((n,i)=>{
    const v=d.freq_values[i], t=(v-fmin)/((fmax-fmin)||1);
    const bright = t>0.55;
    return `<div class="cell" title="Number ${n}: drawn ${v} times"
      style="background:${heatColor(t)};color:${bright?'#0b0f1a':'#0b0f1a'}">${pad(n)}<small>${v}</small></div>`;
  }).join('');

  const predRows = Object.entries(d.predictions||{}).map(([s,line])=>
    `<div class="row"><div class="tag">${s}</div>${balls(line,null,true)}</div>`).join('');

  let mlCard = '';
  if(d.ml){
    const m = d.ml;
    const np = m.next_prediction;
    // leaderboard: rank every predictor by the k/6 position score, then hits
    const ranked = Object.entries(m.models||{})
      .sort((a,b)=> (b[1].mean_pos_score - a[1].mean_pos_score) || (b[1].mean_hits - a[1].mean_hits));
    const rows = ranked.map(([k,v],i)=>{
      const isC = k==='consensus';
      const lead = i===0 ? 'style="background:rgba(55,224,166,.10)"' : (isC?'style="background:rgba(247,201,72,.10)"':'');
      const medal = i===0?'🥇':i===1?'🥈':i===2?'🥉':(i+1)+'.';
      const pct = (v.mean_pos_score*100).toFixed(1);
      return `<tr ${lead}><td>${medal}</td><td>${isC?'⭐ ':''}<b${isC?' style="color:var(--gold)"':''}>${k}</b></td><td>${v.scored}</td>
        <td><b>${v.mean_pos_score.toFixed(3)}</b> <span style="color:var(--faint)">(${pct}%)</span></td>
        <td style="color:var(--muted)">${v.mean_hits.toFixed(2)}</td>
        <td style="color:var(--faint)">${v.best_pos_hits}/6</td></tr>`;
    }).join('');
    // order next-draw tickets by track record: best k/6, then how often it hit
    // that best (falls back to plain key order for older scorecards)
    const npOrder = (np && np.model_order) ? np.model_order : (np ? Object.keys(np.by_model) : []);
    const npStats = (np && np.model_stats) ? np.model_stats : {};
    const nextLines = np ? npOrder.map((k,i)=>{
        const line = np.by_model[k]; if(!line) return '';
        const st = npStats[k];
        const badge = st
          ? `<span style="color:var(--faint);font-size:11px;margin-left:6px" title="best ${st.best_pos_hits}/6, reached ${st.best_pos_count}x over ${st.scored} scored draws">${st.best_pos_hits}/6 &times;${st.best_pos_count}</span>`
          : `<span style="color:var(--faint);font-size:11px;margin-left:6px">no record yet</span>`;
        const top = (i===0 && st) ? ' style="background:rgba(55,224,166,.10)"' : '';
        return `<div class="row"${top}><div class="tag">${k}${badge}</div>${balls(line,null,true)}</div>`;
      }).join('') : '';
    const scored = m.total_scored||0;
    mlCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--violet)"></span>Leaderboard ${scored?`· ${scored} predictions scored`:'· awaiting first results'}</h3>
        ${rows ? `<table><thead><tr><th>#</th><th>Predictor</th><th>Scored</th><th>Score (k/6)</th><th>Hits</th><th>Best</th></tr></thead><tbody>${rows}</tbody></table>
          <div style="color:var(--faint);font-size:11px;margin-top:10px"><b>Score = k / 6</b>, where k = correct number at the correct sorted position, averaged over scored draws (e.g. actual 1-2-3-4-5-6 vs guess 1-19-29-37-36-55 scores 1/6 ≈ 0.167). Best-possible if you always guessed each position's mode ≈ <b>${(m.pos_baseline_score||0).toFixed(3)}</b>. Hits = number overlap. The leader is luck: over enough draws every predictor converges — there is no real edge.</div>`
          : `<div style="color:var(--muted);font-size:13px">No predictions scored yet. After the next draw is crawled and scored, the ranking appears here.</div>`}
        ${np && np.consensus ? (()=>{
          const maxV = np.consensus[0][1];
          const chips = np.consensus.slice(0,12).map(([num,c])=>
            `<span class="cnum" style="background:${heatColor(c/maxV)}" title="${c} of ${np.n_models} predictors">${pad(num)}<small>${c}</small></span>`).join('');
          return `<h3 style="margin-top:18px"><span class="ic" style="background:var(--gold)"></span>Consensus for next draw · ${np.target_date}</h3>
            <div style="font-size:12px;color:var(--muted);margin-bottom:8px">Top-6 by votes: ${balls(np.consensus_ticket,null,true)}</div>
            <div class="cnums">${chips}</div>
            <div style="color:var(--faint);font-size:11px;margin-top:8px">Numbers the most predictors (of ${np.n_models}) agree on. The small figure is the vote count. Consensus is just aggregation — it still can't beat the odds.</div>`;
        })() : ''}
        ${np ? `<h3 style="margin-top:18px"><span class="ic" style="background:var(--mint)"></span>Each predictor · ${np.target_date}</h3><div class="pred">${nextLines}</div>
          <div style="color:var(--faint);font-size:11px;margin-top:8px">Ordered by track record: highest best-ever k/6 first, then how many times that best was reached. The badge reads best/6 &times; times. Ordering is cosmetic — a predictor's past position-accuracy says nothing about the next draw.</div>` : ''}
      </div>`;
  }

  let trendCard = '';
  if(d.ml && d.ml.trend && (d.ml.trend.labels||[]).length>1){
    trendCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--cold)"></span>Accuracy trend · running score (k/6) over draws</h3>
        <div class="chartbox"><canvas id="trend-${k}"></canvas></div>
        <div style="color:var(--faint);font-size:11px;margin-top:8px">Each predictor's running mean position-score as draws accumulate. They wobble early, then converge toward the dashed mode-baseline (${(d.ml.pos_baseline_score||0).toFixed(3)}) — the "leader" keeps changing, which is exactly what no-edge looks like.</div>
      </div>`;
  }

  let histCard = '';
  if(d.ml && (d.ml.history||[]).length){
    const draws = d.ml.history.map(h=>{
      const k = h.actual.length;
      const rows = h.preds.map(pr=>{
        const isC = pr.model==='consensus';
        return `<tr ${isC?'style="background:rgba(247,201,72,.14)"':''}>
        <td>${isC?'⭐ ':''}<b${isC?' style="color:var(--gold)"':'' }>${pr.model}</b></td>
        <td>${ticketBalls(pr.ticket, h.actual)}</td>
        <td class="sc">${pr.pos}/${k}</td>
        <td style="color:var(--muted)">${pr.hits} hits</td></tr>`;}).join('');
      return `<div class="hdraw">
        <div class="hhead"><span class="hdate">${h.date}</span>
          <span style="color:var(--faint);font-size:11px">actual</span>${balls(h.actual,null,true)}</div>
        <table class="htab"><tbody>${rows}</tbody></table>
      </div>`;
    }).join('');
    histCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--gold)"></span>Past predictions vs actual result</h3>
        <div class="hscroll">${draws}</div>
        <div style="color:var(--faint);font-size:11px">
          <span style="color:var(--mint)">green</span> = correct number at the correct sorted position ·
          <span style="color:var(--gold)">amber</span> = right number, wrong position · score = correct-position / 6.
        </div>
      </div>`;
  }

  let jointCard = '';
  if(d.joint){
    const N=d.range.max, mx=d.joint.max, gt=d.joint.gridT, K=gt.length;
    let rows='';
    for(let pp=0; pp<K; pp++){
      let line='';
      for(let i=0;i<N;i++){
        const v=gt[pp][i];
        line+=`<div class="jcell" title="number ${i+1} at position p${pp+1}: ${(v*100).toFixed(1)}%" style="background:${heatColor(v/mx)}"></div>`;
      }
      rows+=`<div class="jrow"><div class="jlab">p${pp+1}</div><div class="jline" style="grid-template-columns:repeat(${N},1fr)">${line}</div></div>`;
    }
    jointCard=`
      <div class="card col12">
        <h3><span class="ic" style="background:var(--violet)"></span>Number × position map — P(each number lands at each ordered slot)</h3>
        ${rows}
        <div class="jaxis"><span>number 1</span><span>${N}</span></div>
        <div class="row" style="margin-top:14px"><div class="tag">model pick</div>${balls(d.joint.ticket,null,true)}</div>
        <div style="color:var(--faint);font-size:11px;margin-top:10px">Each row is one ordered position's distribution over the numbers. The diagonal band is the order-statistic law of a random draw — identical every time, so it's the best view but carries no edge.</div>
      </div>`;
  }

  let jackCard = '';
  if(d.jackpot){
    const j = d.jackpot;
    const bil = n => (n/1e9).toFixed(0)+'B';
    jackCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--gold)"></span>How long to win the jackpot?</h3>
        <div class="kpis">
          <div class="kpi"><div class="l">Odds per line</div><div class="n small">1 in ${j.one_in.toLocaleString()}</div></div>
          <div class="kpi"><div class="l">Expected wait</div><div class="n">${j.expected_years.toLocaleString()} yrs</div></div>
          <div class="kpi"><div class="l">Expected spend to win once</div><div class="n small">${bil(j.expected_cost)} VND</div></div>
          <div class="kpi"><div class="l">vs the jackpot</div><div class="n">${j.cost_vs_jackpot}×</div></div>
        </div>
        <div style="color:var(--faint);font-size:11px;margin-top:10px">Buying one line every draw, you'd wait ~${j.expected_years.toLocaleString()} years and spend ~${bil(j.expected_cost)} VND to win the jackpot once on average — about <b>${j.cost_vs_jackpot}×</b> the prize itself. You're ~<b>${j.lightning_ratio}×</b> more likely to be struck by lightning this year than to win with a single line.</div>
      </div>`;
  }

  let bankCard = '';
  if(d.bankroll && d.bankroll.draws){
    const b = d.bankroll;
    const fmt = n => (n/1e6).toFixed(2)+'M';
    const trows = Object.entries(b.totals).map(([s,v])=>
      `<tr><td>${s}</td><td>${fmt(v.spent)}</td><td>${fmt(v.won)}</td>
        <td class="sc" style="color:var(--hot)">${fmt(v.net)}</td>
        <td style="color:var(--hot)">${v.return_pct}%</td></tr>`).join('');
    // real bankroll for every predictor from its logged results (grows over time)
    let realTable = '';
    if(d.ml && d.ml.models){
      const rr = Object.entries(d.ml.models)
        .filter(([,v])=>v.scored>0)
        .sort((a,b)=> b[1].net - a[1].net);
      if(rr.length){
        const vnd = n => (n/1e3).toFixed(0)+'k';
        realTable = `<h3 style="margin-top:18px"><span class="ic" style="background:var(--gold)"></span>Real P/L to date · every predictor (${d.ml.total_scored} scored)</h3>
          <table><thead><tr><th>Predictor</th><th>Draws</th><th>Spent</th><th>Won</th><th>Net</th><th>Return</th></tr></thead><tbody>`
          + rr.map(([m,v])=>`<tr ${m==='consensus'?'style="background:rgba(247,201,72,.10)"':''}>
              <td>${m==='consensus'?'⭐ ':''}<b>${m}</b></td><td>${v.scored}</td>
              <td>${vnd(v.spent)}</td><td>${vnd(v.won)}</td>
              <td class="sc" style="color:${v.net<0?'var(--hot)':'var(--mint)'}">${vnd(v.net)}</td>
              <td style="color:${v.return_pct<0?'var(--hot)':'var(--mint)'}">${v.return_pct}%</td></tr>`).join('')
          + `</tbody></table>
          <div style="color:var(--faint);font-size:11px;margin-top:6px">Real out-of-sample results from each logged prediction (models, fun lines, and consensus). Grows with every draw.</div>`;
      }
    }
    bankCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--hot)"></span>If you actually played every draw · bankroll</h3>
        <div style="color:var(--faint);font-size:11px;margin-bottom:6px">Simulated over all ${b.draws.toLocaleString()} draws (representative strategies), 1 line/draw @ ${(b.cost/1000)}k VND:</div>
        <div class="chartbox"><canvas id="bank-${k}"></canvas></div>
        <table style="margin-top:12px"><thead><tr><th>Strategy</th><th>Spent</th><th>Won</th><th>Net</th><th>Return</th></tr></thead><tbody>${trows}</tbody></table>
        ${realTable}
        <div style="color:var(--faint);font-size:11px;margin-top:8px">Every line trends down — that's the house edge. No ticket, model or system changes it.</div>
      </div>`;
  }

  let randCard = '';
  if(d.randomness && d.randomness.uniformity){
    const R=d.randomness, u=R.uniformity, oe=R.odd_even, rp=R.repeats;
    const ok = u.p > 0.05;
    randCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:${ok?'var(--mint)':'var(--hot)'}"></span>Is the draw actually random?</h3>
        <table><tbody>
          <tr><td>Chi-square uniformity</td><td>X² = ${u.chi2} (dof ${u.dof})</td>
              <td class="sc" style="color:${ok?'var(--mint)':'var(--hot)'}">p = ${u.p}</td></tr>
          <tr><td>Odd / even balance</td><td>${oe.odd} odd · ${oe.even} even</td>
              <td class="sc">p = ${oe.p}</td></tr>
          <tr><td>Repeats vs previous draw</td><td>mean ${rp.mean_repeat}</td>
              <td style="color:var(--muted)">expected ${rp.expected}</td></tr>
        </tbody></table>
        <div style="color:var(--faint);font-size:11px;margin-top:8px">Over ${R.draws.toLocaleString()} draws. A high p-value means we <b>cannot reject</b> a fair uniform draw. ${R.verdict}</div>
      </div>`;
  }

  let properCard = '';
  if(d.proper && d.proper.models){
    const P = d.proper, floor = P.entropy_floor;
    const rows = Object.entries(P.models).sort((a,b)=> a[1].logloss - b[1].logloss);
    const labels = {
      'uniform':'Uniform (knows nothing)',
      'theory':'Theory (closed-form law)',
      'empirical':'Learned from all history',
      'empirical-100':'Learned from last 100 draws',
      'shrunk':'Shrunk toward theory'
    };
    const body = rows.map(([k,v],i)=>{
      const good = i===0 ? 'style="background:rgba(55,224,166,.10)"' : '';
      const ex = v.excess>=0 ? `+${v.excess.toFixed(4)}` : v.excess.toFixed(4);
      return `<tr ${good}><td>${labels[k]||k}</td>
        <td class="sc"><b>${v.logloss.toFixed(4)}</b></td>
        <td class="sc" style="color:var(--faint)">[${v.logloss_lo.toFixed(3)}, ${v.logloss_hi.toFixed(3)}]</td>
        <td class="sc" style="color:var(--muted)">${ex}</td>
        <td class="sc" style="color:var(--muted)">${v.brier.toFixed(4)}</td></tr>`;
    }).join('');
    properCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--violet)"></span>How much does each model really know? · log-loss</h3>
        <table><thead><tr><th>Model</th><th class="sc">Log-loss</th><th class="sc">95% CI</th><th class="sc">vs floor</th><th class="sc">Brier</th></tr></thead>
        <tbody>${body}</tbody></table>
        <div style="color:var(--faint);font-size:11px;margin-top:8px">
          Instead of asking "did it guess the number", this grades the whole probability
          the model put on the number that actually landed at each position — a
          <b>proper scoring rule</b>, far less noisy than k/6. Lower is better, in nats, over ${P.tested} draws.
          The <b>floor is ${floor.toFixed(4)}</b>: the entropy of the true position law, i.e. the irreducible
          uncertainty of a fair draw. Nothing that only reads past draws can beat it — and nothing does.
          "Uniform" is much worse, which proves the position law is <i>real</i> structure (position 1 really is
          usually small); but that structure is identical every draw, so it still predicts nothing.
        </div>
      </div>`;
  }

  let ceilCard = '';
  if(d.ceiling && d.ceiling.ceiling_score!=null){
    const C = d.ceiling;
    const obs = (C.observed||[]);
    // scale bars against the widest band so the noise is visually obvious
    const top = Math.max(C.ceiling_score*1.6, ...obs.map(o=>o.band_hi||0), 0.02);
    const bars = obs.map(o=>{
      const l = 100*o.band_lo/top, w = Math.max(100*(o.band_hi-o.band_lo)/top, 0.6);
      const dot = 100*Math.min(o.score,top)/top;
      const col = o.in_band ? 'var(--mint)' : 'var(--hot)';
      return `<tr><td style="white-space:nowrap">${o.model}</td>
        <td class="sc" style="color:var(--muted)">${o.scored}</td>
        <td class="sc"><b>${o.score.toFixed(4)}</b></td>
        <td class="sc" style="color:${o.p_value<0.05?'var(--gold)':'var(--faint)'}">${o.p_value.toFixed(3)}</td>
        <td style="width:55%">
          <div style="position:relative;height:14px;background:rgba(255,255,255,.04);border-radius:7px">
            <div style="position:absolute;left:${l}%;width:${w}%;top:0;bottom:0;background:rgba(255,255,255,.13);border-radius:7px"></div>
            <div style="position:absolute;left:${100*C.ceiling_score/top}%;top:-2px;bottom:-2px;width:2px;background:var(--gold)"></div>
            <div style="position:absolute;left:calc(${dot}% - 3px);top:3px;width:8px;height:8px;border-radius:50%;background:${col}"></div>
          </div></td></tr>`;
    }).join('');
    ceilCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--gold)"></span>How good could a <i>perfect</i> model be?</h3>
        <div class="kpis">
          <div class="kpi"><div class="l">Ceiling (knows the exact law)</div><div class="n">${C.ceiling_score.toFixed(4)}</div></div>
          <div class="kpi"><div class="l">A random ticket</div><div class="n">${C.random_score.toFixed(4)}</div></div>
          <div class="kpi"><div class="l">Worth of perfect knowledge</div><div class="n">+${(C.ceiling_score-C.random_score).toFixed(4)}</div></div>
          <div class="kpi"><div class="l">Optimal ticket</div><div class="n small">${C.optimal_ticket.join(' · ')}</div></div>
        </div>
        ${bars ? `<table style="margin-top:12px"><thead><tr><th>Predictor</th><th class="sc">n</th><th class="sc">Score</th><th class="sc">p</th>
          <th>Where a perfect model would land over the same n draws</th></tr></thead><tbody>${bars}</tbody></table>
          <div style="color:var(--faint);font-size:11px;margin-top:6px">
            <b>p</b> = the chance a skill-free player (a random ticket) scores at least that well.
            Best p here is <b>${C.min_p.toFixed(4)}</b> — which looks impressive until you remember we are testing
            <b>${C.n_models}</b> predictors at once. Corrected for that, Šidák p = <b>${C.sidak_p.toFixed(4)}</b>:
            ${C.any_significant ? 'significant — worth investigating.' : 'nothing significant. Test enough candidates and one always looks gifted.'}
          </div>` : ''}
        <div style="color:var(--faint);font-size:11px;margin-top:8px">
          The gold line is the ceiling: the best expected k/6 anyone could reach <b>even knowing the draw law exactly</b>
          (${C.ceiling_score.toFixed(4)} vs ${C.random_score.toFixed(4)} for a random ticket — perfect knowledge is worth
          almost nothing). The grey bar is the 95% range a <i>perfect</i> model would still land in over that predictor's
          own number of scored draws; the dot is where it actually landed. Every dot sits inside its bar, so no predictor
          is doing anything a perfect one wouldn't do by luck alone — and the bars are far wider than the gaps between
          predictors, which is why the leaderboard keeps changing hands.
        </div>
      </div>`;
  }

  let overfitCard = '';
  if(d.overfit && (d.overfit.rows||[]).length){
    const O = d.overfit;
    const rws = O.rows.map(x=>{
      const gap = x.train_score - x.test_score;
      const over = x.train_score > O.ceiling_score;
      return `<tr><td class="sc">${x.n_train}</td>
        <td class="sc" style="color:${over?'var(--hot)':'var(--muted)'}"><b>${x.train_score.toFixed(4)}</b></td>
        <td class="sc" style="color:var(--muted)">${x.test_score.toFixed(4)}</td>
        <td class="sc" style="color:var(--faint)">+${gap.toFixed(4)}</td></tr>`;
    }).join('');
    overfitCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--hot)"></span>Why "it worked on past draws" means nothing</h3>
        <div class="chartbox"><canvas id="over-${k}"></canvas></div>
        <table style="margin-top:12px"><thead><tr><th class="sc">Training draws</th><th class="sc">Score on them</th>
          <th class="sc">Score on unseen</th><th class="sc">Gap</th></tr></thead><tbody>${rws}</tbody></table>
        <div style="color:var(--faint);font-size:11px;margin-top:8px">
          A ticket was optimised against N past draws, then scored on ${O.n_test} draws it never saw.
          With only 10 training draws it scores <b>${O.rows[0].train_score.toFixed(4)}</b> on them —
          ${(O.rows[0].train_score/O.ceiling_score).toFixed(1)}&times; the theoretical ceiling of
          ${O.ceiling_score.toFixed(4)}, which no genuine knowledge could ever pass — and then delivers
          <b>${O.rows[0].test_score.toFixed(4)}</b> on new draws, at or below a random ticket
          (${O.random_score.toFixed(4)}). As the training set grows the gap closes and the "discovered"
          ticket turns out to be the plain theoretical one. Overfitting is not a flaw of one algorithm;
          it is the ratio of freedom to evidence. Any backtest without held-out data can be made to look
          like this.
        </div>
      </div>`;
  }

  let evCard = '';
  if(d.ev && d.ev.tiers){
    const E = d.ev;
    const trows = E.tiers.map(t=>
      `<tr><td>${t.matches} of ${E.main_count}</td>
       <td class="sc" style="color:var(--muted)">1 in ${Math.round(t.one_in).toLocaleString()}</td>
       <td class="sc">${vnd(t.prize)}</td>
       <td class="sc" style="color:var(--faint)">${Math.round(t.contribution).toLocaleString()} VND</td></tr>`).join('');
    evCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:var(--hot)"></span>What is your line actually worth?</h3>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:10px">
          <span style="color:var(--muted);font-size:12px;margin-right:4px">Your ${E.main_count} numbers (1&ndash;${E.max_value}):</span>
          ${Array.from({length:E.main_count},(_,i)=>
            `<input id="ev-${k}-${i}" type="number" min="1" max="${E.max_value}" placeholder="?"
              style="width:56px;padding:6px 8px;border-radius:8px;border:1px solid rgba(255,255,255,.14);
                     background:rgba(255,255,255,.05);color:inherit;text-align:center;font-size:14px">`).join('')}
          <button id="ev-${k}-pick" style="margin-left:6px;padding:6px 12px;border-radius:8px;cursor:pointer;
            border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.07);color:inherit;font-size:12px">Quick pick</button>
        </div>
        <div id="ev-${k}-out" style="font-size:12px;color:var(--muted);margin-bottom:12px">Type a line to check it.</div>
        <table><thead><tr><th>Match</th><th class="sc">Chance</th><th class="sc">Prize</th><th class="sc">Adds to EV</th></tr></thead>
          <tbody>${trows}</tbody></table>
        <div class="kpis" style="margin-top:12px">
          <div class="kpi"><div class="l">Line costs</div><div class="n">${E.ticket_cost.toLocaleString()}</div></div>
          <div class="kpi"><div class="l">Expected return</div><div class="n">${Math.round(E.gross_ev).toLocaleString()}</div></div>
          <div class="kpi"><div class="l">Expected loss</div><div class="n" style="color:var(--hot)">${Math.round(E.net_ev).toLocaleString()}</div></div>
          <div class="kpi"><div class="l">Return</div><div class="n" style="color:var(--hot)">${E.return_pct.toFixed(1)}%</div></div>
        </div>
        <div style="color:var(--faint);font-size:11px;margin-top:8px">
          Chance of winning <i>anything</i>: ${(100*E.p_any_prize).toFixed(2)}% (1 in ${(1/E.p_any_prize).toFixed(0)}).
          Notice what the table above does when you change your numbers: <b>nothing</b>. The probabilities depend only on
          how many numbers you pick from how many — never on which ones. All ${E.total_tickets.toLocaleString()} possible
          tickets carry the identical expected value of <b>${Math.round(E.net_ev).toLocaleString()} VND</b> per line.
          Every strategy on this page, mine included, is decoration on top of that one number.
        </div>
      </div>`;
  }

  let resCard = '';
  if(d.residual && d.residual.positions){
    const R2 = d.residual;
    const prow = R2.positions.map(p=>
      `<tr><td>Position ${p.position}</td>
       <td class="sc" style="color:var(--muted)">${p.chi2.toFixed(1)} / ${p.dof}</td>
       <td class="sc" style="color:${p.ok?'var(--mint)':'var(--gold)'}">${p.p.toFixed(4)}</td></tr>`).join('');
    resCard = `
      <div class="card col12">
        <h3><span class="ic" style="background:${R2.clean!==false?'var(--mint)':'var(--hot)'}"></span>Does "number K at position I" match theory?</h3>
        <div class="kpis">
          <div class="kpi"><div class="l">Cells beyond 2&sigma;</div><div class="n">${(100*R2.frac_gt2).toFixed(1)}%</div><div class="l">expect ~4.6%</div></div>
          <div class="kpi"><div class="l">Largest |z|</div><div class="n">${R2.max_abs_z.toFixed(2)}</div><div class="l">p = ${R2.max_z_p.toFixed(3)}</div></div>
          <div class="kpi"><div class="l">Mean z</div><div class="n">${R2.mean_z.toFixed(3)}</div><div class="l">expect ~0</div></div>
          <div class="kpi"><div class="l">Šidák p</div><div class="n">${R2.sidak_p.toFixed(3)}</div><div class="l">${R2.n_positions_failing}/6 raw hits</div></div>
        </div>
        <table style="margin-top:12px"><thead><tr><th>Position</th><th class="sc">&chi;&sup2; / dof</th><th class="sc">p</th></tr></thead><tbody>${prow}</tbody></table>
        <div style="color:var(--faint);font-size:11px;margin-top:8px">
          Each cell of the number&times;position grid is compared with its exact theoretical rate as
          z = (observed &minus; expected)/sd, over ${R2.draws.toLocaleString()} draws. Numbers are binned so every
          &chi;&sup2; bin expects at least 5 — without that the test would look rigorous and mean nothing.
          ${R2.verdict}
        </div>
      </div>`;
  }

  const recent = d.recent.map(r=>`<tr>
      <td>${r.date}</td><td>#${r.id}</td>
      <td class="recent-nums">${r.main.map(pad).join(' ')}${r.bonus!=null?` <span class="b">| ${pad(r.bonus)}</span>`:''}</td>
    </tr>`).join('');

  p.innerHTML=`
    <div class="kpis">
      <div class="kpi"><div class="l">Draws on record</div><div class="n">${d.draws.toLocaleString()}</div></div>
      <div class="kpi"><div class="l">History since</div><div class="n small">${d.date_range[0]}</div></div>
      <div class="kpi"><div class="l">Latest draw</div><div class="n">#${d.latest.id}</div></div>
      <div class="kpi"><div class="l">Number pool</div><div class="n">1–${d.range.max}</div></div>
    </div>

    <div class="grid">
      <div class="card col12">
        <h3><span class="ic" style="background:var(--gold)"></span>Latest result · ${d.latest.date}</h3>
        ${balls(d.latest.main, d.latest.bonus)}
      </div>

      <div class="card col8">
        <h3><span class="ic" style="background:var(--cold)"></span>All-time frequency</h3>
        <div class="chartbox"><canvas id="chart-${k}"></canvas></div>
      </div>
      <div class="card col4">
        <h3><span class="ic" style="background:var(--mint)"></span>For-fun lines · next draw ${d.next_draw}</h3>
        <div class="pred">${predRows}</div>
      </div>

      ${mlCard}

      ${trendCard}

      ${histCard}

      <div class="card col12">
        <h3><span class="ic" style="background:var(--violet)"></span>Number heatmap — every number, shaded by how often it's drawn</h3>
        <div class="heat">${heat}</div>
        <div class="legend"><span>less</span><div class="bar"></div><span>more</span></div>
      </div>

      ${jointCard}

      <div class="card col4">
        <h3><span class="ic" style="background:var(--hot)"></span>Hot · last 30 draws</h3>
        <table><tbody>${rowsTable(d.hot,'hot')}</tbody></table>
      </div>
      <div class="card col4">
        <h3><span class="ic" style="background:var(--cold)"></span>Cold · last 30 draws</h3>
        <table><tbody>${rowsTable(d.cold,'cold')}</tbody></table>
      </div>
      <div class="card col4">
        <h3><span class="ic" style="background:var(--gold)"></span>Most overdue</h3>
        <table><tbody>${rowsTable(d.overdue,'over','d')}</tbody></table>
      </div>

      ${jackCard}

      ${bankCard}

      ${randCard}

      ${properCard}

      ${resCard}

      ${ceilCard}

      ${overfitCard}

      ${evCard}

      <div class="card col12">
        <h3><span class="ic" style="background:#fff"></span>Recent draws</h3>
        <table><thead><tr><th>Date</th><th>Draw</th><th>Numbers</th></tr></thead><tbody>${recent}</tbody></table>
      </div>
    </div>`;
  main.appendChild(p);
  wireEvCalc(k);
});

// Interactive line checker. Validates the entry, then reports the same expected
// value no matter what was typed — which is the entire point of the exercise.
function wireEvCalc(k){
  const d = DATA[k]; if(!d.ev) return;
  const E = d.ev, N = E.max_value, K = E.main_count;
  const out = document.getElementById('ev-'+k+'-out');
  const boxes = Array.from({length:K},(_,i)=>document.getElementById('ev-'+k+'-'+i));
  if(!out || boxes.some(b=>!b)) return;

  function check(){
    const raw = boxes.map(b=>b.value.trim()).filter(v=>v!=='');
    if(raw.length < K){
      out.style.color='var(--muted)';
      out.textContent = `Enter ${K-raw.length} more number${K-raw.length>1?'s':''} to check a line.`;
      return;
    }
    const nums = raw.map(Number);
    if(nums.some(n=>!Number.isInteger(n) || n<1 || n>N)){
      out.style.color='var(--hot)';
      out.textContent = `Numbers must be whole, between 1 and ${N}.`;
      return;
    }
    if(new Set(nums).size !== K){
      out.style.color='var(--hot)';
      out.textContent = 'Numbers must all be different.';
      return;
    }
    const line = nums.slice().sort((a,b)=>a-b);
    out.style.color='var(--muted)';
    out.innerHTML = `Your line <b style="color:var(--gold)">${line.map(pad).join(' · ')}</b> — `
      + `expected value <b style="color:var(--hot)">${Math.round(E.net_ev).toLocaleString()} VND</b> per draw. `
      + `Exactly the same as every other line. Try changing a number.`;
  }

  boxes.forEach(b=>b.addEventListener('input', check));
  const pick = document.getElementById('ev-'+k+'-pick');
  if(pick) pick.addEventListener('click', ()=>{
    const pool = Array.from({length:N},(_,i)=>i+1);
    for(let i=pool.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [pool[i],pool[j]]=[pool[j],pool[i]]; }
    pool.slice(0,K).sort((a,b)=>a-b).forEach((v,i)=>{ boxes[i].value = v; });
    check();
  });
}

function drawChart(k){
  const d=DATA[k];
  if(!d.draws||charts[k]||!window.Chart) return;
  const ctx=document.getElementById('chart-'+k); if(!ctx) return;
  const avg=d.freq_values.reduce((a,b)=>a+b,0)/d.freq_values.length;
  const g=ctx.getContext('2d').createLinearGradient(0,0,0,260);
  g.addColorStop(0,'rgba(247,201,72,.95)'); g.addColorStop(1,'rgba(255,159,28,.35)');
  const gcold=ctx.getContext('2d').createLinearGradient(0,0,0,260);
  gcold.addColorStop(0,'rgba(77,166,255,.8)'); gcold.addColorStop(1,'rgba(77,166,255,.2)');
  charts[k]=new Chart(ctx,{type:'bar',
    data:{labels:d.freq_labels, datasets:[{data:d.freq_values, borderRadius:5, borderSkipped:false,
      backgroundColor:d.freq_values.map(v=> v>=avg? g : gcold)}]},
    options:{maintainAspectRatio:false, plugins:{legend:{display:false},
      tooltip:{callbacks:{title:it=>'Number '+it[0].label, label:it=>it.raw+' times'}}},
      scales:{x:{ticks:{color:'#5f6b85',autoSkip:true,maxTicksLimit:28,font:{size:10}},grid:{display:false}},
        y:{ticks:{color:'#5f6b85'},grid:{color:'rgba(255,255,255,.06)'}}}}});
}
const bankColors={random:'#4da6ff',hot:'#ff5d6c',cold:'#37e0a6',overdue:'#f7c948',balanced:'#9b6dff'};
function drawBankroll(k){
  const d=DATA[k], b=d.bankroll;
  if(!b||!b.draws||charts['bank-'+k]||!window.Chart) return;
  const ctx=document.getElementById('bank-'+k); if(!ctx) return;
  const ds=Object.entries(b.chart.series).map(([s,arr])=>({
    label:s, data:arr.map(v=>v/1e6), borderColor:bankColors[s]||'#8b97ad',
    backgroundColor:'transparent', borderWidth:2, pointRadius:0, tension:.15}));
  charts['bank-'+k]=new Chart(ctx,{type:'line',
    data:{labels:b.chart.labels, datasets:ds},
    options:{maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#93a0bd',boxWidth:12,font:{size:11}}},
        tooltip:{callbacks:{label:it=>it.dataset.label+': '+it.raw.toFixed(1)+'M VND'}}},
      scales:{x:{ticks:{color:'#5f6b85',autoSkip:true,maxTicksLimit:8,font:{size:10}},grid:{display:false}},
        y:{ticks:{color:'#5f6b85',callback:v=>v+'M'},grid:{color:'rgba(255,255,255,.06)'},
           title:{display:true,text:'cumulative VND',color:'#5f6b85',font:{size:10}}}}}});
}
const trendPalette=['#f7c948','#ff5d6c','#4da6ff','#37e0a6','#9b6dff','#ff9f1c','#8cc4ff','#ff8a95','#7ee0bf','#c9a6ff','#ffd970','#6b7794','#e8ecf4'];
function drawTrend(k){
  const d=DATA[k], t=d.ml&&d.ml.trend;
  if(!t||!(t.labels||[]).length||charts['trend-'+k]||!window.Chart) return;
  const ctx=document.getElementById('trend-'+k); if(!ctx) return;
  const base=(d.ml.pos_baseline_score)||0;
  const ds=Object.entries(t.series).map(([m,arr],i)=>({
    label:m, data:arr, borderColor:trendPalette[i%trendPalette.length],
    backgroundColor:'transparent', borderWidth: m==='consensus'?3:1.5,
    pointRadius:0, tension:.2, spanGaps:true}));
  ds.push({label:'baseline', data:t.labels.map(()=>base), borderColor:'#5f6b85',
    borderDash:[5,5], borderWidth:1, pointRadius:0});
  charts['trend-'+k]=new Chart(ctx,{type:'line',
    data:{labels:t.labels, datasets:ds},
    options:{maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#93a0bd',boxWidth:10,font:{size:10}}},
        tooltip:{callbacks:{label:it=>it.dataset.label+': '+(it.raw==null?'-':it.raw.toFixed(3))}}},
      scales:{x:{ticks:{color:'#5f6b85',autoSkip:true,maxTicksLimit:8,font:{size:10}},grid:{display:false}},
        y:{ticks:{color:'#5f6b85'},grid:{color:'rgba(255,255,255,.06)'},title:{display:true,text:'running k/6',color:'#5f6b85',font:{size:10}}}}}});
}
function drawOverfit(k){
  const d=DATA[k], o=d.overfit;
  if(!o||!(o.rows||[]).length||charts['over-'+k]||!window.Chart) return;
  const ctx=document.getElementById('over-'+k); if(!ctx) return;
  const labels=o.rows.map(r=>r.n_train);
  charts['over-'+k]=new Chart(ctx,{type:'line',
    data:{labels, datasets:[
      {label:'score on training draws', data:o.rows.map(r=>r.train_score),
       borderColor:'#ff5d6c', backgroundColor:'transparent', borderWidth:2.5,
       pointRadius:3, tension:.2},
      {label:'score on unseen draws', data:o.rows.map(r=>r.test_score),
       borderColor:'#37e0a6', backgroundColor:'transparent', borderWidth:2.5,
       pointRadius:3, tension:.2},
      {label:'theoretical ceiling', data:labels.map(()=>o.ceiling_score),
       borderColor:'#f7c948', borderDash:[5,5], borderWidth:1.5, pointRadius:0},
      {label:'random ticket', data:labels.map(()=>o.random_score),
       borderColor:'#5f6b85', borderDash:[3,3], borderWidth:1, pointRadius:0}]},
    options:{maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#93a0bd',boxWidth:12,font:{size:11}}},
        tooltip:{callbacks:{title:it=>it[0].label+' training draws',
          label:it=>it.dataset.label+': '+it.raw.toFixed(4)}}},
      scales:{x:{ticks:{color:'#5f6b85',font:{size:10}},grid:{display:false},
          title:{display:true,text:'training draws',color:'#5f6b85',font:{size:10}}},
        y:{ticks:{color:'#5f6b85'},grid:{color:'rgba(255,255,255,.06)'},
           title:{display:true,text:'k/6 score',color:'#5f6b85',font:{size:10}}}}}});
}
function activate(k){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',keys[i]===k));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-'+k));
  drawChart(k); drawBankroll(k); drawTrend(k); drawOverfit(k);
}
window.addEventListener('load',()=>{drawChart(keys[0]); drawBankroll(keys[0]); drawTrend(keys[0]); drawOverfit(keys[0]);});
</script>
</body>
</html>
"""


def build(output_path=None) -> str:
    scorecard = load_scorecard()
    payload = {name: _product_payload(name, scorecard) for name in PRODUCTS}
    favicon = "data:image/svg+xml;base64," + base64.b64encode(
        LOGO_SVG.encode("utf-8")).decode("ascii")
    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
            .replace("__GENERATED__",
                     datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M") + " (Vietnam time)")
            .replace("__FAVICON__", favicon)
            .replace("__LOGO__", LOGO_SVG))
    out = output_path or (REPORTS_DIR / "dashboard.html")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return str(out)


if __name__ == "__main__":
    print("wrote", build())
