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

import json
from datetime import datetime

from analyze import hot_cold, load_draws, summary
from config import PRODUCTS, REPORTS_DIR, get_product
from predict import suggest_all

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
        "ml": ml,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lucky Predictions Dashboard</title>
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
  .hero::after{content:"🎰"; position:absolute; right:18px; top:-14px; font-size:120px; opacity:.09; transform:rotate(8deg)}
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
    <h1>Lucky Predictions Dashboard</h1>
    <div class="sub">Power 6/55 &amp; 6/45 &middot; generated __GENERATED__</div>
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
    const nextLines = np ? Object.entries(np.by_model).map(([k,line])=>
        `<div class="row"><div class="tag">${k}</div>${balls(line,null,true)}</div>`).join('') : '';
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
        ${np ? `<h3 style="margin-top:18px"><span class="ic" style="background:var(--mint)"></span>Each predictor · ${np.target_date}</h3><div class="pred">${nextLines}</div>` : ''}
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

      <div class="card col12">
        <h3><span class="ic" style="background:#fff"></span>Recent draws</h3>
        <table><thead><tr><th>Date</th><th>Draw</th><th>Numbers</th></tr></thead><tbody>${recent}</tbody></table>
      </div>
    </div>`;
  main.appendChild(p);
});

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
function activate(k){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',keys[i]===k));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id==='panel-'+k));
  drawChart(k);
}
window.addEventListener('load',()=>drawChart(keys[0]));
</script>
</body>
</html>
"""


def build(output_path=None) -> str:
    scorecard = load_scorecard()
    payload = {name: _product_payload(name, scorecard) for name in PRODUCTS}
    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
            .replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M")))
    out = output_path or (REPORTS_DIR / "dashboard.html")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return str(out)


if __name__ == "__main__":
    print("wrote", build())
