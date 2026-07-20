"""Generate a self-contained HTML dashboard from Vietlott draw data.

Produces a single `reports/dashboard.html` with:
  - key facts (draw count, date range, latest draw)
  - an all-time number frequency bar chart (Chart.js from CDN)
  - hot / cold (last 30 draws) and most-overdue tables
  - the 10 most recent draws

All data is embedded as JSON, so the file works by just opening it in a
browser — no server needed. Chart.js is loaded from a CDN (needs internet
the first time it's opened; everything else is offline).

⚠️ For fun/learning only — lottery draws are random.
"""
from __future__ import annotations

import json
from datetime import datetime

from analyze import hot_cold, load_draws, summary
from config import PRODUCTS, REPORTS_DIR, get_product


def _product_payload(name: str) -> dict:
    product = get_product(name)
    draws = load_draws(product)
    if not draws:
        return {"label": product.label, "draws": 0}

    s = summary(name)
    hot30, cold30 = hot_cold(draws, product, 30)
    freq = s["frequency"]
    nums = list(range(product.min_value, product.max_value + 1))
    recent = [
        {"date": d["date"], "id": d["id"], "main": d["main"],
         "bonus": d["result"][-1] if len(d["result"]) > product.main_count else None}
        for d in reversed(draws[-10:])
    ]
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
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vietlott Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root { --bg:#0f1420; --card:#1a2130; --ink:#e8ecf4; --muted:#8b97ad;
          --accent:#f5b301; --hot:#ff6b6b; --cold:#4dabf7; --line:#2a3346; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;
         background:var(--bg); color:var(--ink); }
  header { padding:20px 24px; border-bottom:1px solid var(--line); }
  h1 { margin:0; font-size:20px; }
  .sub { color:var(--muted); font-size:13px; margin-top:4px; }
  .warn { color:var(--accent); font-size:12px; margin-top:8px; }
  .tabs { display:flex; gap:8px; padding:16px 24px 0; }
  .tab { padding:8px 16px; border:1px solid var(--line); border-bottom:none;
         border-radius:8px 8px 0 0; cursor:pointer; color:var(--muted);
         background:transparent; font-size:14px; }
  .tab.active { color:var(--ink); background:var(--card); }
  main { padding:0 24px 32px; }
  .panel { display:none; }
  .panel.active { display:block; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
          gap:16px; margin-top:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:16px; }
  .card h3 { margin:0 0 12px; font-size:13px; text-transform:uppercase;
             letter-spacing:.5px; color:var(--muted); }
  .facts { display:flex; flex-wrap:wrap; gap:24px; margin-top:16px; }
  .fact .n { font-size:22px; font-weight:700; }
  .fact .l { font-size:12px; color:var(--muted); }
  .balls span { display:inline-flex; align-items:center; justify-content:center;
                width:34px; height:34px; border-radius:50%; margin:2px;
                background:#20293b; border:1px solid var(--line); font-weight:700; }
  .balls .bonus { background:var(--accent); color:#1a1a1a; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  td, th { padding:6px 8px; text-align:left; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  .chip { display:inline-block; min-width:26px; text-align:center; padding:2px 6px;
          border-radius:6px; font-weight:700; }
  .chip.hot { background:rgba(255,107,107,.18); color:var(--hot); }
  .chip.cold { background:rgba(77,171,247,.18); color:var(--cold); }
  .card.chartcard { grid-column:1/-1; }
  canvas { max-height:280px; }
</style>
</head>
<body>
<header>
  <h1>🎰 Vietlott Dashboard</h1>
  <div class="sub">Generated __GENERATED__</div>
  <div class="warn">⚠️ Lottery draws are random — these stats describe the past and cannot predict the future. For fun only.</div>
</header>
<div class="tabs" id="tabs"></div>
<main id="main"></main>
<script>
const DATA = __DATA__;

function balls(main, bonus){
  let h = main.map(n => `<span>${String(n).padStart(2,'0')}</span>`).join('');
  if(bonus!=null) h += `<span class="bonus">${String(bonus).padStart(2,'0')}</span>`;
  return `<div class="balls">${h}</div>`;
}
function rows(list, cls){
  return list.map(([n,v]) => `<tr><td><span class="chip ${cls}">${n}</span></td><td>${v}</td></tr>`).join('');
}

const keys = Object.keys(DATA);
const tabs = document.getElementById('tabs');
const main = document.getElementById('main');
const charts = {};

keys.forEach((k,i) => {
  const d = DATA[k];
  const t = document.createElement('button');
  t.className = 'tab' + (i===0?' active':''); t.textContent = d.label;
  t.onclick = () => activate(k);
  tabs.appendChild(t);

  const p = document.createElement('div');
  p.className = 'panel' + (i===0?' active':''); p.id = 'panel-'+k;
  if(!d.draws){ p.innerHTML = `<div class="card" style="margin-top:16px">No data yet for ${d.label}.</div>`; }
  else {
    p.innerHTML = `
      <div class="facts">
        <div class="fact"><div class="n">${d.draws.toLocaleString()}</div><div class="l">draws on record</div></div>
        <div class="fact"><div class="n">${d.date_range[0]} → ${d.date_range[1]}</div><div class="l">date range</div></div>
        <div class="fact"><div class="n">#${d.latest.id}</div><div class="l">latest draw (${d.latest.date})</div></div>
      </div>
      ${balls(d.latest.main, d.latest.bonus)}
      <div class="grid">
        <div class="card chartcard"><h3>All-time frequency</h3><canvas id="chart-${k}"></canvas></div>
        <div class="card"><h3>Hot — last 30 draws</h3><table><thead><tr><th>Number</th><th>Times</th></tr></thead><tbody>${rows(d.hot,'hot')}</tbody></table></div>
        <div class="card"><h3>Cold — last 30 draws</h3><table><thead><tr><th>Number</th><th>Times</th></tr></thead><tbody>${rows(d.cold,'cold')}</tbody></table></div>
        <div class="card"><h3>Most overdue</h3><table><thead><tr><th>Number</th><th>Days since</th></tr></thead><tbody>${rows(d.overdue,'cold')}</tbody></table></div>
        <div class="card"><h3>Recent draws</h3><table><thead><tr><th>Date</th><th>#</th><th>Numbers</th></tr></thead><tbody>
          ${d.recent.map(r=>`<tr><td>${r.date}</td><td>${r.id}</td><td>${r.main.join(' ')}${r.bonus!=null?' | '+r.bonus:''}</td></tr>`).join('')}
        </tbody></table></div>
      </div>`;
  }
  main.appendChild(p);
});

function drawChart(k){
  const d = DATA[k];
  if(!d.draws || charts[k]) return;
  const ctx = document.getElementById('chart-'+k);
  if(!ctx || !window.Chart) return;
  const avg = d.freq_values.reduce((a,b)=>a+b,0)/d.freq_values.length;
  charts[k] = new Chart(ctx, {
    type:'bar',
    data:{ labels:d.freq_labels,
      datasets:[{ label:'Times drawn', data:d.freq_values,
        backgroundColor:d.freq_values.map(v=> v>=avg ? 'rgba(245,179,1,.8)' : 'rgba(77,171,247,.6)') }]},
    options:{ plugins:{legend:{display:false}}, scales:{
      x:{ ticks:{color:'#8b97ad',autoSkip:true,maxTicksLimit:28}, grid:{display:false} },
      y:{ ticks:{color:'#8b97ad'}, grid:{color:'#2a3346'} } } }
  });
}
function activate(k){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active', keys[i]===k));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active', p.id==='panel-'+k));
  drawChart(k);
}
window.addEventListener('load', ()=>drawChart(keys[0]));
</script>
</body>
</html>
"""


def build(output_path=None) -> str:
    payload = {name: _product_payload(name) for name in PRODUCTS}
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
