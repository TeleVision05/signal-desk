const $ = (id) => document.getElementById(id);
let chart;

function fmtMoney(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(n);
}
function fmtPct(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return `${(n * 100).toFixed(2)}%`;
}
function fmtNum(n, d = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toFixed(d);
}

function tickClock() {
  $("clock").textContent = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

async function loadState() {
  const url = window.location.hostname.endsWith("github.io")
    ? `./state.json?t=${Date.now()}`
    : "/api/state";
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error("state failed");
  return res.json();
}

function render(state) {
  const paper = state.paper || {};
  const pos = (state.ledger && state.ledger.position) || {};
  const strat = state.strategy || {};
  const promo = state.promoted_metrics || {};
  const bt = state.backtest || {};

  const equity = paper.equity != null ? paper.equity : (pos.cash != null ? Number(pos.cash) + (Number(pos.shares || 0) * Number(paper.mark || 0)) : null);
  $("mEquity").textContent = fmtMoney(equity);
  $("mEquity").className = "mono";
  $("mEquitySub").textContent = paper.synced_at
    ? `synced ${new Date(paper.synced_at).toISOString().slice(0, 16)}Z`
    : "no live sync yet — showing ledger / backtest";

  const sig = paper.signal;
  if (sig === 1) {
    $("mSignal").textContent = "LONG";
    $("mSignal").className = "mono up";
  } else if (sig === 0) {
    $("mSignal").textContent = "FLAT";
    $("mSignal").className = "mono flat";
  } else {
    $("mSignal").textContent = "—";
    $("mSignal").className = "mono";
  }
  $("mSignalSub").textContent = paper.proba != null ? `p(up)=${fmtNum(paper.proba, 3)}` : "awaiting probability";

  if (pos.asset) {
    $("mPos").textContent = pos.asset;
    $("mPosSub").textContent = `${fmtNum(pos.shares, 4)} sh · mark ${fmtMoney(paper.mark)}`;
  } else {
    $("mPos").textContent = "CASH";
    $("mPosSub").textContent = fmtMoney(pos.cash ?? equity);
  }

  // 3m reference from findings / promo if present
  const l3m = promo.l3m_return ?? promo.last3m_1k?.return;
  const l3mProfit = promo.l3m_profit ?? promo.last3m_1k?.profit;
  if (l3m != null) {
    $("m3m").textContent = `${l3m >= 0 ? "+" : ""}${fmtPct(l3m)}`;
    $("m3m").className = `mono ${l3m >= 0 ? "up" : "down"}`;
  } else if (bt.total_return != null) {
    $("m3m").textContent = fmtPct(bt.total_return);
  }

  // strategy card
  const card = [
    ["Name", strat.name || "—"],
    ["Ticker", strat.ticker || "TQQQ"],
    ["Model", strat.model || "HGB"],
    ["Horizon", `${strat.horizon_days ?? 5}d forward`],
    ["Threshold", `p ≥ ${strat.proba_threshold ?? 0.55}`],
    ["Hard stop", fmtPct(strat.hard_stop_pct ?? 0.08)],
    ["Trail stop", fmtPct(strat.trailing_stop_pct ?? 0.12)],
    ["Train window", `${strat.train_days ?? 504} days / fold`],
  ];
  $("strategyCard").innerHTML = card.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("featureList").innerHTML = (strat.features || []).map((f) => `<span>${f}</span>`).join("");

  // fills
  const fills = ((state.ledger && state.ledger.fills) || []).slice().reverse();
  $("fillsBody").innerHTML = fills.length
    ? fills.map((f) => `
      <tr>
        <td>${(f.ts || "").replace("T", " ").slice(0, 19)}</td>
        <td class="${(f.side || "").toLowerCase()}">${f.side}</td>
        <td>${f.source || "—"}</td>
        <td>${fmtNum(f.shares, 4)}</td>
        <td>${fmtMoney(f.price)}</td>
        <td>${fmtMoney(f.cost)}</td>
      </tr>`).join("")
    : `<tr><td colspan="6">No paper fills yet. Hit Run sync after market data is available.</td></tr>`;

  // promo metrics
  const metrics = [
    ["Full CAGR", fmtPct(promo.cagr)],
    ["Last 3y CAGR", fmtPct(promo.l3y_cagr)],
    ["vs SPY (3y)", fmtPct(promo.l3y_excess_cagr_vs_spy)],
    ["Max DD", fmtPct(promo.max_drawdown ?? promo.l3y_max_drawdown)],
    ["Green weeks", fmtPct(promo.positive_week_pct)],
    ["3m $1k P&amp;L", l3mProfit != null ? fmtMoney(l3mProfit) : "—"],
  ];
  $("promoMetrics").innerHTML = metrics.map(([k, v]) => `<div><span>${k}</span><strong>${v}</strong></div>`).join("");
  $("genAt").textContent = `State generated ${state.generated_at || "—"} · mode ${state.mode || "paper"}`;

  // chart: prefer live equity marks, else backtest weekly
  const liveEq = ((state.ledger && state.ledger.equity) || []).map((e) => ({
    date: (e.ts || "").slice(0, 10),
    equity: Number(e.equity),
  })).filter((e) => e.date && !Number.isNaN(e.equity));
  const series = liveEq.length > 1 ? liveEq : (bt.equity_weekly || []);
  $("chartHint").textContent = liveEq.length > 1 ? "paper ledger marks" : "walk-forward backtest (weekly)";
  drawChart(series);
}

function drawChart(series) {
  const labels = series.map((p) => p.date);
  const data = series.map((p) => p.equity);
  const ctx = $("equityChart").getContext("2d");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data,
        borderColor: "#c8f542",
        backgroundColor: "rgba(200,245,66,0.08)",
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.2,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "#8a9a92", maxTicksLimit: 8, font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: "rgba(232,240,230,0.06)" },
        },
        y: {
          ticks: { color: "#8a9a92", font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: "rgba(232,240,230,0.06)" },
        },
      },
    },
  });
}

async function syncNow() {
  const btn = $("syncBtn");
  btn.disabled = true;
  btn.textContent = "Syncing…";
  try {
    const secret = window.localStorage.getItem("CRON_SECRET") || "dev-paper-secret";
    const res = await fetch("/api/sync", {
      method: "POST",
      headers: { "X-Cron-Secret": secret },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.detail || `Sync failed (${res.status})`);
    }
    const state = await loadState();
    render(state);
  } catch (e) {
    alert(String(e));
  } finally {
    btn.disabled = false;
    btn.textContent = "Run sync";
  }
}

async function boot() {
  tickClock();
  setInterval(tickClock, 1000);
  $("syncBtn").addEventListener("click", syncNow);
  try {
    render(await loadState());
  } catch (e) {
    $("mEquitySub").textContent = String(e);
  }
}
boot();
