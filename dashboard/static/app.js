const $ = (id) => document.getElementById(id);
let chart;
const DEFAULT_CASH = 1000;
const RENDER_API = "https://signal-desk-upb4.onrender.com";

function apiRoot() {
  if (window.location.hostname.endsWith("github.io")) return RENDER_API;
  return "";
}

function fmtMoney(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(Number(n));
}
function fmtPct(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return `${(Number(n) * 100).toFixed(2)}%`;
}
function fmtNum(n, d = 2) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(d);
}

function tickClock() {
  $("clock").textContent = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

async function loadState() {
  const root = apiRoot();
  if (root) {
    try {
      const live = await fetch(`${root}/api/state?t=${Date.now()}`, { cache: "no-store" });
      if (live.ok) {
        const data = await live.json();
        const bt = (data.backtest && data.backtest.equity_weekly) || [];
        const paper = data.paper || {};
        // Prefer live Render when it has real paper/backtest; else fall through to Pages snapshot
        if (bt.length || paper.equity != null || paper.signal != null) return data;
      }
    } catch (_) {
      /* cold start / CORS — use static snapshot */
    }
    const snap = await fetch(`./state.json?t=${Date.now()}`, { cache: "no-store" });
    if (!snap.ok) throw new Error(`state failed (${snap.status})`);
    return snap.json();
  }
  const res = await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`state failed (${res.status})`);
  return res.json();
}

function render(state) {
  const paper = state.paper || {};
  const pos = (state.ledger && state.ledger.position) || {};
  const strat = state.strategy || {};
  const promo = state.promoted_metrics || {};
  const bt = state.backtest || {};

  let equity = null;
  if (paper.equity != null) equity = Number(paper.equity);
  else if (pos.cash != null || pos.shares) {
    equity = Number(pos.cash || 0) + Number(pos.shares || 0) * Number(paper.mark || 0);
  } else {
    equity = DEFAULT_CASH;
  }

  $("mEquity").textContent = fmtMoney(equity);
  $("mEquity").className = "mono";
  $("mEquitySub").textContent = paper.synced_at
    ? `synced ${String(paper.synced_at).replace("T", " ").slice(0, 16)}Z`
    : state.last_sync_at
      ? `last sync ${String(state.last_sync_at).replace("T", " ").slice(0, 16)}Z`
      : "waiting for first paper sync";

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
  $("mSignalSub").textContent =
    paper.proba != null
      ? `p(up)=${fmtNum(paper.proba, 3)}`
      : paper.action
        ? `last action: ${paper.action}`
        : "awaiting probability";

  if (pos.asset) {
    $("mPos").textContent = pos.asset;
    $("mPosSub").textContent = `${fmtNum(pos.shares, 4)} sh · mark ${fmtMoney(paper.mark)}`;
  } else {
    $("mPos").textContent = "CASH";
    $("mPosSub").textContent = fmtMoney(pos.cash != null ? pos.cash : equity);
  }

  const l3m = promo.l3m_return ?? promo.last3m_1k?.return;
  const l3mProfit = promo.l3m_profit ?? promo.last3m_1k?.profit;
  if (l3m != null) {
    $("m3m").textContent = `${l3m >= 0 ? "+" : ""}${fmtPct(l3m)}`;
    $("m3m").className = `mono ${l3m >= 0 ? "up" : "down"}`;
  } else if (bt.total_return != null) {
    $("m3m").textContent = fmtPct(bt.total_return);
    $("m3m").className = "mono";
  } else {
    $("m3m").textContent = "—";
  }

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
  $("featureList").innerHTML = (strat.features || [])
    .map((f) => `<span>${f}</span>`)
    .join("");

  const fills = ((state.ledger && state.ledger.fills) || []).slice().reverse();
  $("fillsBody").innerHTML = fills.length
    ? fills
        .map(
          (f) => `
      <tr>
        <td>${String(f.ts || "").replace("T", " ").slice(0, 19)}</td>
        <td class="${String(f.side || "").toLowerCase()}">${f.side}</td>
        <td>${f.source || "—"}</td>
        <td>${fmtNum(f.shares, 4)}</td>
        <td>${fmtMoney(f.price)}</td>
        <td>${fmtMoney(f.cost)}</td>
      </tr>`
        )
        .join("")
    : `<tr><td colspan="6">No paper fills yet. Sync runs on market open / hourly RTH / after close, or click Run sync.</td></tr>`;

  const metrics = [
    ["Full CAGR", fmtPct(promo.cagr)],
    ["Last 3y CAGR", fmtPct(promo.l3y_cagr)],
    ["vs SPY (3y)", fmtPct(promo.l3y_excess_cagr_vs_spy)],
    ["Max DD", fmtPct(promo.max_drawdown ?? promo.l3y_max_drawdown)],
    ["Green weeks", fmtPct(promo.positive_week_pct)],
    ["3m $1k P&L", l3mProfit != null ? fmtMoney(l3mProfit) : "—"],
  ];
  $("promoMetrics").innerHTML = metrics
    .map(([k, v]) => `<div><span>${k}</span><strong>${v}</strong></div>`)
    .join("");
  $("genAt").textContent = `State generated ${state.generated_at || "—"} · mode ${state.mode || "paper"} · ${state.last_sync_reason || "no sync yet"}`;

  const liveEq = ((state.ledger && state.ledger.equity) || [])
    .map((e) => ({ date: String(e.ts || "").slice(0, 10), equity: Number(e.equity) }))
    .filter((e) => e.date && !Number.isNaN(e.equity));
  const btSeries = bt.equity_weekly || [];
  // Prefer full walk-forward curve; paper ledger only once it has a real history
  const usePaperPath =
    liveEq.length >= 8 &&
    liveEq[0].date !== liveEq[liveEq.length - 1].date &&
    btSeries.length === 0;
  const series = usePaperPath ? liveEq : btSeries.length ? btSeries : liveEq;
  $("chartHint").textContent = usePaperPath
    ? "paper ledger marks"
    : btSeries.length
      ? "walk-forward backtest (weekly)"
      : "paper ledger marks";
  try {
    drawChart(series);
  } catch (err) {
    console.error("chart", err);
    $("chartHint").textContent = "chart unavailable";
  }
}

function drawChart(series) {
  const canvas = $("equityChart");
  if (!canvas || typeof Chart === "undefined") return;
  if (!series || !series.length) {
    if (chart) chart.destroy();
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const labels = series.map((p) => p.date);
  const data = series.map((p) => p.equity);
  const ctx = canvas.getContext("2d");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          data,
          borderColor: "#c8f542",
          backgroundColor: "rgba(200,245,66,0.08)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
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
    const secret = window.localStorage.getItem("CRON_SECRET") || "signal-desk-paper-sync";
    const root = apiRoot();
    const res = await fetch(`${root}/api/sync`, {
      method: "POST",
      headers: { "X-Cron-Secret": secret },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      // On Pages, refresh from committed snapshot if live sync is still on old Render build
      if (root) {
        try {
          render(await (await fetch(`./state.json?t=${Date.now()}`, { cache: "no-store" })).json());
        } catch (_) {}
      }
      alert(
        err.detail ||
          `Sync failed (${res.status}). Redeploy Render to main (ALLOW_PUBLIC_SYNC=1) or wait for the weekday Pages/Action sync.`
      );
    } else {
      render(await loadState());
    }
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
    console.error(e);
    $("mEquitySub").textContent = String(e);
  }
}
boot();
