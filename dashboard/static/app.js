const $ = (id) => document.getElementById(id);
let chart;
const DEFAULT_CASH = 1000;
const POLL_MS = 30_000;

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
function fmtPct1(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return `${(Number(n) * 100).toFixed(1)}%`;
}
function fmtPts(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  const v = Number(n) * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)} pts`;
}
function fmtNum(n, d = 2) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(d);
}
function fmtTs(s) {
  if (!s) return "—";
  return String(s).replace("T", " ").slice(0, 16) + "Z";
}
function actionLabel(a) {
  return (
    {
      hold: "HOLD",
      signal_entry: "BUY",
      signal_exit: "SELL",
      stop_exit: "STOP SELL",
    }[a] || String(a || "—").toUpperCase()
  );
}

function tickClock() {
  $("clock").textContent = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

async function loadState() {
  const res = await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`state failed (${res.status})`);
  return res.json();
}

function renderComparison(paper, strat) {
  const thr = Number(paper.threshold ?? strat.proba_threshold ?? 0.55);
  const p = paper.proba != null ? Number(paper.proba) : null;
  const pass = p != null && p >= thr;

  $("thrLabel").textContent = `threshold ${(thr * 100).toFixed(0)}%`;
  $("probaThreshold").style.left = `${thr * 100}%`;

  if (p == null) {
    $("mProba").textContent = "—";
    $("mProba").className = "mono";
    $("mProbaSub").textContent = `need ≥ ${fmtPct1(thr)} to go long`;
    $("probaFill").style.width = "0%";
    $("probaFill").className = "proba-fill";
    $("probaMarker").style.left = "0%";
    $("decisionExplain").textContent =
      paper.explanation || "Waiting for sync to compute p(up)…";
    $("decisionBarHint").textContent = "p(up) unavailable";
  } else {
    $("mProba").textContent = fmtPct1(p);
    $("mProba").className = `mono ${pass ? "up" : "down"}`;
    $("mProbaSub").textContent = pass
      ? `${fmtPts(p - thr)} over ${fmtPct1(thr)} threshold`
      : `${fmtPts(p - thr)} under ${fmtPct1(thr)} threshold`;
    $("probaFill").style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
    $("probaFill").className = `proba-fill ${pass ? "" : "fail"}`.trim();
    $("probaMarker").style.left = `${Math.max(0, Math.min(100, p * 100))}%`;
    $("decisionExplain").textContent = paper.explanation || "";
    $("decisionBarHint").textContent = `bar ${paper.bar_date || "—"} · ${pass ? "PASS" : "FAIL"}`;
  }

  const stats = [
    ["p(up)", p == null ? "—" : fmtPct1(p), pass ? "pass" : p == null ? "" : "fail"],
    ["Threshold", fmtPct1(thr), ""],
    ["Gap", p == null ? "—" : fmtPts(p - thr), pass ? "pass" : p == null ? "" : "fail"],
    ["Signal", paper.signal === 1 ? "LONG" : paper.signal === 0 ? "FLAT" : "—", ""],
    ["Action", actionLabel(paper.action), ""],
  ];
  $("decisionStats").innerHTML = stats
    .map(([k, v, cls]) => `<div><span>${k}</span><strong class="${cls}">${v}</strong></div>`)
    .join("");
}

function renderStatus(state) {
  const paper = state.paper || {};
  const notify = state.notify || {};
  const open = state.market_open_et;
  const el = $("liveStatus");
  if (!el) return;

  const bits = [];
  bits.push(open === true ? "Market OPEN" : open === false ? "Market CLOSED" : "Market —");
  bits.push(paper.synced_at ? `synced ${fmtTs(paper.synced_at)}` : "not synced");
  if (notify.configured === false || (!notify.ok && String(notify.error || "").includes("not configured"))) {
    bits.push("EMAIL NOT CONFIGURED");
    el.className = "live-status warn";
  } else if (notify.ok) {
    bits.push(`last alert via ${notify.via}`);
    el.className = "live-status ok";
  } else if (notify.error && !String(notify.error).startsWith("skipped")) {
    bits.push(`alert issue: ${notify.error}`);
    el.className = "live-status warn";
  } else {
    bits.push("alerts armed");
    el.className = "live-status";
  }
  el.textContent = bits.join(" · ");
}

function render(state) {
  const paper = state.paper || {};
  const pos = (state.ledger && state.ledger.position) || {};
  const strat = state.strategy || {};
  const promo = state.promoted_metrics || {};
  const bt = state.backtest || {};
  const decisions = state.decisions || [];

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
    ? `synced ${fmtTs(paper.synced_at)}`
    : state.last_sync_at
      ? `last sync ${fmtTs(state.last_sync_at)}`
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
  $("mSignalSub").textContent = paper.action
    ? `last action: ${actionLabel(paper.action)}`
    : "awaiting sync";

  if (pos.asset || (paper.shares && Number(paper.shares) > 0)) {
    $("mPos").textContent = paper.asset || pos.asset || "TQQQ";
    $("mPosSub").textContent = `${fmtNum(paper.shares ?? pos.shares, 4)} sh · mark ${fmtMoney(paper.mark)}`;
  } else {
    $("mPos").textContent = "CASH";
    $("mPosSub").textContent = fmtMoney(
      paper.cash != null ? paper.cash : pos.cash != null ? pos.cash : equity
    );
  }

  renderComparison(paper, strat);
  renderStatus(state);

  const l3m = promo.l3m_return ?? promo.last3m_1k?.return;
  const l3mProfit = promo.l3m_profit ?? promo.last3m_1k?.profit;
  if (l3m != null) {
    $("m3m").textContent = `${l3m >= 0 ? "+" : ""}${fmtPct(l3m)}`;
    $("m3m").className = `mono ${l3m >= 0 ? "up" : "down"}`;
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
  $("featureList").innerHTML = (strat.features || []).map((f) => `<span>${f}</span>`).join("");

  $("decisionsBody").innerHTML = decisions.length
    ? decisions
        .map((d) => {
          const pass =
            d.comparison === "pass" || (d.proba != null && d.proba >= (d.threshold ?? 0.55));
          const sigCls = d.signal === 1 ? "long" : "flat";
          return `
      <tr>
        <td>${fmtTs(d.synced_at)}</td>
        <td>${d.bar_date || "—"}</td>
        <td>${d.proba == null ? "—" : fmtPct1(d.proba)}</td>
        <td>${fmtPct1(d.threshold ?? 0.55)}</td>
        <td>${d.gap_to_threshold == null ? "—" : fmtPts(d.gap_to_threshold)}</td>
        <td><span class="badge ${pass ? "pass" : "fail"}">${pass ? "PASS" : "FAIL"}</span></td>
        <td><span class="badge ${sigCls}">${d.signal === 1 ? "LONG" : d.signal === 0 ? "FLAT" : "—"}</span></td>
        <td>${actionLabel(d.action)}</td>
        <td>${fmtMoney(d.equity)}</td>
        <td class="wrap">${d.explanation || "—"}</td>
      </tr>`;
        })
        .join("")
    : `<tr><td colspan="10">No decisions yet — waiting for next sync.</td></tr>`;

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
    : `<tr><td colspan="6">No fills yet — holds do not create trades.</td></tr>`;

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

  const notify = state.notify || {};
  $("genAt").textContent = [
    `State ${state.generated_at || "—"}`,
    `mode ${state.mode || "paper"}`,
    state.last_sync_reason || "no sync",
    notify.configured === false
      ? "EMAIL NOT CONFIGURED"
      : notify.ok
        ? `last email via ${notify.via}`
        : notify.error || "alerts ready",
  ].join(" · ");

  const liveEq = ((state.ledger && state.ledger.equity) || [])
    .map((e) => ({ date: String(e.ts || "").slice(0, 10), equity: Number(e.equity) }))
    .filter((e) => e.date && !Number.isNaN(e.equity));
  const btSeries = bt.equity_weekly || [];
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
    const res = await fetch(`/api/sync`, {
      method: "POST",
      headers: { "X-Cron-Secret": secret },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.detail || `Sync failed (${res.status})`);
    }
    render(await loadState());
  } catch (e) {
    alert(String(e));
  } finally {
    btn.disabled = false;
    btn.textContent = "Run sync";
  }
}

async function refresh() {
  try {
    const state = await loadState();
    // enrich with market flag from config if missing
    if (state.market_open_et == null) {
      try {
        const cfg = await (await fetch("/api/config", { cache: "no-store" })).json();
        state.market_open_et = cfg.market_open_et;
      } catch (_) {}
    }
    render(state);
  } catch (e) {
    console.error(e);
    if ($("mEquitySub")) $("mEquitySub").textContent = String(e);
  }
}

async function boot() {
  tickClock();
  setInterval(tickClock, 1000);
  $("syncBtn").addEventListener("click", syncNow);
  await refresh();
  setInterval(refresh, POLL_MS);
}
boot();
