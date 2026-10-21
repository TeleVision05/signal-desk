const $ = (id) => document.getElementById(id);
let chart;
const DEFAULT_CASH = 1000;
const POLL_MS = 30_000;
const BOOK_KEY = "sd_book";

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
function fmtNum(n, d = 2) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(d);
}
function fmtWhen(s) {
  if (!s) return "—";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return String(s).replace("T", " ").slice(0, 16);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
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
function whyTrade(source, side) {
  const s = String(source || "").toLowerCase();
  if (s.includes("stop")) return "Stop loss — cut losses";
  if (s.includes("signal") && String(side).toUpperCase() === "BUY") return "Bot was confident enough to buy";
  if (s.includes("signal")) return "Bot confidence dropped — sold";
  if (String(side).toUpperCase() === "BUY") return "Bought";
  if (String(side).toUpperCase() === "SELL") return "Sold";
  return source || "—";
}
function confBucket(p, thr) {
  if (p == null) return { label: "Waiting", key: "wait" };
  if (p >= thr) return { label: "High", key: "high" };
  if (p >= thr - 0.1) return { label: "Maybe", key: "maybe" };
  return { label: "Low", key: "low" };
}
function currentBook() {
  try {
    return localStorage.getItem(BOOK_KEY) || "tqqq";
  } catch (_) {
    return "tqqq";
  }
}
function setBook(id) {
  const book = id === "btc" ? "btc" : "tqqq";
  try {
    localStorage.setItem(BOOK_KEY, book);
  } catch (_) {}
  document.querySelectorAll(".book-btn").forEach((btn) => {
    const on = btn.dataset.book === book;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
  return book;
}
function assetName(state) {
  return (state.paper && state.paper.asset) || (state.strategy && state.strategy.ticker) || "TQQQ";
}
function updateLogLine(d, asset) {
  const a = asset || d.asset || "TQQQ";
  const action = d.action || "hold";
  if (action === "signal_entry") return `Bought ${a} (paper) — confidence was high enough.`;
  if (action === "signal_exit") return `Sold ${a} (paper) — confidence dropped.`;
  if (action === "stop_exit") return `Sold ${a} (paper) — stop loss triggered.`;
  if (d.signal === 1) return "Checked market — still invested (no new paper trade).";
  return "Checked market — still in cash (no new paper trade).";
}

function setView(name) {
  const simple = name === "simple";
  $("view-simple").classList.toggle("active", simple);
  $("view-simple").hidden = !simple;
  $("view-advanced").classList.toggle("active", !simple);
  $("view-advanced").hidden = simple;
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === name);
  });
  try {
    localStorage.setItem("sd_view", name);
  } catch (_) {}
}

function bindNav() {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.view));
  });
  let start = "simple";
  try {
    start = localStorage.getItem("sd_view") || "simple";
  } catch (_) {}
  setView(start === "advanced" ? "advanced" : "simple");
}

function bindBooks() {
  document.querySelectorAll(".book-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      setBook(btn.dataset.book);
      await refresh();
    });
  });
  setBook(currentBook());
}

function bindTooltip() {
  const btn = $("assetTip");
  const pop = $("tipPop");
  if (!btn || !pop) return;
  const place = () => {
    const r = btn.getBoundingClientRect();
    pop.style.left = `${Math.min(window.innerWidth - 300, Math.max(8, r.left))}px`;
    pop.style.top = `${r.bottom + 8}px`;
  };
  const show = () => {
    place();
    pop.hidden = false;
  };
  const hide = () => {
    pop.hidden = true;
  };
  btn.addEventListener("mouseenter", show);
  btn.addEventListener("mouseleave", hide);
  btn.addEventListener("focus", show);
  btn.addEventListener("blur", hide);
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    if (pop.hidden) show();
    else hide();
  });
}

function tickClock() {
  const el = $("clock");
  if (el) el.textContent = new Date().toLocaleString();
}

async function loadState() {
  const book = currentBook();
  const res = await fetch(`/api/state?book=${encodeURIComponent(book)}&t=${Date.now()}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Could not load desk (${res.status})`);
  return res.json();
}

function renderSimple(state) {
  const paper = state.paper || {};
  const pos = (state.ledger && state.ledger.position) || {};
  const strat = state.strategy || {};
  const notify = state.notify || {};
  const decisions = state.decisions || [];
  const thr = Number(paper.threshold ?? strat.proba_threshold ?? 0.55);
  const p = paper.proba != null ? Number(paper.proba) : null;
  const shares = Number(paper.shares ?? pos.shares ?? 0);
  const invested = shares > 0 || paper.signal === 1;
  const asset = assetName(state);
  const isBtc = state.active_book === "btc";

  let equity = null;
  if (paper.equity != null) equity = Number(paper.equity);
  else if (pos.cash != null || shares) {
    equity = Number(pos.cash || 0) + shares * Number(paper.mark || 0);
  } else {
    equity = DEFAULT_CASH;
  }

  if (isBtc) {
    $("marketLine").textContent = "BTC paper book · 24/7 (fake money)";
  } else {
    $("marketLine").textContent =
      state.market_open_et === true
        ? "US market is open · TQQQ paper"
        : state.market_open_et === false
          ? "US market is closed · TQQQ paper"
          : "TQQQ paper book";
  }

  $("heroStatus").textContent = invested ? "We're currently invested" : "We're sitting in cash";
  $("heroAsset").textContent = asset;
  $("heroEquity").textContent = fmtMoney(equity);
  if ($("heroMoneySub")) {
    $("heroMoneySub").textContent = isBtc
      ? "Fake money — BTC paper book (no live exchange orders)"
      : "Fake money in this practice account";
  }

  const tip = state.book_tip || (isBtc
    ? "BTCUSDT paper book uses Binance public prices and simulates fills in our ledger. No live crypto orders."
    : "TQQQ is a fund that aims to move about 3× the daily return of the Nasdaq-100. Practice trades only.");
  const tipBtn = $("assetTip");
  const tipPop = $("tipPop");
  if (tipBtn) tipBtn.title = tip;
  if (tipPop) tipPop.textContent = tip;

  $("lastUpdate").textContent = fmtWhen(paper.synced_at || state.last_sync_at);
  $("nextUpdate").textContent = fmtWhen(state.next_sync_at);

  const emailEl = $("emailLine");
  if (notify.configured === false) {
    emailEl.textContent = "Trade emails: not set up";
    emailEl.className = "email-line warn";
  } else if (notify.ok) {
    emailEl.textContent = "Paper trade emails: working";
    emailEl.className = "email-line ok";
  } else if (notify.error && !String(notify.error).startsWith("skipped")) {
    emailEl.textContent = `Trade emails: issue (${notify.error})`;
    emailEl.className = "email-line warn";
  } else {
    emailEl.textContent = "Paper trade emails: working";
    emailEl.className = "email-line ok";
  }

  const bucket = confBucket(p, thr);
  $("confLevel").textContent = bucket.label;
  $("confThreshold").style.left = `${thr * 100}%`;
  if (p == null) {
    $("confSentence").textContent = "Waiting for the next update to score confidence…";
    $("confFill").style.width = "0%";
    $("confMarker").style.left = "0%";
    $("confAction").textContent = "";
  } else {
    const pct = (p * 100).toFixed(1);
    const thrPct = (thr * 100).toFixed(0);
    $("confSentence").textContent = `Bot is ${pct}% sure prices go up; it paper-buys above ${thrPct}%.`;
    const left = Math.max(0, Math.min(100, p * 100));
    $("confFill").style.width = `${left}%`;
    $("confMarker").style.left = `${left}%`;
    if (paper.action === "signal_entry") {
      $("confAction").textContent = "Last move: paper buy because confidence cleared the bar.";
    } else if (paper.action === "signal_exit") {
      $("confAction").textContent = "Last move: paper sell because confidence dropped.";
    } else if (paper.action === "stop_exit") {
      $("confAction").textContent = "Last move: paper sell on a stop loss to limit damage.";
    } else if (invested) {
      $("confAction").textContent = "Holding the paper position — no new trade this update.";
    } else {
      $("confAction").textContent = "Staying in cash until confidence is High.";
    }
  }

  const fills = ((state.ledger && state.ledger.fills) || []).slice().reverse();
  $("tradesBody").innerHTML = fills.length
    ? fills
        .map((f) => {
          const side = String(f.side || "").toUpperCase();
          const cls = side === "BUY" ? "buy" : side === "SELL" ? "sell" : "";
          const label = side === "BUY" ? "Bought" : side === "SELL" ? "Sold" : side || "—";
          return `<tr>
            <td>${fmtWhen(f.ts)}</td>
            <td class="${cls}">${label}</td>
            <td class="wrap">${whyTrade(f.source, side)}</td>
            <td>${fmtMoney(f.cost)}</td>
          </tr>`;
        })
        .join("")
    : `<tr><td colspan="4">No paper trades yet — the bot is waiting for a High confidence signal.</td></tr>`;

  const log = decisions.slice(0, 10);
  $("updateLog").innerHTML = log.length
    ? log
        .map(
          (d) => `<li>
            <time datetime="${d.synced_at || ""}">${fmtWhen(d.synced_at)}</time>
            <span>${updateLogLine(d, asset)}</span>
          </li>`
        )
        .join("")
    : `<li><span class="muted">No updates yet — waiting for the first scheduled check.</span></li>`;

  const liveEq = ((state.ledger && state.ledger.equity) || [])
    .map((e) => ({ date: String(e.ts || "").slice(0, 10), equity: Number(e.equity) }))
    .filter((e) => e.date && !Number.isNaN(e.equity));
  $("chartHint").textContent = liveEq.length
    ? `Fake money over time (${isBtc ? "BTC" : "TQQQ"} paper account)`
    : "No paper marks yet — chart fills in after updates";
  try {
    drawChart(liveEq);
  } catch (err) {
    console.error("chart", err);
    $("chartHint").textContent = "Chart unavailable";
  }
}

function renderAdvanced(state) {
  const paper = state.paper || {};
  const strat = state.strategy || {};
  const promo = state.promoted_metrics || {};
  const decisions = state.decisions || [];
  const notify = state.notify || {};
  const isBtc = state.active_book === "btc";

  const l3m = promo.l3m_return ?? promo.last3m_1k?.return;
  const l3mProfit = promo.l3m_profit ?? promo.last3m_1k?.profit;
  if (!isBtc && l3m != null) {
    $("m3m").textContent = `${l3m >= 0 ? "+" : ""}${fmtPct(l3m)}`;
    $("m3m").className = `mono ${l3m >= 0 ? "up" : "down"}`;
  } else {
    $("m3m").textContent = isBtc ? "paper only" : "—";
    $("m3m").className = "mono";
  }

  const card = [
    ["Book", isBtc ? "BTC paper (24/7)" : "TQQQ paper (US RTH)"],
    ["Mode", "paper_only — no live orders"],
    ["Name", strat.name || "—"],
    ["Ticker", strat.ticker || "TQQQ"],
    ["Model", strat.model || "HGB"],
    ["Horizon", `${strat.horizon_days ?? 5}d forward`],
    ["Threshold", `p ≥ ${strat.proba_threshold ?? 0.55}`],
    ["Hard stop", fmtPct(strat.hard_stop_pct ?? 0.08)],
    ["Trail stop", fmtPct(strat.trailing_stop_pct ?? 0.12)],
    ["Train window", `${strat.train_days ?? 504} days / fold`],
  ];
  if (strat.cost_bps != null) card.push(["Cost model", `${strat.cost_bps} bps (paper)`]);
  $("strategyCard").innerHTML = card.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("featureList").innerHTML = (strat.features || []).map((f) => `<span>${f}</span>`).join("");

  $("decisionsBody").innerHTML = decisions.length
    ? decisions
        .map((d) => {
          const pass =
            d.comparison === "pass" || (d.proba != null && d.proba >= (d.threshold ?? 0.55));
          const sigCls = d.signal === 1 ? "long" : "flat";
          const gap =
            d.gap_to_threshold == null
              ? "—"
              : `${(Number(d.gap_to_threshold) * 100).toFixed(1)} pts`;
          return `<tr>
            <td>${fmtWhen(d.synced_at)}</td>
            <td>${d.bar_date || "—"}</td>
            <td>${d.proba == null ? "—" : fmtPct1(d.proba)}</td>
            <td>${fmtPct1(d.threshold ?? 0.55)}</td>
            <td>${gap}</td>
            <td><span class="badge ${pass ? "pass" : "fail"}">${pass ? "PASS" : "FAIL"}</span></td>
            <td><span class="badge ${sigCls}">${d.signal === 1 ? "LONG" : d.signal === 0 ? "FLAT" : "—"}</span></td>
            <td>${actionLabel(d.action)}</td>
            <td>${fmtMoney(d.equity)}</td>
            <td class="wrap">${d.explanation || "—"}</td>
          </tr>`;
        })
        .join("")
    : `<tr><td colspan="10">No paper decisions yet.</td></tr>`;

  const fills = ((state.ledger && state.ledger.fills) || []).slice().reverse();
  $("fillsBody").innerHTML = fills.length
    ? fills
        .map(
          (f) => `<tr>
            <td>${fmtWhen(f.ts)}</td>
            <td class="${String(f.side || "").toLowerCase() === "buy" ? "buy" : "sell"}">${f.side}</td>
            <td>${f.source || "—"}</td>
            <td>${fmtNum(f.shares, 6)}</td>
            <td>${fmtMoney(f.price)}</td>
            <td>${fmtMoney(f.cost)}</td>
          </tr>`
        )
        .join("")
    : `<tr><td colspan="6">No paper fills yet.</td></tr>`;

  const metrics = isBtc
    ? [
        ["Mode", "paper_only"],
        ["Data", "Binance public klines"],
        ["Fills", "Local Postgres ledger"],
        ["Live orders", "None"],
        ["Starting cash", fmtMoney(DEFAULT_CASH)],
        ["Last mark", paper.mark != null ? fmtMoney(paper.mark) : "—"],
      ]
    : [
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

  $("genAt").textContent = [
    `State ${state.generated_at || "—"}`,
    `book ${state.active_book || "tqqq"}`,
    `mode ${state.mode || "paper"}`,
    state.last_sync_reason || "no sync",
    `v${state.version || "—"}`,
    notify.configured === false
      ? "EMAIL NOT CONFIGURED"
      : notify.ok
        ? `last email via ${notify.via}`
        : notify.error || "alerts ready",
  ].join(" · ");
}

function drawChart(series) {
  const canvas = $("equityChart");
  if (!canvas || typeof Chart === "undefined") return;
  if (!series || !series.length) {
    if (chart) {
      chart.destroy();
      chart = null;
    }
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
          borderColor: "#c4a574",
          backgroundColor: "rgba(196,165,116,0.1)",
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "#9a9186", maxTicksLimit: 8, font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: "rgba(240,235,227,0.06)" },
        },
        y: {
          ticks: { color: "#9a9186", font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: "rgba(240,235,227,0.06)" },
        },
      },
    },
  });
}

function render(state) {
  if ($("sideVersion")) $("sideVersion").textContent = `v${state.version || "0.6.0"}`;
  renderSimple(state);
  renderAdvanced(state);
}

async function refresh() {
  try {
    render(await loadState());
  } catch (e) {
    console.error(e);
    if ($("heroStatus")) $("heroStatus").textContent = "Couldn’t reach the desk";
    if ($("confSentence")) $("confSentence").textContent = String(e);
  }
}

async function boot() {
  bindNav();
  bindBooks();
  bindTooltip();
  tickClock();
  setInterval(tickClock, 1000);
  await refresh();
  setInterval(refresh, POLL_MS);
}
boot();
