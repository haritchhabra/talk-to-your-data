const API_BASE = "http://localhost:8000";

const form = document.getElementById("queryForm");
const input = document.getElementById("queryInput");
const submitBtn = document.getElementById("submitBtn");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("statusText");
const chips = document.querySelectorAll(".chip");

const statusLine = document.getElementById("statusLine");
const statusCursor = document.getElementById("statusCursor");

const tileText = document.getElementById("tileText");
const tileTextBody = document.getElementById("tileTextBody");
const tileNumeric = document.getElementById("tileNumeric");
const tileNumericBody = document.getElementById("tileNumericBody");
const tileChart = document.getElementById("tileChart");
const tileChartBody = document.getElementById("tileChartBody");

let chartInstance = null;

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    statusEl.classList.add("online");
    statusEl.classList.remove("offline");
    statusText.textContent = data.groq_configured
      ? `Connected · ${data.rows} records`
      : `Connected · no GROQ_API_KEY set`;
  } catch (e) {
    statusEl.classList.add("offline");
    statusEl.classList.remove("online");
    statusText.textContent = "Backend unreachable";
  }
}
checkHealth();

async function loadKpis() {
  try {
    const res = await fetch(`${API_BASE}/api/stats`);
    const s = await res.json();
    document.getElementById("kpiTotal").textContent = s.total_tickets ?? "—";
    document.getElementById("kpiOpen").textContent = s.open_tickets ?? "—";
    document.getElementById("kpiRes").textContent = s.avg_resolution_hrs ?? "—";
    document.getElementById("kpiCsat").textContent = s.avg_satisfaction ?? "—";
    document.getElementById("kpiCritical").textContent = s.critical_tickets ?? "—";
  } catch (e) {
    // Instrument strip is decorative context; fail silently if backend isn't up yet.
  }
}
loadKpis();

chips.forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.q;
    form.requestSubmit();
  });
});

// ---------- Status line (terminal-style stage indicator) ----------
const STAGES = ["classify", "compute", "synthesize"];

function renderStatusLine(activeIndex, doneUpTo) {
  statusLine.innerHTML = "";
  STAGES.forEach((stage, i) => {
    if (i > 0) {
      const sep = document.createElement("span");
      sep.className = "sep";
      sep.textContent = "›";
      statusLine.appendChild(sep);
    }
    const span = document.createElement("span");
    span.className = "stage";
    if (i < doneUpTo) span.classList.add("done");
    if (i === activeIndex) span.classList.add("active");
    span.textContent = stage;
    statusLine.appendChild(span);
  });
  statusLine.appendChild(statusCursor);
  statusCursor.classList.remove("hidden");
}

function clearStatusLine() {
  statusLine.innerHTML = "";
  statusCursor.classList.add("hidden");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  submitBtn.disabled = true;
  setLoadingState();
  renderStatusLine(0, 0);
  const computeTimer = setTimeout(() => renderStatusLine(1, 1), 350);

  const controller = new AbortController();
  const requestTimeout = setTimeout(() => controller.abort(), 30000);

  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    clearTimeout(requestTimeout);
    if (!res.ok) {
      throw new Error(`Backend returned ${res.status}`);
    }
    const data = await res.json();
    clearTimeout(computeTimer);
    renderStatusLine(2, 2);
    setTimeout(() => {
      statusLine.querySelectorAll(".stage").forEach((s) => s.classList.add("done"));
      statusLine.querySelectorAll(".stage.active").forEach((s) => s.classList.remove("active"));
      statusCursor.classList.add("hidden");
    }, 400);
    renderResult(data);
  } catch (err) {
    clearTimeout(computeTimer);
    clearTimeout(requestTimeout);
    clearStatusLine();
    const message = err.name === "AbortError"
      ? "The request timed out after 30s — the model may be slow or unreachable. Try again."
      : "Couldn't reach the backend. Is it running on localhost:8000?";
    renderErrorAllTiles(message);
  } finally {
    submitBtn.disabled = false;
  }
});

function setLoadingState() {
  [tileText, tileNumeric, tileChart].forEach((t) => {
    t.classList.remove("is-empty", "is-filled");
    t.classList.add("is-loading");
  });
  const loadingHtml = `<div class="loading"><span class="spin"></span> working</div>`;
  tileTextBody.innerHTML = loadingHtml;
  tileNumericBody.innerHTML = loadingHtml;
  tileChartBody.innerHTML = loadingHtml;
}

function renderErrorAllTiles(message) {
  [tileText, tileNumeric, tileChart].forEach((t) => {
    t.classList.remove("is-loading", "is-filled");
    t.classList.add("is-empty");
  });
  tileTextBody.innerHTML = `<p class="error-text">${escapeHtml(message)}</p>`;
  tileNumericBody.innerHTML = `<p class="placeholder">—</p>`;
  tileChartBody.innerHTML = `<p class="placeholder">—</p>`;
}

function renderResult(data) {
  // --- text tile: always filled ---
  tileText.classList.remove("is-loading");
  if (data.text) {
    tileText.classList.add("is-filled");
    tileText.classList.remove("is-empty");
    tileTextBody.innerHTML = `<p class="answer-text">${escapeHtml(data.text)}</p>`;
  } else {
    tileText.classList.add("is-empty");
    tileTextBody.innerHTML = `<p class="placeholder">No text answer returned.</p>`;
  }

  // --- numeric tile: optional ---
  tileNumeric.classList.remove("is-loading");
  if (data.numeric && data.numeric.value !== undefined && data.numeric.value !== null) {
    tileNumeric.classList.add("is-filled");
    tileNumeric.classList.remove("is-empty");
    const val = typeof data.numeric.value === "object"
      ? JSON.stringify(data.numeric.value)
      : data.numeric.value;
    const valStr = String(val);
    const sizeClass = valStr.length > 40 ? "is-tiny" : valStr.length > 18 ? "is-small" : "";
    tileNumericBody.innerHTML = `
      <div class="answer-numeric ${sizeClass}">${escapeHtml(valStr)}</div>
      ${data.numeric.explanation ? `<p class="answer-explain">${escapeHtml(data.numeric.explanation)}</p>` : ""}
    `;
  } else {
    tileNumeric.classList.add("is-empty");
    tileNumeric.classList.remove("is-filled");
    tileNumericBody.innerHTML = `<p class="placeholder">Not applicable to this question.</p>`;
  }

  // --- chart tile: optional ---
  tileChart.classList.remove("is-loading");
  if (data.chart && data.chart.labels && data.chart.labels.length) {
    tileChart.classList.add("is-filled");
    tileChart.classList.remove("is-empty");
    tileChartBody.innerHTML = `
      <p class="chart-title">${escapeHtml(data.chart.title || "")}</p>
      <div class="chart-wrap"><canvas id="chartCanvas"></canvas></div>
    `;
    drawChart(data.chart);
  } else {
    tileChart.classList.add("is-empty");
    tileChart.classList.remove("is-filled");
    tileChartBody.innerHTML = `<p class="placeholder">Not applicable to this question.</p>`;
    if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  }
}

function drawChart(chart) {
  const canvas = document.getElementById("chartCanvas");
  if (!canvas) return;

  if (typeof Chart === "undefined") {
    canvas.replaceWith(Object.assign(document.createElement("p"), {
      className: "error-text",
      textContent: "Chart library failed to load — check your network or ad blocker.",
    }));
    return;
  }

  const ctx = canvas.getContext("2d");
  if (chartInstance) chartInstance.destroy();

  const palette = ["#2b4257", "#8a5a2b", "#4b6358", "#6b5a8a", "#a13f3f", "#7a7440"];
  const chartType = chart.chart_type === "pie" ? "pie" : chart.chart_type === "line" ? "line" : "bar";

  chartInstance = new Chart(ctx, {
    type: chartType,
    data: {
      labels: chart.labels,
      datasets: [
        {
          label: chart.title,
          data: chart.values,
          backgroundColor: chartType === "pie" ? palette : "rgba(43,66,87,0.78)",
          hoverBackgroundColor: chartType === "pie" ? palette : "rgba(43,66,87,0.95)",
          borderColor: chartType === "line" ? "#2b4257" : "transparent",
          borderWidth: chartType === "line" ? 2 : 0,
          borderRadius: chartType === "bar" ? 2 : 0,
          maxBarThickness: 32,
          pointRadius: chartType === "line" ? 3 : 0,
          pointBackgroundColor: "#2b4257",
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 4, right: 4, bottom: 0, left: 0 } },
      animation: { duration: 300, easing: "easeOutQuart" },
      plugins: {
        legend: {
          display: chartType === "pie",
          position: "bottom",
          labels: { color: "#575e64", font: { family: "Fragment Mono", size: 10 }, boxWidth: 9, padding: 10 },
        },
        tooltip: {
          backgroundColor: "#14171a",
          titleColor: "#eef0ee",
          bodyColor: "#eef0ee",
          borderColor: "#2b4257",
          borderWidth: 1,
          padding: 10,
          titleFont: { family: "Fragment Mono", size: 11 },
          bodyFont: { family: "Fragment Mono", size: 11 },
          displayColors: false,
        },
      },
      scales:
        chartType === "pie"
          ? {}
          : {
              x: {
                ticks: { color: "#575e64", font: { family: "Fragment Mono", size: 10 }, maxRotation: 40, minRotation: 0 },
                grid: { display: false },
                border: { color: "#d7dad6" },
              },
              y: {
                ticks: { color: "#575e64", font: { family: "Fragment Mono", size: 10 } },
                grid: { color: "#e4e6e2", drawTicks: false },
                border: { display: false },
              },
            },
    },
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
