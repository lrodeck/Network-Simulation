// Run monitor anywidget front end — plain canvas line charts, no build step.
// Traits synced from Python: ticks (int[]), series ({name: number[]}), current_tick.

const C = {
  bg: "#14171c", panel: "#1b1f26", grid: "#2a3039", ink: "#dce1e9",
  muted: "#7c8797", faint: "#4a535f", line: "#4fb5c9", warn: "#d8695e",
};
const FONT = "'Space Grotesk', ui-sans-serif, system-ui, sans-serif";
const MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, monospace";

const PANELS = [
  { key: "attention_gini", label: "attention gini", range: [0, 1] },
  { key: "bubble_index", label: "bubble index", range: [0, 1] },
  { key: "agreement", label: "stance agreement", range: null },
  { key: "salience", label: "salience", range: [0, 1] },
  { key: "r_eff", label: "R_eff", range: null, warnAbove: 1 },
  { key: "n_engagements", label: "engagements / tick", range: null },
];

function el(tag, attrs = {}) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(node.style, v);
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  return node;
}

function drawSeries(canvas, values, range, warnAbove) {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const W = rect.width || 260, H = 90;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return;

  let [lo, hi] = range || [Math.min(...finite), Math.max(...finite)];
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const px = (i) => (i / Math.max(values.length - 1, 1)) * W;
  const py = (v) => H - 4 - ((v - lo) / (hi - lo)) * (H - 8);

  g.strokeStyle = C.grid; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, H - 4); g.lineTo(W, H - 4); g.stroke();

  if (warnAbove !== undefined && warnAbove >= lo && warnAbove <= hi) {
    g.strokeStyle = C.warn; g.setLineDash([3, 3]);
    g.beginPath(); g.moveTo(0, py(warnAbove)); g.lineTo(W, py(warnAbove)); g.stroke();
    g.setLineDash([]);
  }

  g.strokeStyle = C.line; g.lineWidth = 1.75; g.lineJoin = "round";
  g.beginPath();
  let started = false;
  values.forEach((v, i) => {
    if (!Number.isFinite(v)) return;
    const x = px(i), y = py(v);
    if (!started) { g.moveTo(x, y); started = true; } else { g.lineTo(x, y); }
  });
  g.stroke();
}

function render({ model, el: root }) {
  root.innerHTML = "";
  Object.assign(root.style, { background: C.bg, color: C.ink, fontFamily: FONT, padding: "16px" });

  const header = el("div", { style: { display: "flex", justifyContent: "space-between", marginBottom: "12px" } });
  header.appendChild(el("h1", { text: "Run monitor", style: { font: `500 16px ${FONT}`, margin: 0 } }));
  const tickLabel = el("span", { style: { fontFamily: MONO, fontSize: "11px", color: C.faint } });
  header.appendChild(tickLabel);
  root.appendChild(header);

  const grid = el("div", { style: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "10px" } });
  root.appendChild(grid);

  const canvases = {};
  for (const panel of PANELS) {
    const box = el("div", {
      style: { background: C.panel, border: `1px solid ${C.grid}`, borderRadius: "3px", padding: "8px 10px" },
    });
    box.appendChild(el("div", { text: panel.label, style: { fontSize: "11px", color: C.muted, marginBottom: "4px" } }));
    const wrap = el("div", { style: { position: "relative", width: "100%" } });
    const canvas = el("canvas", { style: { display: "block" } });
    wrap.appendChild(canvas);
    box.appendChild(wrap);
    grid.appendChild(box);
    canvases[panel.key] = canvas;
  }

  const redraw = () => {
    const series = model.get("series") || {};
    const ticks = model.get("ticks") || [];
    tickLabel.textContent = ticks.length ? `tick ${model.get("current_tick")} (${ticks.length} logged)` : "waiting for ticks...";
    for (const panel of PANELS) {
      drawSeries(canvases[panel.key], series[panel.key] || [], panel.range, panel.warnAbove);
    }
  };

  redraw();
  model.on("change:series change:ticks change:current_tick", redraw);
  new ResizeObserver(redraw).observe(grid);
}

export default { render };
