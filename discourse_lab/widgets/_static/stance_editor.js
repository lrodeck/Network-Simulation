// Stance editor anywidget front end — plain DOM/canvas, no build step.
// Ported from stance-editor.jsx; traits synced: name, axes, seed, show_samples.
// Each axis: {id, name, pole_neg, pole_pos, density[128], floor, cost_neg, cost_pos}.

const BINS = 128;
const SAMPLES = 6000;

const C = {
  bg: "#14171c", panel: "#1b1f26", panelHi: "#222732", grid: "#2a3039",
  gridStrong: "#3a424e", ink: "#dce1e9", muted: "#7c8797", faint: "#4a535f",
  drawn: "#e8a33d", drawnSoft: "rgba(232,163,61,0.14)",
  sampled: "#4fb5c9", sampledSoft: "rgba(79,181,201,0.30)", warn: "#d8695e",
};
const FONT = "'Space Grotesk', ui-sans-serif, system-ui, sans-serif";
const MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, monospace";

const binCentre = (i) => -1 + (2 * (i + 0.5)) / BINS;

function gaussMix(components) {
  const d = new Float64Array(BINS);
  for (let i = 0; i < BINS; i++) {
    const x = binCentre(i);
    let v = 0;
    for (const [mu, sd, w] of components) v += (w * Math.exp(-((x - mu) ** 2) / (2 * sd * sd))) / sd;
    d[i] = v;
  }
  const max = Math.max(...d) || 1;
  for (let i = 0; i < BINS; i++) d[i] /= max;
  return d;
}

const PRESETS = {
  symmetric: () => gaussMix([[0, 0.34, 1]]),
  "majority left": () => gaussMix([[-0.42, 0.32, 1], [0.55, 0.22, 0.28]]),
  polarized: () => gaussMix([[-0.6, 0.2, 1], [0.6, 0.2, 1]]),
  "skewed tail": () => gaussMix([[-0.55, 0.22, 1], [0.1, 0.45, 0.45]]),
  flat: () => Float64Array.from({ length: BINS }, () => 0.7),
};

function normalise(density, floor) {
  const p = new Float64Array(BINS);
  let sum = 0;
  for (let i = 0; i < BINS; i++) { p[i] = Math.max(density[i], floor); sum += p[i]; }
  if (sum <= 0) { p.fill(1 / BINS); return p; }
  for (let i = 0; i < BINS; i++) p[i] /= sum;
  return p;
}

function buildCdf(p) {
  const cdf = new Float64Array(BINS + 1);
  for (let i = 0; i < BINS; i++) cdf[i + 1] = cdf[i] + p[i];
  return cdf;
}

function sampleFrom(cdf, n, seed) {
  let s = (seed >>> 0) || 1;
  const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const u = rand();
    let lo = 0, hi = BINS;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid + 1] < u) lo = mid + 1; else hi = mid; }
    const within = (u - cdf[lo]) / Math.max(cdf[lo + 1] - cdf[lo], 1e-12);
    out[k] = -1 + (2 * (lo + within)) / BINS;
  }
  return out;
}

function moments(p) {
  let mean = 0;
  for (let i = 0; i < BINS; i++) mean += p[i] * binCentre(i);
  let v = 0, m3 = 0;
  for (let i = 0; i < BINS; i++) {
    const d = binCentre(i) - mean;
    v += p[i] * d * d; m3 += p[i] * d * d * d;
  }
  const sd = Math.sqrt(v);
  const skew = sd > 1e-9 ? m3 / sd ** 3 : 0;
  let below = 0;
  for (let i = 0; i < BINS; i++) if (binCentre(i) < 0) below += p[i];
  return { mean, sd, skew, below };
}

function smooth(density) {
  const out = new Float64Array(BINS);
  for (let i = 0; i < BINS; i++) {
    const a = density[Math.max(0, i - 1)], b = density[i], c = density[Math.min(BINS - 1, i + 1)];
    out[i] = (a + 2 * b + c) / 4;
  }
  return out;
}

function mirror(density) {
  const out = new Float64Array(BINS);
  for (let i = 0; i < BINS; i++) out[i] = density[BINS - 1 - i];
  return out;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(node.style, v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.appendChild(c);
  return node;
}

function button(label, onClick, style = {}) {
  return el("button", {
    text: label, onclick: onClick,
    style: {
      background: C.panelHi, border: `1px solid ${C.grid}`, borderRadius: "2px",
      color: C.muted, font: `11.5px ${FONT}`, padding: "6px 11px", cursor: "pointer", ...style,
    },
  });
}

function drawPlot(canvas, density, samples, showSamples) {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const W = rect.width || 600, H = 210;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  const px = (x) => ((x + 1) / 2) * W;
  const py = (v) => H - 12 - v * (H - 26);

  g.strokeStyle = C.grid; g.lineWidth = 1;
  for (let f = 0.25; f <= 1.001; f += 0.25) {
    g.beginPath(); g.moveTo(0, Math.round(py(f)) + 0.5); g.lineTo(W, Math.round(py(f)) + 0.5); g.stroke();
  }
  for (const x of [-1, -0.5, 0.5, 1]) {
    g.beginPath(); g.moveTo(Math.round(px(x)) + 0.5, 0); g.lineTo(Math.round(px(x)) + 0.5, H - 12); g.stroke();
  }
  g.strokeStyle = C.gridStrong;
  g.beginPath(); g.moveTo(Math.round(px(0)) + 0.5, 0); g.lineTo(Math.round(px(0)) + 0.5, H - 12); g.stroke();

  g.strokeStyle = C.faint;
  g.beginPath(); g.moveTo(0, py(0) + 0.5); g.lineTo(W, py(0) + 0.5); g.stroke();

  if (showSamples && samples) {
    const hist = new Float64Array(BINS);
    for (let k = 0; k < samples.length; k++) {
      let idx = Math.floor(((samples[k] + 1) / 2) * BINS);
      idx = Math.max(0, Math.min(BINS - 1, idx));
      hist[idx]++;
    }
    const max = Math.max(...hist) || 1;
    g.fillStyle = C.sampledSoft;
    const bw = W / BINS;
    for (let i = 0; i < BINS; i++) {
      const h = (hist[i] / max) * (H - 26);
      g.fillRect(i * bw, py(0) - h, Math.max(bw - 0.6, 0.6), h);
    }
  }

  g.beginPath(); g.moveTo(0, py(density[0]));
  for (let i = 0; i < BINS; i++) g.lineTo(px(binCentre(i)), py(density[i]));
  g.lineTo(W, py(density[BINS - 1])); g.lineTo(W, py(0)); g.lineTo(0, py(0)); g.closePath();
  g.fillStyle = C.drawnSoft; g.fill();

  g.beginPath(); g.moveTo(0, py(density[0]));
  for (let i = 0; i < BINS; i++) g.lineTo(px(binCentre(i)), py(density[i]));
  g.lineTo(W, py(density[BINS - 1]));
  g.strokeStyle = C.drawn; g.lineWidth = 1.75; g.lineJoin = "round"; g.stroke();

  g.fillStyle = C.faint; g.font = `10px ${MONO}`;
  for (const x of [-1, -0.5, 0, 0.5, 1]) {
    const label = x.toFixed(1);
    const w = g.measureText(label).width;
    const tx = Math.max(1, Math.min(W - w - 1, px(x) - w / 2));
    g.fillText(label, tx, H - 1);
  }
}

function renderAxis(model, axis, ctx) {
  const section = el("section", {
    style: {
      background: C.panel, border: `1px solid ${C.grid}`, borderRadius: "3px",
      padding: "16px 18px 18px", marginBottom: "14px",
    },
  });

  const header = el("div", { style: { display: "flex", gap: "10px", alignItems: "flex-end", marginBottom: "14px" } });
  const mkField = (label, value, onInput) => {
    const wrap = el("label", { style: { flex: "1 1 160px" } });
    wrap.appendChild(el("div", { text: label, style: { fontSize: "10.5px", color: C.muted, marginBottom: "4px" } }));
    const input = el("input", { value });
    Object.assign(input.style, {
      width: "100%", background: C.panelHi, border: `1px solid ${C.grid}`, borderRadius: "2px",
      color: C.ink, font: `13px ${FONT}`, padding: "7px 9px", boxSizing: "border-box",
    });
    input.addEventListener("input", (e) => onInput(e.target.value));
    wrap.appendChild(input);
    return wrap;
  };
  header.appendChild(mkField("Axis", axis.name, (v) => ctx.update(axis.id, { name: v })));
  header.appendChild(mkField("Left pole (-1)", axis.pole_neg, (v) => ctx.update(axis.id, { pole_neg: v })));
  header.appendChild(mkField("Right pole (+1)", axis.pole_pos, (v) => ctx.update(axis.id, { pole_pos: v })));
  if (ctx.canRemove) {
    header.appendChild(button("Remove", () => ctx.remove(axis.id), { color: C.warn, borderColor: "#3a2a2a" }));
  }
  section.appendChild(header);

  const box = el("div", { style: { position: "relative", width: "100%" } });
  const canvas = el("canvas", {
    style: { display: "block", cursor: "crosshair", touchAction: "none", borderRadius: "2px", background: C.panelHi },
  });
  box.appendChild(canvas);
  section.appendChild(box);

  const p = normalise(axis.density, axis.floor);
  const cdf = buildCdf(p);
  const samples = ctx.showSamples ? sampleFrom(cdf, SAMPLES, ctx.seed + axis.id * 7919) : null;
  const redraw = () => drawPlot(canvas, axis.density, samples, ctx.showSamples);
  requestAnimationFrame(redraw);
  new ResizeObserver(redraw).observe(box);

  let last = null;
  const paintAt = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const yRaw = (e.clientY - r.top - 12) / (210 - 26);
    const v = Math.max(0, Math.min(1, 1 - yRaw));
    const idx = Math.max(0, Math.min(BINS - 1, Math.floor(x * BINS)));
    const d = Array.from(axis.density);
    if (last && last.idx !== idx) {
      const step = last.idx < idx ? 1 : -1;
      const span = Math.abs(idx - last.idx);
      for (let k = 0; k <= span; k++) {
        const i = last.idx + step * k;
        d[i] = last.v + ((v - last.v) * k) / span;
      }
    } else d[idx] = v;
    last = { idx, v };
    ctx.update(axis.id, { density: d });
  };
  canvas.addEventListener("pointerdown", (e) => { last = null; canvas.setPointerCapture(e.pointerId); paintAt(e); });
  canvas.addEventListener("pointermove", (e) => { if (e.buttons & 1) paintAt(e); });
  canvas.addEventListener("pointerup", () => { last = null; });
  canvas.addEventListener("pointerleave", () => { last = null; });

  const poles = el("div", {
    style: { display: "flex", justifyContent: "space-between", fontSize: "11px", color: C.muted, marginTop: "7px" },
  });
  poles.appendChild(el("span", { text: axis.pole_neg }));
  poles.appendChild(el("span", { text: axis.pole_pos }));
  section.appendChild(poles);

  const presetsRow = el("div", { style: { display: "flex", gap: "6px", flexWrap: "wrap", margin: "14px 0 12px" } });
  for (const name of Object.keys(PRESETS)) {
    presetsRow.appendChild(button(name, () => ctx.update(axis.id, { density: Array.from(PRESETS[name]()) })));
  }
  presetsRow.appendChild(button("smooth", () => ctx.update(axis.id, { density: Array.from(smooth(axis.density)) })));
  presetsRow.appendChild(button("mirror", () => ctx.update(axis.id, { density: Array.from(mirror(axis.density)) }),
    { color: C.drawn, borderColor: "#3a3226" }));
  section.appendChild(presetsRow);

  const m = moments(p);
  const stats = el("div", {
    style: { display: "flex", gap: "22px", flexWrap: "wrap", padding: "12px 0", borderTop: `1px solid ${C.grid}` },
  });
  const stat = (label, value, tone) => {
    const s = el("div", { style: { minWidth: "74px" } });
    s.appendChild(el("div", { text: label, style: { fontSize: "10.5px", color: C.muted } }));
    s.appendChild(el("div", { text: value, style: { fontFamily: MONO, fontSize: "15px", color: tone || C.ink, marginTop: "2px" } }));
    return s;
  };
  stats.appendChild(stat("mean", m.mean.toFixed(3)));
  stats.appendChild(stat("spread", m.sd.toFixed(3)));
  stats.appendChild(stat("skew", m.skew.toFixed(3)));
  stats.appendChild(stat(`share on ${axis.pole_neg.slice(0, 14)}`, (m.below * 100).toFixed(1) + "%", C.drawn));
  section.appendChild(stats);

  const sliders = el("div", { style: { display: "flex", gap: "20px", flexWrap: "wrap", marginTop: "6px" } });
  const slider = (label, value, min, max, step, onChange, hint) => {
    const wrap = el("label", { style: { display: "block", flex: "1", minWidth: "150px" } });
    const top = el("div", { style: { display: "flex", justifyContent: "space-between", fontSize: "11.5px", color: C.muted, marginBottom: "5px" } });
    top.appendChild(el("span", { text: label }));
    const valSpan = el("span", { text: value.toFixed(2), style: { fontFamily: MONO, color: C.ink } });
    top.appendChild(valSpan);
    wrap.appendChild(top);
    const input = el("input", { type: "range", min, max, step, value });
    input.style.width = "100%";
    input.addEventListener("input", (e) => { valSpan.textContent = parseFloat(e.target.value).toFixed(2); onChange(parseFloat(e.target.value)); });
    wrap.appendChild(input);
    if (hint) wrap.appendChild(el("div", { text: hint, style: { fontSize: "10.5px", color: C.faint, marginTop: "3px", lineHeight: "1.4" } }));
    return wrap;
  };
  sliders.appendChild(slider("Density floor", axis.floor, 0, 0.05, 0.001, (v) => ctx.update(axis.id, { floor: v }),
    "Lifts empty regions so no stance is impossible to hold."));
  sliders.appendChild(slider(`Cost of stating ${axis.pole_neg.slice(0, 16)}`, axis.cost_neg, 0, 1, 0.01,
    (v) => ctx.update(axis.id, { cost_neg: v }), "Extra hostility and lost reach."));
  sliders.appendChild(slider(`Cost of stating ${axis.pole_pos.slice(0, 16)}`, axis.cost_pos, 0, 1, 0.01,
    (v) => ctx.update(axis.id, { cost_pos: v })));
  section.appendChild(sliders);

  return section;
}

function render({ model, el: root }) {
  root.innerHTML = "";
  Object.assign(root.style, { background: C.bg, color: C.ink, fontFamily: FONT, padding: "20px" });

  const header = el("div", { style: { marginBottom: "18px" } });
  header.appendChild(el("h1", { text: "Stance distributions", style: { font: `500 22px ${FONT}`, margin: 0 } }));
  header.appendChild(el("p", {
    text: "Drag on a curve to shape how the population sits on that axis. The cyan bars are what the sampler actually draws.",
    style: { color: C.muted, fontSize: "12.5px", lineHeight: "1.6", margin: "8px 0 0", maxWidth: "620px" },
  }));
  root.appendChild(header);

  const toolbar = el("div", { style: { display: "flex", gap: "10px", alignItems: "center", marginBottom: "16px" } });
  const toggleBtn = button(model.get("show_samples") ? "Hide draw" : "Show draw", () => {
    model.set("show_samples", !model.get("show_samples"));
    model.save_changes();
  });
  toolbar.appendChild(toggleBtn);
  toolbar.appendChild(button("Resample", () => { model.set("seed", model.get("seed") + 1); model.save_changes(); }));
  const seedLabel = el("span", {
    text: `${SAMPLES.toLocaleString()} users · seed ${model.get("seed")}`,
    style: { fontSize: "11px", color: C.faint, fontFamily: MONO },
  });
  toolbar.appendChild(seedLabel);
  root.appendChild(toolbar);

  const list = el("div", {});
  root.appendChild(list);

  let nextId = Math.max(-1, ...model.get("axes").map((a) => a.id)) + 1;

  const update = (id, patch) => {
    const axes = model.get("axes").map((a) => (a.id === id ? { ...a, ...patch } : a));
    model.set("axes", axes);
    model.save_changes();
  };
  const remove = (id) => {
    model.set("axes", model.get("axes").filter((a) => a.id !== id));
    model.save_changes();
  };

  const renderList = () => {
    list.innerHTML = "";
    const axes = model.get("axes");
    const ctx = { seed: model.get("seed"), showSamples: model.get("show_samples"), canRemove: axes.length > 1, update, remove };
    for (const axis of axes) list.appendChild(renderAxis(model, axis, ctx));

    const addBtn = button("Add axis", () => {
      const axes2 = [...model.get("axes"), {
        id: nextId++, name: "new axis", pole_neg: "one side", pole_pos: "other side",
        density: Array.from(PRESETS.symmetric()), floor: 0.004, cost_neg: 0, cost_pos: 0,
      }];
      model.set("axes", axes2);
      model.save_changes();
    }, { padding: "9px 14px" });
    list.appendChild(addBtn);
  };

  renderList();
  model.on("change:axes change:seed change:show_samples", () => {
    toggleBtn.textContent = model.get("show_samples") ? "Hide draw" : "Show draw";
    seedLabel.textContent = `${SAMPLES.toLocaleString()} users · seed ${model.get("seed")}`;
    renderList();
  });
}

export default { render };
