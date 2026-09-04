import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";

const BINS = 128;
const SAMPLES = 6000;

const C = {
  bg: "#14171c",
  panel: "#1b1f26",
  panelHi: "#222732",
  grid: "#2a3039",
  gridStrong: "#3a424e",
  ink: "#dce1e9",
  muted: "#7c8797",
  faint: "#4a535f",
  drawn: "#e8a33d",
  drawnSoft: "rgba(232,163,61,0.14)",
  sampled: "#4fb5c9",
  sampledSoft: "rgba(79,181,201,0.30)",
  warn: "#d8695e",
};

const FONT = `'Space Grotesk', ui-sans-serif, system-ui, sans-serif`;
const MONO = `'JetBrains Mono', ui-monospace, SFMono-Regular, monospace`;

/* ---------- distribution helpers ---------- */

const binCentre = (i) => -1 + (2 * (i + 0.5)) / BINS;

function gaussMix(components) {
  const d = new Float64Array(BINS);
  for (let i = 0; i < BINS; i++) {
    const x = binCentre(i);
    let v = 0;
    for (const [mu, sd, w] of components) {
      v += (w * Math.exp(-((x - mu) ** 2) / (2 * sd * sd))) / sd;
    }
    d[i] = v;
  }
  const max = Math.max(...d);
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
  for (let i = 0; i < BINS; i++) {
    p[i] = Math.max(density[i], floor);
    sum += p[i];
  }
  if (sum <= 0) return { p: p.fill(1 / BINS), sum: 1 };
  for (let i = 0; i < BINS; i++) p[i] /= sum;
  return { p, sum };
}

function buildCdf(p) {
  const cdf = new Float64Array(BINS + 1);
  for (let i = 0; i < BINS; i++) cdf[i + 1] = cdf[i] + p[i];
  return cdf;
}

function sampleFrom(cdf, n, seed) {
  // deterministic LCG so the preview is stable between renders
  let s = seed >>> 0 || 1;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const u = rand();
    let lo = 0,
      hi = BINS;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid + 1] < u) lo = mid + 1;
      else hi = mid;
    }
    const within = (u - cdf[lo]) / Math.max(cdf[lo + 1] - cdf[lo], 1e-12);
    out[k] = -1 + (2 * (lo + within)) / BINS;
  }
  return out;
}

function moments(p) {
  let mean = 0;
  for (let i = 0; i < BINS; i++) mean += p[i] * binCentre(i);
  let v = 0,
    m3 = 0;
  for (let i = 0; i < BINS; i++) {
    const d = binCentre(i) - mean;
    v += p[i] * d * d;
    m3 += p[i] * d * d * d;
  }
  const sd = Math.sqrt(v);
  const skew = sd > 1e-9 ? m3 / sd ** 3 : 0;
  let below = 0;
  for (let i = 0; i < BINS; i++) if (binCentre(i) < 0) below += p[i];
  return { mean, sd, skew, below };
}

function gapCount(density, floor) {
  if (floor > 0) return 0;
  let runs = 0,
    inRun = false;
  for (let i = 0; i < BINS; i++) {
    if (density[i] <= 1e-6) {
      if (!inRun) {
        inRun = true;
        runs++;
      }
    } else inRun = false;
  }
  return runs;
}

function smooth(density) {
  const out = new Float64Array(BINS);
  for (let i = 0; i < BINS; i++) {
    const a = density[Math.max(0, i - 1)];
    const b = density[i];
    const c = density[Math.min(BINS - 1, i + 1)];
    out[i] = (a + 2 * b + c) / 4;
  }
  return out;
}

function mirror(density) {
  const out = new Float64Array(BINS);
  for (let i = 0; i < BINS; i++) out[i] = density[BINS - 1 - i];
  return out;
}

/* ---------- canvas ---------- */

function Plot({ density, samples, onPaint, showSamples }) {
  const ref = useRef(null);
  const box = useRef(null);
  const painting = useRef(false);
  const last = useRef(null);

  const hist = useMemo(() => {
    if (!showSamples) return null;
    const h = new Float64Array(BINS);
    for (let k = 0; k < samples.length; k++) {
      let idx = Math.floor(((samples[k] + 1) / 2) * BINS);
      if (idx < 0) idx = 0;
      if (idx >= BINS) idx = BINS - 1;
      h[idx]++;
    }
    const max = Math.max(...h) || 1;
    for (let i = 0; i < BINS; i++) h[i] /= max;
    return h;
  }, [samples, showSamples]);

  const draw = useCallback(() => {
    const cv = ref.current;
    if (!cv) return;
    const rect = cv.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const W = rect.width;
    const H = 210;
    cv.width = W * dpr;
    cv.height = H * dpr;
    cv.style.width = W + "px";
    cv.style.height = H + "px";
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    const px = (x) => ((x + 1) / 2) * W;
    const py = (v) => H - 12 - v * (H - 26);

    // horizontal guides
    g.strokeStyle = C.grid;
    g.lineWidth = 1;
    for (let f = 0.25; f <= 1.001; f += 0.25) {
      g.beginPath();
      g.moveTo(0, Math.round(py(f)) + 0.5);
      g.lineTo(W, Math.round(py(f)) + 0.5);
      g.stroke();
    }
    // vertical guides
    for (const x of [-1, -0.5, 0.5, 1]) {
      g.beginPath();
      g.moveTo(Math.round(px(x)) + 0.5, 0);
      g.lineTo(Math.round(px(x)) + 0.5, H - 12);
      g.stroke();
    }
    // centre
    g.strokeStyle = C.gridStrong;
    g.beginPath();
    g.moveTo(Math.round(px(0)) + 0.5, 0);
    g.lineTo(Math.round(px(0)) + 0.5, H - 12);
    g.stroke();

    // baseline
    g.strokeStyle = C.faint;
    g.beginPath();
    g.moveTo(0, py(0) + 0.5);
    g.lineTo(W, py(0) + 0.5);
    g.stroke();

    // sampled histogram
    if (hist) {
      g.fillStyle = C.sampledSoft;
      const bw = W / BINS;
      for (let i = 0; i < BINS; i++) {
        const h = hist[i] * (H - 26);
        g.fillRect(i * bw, py(0) - h, Math.max(bw - 0.6, 0.6), h);
      }
    }

    // drawn density
    g.beginPath();
    g.moveTo(0, py(density[0]));
    for (let i = 0; i < BINS; i++) g.lineTo(px(binCentre(i)), py(density[i]));
    g.lineTo(W, py(density[BINS - 1]));
    g.lineTo(W, py(0));
    g.lineTo(0, py(0));
    g.closePath();
    g.fillStyle = C.drawnSoft;
    g.fill();

    g.beginPath();
    g.moveTo(0, py(density[0]));
    for (let i = 0; i < BINS; i++) g.lineTo(px(binCentre(i)), py(density[i]));
    g.lineTo(W, py(density[BINS - 1]));
    g.strokeStyle = C.drawn;
    g.lineWidth = 1.75;
    g.lineJoin = "round";
    g.stroke();

    // axis ticks
    g.fillStyle = C.faint;
    g.font = `10px ${MONO}`;
    for (const x of [-1, -0.5, 0, 0.5, 1]) {
      const label = x.toFixed(1);
      const w = g.measureText(label).width;
      let tx = px(x) - w / 2;
      tx = Math.max(1, Math.min(W - w - 1, tx));
      g.fillText(label, tx, H - 1);
    }
  }, [density, hist]);

  useEffect(() => {
    draw();
    const ro = new ResizeObserver(draw);
    if (box.current) ro.observe(box.current);
    return () => ro.disconnect();
  }, [draw]);

  const paintAt = (e) => {
    const cv = ref.current;
    const r = cv.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const yRaw = (e.clientY - r.top - 12) / (210 - 26);
    const v = Math.max(0, Math.min(1, 1 - yRaw));
    const idx = Math.max(0, Math.min(BINS - 1, Math.floor(x * BINS)));
    onPaint(idx, v, last.current);
    last.current = { idx, v };
  };

  return (
    <div ref={box} style={{ position: "relative", width: "100%" }}>
      <canvas
        ref={ref}
        style={{
          display: "block",
          cursor: "crosshair",
          touchAction: "none",
          borderRadius: 2,
          background: C.panelHi,
        }}
        onPointerDown={(e) => {
          painting.current = true;
          last.current = null;
          e.currentTarget.setPointerCapture(e.pointerId);
          paintAt(e);
        }}
        onPointerMove={(e) => painting.current && paintAt(e)}
        onPointerUp={() => {
          painting.current = false;
          last.current = null;
        }}
        onPointerLeave={() => {
          painting.current = false;
          last.current = null;
        }}
      />
    </div>
  );
}

/* ---------- readouts ---------- */

function Stat({ label, value, tone }) {
  return (
    <div style={{ minWidth: 74 }}>
      <div style={{ fontSize: 10.5, color: C.muted, letterSpacing: 0.2 }}>{label}</div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 15,
          color: tone || C.ink,
          marginTop: 2,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, hint }) {
  return (
    <label style={{ display: "block", flex: 1, minWidth: 150 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11.5,
          color: C.muted,
          marginBottom: 5,
        }}
      >
        <span>{label}</span>
        <span style={{ fontFamily: MONO, color: C.ink }}>{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: C.drawn, height: 18 }}
      />
      {hint && (
        <div style={{ fontSize: 10.5, color: C.faint, marginTop: 3, lineHeight: 1.4 }}>{hint}</div>
      )}
    </label>
  );
}

/* ---------- axis panel ---------- */

function AxisPanel({ axis, update, remove, canRemove, showSamples, seed }) {
  const { p } = useMemo(() => normalise(axis.density, axis.floor), [axis.density, axis.floor]);
  const cdf = useMemo(() => buildCdf(p), [p]);
  const samples = useMemo(() => sampleFrom(cdf, SAMPLES, seed + axis.id * 7919), [cdf, seed, axis.id]);
  const m = useMemo(() => moments(p), [p]);
  const gaps = gapCount(axis.density, axis.floor);

  const paint = (idx, v, prev) => {
    const d = Float64Array.from(axis.density);
    if (prev && prev.idx !== idx) {
      const step = prev.idx < idx ? 1 : -1;
      const span = Math.abs(idx - prev.idx);
      for (let k = 0; k <= span; k++) {
        const i = prev.idx + step * k;
        d[i] = prev.v + ((v - prev.v) * k) / span;
      }
    } else {
      d[idx] = v;
    }
    update({ ...axis, density: d });
  };

  return (
    <section
      style={{
        background: C.panel,
        border: `1px solid ${C.grid}`,
        borderRadius: 3,
        padding: "16px 18px 18px",
        marginBottom: 14,
      }}
    >
      <div style={{ display: "flex", gap: 10, alignItems: "flex-end", marginBottom: 14 }}>
        <label style={{ flex: "1 1 180px" }}>
          <div style={{ fontSize: 10.5, color: C.muted, marginBottom: 4 }}>Axis</div>
          <input
            value={axis.name}
            onChange={(e) => update({ ...axis, name: e.target.value })}
            style={inputStyle}
          />
        </label>
        <label style={{ flex: "1 1 150px" }}>
          <div style={{ fontSize: 10.5, color: C.muted, marginBottom: 4 }}>Left pole (−1)</div>
          <input
            value={axis.poleNeg}
            onChange={(e) => update({ ...axis, poleNeg: e.target.value })}
            style={inputStyle}
          />
        </label>
        <label style={{ flex: "1 1 150px" }}>
          <div style={{ fontSize: 10.5, color: C.muted, marginBottom: 4 }}>Right pole (+1)</div>
          <input
            value={axis.polePos}
            onChange={(e) => update({ ...axis, polePos: e.target.value })}
            style={inputStyle}
          />
        </label>
        {canRemove && (
          <button onClick={remove} style={{ ...btnStyle, color: C.warn, borderColor: "#3a2a2a" }}>
            Remove
          </button>
        )}
      </div>

      <Plot density={axis.density} samples={samples} onPaint={paint} showSamples={showSamples} />

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: C.muted,
          marginTop: 7,
        }}
      >
        <span>{axis.poleNeg}</span>
        <span>{axis.polePos}</span>
      </div>

      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "14px 0 12px" }}>
        {Object.keys(PRESETS).map((k) => (
          <button
            key={k}
            onClick={() => update({ ...axis, density: PRESETS[k]() })}
            style={btnStyle}
          >
            {k}
          </button>
        ))}
        <button onClick={() => update({ ...axis, density: smooth(axis.density) })} style={btnStyle}>
          smooth
        </button>
        <button
          onClick={() => update({ ...axis, density: mirror(axis.density) })}
          style={{ ...btnStyle, color: C.drawn, borderColor: "#3a3226" }}
          title="Move the population to the other pole. Labels and expression costs stay put."
        >
          mirror
        </button>
      </div>

      <div
        style={{
          display: "flex",
          gap: 22,
          flexWrap: "wrap",
          padding: "12px 0",
          borderTop: `1px solid ${C.grid}`,
        }}
      >
        <Stat label="mean" value={m.mean.toFixed(3)} />
        <Stat label="spread" value={m.sd.toFixed(3)} />
        <Stat label="skew" value={m.skew.toFixed(3)} />
        <Stat
          label={`share on ${axis.poleNeg.slice(0, 14)}`}
          value={(m.below * 100).toFixed(1) + "%"}
          tone={C.drawn}
        />
      </div>

      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginTop: 6 }}>
        <Slider
          label="Density floor"
          value={axis.floor}
          min={0}
          max={0.05}
          step={0.001}
          onChange={(v) => update({ ...axis, floor: v })}
          hint="Lifts empty regions so no stance is impossible to hold."
        />
        <Slider
          label={`Cost of stating ${axis.poleNeg.slice(0, 16)}`}
          value={axis.costNeg}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => update({ ...axis, costNeg: v })}
          hint="Extra hostility and lost reach. Independent of how many hold it."
        />
        <Slider
          label={`Cost of stating ${axis.polePos.slice(0, 16)}`}
          value={axis.costPos}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => update({ ...axis, costPos: v })}
        />
      </div>

      {gaps > 0 && (
        <div
          style={{
            marginTop: 12,
            fontSize: 11.5,
            color: C.warn,
            lineHeight: 1.5,
            borderLeft: `2px solid ${C.warn}`,
            paddingLeft: 10,
          }}
        >
          {gaps} region{gaps > 1 ? "s" : ""} of this axis has zero density. No user will ever be
          sampled there, and the sampler jumps across the gap. Raise the density floor if that is
          not what you want.
        </div>
      )}
    </section>
  );
}

const inputStyle = {
  width: "100%",
  background: C.panelHi,
  border: `1px solid ${C.grid}`,
  borderRadius: 2,
  color: C.ink,
  font: `13px ${FONT}`,
  padding: "7px 9px",
  outline: "none",
  boxSizing: "border-box",
};

const btnStyle = {
  background: C.panelHi,
  border: `1px solid ${C.grid}`,
  borderRadius: 2,
  color: C.muted,
  font: `11.5px ${FONT}`,
  padding: "6px 11px",
  cursor: "pointer",
};

/* ---------- app ---------- */

let nextId = 3;

export default function StanceEditor() {
  const [axes, setAxes] = useState([
    {
      id: 0,
      name: "provision",
      poleNeg: "market",
      polePos: "state",
      density: PRESETS["majority left"](),
      floor: 0.004,
      costNeg: 0.0,
      costPos: 0.35,
    },
    {
      id: 1,
      name: "openness",
      poleNeg: "closed",
      polePos: "open",
      density: PRESETS.polarized(),
      floor: 0.004,
      costNeg: 0.3,
      costPos: 0.1,
    },
    {
      id: 2,
      name: "institutional trust",
      poleNeg: "distrust",
      polePos: "trust",
      density: PRESETS.symmetric(),
      floor: 0.004,
      costNeg: 0,
      costPos: 0,
    },
  ]);
  const [showSamples, setShowSamples] = useState(true);
  const [seed, setSeed] = useState(12345);
  const [copied, setCopied] = useState(false);

  const config = useMemo(
    () => ({
      scenario: {
        stance_axes: axes.map((a) => {
          const { p } = normalise(a.density, a.floor);
          const m = moments(p);
          return {
            name: a.name,
            pole_neg: a.poleNeg,
            pole_pos: a.polePos,
            marginal: {
              kind: "empirical",
              bins: BINS,
              support: [-1, 1],
              density: Array.from(p, (v) => +(v * BINS).toFixed(5)),
            },
            expression_cost: { neg: +a.costNeg.toFixed(3), pos: +a.costPos.toFixed(3) },
            derived: {
              mean: +m.mean.toFixed(4),
              sd: +m.sd.toFixed(4),
              skew: +m.skew.toFixed(4),
              share_neg: +m.below.toFixed(4),
            },
          };
        }),
      },
    }),
    [axes]
  );

  const json = JSON.stringify(config, null, 2);

  const copy = () => {
    const ta = document.createElement("textarea");
    ta.value = json;
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch (e) {
      /* selection left in place for manual copy */
    }
    document.body.removeChild(ta);
  };

  return (
    <div
      style={{
        background: C.bg,
        color: C.ink,
        fontFamily: FONT,
        minHeight: "100%",
        padding: "26px 22px 40px",
      }}
    >
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        input[type=range]{-webkit-appearance:none;background:transparent}
        input[type=range]::-webkit-slider-runnable-track{height:2px;background:${C.gridStrong}}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:11px;height:11px;border-radius:50%;background:${C.drawn};margin-top:-4.5px}
        button:hover{border-color:${C.faint} !important;color:${C.ink} !important}
        input:focus{border-color:${C.faint} !important}
        *:focus-visible{outline:2px solid ${C.sampled};outline-offset:1px}
      `}</style>

      <div style={{ maxWidth: 880, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <h1 style={{ font: "500 27px/1.15 " + FONT, margin: 0, letterSpacing: -0.4 }}>
            Stance distributions
          </h1>
          <p
            style={{
              color: C.muted,
              fontSize: 13.5,
              lineHeight: 1.6,
              margin: "10px 0 0",
              maxWidth: 620,
            }}
          >
            Drag on a curve to shape how the population sits on that axis. The cyan bars are
            what the sampler actually draws from your curve — if they disagree with the line, the
            curve is doing something you did not intend.
          </p>
        </header>

        <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 18 }}>
          <button onClick={() => setShowSamples((s) => !s)} style={btnStyle}>
            {showSamples ? "Hide draw" : "Show draw"}
          </button>
          <button onClick={() => setSeed((s) => s + 1)} style={btnStyle}>
            Resample
          </button>
          <span style={{ fontSize: 11, color: C.faint, fontFamily: MONO }}>
            {SAMPLES.toLocaleString()} users · seed {seed}
          </span>
        </div>

        {axes.map((a) => (
          <AxisPanel
            key={a.id}
            axis={a}
            seed={seed}
            showSamples={showSamples}
            canRemove={axes.length > 1}
            update={(next) => setAxes((xs) => xs.map((x) => (x.id === a.id ? next : x)))}
            remove={() => setAxes((xs) => xs.filter((x) => x.id !== a.id))}
          />
        ))}

        <button
          onClick={() =>
            setAxes((xs) => [
              ...xs,
              {
                id: nextId++,
                name: "new axis",
                poleNeg: "one side",
                polePos: "other side",
                density: PRESETS.symmetric(),
                floor: 0.004,
                costNeg: 0,
                costPos: 0,
              },
            ])
          }
          style={{ ...btnStyle, padding: "9px 14px" }}
        >
          Add axis
        </button>

        <section style={{ marginTop: 34 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 10,
            }}
          >
            <h2 style={{ font: "500 15px " + FONT, margin: 0 }}>Scenario config</h2>
            <button onClick={copy} style={btnStyle}>
              {copied ? "Copied" : "Copy JSON"}
            </button>
          </div>
          <pre
            style={{
              background: C.panel,
              border: `1px solid ${C.grid}`,
              borderRadius: 3,
              padding: 14,
              margin: 0,
              maxHeight: 300,
              overflow: "auto",
              font: `11.5px/1.6 ${MONO}`,
              color: C.muted,
            }}
          >
            {json}
          </pre>
          <p style={{ fontSize: 11.5, color: C.faint, lineHeight: 1.6, marginTop: 10 }}>
            Density is emitted normalised to the bin width, so it integrates to 1 over [−1, 1].
            Build the empirical inverse CDF from it and pass that to the copula as the marginal for
            this axis. Expression cost is separate and belongs to the kernel and ranker.
          </p>
        </section>
      </div>
    </div>
  );
}
