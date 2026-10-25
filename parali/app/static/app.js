"use strict";

// Probability bands for the map (calibrated probability of a fire in the next 3 days).
// One orange hue, brighter = higher risk (validated as an ordinal ramp on the dark map).
const BANDS = [
  { min: 0.35, color: "#ffb36b", label: "35% or more" },
  { min: 0.20, color: "#ec7a2e", label: "20 to 35%" },
  { min: 0.10, color: "#c2541a", label: "10 to 20%" },
  { min: 0.05, color: "#8a3a12", label: "5 to 10%" },
];
const ACCENT = "#76b900";
// Below 5% the cell is left clear so the map highlights where action is needed.

const state = { dates: [], index: 0, selected: null, metrics: null, bulletin: null, lang: "en", season: [], timer: null };
const $ = (id) => document.getElementById(id);
const pct = (x, digits = 0) => `${(100 * x).toFixed(digits)}%`;

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      base: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        maxzoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      },
    },
    // OpenStreetMap tiles turned into a dark basemap: grey, brightness inverted, low contrast
    layers: [{
      id: "base", type: "raster", source: "base",
      paint: { "raster-saturation": -1, "raster-brightness-min": 0.86, "raster-brightness-max": 0.04, "raster-contrast": 0.15 },
    }],
  },
  center: [75.7, 30.3],
  zoom: 6.4,
  attributionControl: { compact: true },
});
map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

function fillColor() {
  const expr = ["case"];
  for (const b of BANDS) expr.push([">=", ["get", "prob"], b.min], b.color);
  expr.push("rgba(0,0,0,0)");
  return expr;
}

function buildLegend() {
  $("legend").innerHTML = BANDS.map(
    (b) => `<li><span class="swatch" style="background:${b.color}"></span>${b.label}</li>`
  ).join("");
}

async function loadDay() {
  const d = state.dates[state.index];
  $("date-label").textContent = new Date(d + "T00:00:00").toLocaleDateString("en-IN", {
    weekday: "short", day: "numeric", month: "short", year: "numeric",
  });
  $("date-range").value = state.index;
  renderTimeline();
  const [risk, fires, top] = await Promise.all([
    getJSON(`/api/risk?date=${d}&min_prob=0.05`),
    getJSON(`/api/fires?date=${d}`),
    getJSON(`/api/top?date=${d}&n=10`),
  ]);
  map.getSource("risk").setData(risk);
  map.getSource("fires").setData(fires);
  renderTop(top);
  if (state.selected) selectCell(state.selected, false);
  loadBulletin(d);
}

async function loadBulletin(d) {
  try {
    state.bulletin = await getJSON(`/api/bulletin?date=${d}`);
  } catch {
    state.bulletin = null;
  }
  renderBulletin();
}

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderBulletin() {
  const b = state.bulletin;
  if (!b) { $("b-headline").textContent = "Bulletin unavailable for this date."; return; }
  const t = b[state.lang];
  $("b-headline").textContent = t.headline;
  $("b-summary").textContent = t.summary;
  $("b-actions").innerHTML = t.actions.map((a) => `<li>${esc(a)}</li>`).join("");
  $("b-districts").innerHTML = b.districts
    .map((d) => `<li><span>${esc(d.district)}, ${esc(d.state)} · ${d.fires_last_3d} fires in 3 days (${d.trend})</span><span class="lvl ${d.level}">${d.level}</span></li>`)
    .join("");
  $("b-status").classList.toggle("approved", b.status === "approved");
  $("b-status").textContent = b.status === "approved"
    ? `Approved by ${b.approved_by}. Written by: ${b.writer === "llm" ? "AI agents" : "agents (template)"}.`
    : `Draft, waiting for an official's approval. Written by: ${b.writer === "llm" ? "AI agents" : "agents (template)"}.`;
  $("approve-form").hidden = b.status === "approved";
}

function setLang(lang) {
  state.lang = lang;
  $("lang-en").classList.toggle("active", lang === "en");
  $("lang-hi").classList.toggle("active", lang === "hi");
  $("lang-en").setAttribute("aria-pressed", lang === "en");
  $("lang-hi").setAttribute("aria-pressed", lang === "hi");
  renderBulletin();
}

function renderTop(rows) {
  $("top-list").innerHTML = rows
    .map(
      (r, i) => `<li><button data-cell="${r.cell}" data-lon="${r.lon}" data-lat="${r.lat}">
        <span class="rank">${String(i + 1).padStart(2, "0")}</span>
        <span>${esc(r.district)}, ${esc(r.state)}</span><span class="pct">${pct(r.prob)}</span></button></li>`
    )
    .join("");
  $("top-list").querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => {
      map.flyTo({ center: [Number(btn.dataset.lon), Number(btn.dataset.lat)], zoom: 7.6 });
      selectCell(btn.dataset.cell, true);
    })
  );
}

function highlight(cell) {
  for (const id of ["risk-selected", "risk-selected-halo"]) map.setFilter(id, ["==", ["get", "cell"], cell]);
}

async function selectCell(cell, scroll) {
  state.selected = cell;
  highlight(cell);
  const d = state.dates[state.index];
  let info;
  try {
    info = await getJSON(`/api/cell/${cell}?date=${d}`);
  } catch {
    return;
  }
  showTab("risk");
  $("detail").hidden = false;
  $("d-district").textContent = `${info.district}, ${info.state}`;
  $("d-meta").textContent = `Cell ${info.cell} · ${info.lat.toFixed(3)}, ${info.lon.toFixed(3)}`;
  $("d-prob").textContent = pct(info.prob);
  $("d-rank").textContent = `Top ${Math.max(1, Math.ceil(100 * (1 - info.pct_rank)))}%`;
  const maxShap = Math.max(...info.reasons.map((r) => Math.abs(r.shap)), 1e-6);
  $("d-reasons").innerHTML = info.reasons
    .map((r) => {
      const width = Math.round((100 * Math.abs(r.shap)) / maxShap);
      const dir = r.shap >= 0 ? "raises risk" : "lowers risk";
      return `<li>${r.text} <span class="muted">(${dir})</span>
        <div class="reason-bar ${r.shap < 0 ? "neg" : ""}" style="width:${width}%"></div></li>`;
    })
    .join("");
  const maxFires = Math.max(...info.history.map((h) => h.fires), 1);
  $("d-history").innerHTML = info.history
    .map((h) => {
      const height = Math.round((100 * h.fires) / maxFires);
      return `<span class="${h.fires ? "" : "zero"}" style="height:${Math.max(height, 4)}%" title="${h.date}: ${h.fires} fire(s)"></span>`;
    })
    .join("");
  const n = info.fires_next3_actual;
  $("d-outcome").textContent =
    n > 0
      ? `What actually happened: ${n} fire(s) were detected here in the next 3 days.`
      : "What actually happened: no fires were detected here in the next 3 days.";
  if (scroll) $("detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

map.on("load", async () => {
  buildLegend();
  map.addSource("risk", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addSource("fires", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addSource("districts", { type: "geojson", data: "/api/districts" });

  map.addLayer({ id: "risk-fill", type: "fill", source: "risk", paint: { "fill-color": fillColor(), "fill-opacity": 0.82, "fill-outline-color": "rgba(0,0,0,0.45)" } });
  map.addLayer({ id: "districts-line", type: "line", source: "districts", paint: { "line-color": "#8a8a8a", "line-width": 0.6, "line-opacity": 0.55 } });
  map.addLayer({ id: "risk-selected-halo", type: "line", source: "risk", filter: ["==", ["get", "cell"], ""], paint: { "line-color": ACCENT, "line-width": 9, "line-opacity": 0.3, "line-blur": 3 } });
  map.addLayer({ id: "risk-selected", type: "line", source: "risk", filter: ["==", ["get", "cell"], ""], paint: { "line-color": ACCENT, "line-width": 3 } });
  map.addLayer({
    id: "fires", type: "circle", source: "fires",
    paint: { "circle-radius": 2.8, "circle-color": "#ffffff", "circle-stroke-color": "#ff5a36", "circle-stroke-width": 1.2 },
  });

  map.on("click", "risk-fill", (e) => selectCell(e.features[0].properties.cell, true));
  map.on("mouseenter", "risk-fill", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "risk-fill", () => (map.getCanvas().style.cursor = ""));

  const meta = await getJSON("/api/meta");
  if (meta.bbox) map.fitBounds([[meta.bbox[0], meta.bbox[1]], [meta.bbox[2], meta.bbox[3]]], { padding: 24, duration: 0 });
  state.dates = meta.dates;
  state.index = Math.max(0, meta.dates.indexOf(meta.default_date));
  $("date-range").max = meta.dates.length - 1;
  if (meta.metrics && meta.metrics.test) {
    const m = meta.metrics.test;
    const rule = meta.metrics.best_rule;
    $("metrics-line").textContent =
      `On the unseen 2025 season, the 10% of cells ranked highest each day held ${pct(m.fire_capture_top10pct, 1)} ` +
      `of the next 3 days' fires (best simple rule: ${pct(rule.fire_capture_top10pct, 1)}).`;
    $("kpi-capture").textContent = pct(m.fire_capture_top10pct, 1);
    $("kpi-capture-rule").textContent = `best rule ${pct(rule.fire_capture_top10pct, 1)}`;
    $("kpi-prauc").textContent = m.pr_auc.toFixed(3);
    $("kpi-prauc-rule").textContent = `best rule ${rule.pr_auc.toFixed(3)}`;
  }
  try { state.season = await getJSON("/api/season"); } catch { state.season = []; }
  await loadDay();
});

$("date-range").addEventListener("input", (e) => { state.index = Number(e.target.value); loadDay(); });
function step(delta) {
  const next = state.index + delta;
  if (next < 0 || next > state.dates.length - 1) return false;
  state.index = next;
  loadDay();
  return true;
}
$("prev-day").addEventListener("click", () => step(-1));
$("next-day").addEventListener("click", () => step(1));
// arrow keys move through the season unless the user is typing
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, textarea, select") || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.key === "ArrowLeft") step(-1);
  if (e.key === "ArrowRight") step(1);
});

// Play the season: one day per second, stops at the last day.
const PLAY = '<path d="M7 5v14l12-7z" fill="currentColor"/>';
const PAUSE = '<path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor"/>';
function setPlaying(on) {
  clearInterval(state.timer);
  state.timer = on ? setInterval(() => { if (!step(1)) setPlaying(false); }, 1000) : null;
  $("play").classList.toggle("playing", on);
  $("play-icon").innerHTML = on ? PAUSE : PLAY;
  $("play").setAttribute("aria-label", on ? "Pause" : "Play the season");
}
$("play").addEventListener("click", () => {
  if (!state.timer && state.index >= state.dates.length - 1) state.index = 0;
  setPlaying(!state.timer);
});

// Season timeline: fires detected per day, the selected day in the accent colour.
function renderTimeline() {
  const svg = $("timeline");
  const rows = state.season;
  if (!rows.length) { svg.innerHTML = ""; return; }
  const w = svg.clientWidth || 600, h = 46, top = 2, axis = 12;
  const max = Math.max(...rows.map((r) => r.fires), 1);
  const bw = w / rows.length, gap = Math.min(2, bw * 0.25);
  const ticks = [];
  rows.forEach((r, i) => {
    if (r.date.endsWith("-01")) ticks.push(`<text class="tick" x="${i * bw}" y="${h}">${new Date(r.date + "T00:00:00").toLocaleDateString("en-IN", { month: "short" })}</text>`);
  });
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML =
    rows.map((r, i) => {
      const bh = Math.max(1, ((h - axis - top) * r.fires) / max);
      const cls = i === state.index ? "bar current" : r.fires ? "bar fire" : "bar";
      return `<rect class="${cls}" data-i="${i}" x="${i * bw + gap / 2}" y="${h - axis - bh}" width="${Math.max(1, bw - gap)}" height="${bh}" rx="1"></rect>`;
    }).join("") +
    `<line class="base" x1="0" x2="${w}" y1="${h - axis + 0.5}" y2="${h - axis + 0.5}"></line>` + ticks.join("");
  const r = rows[state.index];
  if (r) {
    $("timeline-readout").innerHTML = `<b>${r.fires}</b> fires detected · <b>${r.expected_cells}</b> cells expected to burn · <b>${r.high_risk_cells}</b> at high risk`;
  }
}
window.addEventListener("resize", renderTimeline);

// Hover tooltip on the timeline (the transparent slider sits on top and receives the pointer).
const tip = document.createElement("div");
tip.className = "tooltip";
tip.hidden = true;
document.querySelector(".timeline-wrap").appendChild(tip);
$("date-range").addEventListener("pointermove", (e) => {
  const rows = state.season;
  if (!rows.length) return;
  const box = $("timeline").getBoundingClientRect();
  const i = Math.min(rows.length - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * rows.length)));
  const r = rows[i];
  $("timeline").querySelectorAll(".bar.hover").forEach((b) => b.classList.remove("hover"));
  const bar = $("timeline").querySelector(`.bar[data-i="${i}"]`);
  if (bar && i !== state.index) bar.classList.add("hover");
  const d = new Date(r.date + "T00:00:00").toLocaleDateString("en-IN", { day: "numeric", month: "short" });
  tip.innerHTML = `<b>${d}</b> · ${r.fires} fires · ${r.expected_cells} cells expected`;
  tip.hidden = false;
  const wrap = document.querySelector(".timeline-wrap").getBoundingClientRect();
  const x = Math.min(Math.max(e.clientX - wrap.left - 70, 0), wrap.width - 200);
  tip.style.left = `${x}px`;
  tip.style.top = `-6px`;
});
$("date-range").addEventListener("pointerleave", () => {
  tip.hidden = true;
  $("timeline").querySelectorAll(".bar.hover").forEach((b) => b.classList.remove("hover"));
});

// Tabs
function showTab(name) {
  for (const t of ["risk", "bulletin", "ask"]) {
    const on = t === name;
    $(`tab-${t}`).classList.toggle("active", on);
    $(`tab-${t}`).setAttribute("aria-selected", on);
    $(`pane-${t}`).hidden = !on;
  }
}
for (const t of ["risk", "bulletin", "ask"]) $(`tab-${t}`).addEventListener("click", () => showTab(t));

// Example questions
document.querySelectorAll("#examples .chip").forEach((c) =>
  c.addEventListener("click", () => { $("ask-input").value = c.textContent; $("ask-form").requestSubmit(); })
);
$("show-fires").addEventListener("change", (e) => map.setLayoutProperty("fires", "visibility", e.target.checked ? "visible" : "none"));
$("close-detail").addEventListener("click", () => {
  state.selected = null;
  $("detail").hidden = true;
  highlight("");
});

$("lang-en").addEventListener("click", () => setLang("en"));
$("lang-hi").addEventListener("click", () => setLang("hi"));

$("approve-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("approver").value.trim();
  if (name.length < 2) { $("approver").focus(); return; }
  const res = await fetch(`/api/bulletin/approve?date=${state.dates[state.index]}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ official: name }),
  });
  if (res.ok) { state.bulletin = await res.json(); renderBulletin(); }
});

// Speak an answer with the browser's own text-to-speech (no server model, works offline on phones).
function speak(text) {
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const clean = text.replace(/\[\d+\]/g, "").replace(/\n+/g, ". ");
  const u = new SpeechSynthesisUtterance(clean);
  // choose the voice by the script most of the text is in (quoted sources may be English)
  const deva = (clean.match(/[\u0900-\u097F]/g) || []).length;
  const latin = (clean.match(/[A-Za-z]/g) || []).length;
  u.lang = deva > latin ? "hi-IN" : "en-IN";
  const voice = window.speechSynthesis.getVoices().find((v) => v.lang === u.lang);
  if (voice) u.voice = voice;
  window.speechSynthesis.speak(u);
}

// Voice input: record with MediaRecorder, transcribe on the server (Whisper), then ask.
const rec = { recorder: null, chunks: [] };
$("mic").addEventListener("click", async () => {
  const btn = $("mic");
  if (rec.recorder && rec.recorder.state === "recording") { rec.recorder.stop(); return; }
  if (!navigator.mediaDevices?.getUserMedia) { $("mic-label").textContent = "No mic"; return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch { $("mic-label").textContent = "Mic blocked"; return; }
  rec.chunks = [];
  rec.recorder = new MediaRecorder(stream);
  rec.recorder.ondataavailable = (ev) => ev.data.size && rec.chunks.push(ev.data);
  rec.recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    btn.classList.remove("recording");
    $("mic-label").textContent = "Listening\u2026";
    btn.disabled = true;
    try {
      const blob = new Blob(rec.chunks, { type: rec.recorder.mimeType });
      const res = await fetch("/api/transcribe", { method: "POST", body: blob, headers: { "Content-Type": blob.type || "application/octet-stream" } });
      const t = await res.json();
      if (!res.ok) { $("ask-answer").hidden = false; $("ask-answer").textContent = t.detail || "Could not understand the audio."; return; }
      $("ask-input").value = t.text;
      if (t.text.trim().length >= 3) { state.voiceAsk = true; $("ask-form").requestSubmit(); }
    } finally {
      btn.disabled = false;
      $("mic-label").textContent = "Speak";
    }
  };
  rec.recorder.start();
  btn.classList.add("recording");
  $("mic-label").textContent = "Stop";
});

// Thumbs up / down on an answer. Ratings go into the feedback loop (see docs/OPS.md).
function feedbackRow(traceId) {
  const row = document.createElement("div");
  row.className = "feedback";
  row.innerHTML = `<span>Was this useful?</span>
    <button type="button" class="fb" data-r="1" aria-label="Useful">&#128077;</button>
    <button type="button" class="fb" data-r="-1" aria-label="Not useful">&#128078;</button>`;
  row.querySelectorAll(".fb").forEach((b) => b.addEventListener("click", async () => {
    row.querySelectorAll(".fb").forEach((x) => (x.disabled = true));
    try {
      const res = await fetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, rating: Number(b.dataset.r) }),
      });
      row.querySelector("span").textContent = res.ok ? "Thanks for the feedback." : "Could not save feedback.";
    } catch { row.querySelector("span").textContent = "Could not save feedback."; }
  }));
  return row;
}

// While an answer is being prepared: a live timer, and on the first question an explanation of
// the one-time setup (model download / loading), from /api/assistant/status.
const fmtTime = (s) => (s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`);
const SETUP_NOTES = {
  first_run: "First question on this computer: downloading the search models (about 1 GB, only once) and indexing " +
    "the documents. This usually takes 3 to 10 minutes depending on your internet. You can keep using the map meanwhile.",
  loading: "First question since the app started: loading the search models into memory (about 30 to 60 seconds).",
};
async function showProgress(box) {
  let stage = "ready";
  try { stage = (await getJSON("/api/assistant/status")).stage; } catch { /* status is optional */ }
  const start = Date.now();
  box.innerHTML = `<div class="progress" role="status" aria-live="polite">
      <div class="progress-line"><span class="spinner" aria-hidden="true"></span>
        <span>Searching the documents\u2026 <b class="elapsed">0 s</b></span></div>
      <div class="progress-bar" aria-hidden="true"><i></i></div>
      ${SETUP_NOTES[stage] ? `<p class="progress-note">${SETUP_NOTES[stage]}</p>` : ""}
    </div>`;
  const el = box.querySelector(".elapsed");
  const timer = setInterval(() => { el.textContent = fmtTime(Math.round((Date.now() - start) / 1000)); }, 1000);
  return () => { clearInterval(timer); return Math.max(1, Math.round((Date.now() - start) / 1000)); };
}

// Answer text -> one block per cited statement ("... [2]"), plus the knowledge-graph facts.
function answerHtml(text) {
  const [main, facts] = text.split(/\n\nRelated facts:\n/);
  const lines = main.split("\n");
  let intro = "";
  if (lines.length > 1 && !/\[\d+\]/.test(lines[0])) intro = `<p class="answer-intro">${esc(lines.shift())}</p>`;
  const body = lines.join(" ");
  const parts = [...body.matchAll(/(.+?)\s*\[(\d+)\](?=\s|$)/g)];
  let html = intro;
  if (parts.length) {
    html += `<ul class="quotes">${parts
      .map((m) => `<li><span>${esc(m[1].trim())}</span><span class="cite-chip" title="Source ${m[2]}">${m[2]}</span></li>`)
      .join("")}</ul>`;
  } else {
    html += `<p>${esc(body)}</p>`;
  }
  if (facts) {
    html += `<p class="eyebrow cites-head">Related facts (reviewed knowledge graph)</p><ul class="facts">${facts
      .split("\n").filter(Boolean).map((f) => `<li>${esc(f.replace(/^-\s*/, ""))}</li>`).join("")}</ul>`;
  }
  return html;
}

$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("ask-input").value.trim();
  const box = $("ask-answer");
  if (q.length < 3) return;
  const btn = e.target.querySelector("button");
  btn.disabled = true;
  box.hidden = false;
  const stopProgress = await showProgress(box);
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }),
    });
    const a = await res.json();
    const seconds = stopProgress();
    if (!res.ok) { box.textContent = a.detail || "Something went wrong."; return; }
    const cites = a.citations
      .map((c) => `<li>[${c.n}] ${esc(c.title)}, page ${c.page}</li>`)
      .join("");
    const modes = { llm: "Written by the AI from the sources below.", extractive: "Quoted from the sources below (no AI model connected).", not_found: "" };
    box.innerHTML = `${answerHtml(a.text)}${cites ? `<p class="eyebrow cites-head">Sources</p><ol class="cites">${cites}</ol>` : ""}<div class="mode">${modes[a.mode] || ""}${modes[a.mode] ? " · " : ""}Answered in ${fmtTime(seconds)}</div>`;
    if ("speechSynthesis" in window) {
      const listen = document.createElement("button");
      listen.type = "button";
      listen.className = "btn btn-ghost listen";
      listen.textContent = a.language === "hi" ? "\u0938\u0941\u0928\u0947\u0902 (Listen)" : "Listen";
      listen.addEventListener("click", () => speak(a.text));
      box.appendChild(listen);
      if (state.voiceAsk) speak(a.text);  // asked by voice -> answer by voice
    }
    state.voiceAsk = false;
    if (a.trace_id) box.appendChild(feedbackRow(a.trace_id));
  } catch {
    stopProgress();
    box.textContent = "Could not reach the assistant.";
  } finally {
    btn.disabled = false;
  }
});
