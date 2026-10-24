"use strict";

// Probability bands for the map (calibrated probability of a fire in the next 3 days).
const BANDS = [
  { min: 0.35, color: "#7f1d1d", label: "35% or more" },
  { min: 0.20, color: "#c2410c", label: "20 to 35%" },
  { min: 0.10, color: "#f97316", label: "10 to 20%" },
  { min: 0.05, color: "#fde68a", label: "5 to 10%" },
];
// Below 5% the cell is left clear so the map highlights where action is needed.

const state = { dates: [], index: 0, selected: null, metrics: null, bulletin: null, lang: "en" };
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
    layers: [{ id: "base", type: "raster", source: "base", paint: { "raster-saturation": -0.85, "raster-opacity": 0.85 } }],
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
        <span>${i + 1}. ${r.district}, ${r.state}</span><span class="pct">${pct(r.prob)}</span></button></li>`
    )
    .join("");
  $("top-list").querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => {
      map.flyTo({ center: [Number(btn.dataset.lon), Number(btn.dataset.lat)], zoom: 8.5 });
      selectCell(btn.dataset.cell, true);
    })
  );
}

async function selectCell(cell, scroll) {
  state.selected = cell;
  map.setFilter("risk-selected", ["==", ["get", "cell"], cell]);
  const d = state.dates[state.index];
  let info;
  try {
    info = await getJSON(`/api/cell/${cell}?date=${d}`);
  } catch {
    return;
  }
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

  map.addLayer({ id: "risk-fill", type: "fill", source: "risk", paint: { "fill-color": fillColor(), "fill-opacity": 0.62, "fill-outline-color": "rgba(255,255,255,0.35)" } });
  map.addLayer({ id: "districts-line", type: "line", source: "districts", paint: { "line-color": "#57534e", "line-width": 0.7, "line-opacity": 0.6 } });
  map.addLayer({ id: "risk-selected", type: "line", source: "risk", filter: ["==", ["get", "cell"], ""], paint: { "line-color": "#111827", "line-width": 2.5 } });
  map.addLayer({
    id: "fires", type: "circle", source: "fires",
    paint: { "circle-radius": 3.2, "circle-color": "#7f1d1d", "circle-stroke-color": "#ffffff", "circle-stroke-width": 1 },
  });

  map.on("click", "risk-fill", (e) => selectCell(e.features[0].properties.cell, true));
  map.on("mouseenter", "risk-fill", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "risk-fill", () => (map.getCanvas().style.cursor = ""));

  const meta = await getJSON("/api/meta");
  state.dates = meta.dates;
  state.index = Math.max(0, meta.dates.indexOf(meta.default_date));
  $("date-range").max = meta.dates.length - 1;
  if (meta.metrics && meta.metrics.test) {
    const m = meta.metrics.test;
    $("metrics-line").textContent =
      `On the unseen 2025 season, the 10% of cells ranked highest each day held ${pct(m.fire_capture_top10pct)} ` +
      `of the next 3 days' fires (best simple rule: ${pct(meta.metrics.best_rule.fire_capture_top10pct)}).`;
  }
  await loadDay();
});

$("date-range").addEventListener("input", (e) => { state.index = Number(e.target.value); loadDay(); });
$("prev-day").addEventListener("click", () => { if (state.index > 0) { state.index--; loadDay(); } });
$("next-day").addEventListener("click", () => { if (state.index < state.dates.length - 1) { state.index++; loadDay(); } });
$("show-fires").addEventListener("change", (e) => map.setLayoutProperty("fires", "visibility", e.target.checked ? "visible" : "none"));
$("close-detail").addEventListener("click", () => {
  state.selected = null;
  $("detail").hidden = true;
  map.setFilter("risk-selected", ["==", ["get", "cell"], ""]);
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

$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("ask-input").value.trim();
  const box = $("ask-answer");
  if (q.length < 3) return;
  const btn = e.target.querySelector("button");
  btn.disabled = true;
  box.hidden = false;
  box.textContent = "Searching the documents\u2026";
  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }),
    });
    const a = await res.json();
    if (!res.ok) { box.textContent = a.detail || "Something went wrong."; return; }
    const cites = a.citations
      .map((c) => `<li>[${c.n}] ${esc(c.title)}, page ${c.page}</li>`)
      .join("");
    const modes = { llm: "Written by the AI from the sources below.", extractive: "Quoted from the sources below (no AI model connected).", not_found: "" };
    box.innerHTML = `<div>${esc(a.text).replace(/\n/g, "<br>")}</div>${cites ? `<ol class="cites">${cites}</ol>` : ""}<div class="mode">${modes[a.mode] || ""}</div>`;
  } catch {
    box.textContent = "Could not reach the assistant.";
  } finally {
    btn.disabled = false;
  }
});
