// Forecast Workstation — front-end (vanilla ES module, no build step)
import * as live from "./live.js";
import { KINDS as XS_KINDS, LineBox, lineThrough } from "./xsect.js";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = (path) => fetch(path).then(async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});

const WEEK = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"];
const MON = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"];
const PLACES = [
  ["Lisboa", 38.72, -9.14], ["Porto", 41.15, -8.61], ["Faro", 37.02, -7.93],
  ["Coimbra", 40.21, -8.43], ["Bragança", 41.81, -6.76], ["Viseu", 40.66, -7.91],
  ["Évora", 38.57, -7.91], ["Beja", 38.02, -7.86], ["Castelo Branco", 39.82, -7.49],
  ["Penhas Douradas", 40.41, -7.56], ["Sagres", 37.01, -8.94],
  ["Ponta Delgada", 37.74, -25.67], ["Lajes (Terceira)", 38.76, -27.09], ["Horta", 38.53, -28.63],
  ["Funchal", 32.65, -16.91], ["Porto Santo", 33.07, -16.35],
  ["Madrid", 40.42, -3.70], ["Barcelona", 41.39, 2.17], ["Sevilha", 37.39, -5.98],
  ["A Corunha", 43.36, -8.41], ["Valência", 39.47, -0.38], ["Bordéus", 44.84, -0.58],
];
const MODEL_COLORS = { gfs: "#3b82f6", ecmwf: "#ef4444", aifs: "#22c55e", icon_eu: "#a855f7", era5: "#14b8a6" };

const S = {
  tab: "maps", model: "gfs", models: [], runs: [], run: null, steps: [], idx: 0,
  region: "europe", preset: "sfc_precip", layers: null, catalog: null, regions: [],
  lat: 38.72, lon: -9.14, place: "Lisboa", parcel: "ML", playing: false, axes: null,
  cmp: { mode: "maps", models: ["gfs", "ecmwf", "aifs"], preset: "z500_t500", region: "europe",
    times: [], idx: 0, a: "gfs", b: "ecmwf", v: "gh", level: 500, accum: 6, dregion: "europe", xdiff: false,
    axes: {} },
  cs: { eid: null, product: "track", models: ["ecmwf", "gfs", "aifs", "icon_eu"], fc: "ecmwf",
    model: "ecmwf", region: "nea", every: 12, systems: [], station: "08536", time: null, obs: null },
  x: { product: "point", region: "iberia", days: "1-16", sigma: "1p5", view: "sum",
    steps: [], run: null, idx: 0 },
  // vertical cross-section line (shared by the Sondagens and Casos panels)
  xs: { lat0: 32.65, lon0: -16.91, lat1: 38.72, lon1: -9.14, a: "Funchal", b: "Lisboa",
    preset: "L:Madeira → Lisboa", kind: "dyn" },
  sndMode: "point", clickMode: "sounding", xsPending: null,
};
const XS = { fc: null, cs: null, cmp: null };   // LineBox instances
const frames = new Map();     // url -> {src, axes}
let loadToken = 0;

// ------------------------------------------------------------------ helpers
const parseRun = (r) => new Date(Date.UTC(+r.slice(0, 4), +r.slice(4, 6) - 1, +r.slice(6, 8), +r.slice(8, 10)));
const fmt = (d) => `${WEEK[d.getUTCDay()]} ${String(d.getUTCDate()).padStart(2, "0")} ${MON[d.getUTCMonth()]} ${String(d.getUTCHours()).padStart(2, "0")}Z`;
const step = () => S.steps[S.idx] ?? 0;
const qs = (o) => new URLSearchParams(o).toString();
function status(t, cls = "") { const s = $("#status"); s.textContent = t; s.className = "status " + cls; }
const modelName = (id) => (S.models.find((m) => m.id === id) || {}).name || id;
const isCompareTimed = () => S.tab === "compare" && S.cmp.mode !== "meteogram";
const isXTimed = () => S.tab === "extremes" && S.x.product.startsWith("ens") && S.x.view === "step";
const isLive = () => S.tab === "live";
const hasTimeline = () => ["maps", "sounding", "live"].includes(S.tab) || isCompareTimed() || isXTimed();  // not "cases"
const X_MAPS = ["heat", "cold", "tmax_anom", "tmin_anom"];

function mapUrl(i = S.idx) {
  const q = { model: S.model, run: S.run, step: S.steps[i], region: S.region };
  if (S.layers) q.layers = JSON.stringify(S.layers); else q.preset = S.preset;
  return "/api/map?" + qs(q);
}
function soundingUrl(i = S.idx, kind = "png") {
  return `/api/sounding.${kind}?` + qs({ model: S.model, run: S.run, step: S.steps[i], lat: S.lat,
    lon: S.lon, parcel: S.parcel, name: S.place || "" });
}
function xsQuery() {
  const l = S.xs;
  return { lat0: l.lat0, lon0: l.lon0, lat1: l.lat1, lon1: l.lon1, kind: l.kind, a: l.a || "", b: l.b || "" };
}
function xsUrl(i = S.idx) {
  return "/api/xsection.png?" + qs({ model: S.model, run: S.run, step: S.steps[i], ...xsQuery() });
}
function meteogramUrl() {
  return "/api/meteogram.png?" + qs({ model: S.model, run: S.run, lat: S.lat, lon: S.lon,
    hours: $("#mg-hours").value, profile: $("#mg-profile").checked, name: S.place || "" });
}
function cmpUrls(i = S.cmp.idx) {
  const c = S.cmp, t = c.times[i];
  if (c.mode === "meteogram")
    return ["/api/compare/meteogram.png?" + qs({ models: c.models.join(","), lat: S.lat, lon: S.lon,
      hours: $("#cmg-hours").value, name: S.place || "" })];
  if (!t) return [];
  if (c.mode === "maps")
    return c.models.map((m) => "/api/map?" + qs({ model: m, run: t.models[m].run, step: t.models[m].step,
      region: c.region, preset: c.preset }));
  if (c.mode === "diff")
    return ["/api/compare/diff.png?" + qs({ a: c.a, b: c.b, var: c.v, level: c.level, accum: c.accum,
      valid: t.valid, region: c.dregion })];
  if (c.mode === "xsec")
    return ["/api/compare/xsection.png?" + qs({ models: c.models.join(","), valid: t.valid, ...xsQuery(),
      diff: c.xdiff && c.models.length >= 2 ? 1 : 0 })];
  return ["/api/compare/sounding.png?" + qs({ models: c.models.join(","), valid: t.valid,
    lat: S.lat, lon: S.lon, name: S.place || "" })];
}
function xUrls(i = S.x.idx) {
  const x = S.x, [d0, d1] = x.days.split("-");
  if (x.product === "point")
    return ["/api/extremes/point.png?" + qs({ lat: S.lat, lon: S.lon, name: S.place || "" })];
  if (X_MAPS.includes(x.product))
    return ["/api/extremes/map.png?" + qs({ kind: x.product, model: S.model, region: x.region, d0, d1 })];
  const q = { param: (x.product === "ens_warm" ? "p_gt" : "p_lt") + x.sigma, region: x.region, d0, d1 };
  if (x.view === "step") { if (!x.steps.length) return []; q.step = x.steps[i]; }
  return ["/api/extremes/ens.png?" + qs(q)];
}
function caseUrl() {
  const c = S.cs;
  if (!c.eid || c.product === "systems") return [];
  const ms = c.models.join(",");
  const u = {
    track: ["track.png", { models: ms, fc: c.fc, region: c.region }],
    pressure: ["pressure.png", { models: ms, fc: c.fc }],
    sequence: ["sequence.png", { model: c.model, every: c.every, region: c.region }],
    precip: ["totals.png", { model: c.model, kind: "precip", region: c.region }],
    gust: ["totals.png", { model: c.model, kind: "gust", region: c.region }],
    point: ["point.png", { models: ms, lat: S.lat, lon: S.lon, name: S.place || "" }],
    sounding: ["sounding.png", c.station ? { time: c.time || "", station: c.station, models: ms }
      : { time: c.time || "", lat: S.lat, lon: S.lon, name: S.place || "", models: ms }],
    meteogram: ["meteogram.png", { model: c.model, lat: S.lat, lon: S.lon, name: S.place || "" }],
    xsection: ["xsection.png", { model: c.model, time: c.xtime || "", ...xsQuery() }],
    atcf: ["atcf_tracks.png", { base: c.atcfBase || "all" }],
    atcf_lf: ["atcf_landfall.png", {}],
  }[c.product];
  if (c.product === "xsection" && !c.xtime) return [];
  if (c.product === "sounding" && !c.time) return [];
  return [`/api/cases/${c.eid}/${u[0]}?` + qs(u[1])];
}
function urlsAt(i) {
  if (isLive()) return [];
  if (S.tab === "cases") return caseUrl();
  if (S.tab === "extremes") return xUrls(i);
  if (S.tab === "maps") return [mapUrl(i)];
  if (S.tab === "sounding") return [S.sndMode === "xsec" ? xsUrl(i) : soundingUrl(i)];
  if (S.tab === "meteogram") return [meteogramUrl()];
  return cmpUrls(i);
}

async function fetchFrame(url, job) {
  if (frames.has(url)) return frames.get(url);
  const r = await fetch(job ? `${url}&job=${job}` : url);
  if (!r.ok) {
    const err = new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    err.status = r.status;
    throw err;
  }
  const blob = await r.blob();
  const axes = r.headers.get("X-Map-Axes");
  const f = { src: URL.createObjectURL(blob), axes: axes ? axes.split(",").map(Number) : null };
  frames.set(url, f);
  return f;
}

// ------------------------------------------------------------------ progress (slow first loads)
function watchProgress(job, token) {
  const bar = $("#pbar"), fill = $("#pfill");
  const timer = setInterval(async () => {
    if (token !== loadToken) return clearInterval(timer);
    try {
      const p = await api(`/api/progress?job=${job}`);
      if (!p.total) return;
      bar.classList.remove("hidden");
      fill.style.width = (100 * p.done / p.total) + "%";
      const mm = p.models ? ` (modelo ${(p.model_i || 0) + 1}/${p.models})` : "";
      $("#loadmsg").textContent = `${p.model}${mm}: ${p.done}/${p.total} passos`;
    } catch { /* ignore */ }
  }, 700);
  return () => { clearInterval(timer); bar.classList.add("hidden"); fill.style.width = "0"; };
}

// ------------------------------------------------------------------ rendering
async function show() {
  const token = ++loadToken;
  if (isLive()) {
    updateTime();
    try { await live.render(live.idx()); $("#error").classList.add("hidden"); } catch (e) {
      $("#error").textContent = e.message; $("#error").classList.remove("hidden");
    }
    return;
  }
  if (S.tab !== "compare" && (!S.run || !S.steps.length)) return;
  const urls = urlsAt(curIdx());
  updateTime();
  $("#error").classList.add("hidden");
  if (!urls.length) return;
  const slow = S.tab === "meteogram" || (S.tab === "compare" && ["meteogram", "xsec"].includes(S.cmp.mode)) ||
    (S.tab === "cases" && ["pressure", "sequence", "point", "meteogram"].includes(S.cs.product)) ||
    (S.tab === "extremes" && !S.x.product.startsWith("ens"));
  const job = slow ? Math.random().toString(36).slice(2) : null;
  const cached = urls.every((u) => frames.has(u));
  let stopWatch = () => {};
  if (!cached) {
    $("#loader").classList.remove("hidden");
    $("#loadmsg").textContent = { maps: "A gerar mapa…", sounding: S.sndMode === "xsec" ? "A calcular o corte vertical…" : "A calcular sondagem…",
      meteogram: "A preparar meteograma…", compare: S.cmp.mode === "xsec" ? "A calcular os cortes de cada modelo…" : "A comparar modelos…",
      extremes: "A calcular Tmáx/Tmín diárias…", cases: "A analisar o caso a partir do arquivo…" }[S.tab];
    if (job) stopWatch = watchProgress(job, token);
  }
  try {
    const fs = await Promise.all(urls.map((u) => fetchFrame(u, job)));
    if (token !== loadToken) return;
    const grid = S.tab === "compare" && S.cmp.mode === "maps";
    $("#img").classList.toggle("hidden", grid);
    $("#grid").classList.toggle("hidden", !grid);
    if (grid) renderGrid(fs);
    else { $("#img").src = fs[0].src; S.axes = fs[0].axes; }
    $("#canvas").classList.toggle("tall", slow && !(S.tab === "extremes" && S.x.product !== "point"));
    $("#canvas").classList.toggle("clickable", S.tab === "maps" ||
      (S.tab === "cases" && ["track", "precip", "gust"].includes(S.cs.product)) || (S.tab === "compare" && ["maps", "diff"].includes(S.cmp.mode)) ||
      (S.tab === "extremes" && S.x.product !== "point"));
    markCached();
    if (S.tab === "sounding" && S.sndMode === "point") loadSummary();
    $("#crosshair").classList.add("hidden");
  } catch (e) {
    if (token !== loadToken) return;
    if (e.status === 503) {          // climatology being downloaded: wait and retry
      $("#loader").classList.remove("hidden");
      $("#loadmsg").textContent = e.message;
      setTimeout(() => { if (token === loadToken) show(); }, 4000);
      return;
    }
    $("#error").textContent = e.message;
    $("#error").classList.remove("hidden");
  } finally {
    stopWatch();
    if (token === loadToken) $("#loader").classList.add("hidden");
  }
  if (!S.playing && hasTimeline() && !isLive()) prefetch(token);
}

function renderGrid(fs) {
  const g = $("#grid");
  g.style.setProperty("--n", fs.length);
  g.innerHTML = "";
  fs.forEach((f, k) => {
    const m = S.cmp.models[k];
    const cell = document.createElement("figure");
    cell.innerHTML = `<figcaption style="--c:${MODEL_COLORS[m]}">${modelName(m)}</figcaption><img src="${f.src}" data-model="${m}">`;
    S.cmp.axes[m] = f.axes;
    g.appendChild(cell);
  });
}

async function prefetch(token) {
  for (let k = 1; k <= 2; k++) {
    const i = curIdx() + k;
    if (i >= tlLength() || token !== loadToken) return;
    try { await Promise.all(urlsAt(i).map((u) => fetchFrame(u))); markCached(); } catch { return; }
  }
}

// ------------------------------------------------------------------ timeline (model steps or common valid times)
function curIdx() { if (isLive()) return live.idx(); return isXTimed() ? S.x.idx : isCompareTimed() ? S.cmp.idx : S.idx; }
function setIdx(i) { if (isLive()) live.setIdx(i); else if (isXTimed()) S.x.idx = i; else if (isCompareTimed()) S.cmp.idx = i; else S.idx = i; }
function tlLength() { if (isLive()) return live.frames().length; return isXTimed() ? S.x.steps.length : isCompareTimed() ? S.cmp.times.length : S.steps.length; }
function tlValid(i) {
  if (isLive()) return live.frames()[i];
  if (isXTimed()) return new Date(parseRun(S.x.run).getTime() + S.x.steps[i] * 3600e3);
  if (isCompareTimed()) return parseRun(S.cmp.times[i].valid);
  return new Date(parseRun(S.run).getTime() + S.steps[i] * 3600e3);
}

function updateTime() {
  $("#timeline").classList.toggle("hidden", !hasTimeline());
  if (!hasTimeline() || !tlLength()) return;
  const i = curIdx();
  $("#slider").value = i;
  if (isLive()) {
    const t = tlValid(i);
    $("#tstep").textContent = `${String(t.getUTCHours()).padStart(2, "0")}:${String(t.getUTCMinutes()).padStart(2, "0")} UTC`;
    const ago = Math.round((Date.now() - t.getTime()) / 60e3);
    $("#tvalid").textContent = ago > 2880 ? `${fmt(t).slice(0, 10)} · evento arquivado`
      : `${fmt(t).slice(0, 10)} · há ${ago < 120 ? ago + " min" : Math.round(ago / 60) + " h"}`;
  } else if (isXTimed()) {
    $("#tstep").textContent = `+${S.x.steps[i]} h`;
    $("#tvalid").textContent = "ENS · válido " + fmt(tlValid(i));
  } else if (isCompareTimed()) {
    const t = S.cmp.times[i];
    $("#tstep").textContent = fmt(parseRun(t.valid));
    $("#tvalid").textContent = S.cmp.models.map((m) => `${modelName(m).replace("ECMWF ", "")} +${t.models[m].step}h`).join(" · ");
  } else {
    $("#tstep").textContent = `+${step()} h`;
    $("#tvalid").textContent = "Válido " + fmt(tlValid(i));
  }
}

function buildTicks() {
  const t = $("#ticks"); t.innerHTML = "";
  updateTime();
  if (!hasTimeline()) return;
  const n = tlLength() - 1;
  $("#slider").max = Math.max(0, n);
  if (n <= 0) return;
  const days = (tlValid(n) - tlValid(0)) / 864e5;
  const every = Math.max(1, Math.ceil(days / Math.max(4, t.clientWidth / 55)));
  let k = 0;
  const hourly = isLive();
  for (let i = 0; i <= n; i++) {
    const d = tlValid(i);
    const longLive = hourly && (tlValid(n) - tlValid(0)) > 12 * 3600e3;   // archived event
    if (hourly ? (d.getUTCMinutes() === 0 && (!longLive || d.getUTCHours() % 6 === 0))
      : (d.getUTCHours() === 0 && k++ % every === 0)) {
      const el = document.createElement("span");
      el.style.left = (100 * i / n) + "%";
      el.textContent = hourly ? (longLive ? `${d.getUTCDate()} ${String(d.getUTCHours()).padStart(2, "0")}h`
        : `${String(d.getUTCHours()).padStart(2, "0")}h`) : `${WEEK[d.getUTCDay()]} ${d.getUTCDate()}`;
      el.dataset.i = i;
      t.appendChild(el);
    }
  }
  markCached();
}
function markCached() {
  if (isLive()) return;
  $$("#ticks span").forEach((el) => el.classList.toggle("cached", urlsAt(+el.dataset.i).every((u) => frames.has(u))));
}

// ------------------------------------------------------------------ model / run
async function selectModel(id, keepValid = true) {
  const prevValid = S.run && S.steps.length ? new Date(parseRun(S.run).getTime() + step() * 3600e3) : null;
  S.model = id;
  $$("#models button").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
  status("A procurar runs…");
  try {
    S.runs = await api(`/api/models/${id}/runs`);
  } catch (e) { status("Erro: " + e.message, "err"); return; }
  if (!S.runs.length) { status("Sem runs disponíveis", "err"); return; }
  $("#run").innerHTML = S.runs.map((r) => `<option value="${r.run}">${r.label}</option>`).join("");
  S.catalog = await api(`/api/catalog?model=${id}`);
  renderPresets(); fillBuilder();
  await selectRun(S.runs[0].run, keepValid ? prevValid : null);
}

async function selectRun(run, keepValid) {
  S.run = run;
  $("#run").value = run;
  const r = S.runs.find((x) => x.run === run);
  S.steps = r && r.steps ? r.steps : await api(`/api/models/${S.model}/runs/${run}/steps`);
  const runT = parseRun(run).getTime();
  let idx = 0;
  if (keepValid) {
    const target = (keepValid.getTime() - runT) / 3600e3;
    let best = Infinity;
    S.steps.forEach((s, i) => { const d = Math.abs(s - target); if (d < best) { best = d; idx = i; } });
  } else {
    idx = Math.max(0, S.steps.findIndex((s) => s >= 24));
  }
  S.idx = idx;
  buildTicks();
  status(`${S.steps.length} passos · até +${S.steps[S.steps.length - 1]} h`, "ok");
  show();
}

// ------------------------------------------------------------------ presets & builder
function renderPresets() {
  $("#presets").innerHTML = S.catalog.presets
    .map((p) => `<button data-id="${p.id}" class="${!S.layers && p.id === S.preset ? "on" : ""}">${p.name}</button>`).join("");
  $$("#presets button").forEach((b) => b.onclick = () => {
    S.preset = b.dataset.id; S.layers = null;
    $$("#presets button").forEach((x) => x.classList.toggle("on", x === b));
    loadIntoBuilder(S.catalog.presets.find((p) => p.id === S.preset).layers);
    show();
  });
  $("#cmp-preset").innerHTML = S.catalog.presets.map((p) => `<option value="${p.id}">${p.name}</option>`).join("");
  $("#cmp-preset").value = S.cmp.preset;
}

function varOptions(withNone) {
  const groups = {};
  S.catalog.variables.forEach((v) => (groups[v.group] ??= []).push(v));
  let html = withNone ? `<option value="">— nenhum —</option>` : "";
  for (const [g, vs] of Object.entries(groups))
    html += `<optgroup label="${g}">` + vs.map((v) => `<option value="${v.id}">${v.name} (${v.units})</option>`).join("") + "</optgroup>";
  return html;
}
const ACCUM_OPTS = [1, 3, 6, 12, 24, 0].map((h) => `<option value="${h}">${h ? h + " h" : "total"}</option>`).join("");

function fillBuilder() {
  $$(".builder .layer").forEach((el) => {
    const kind = el.dataset.kind;
    if (kind === "wind") {
      $(".wlevel", el).innerHTML = S.catalog.wind_levels.map((l) => `<option value="${l}">${l ? l + " hPa" : "10 m"}</option>`).join("");
      return;
    }
    $(".var", el).innerHTML = varOptions(kind !== "fill");
    $(".var", el).onchange = () => syncLayer(el);
    const acc = $(".accum", el);
    if (acc) acc.innerHTML = ACCUM_OPTS;
    syncLayer(el);
  });
  loadIntoBuilder(S.layers || S.catalog.presets.find((p) => p.id === S.preset)?.layers || []);
}

function levelsFor(varSel, levSel, accSel, catalog = S.catalog) {
  const v = catalog.variables.find((x) => x.id === varSel.value);
  const prev = levSel.value;
  if (v && v.upper) {
    levSel.disabled = false;
    levSel.innerHTML = v.levels.map((l) => `<option value="${l}">${l} hPa</option>`).join("");
    levSel.value = v.levels.includes(+prev) ? prev : (v.levels.includes(500) ? 500 : v.levels[0]);
  } else { levSel.disabled = true; levSel.innerHTML = `<option value="0">—</option>`; }
  if (accSel) accSel.style.display = v && v.id === "precip" ? "" : "none";
}
function syncLayer(el) { levelsFor($(".var", el), $(".level", el), $(".accum", el)); }

function loadIntoBuilder(layers) {
  const fills = layers.filter((l) => l.type === "fill");
  const conts = layers.filter((l) => l.type === "contour");
  const wind = layers.find((l) => l.type === "wind");
  const set = (kind, ly) => {
    const el = $(`.layer[data-kind="${kind}"]`);
    $(".var", el).value = ly ? ly.var : "";
    syncLayer(el);
    if (ly && ly.level) $(".level", el).value = ly.level;
    if (ly && $(".accum", el)) $(".accum", el).value = ly.accum ?? 3;
    if ($(".interval", el)) $(".interval", el).value = ly && ly.interval ? ly.interval : "";
  };
  let base = fills[0], over = fills[1];
  if (fills.length === 1 && ["precip", "refc"].includes(fills[0].var)) { base = null; over = fills[0]; }
  set("fill", base); set("fill2", over); set("contour", conts[0]); set("contour2", conts[1]);
  const wel = $('.layer[data-kind="wind"]');
  $(".wlevel", wel).value = wind ? wind.level : 0;
  $(".wstyle", wel).value = wind ? wind.style : "";
}

function readBuilder() {
  const layers = [];
  for (const kind of ["fill", "fill2", "contour", "contour2"]) {
    const el = $(`.layer[data-kind="${kind}"]`);
    const v = $(".var", el).value;
    if (!v) continue;
    const ly = { type: kind.startsWith("fill") ? "fill" : "contour", var: v, level: +$(".level", el).value || 0 };
    if (v === "precip") ly.accum = +$(".accum", el).value;
    const iv = $(".interval", el);
    if (iv && iv.value) ly.interval = +iv.value;
    if (kind === "contour2") ly.color = "#c1121f";
    layers.push(ly);
  }
  const wel = $('.layer[data-kind="wind"]');
  if ($(".wstyle", wel).value) layers.push({ type: "wind", level: +$(".wlevel", wel).value, style: $(".wstyle", wel).value });
  return layers;
}

// ------------------------------------------------------------------ places (shared by several panels)
function initPlaces() {
  $$(".place-box").forEach((box) => {
    box.appendChild($("#place-tpl").content.cloneNode(true));
    $(".place", box).innerHTML = `<option value="">— coordenadas —</option>` + PLACES.map((p) => `<option>${p[0]}</option>`).join("");
    $(".place", box).onchange = (e) => {
      const p = PLACES.find((x) => x[0] === e.target.value);
      if (!p) return;
      [S.place, S.lat, S.lon] = p;
      syncPlaces(); show();
    };
  });
  $$(".go").forEach((b) => b.onclick = () => {
    const box = b.closest("section").querySelector(".place-box");
    S.lat = +$(".lat", box).value; S.lon = +$(".lon", box).value;
    const p = PLACES.find((x) => x[1] === S.lat && x[2] === S.lon);
    S.place = p ? p[0] : "";
    syncPlaces(); show();
  });
  syncPlaces();
}
function syncPlaces() {
  $$(".place-box").forEach((box) => {
    $(".place", box).value = S.place || "";
    $(".lat", box).value = S.lat; $(".lon", box).value = S.lon;
  });
}

// ------------------------------------------------------------------ map click -> sounding
function clickToFrac(e, axes) {
  const r = e.target.getBoundingClientRect();
  const px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
  const [l, t, w, h] = axes;
  const fx = (px - l) / w, fy = (py - t) / h;
  return fx < 0 || fx > 1 || fy < 0 || fy > 1 ? null : [fx, fy];
}
async function pickPoint(e, axes, region) {
  if (!axes) return false;
  const f = clickToFrac(e, axes);
  if (!f) return false;
  const loc = await api(`/api/locate?region=${region}&fx=${f[0]}&fy=${f[1]}`);
  S.lat = loc.lat; S.lon = loc.lon; S.place = "";
  syncPlaces();
  return true;
}

async function locateClick(e, axes, region) {
  if (!axes) return null;
  const f = clickToFrac(e, axes);
  if (!f) return null;
  return api(`/api/locate?region=${region}&fx=${f[0]}&fy=${f[1]}`);
}
async function xsMapClick(e) {
  const loc = await locateClick(e, S.axes, S.region);
  if (!loc) return;
  const cv = $("#canvas").getBoundingClientRect(), ch = $("#crosshair");
  if (!S.xsPending) {
    S.xsPending = loc;
    ch.style.left = (e.clientX - cv.left + $("#canvas").scrollLeft) + "px";
    ch.style.top = (e.clientY - cv.top + $("#canvas").scrollTop) + "px";
    ch.classList.remove("hidden");
    status(`Ponto A: ${loc.lat.toFixed(2)}, ${loc.lon.toFixed(2)} — clique agora no ponto B`, "ok");
    return;
  }
  Object.assign(S.xs, { lat0: S.xsPending.lat, lon0: S.xsPending.lon, lat1: loc.lat, lon1: loc.lon,
    a: "", b: "", preset: "" });
  S.xsPending = null;
  ch.classList.add("hidden");
  S.sndMode = "xsec";
  await setTab("sounding");
}

$("#img").addEventListener("click", async (e) => {
  if (S.tab === "maps" && S.clickMode === "xsec") { await xsMapClick(e); return; }
  if (S.tab === "cases" && ["track", "precip", "gust"].includes(S.cs.product)) {
    if (await pickPoint(e, S.axes, S.cs.region)) setCaseProduct("point");
    return;
  }
  if (S.tab === "extremes" && S.x.product !== "point" && await pickPoint(e, S.axes, S.x.region)) {
    setXProduct("point");
    return;
  }
  if (S.tab === "maps" && await pickPoint(e, S.axes, S.region)) setTab("sounding");
  else if (S.tab === "compare" && S.cmp.mode === "diff" && await pickPoint(e, S.axes, S.cmp.dregion)) {
    S.cmp.models = [S.cmp.a, S.cmp.b]; setCmpMode("sounding");
  }
});
$("#grid").addEventListener("click", async (e) => {
  if (e.target.tagName !== "IMG") return;
  if (await pickPoint(e, S.cmp.axes[e.target.dataset.model], S.cmp.region)) setCmpMode("sounding");
});

// ------------------------------------------------------------------ sounding summary
async function loadSummary() {
  const box = $("#snd-summary");
  try {
    const d = await api(soundingUrl(S.idx, "json"));
    const p = d.parcels[S.parcel] || {};
    const k = d.kinematics, ix = d.indices;
    const row = (a, b, hi) => `<tr><td>${a}</td><td class="${hi ? "hi" : ""}">${b ?? "—"}</td></tr>`;
    const r0 = (x) => (x == null ? "—" : Math.round(x));
    const r1 = (x) => (x == null ? "—" : x.toFixed(1));
    box.innerHTML = `<h3>Resumo (${S.parcel})</h3><table>
      ${row("CAPE", r0(p.cape) + " J/kg", p.cape > 500)}
      ${row("CIN", r0(p.cin) + " J/kg")}
      ${row("LI", r1(p.li), p.li < -2)}
      ${row("LCL", r0(p.lcl_z) + " m")}
      ${row("Água precipitável", r1(ix.pw) + " mm")}
      ${row("Iso 0 °C", r0(ix.frz_msl) + " m")}
      ${row("Shear 0–6 km", r0(k.shear_0_6) + " kt", k.shear_0_6 > 35)}
      ${row("SRH 0–3 km", r0(k.srh_0_3) + " m²/s²", k.srh_0_3 > 150)}
      ${row("K / TT", r0(ix.k) + " / " + r0(ix.tt))}
      </table>`;
  } catch { box.innerHTML = ""; }
}

// ------------------------------------------------------------------ compare
async function loadCompareTimes() {
  const c = S.cmp;
  const prev = c.times[c.idx] ? c.times[c.idx].valid : null;
  status("A alinhar horas válidas…");
  try {
    c.times = await api(`/api/compare/times?models=${c.models.join(",")}`);
  } catch (e) { status("Erro: " + e.message, "err"); c.times = []; return; }
  let idx = c.times.findIndex((t) => t.valid === prev);
  if (idx < 0) idx = Math.min(c.times.length - 1, 4);
  c.idx = Math.max(0, idx);
  status(`${c.times.length} horas válidas comuns`, "ok");
  buildTicks();
}
function renderCmpModels() {
  $("#cmp-models").innerHTML = S.models.map((m) => `<label class="check" style="--c:${MODEL_COLORS[m.id]}">
    <input type="checkbox" value="${m.id}" ${S.cmp.models.includes(m.id) ? "checked" : ""}><span class="sw"></span>${m.name}</label>`).join("");
  $$("#cmp-models input").forEach((cb) => cb.onchange = async () => {
    const sel = $$("#cmp-models input").filter((x) => x.checked).map((x) => x.value);
    if (!sel.length) { cb.checked = true; return; }
    S.cmp.models = sel;
    xdiffNote();
    await loadCompareTimes(); show();
  });
}
function xdiffNote() {
  const c = S.cmp, el = $("#cmp-xdiff-note");
  if (!c.xdiff) { el.textContent = ""; return; }
  el.textContent = c.models.length >= 2
    ? `Diferença ${modelName(c.models[0])} − ${modelName(c.models[1])} (os dois primeiros modelos assinalados).`
    : "Assinale pelo menos dois modelos para ver a diferença.";
}
async function setCmpMode(mode) {
  S.cmp.mode = mode;
  stop();
  $$("#cmp-mode button").forEach((b) => b.classList.toggle("on", b.dataset.m === mode));
  $$("#side-compare .cmp-only").forEach((el) => el.classList.toggle("hidden", !el.dataset.modes.split(" ").includes(mode)));
  $("#cmp-models").closest("section").classList.toggle("no-models", mode === "diff");
  if (mode === "diff") {
    const want = [S.cmp.a, S.cmp.b];
    if (S.cmp.models.join() !== want.join()) { S.cmp.models = want; renderCmpModels(); await loadCompareTimes(); }
  }
  if (mode === "xsec") { XS.cmp.refresh(); xdiffNote(); }
  buildTicks(); show();
}
async function initDiffControls() {
  const opts = S.models.map((m) => `<option value="${m.id}">${m.name}</option>`).join("");
  $("#diff-a").innerHTML = opts; $("#diff-b").innerHTML = opts;
  $("#diff-a").value = S.cmp.a; $("#diff-b").value = S.cmp.b;
  const [g, e] = await Promise.all([api("/api/catalog?model=gfs"), api("/api/catalog?model=ecmwf")]);
  const ok = new Set(e.variables.map((v) => v.id));
  const cat = { variables: g.variables.filter((v) => ok.has(v.id)) };
  $("#diff-var").innerHTML = cat.variables.map((v) => `<option value="${v.id}">${v.name} (${v.units})</option>`).join("");
  $("#diff-var").value = S.cmp.v;
  $("#diff-accum").innerHTML = ACCUM_OPTS; $("#diff-accum").value = 6;
  const sync = () => levelsFor($("#diff-var"), $("#diff-level"), $("#diff-accum"), cat);
  $("#diff-var").onchange = sync; sync();
  $("#diff-level").value = 500;
  $("#diff-go").onclick = async () => {
    Object.assign(S.cmp, { a: $("#diff-a").value, b: $("#diff-b").value, v: $("#diff-var").value,
      level: +$("#diff-level").value || 0, accum: +$("#diff-accum").value, dregion: $("#diff-region").value });
    if (S.cmp.a === S.cmp.b) { $("#error").textContent = "Escolha dois modelos diferentes."; $("#error").classList.remove("hidden"); return; }
    S.cmp.models = [S.cmp.a, S.cmp.b]; renderCmpModels();
    await loadCompareTimes(); show();
  };
}

// ------------------------------------------------------------------ tabs
async function setTab(tab) {
  S.tab = tab;
  stop();
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  for (const t of ["maps", "sounding", "meteogram", "compare", "extremes", "live", "events", "cases"]) $(`#side-${t}`).classList.toggle("hidden", t !== tab);
  $("#canvas").classList.toggle("hidden", tab === "live" || tab === "events");
  $("#ev-panel").classList.toggle("hidden", tab !== "events");
  $("#model-bar").classList.toggle("hidden", tab === "live" || tab === "events" || tab === "cases");
  $("#timeline").classList.toggle("hidden", tab === "events");
  if (tab === "events") { live.hide(); await showEvents(); return; }
  if (tab === "cases") { live.hide(); await showCases(); return; }
  if (tab === "live") {
    status("A obter imagens de satélite…");
    try { await live.show(); status("Satélite: EUMETSAT · Radar: RainViewer", "ok"); } catch (e) { status("Erro: " + e.message, "err"); }
    buildTicks(); updateTime();
    return;
  }
  live.hide();
  $("#model-bar").classList.toggle("invisible", tab === "compare");
  $("#crosshair").classList.add("hidden");
  if (tab === "compare") {
    if (!S.cmp.times.length) await loadCompareTimes();
    setCmpMode(S.cmp.mode);
    return;
  }
  if (tab === "extremes") { setXProduct(S.x.product); return; }
  if (tab === "sounding") applySndMode();
  status(`${S.steps.length} passos · até +${S.steps[S.steps.length - 1]} h`, "ok");
  buildTicks();
  show();
}

// ------------------------------------------------------------------ heat / cold waves
async function setXProduct(p) {
  S.x.product = p;
  stop();
  $$("#x-products button").forEach((b) => b.classList.toggle("on", b.dataset.p === p));
  $$("#side-extremes .x-only").forEach((el) => el.classList.toggle("hidden", !el.dataset.p.split(" ").includes(p)));
  if (p.startsWith("ens") && !S.x.steps.length) {
    try {
      const r = await api("/api/extremes/ens/steps");
      Object.assign(S.x, { run: r.run, steps: r.steps, idx: Math.min(9, r.steps.length - 1) });
      status(`ECMWF ENS run ${r.label}`, "ok");
    } catch (e) { status("Erro: " + e.message, "err"); }
  }
  buildTicks();
  show();
}
function initExtremes() {
  $$("#x-products button").forEach((b) => b.onclick = () => setXProduct(b.dataset.p));
  $("#x-region").value = S.x.region;
  $("#x-region").onchange = (e) => { S.x.region = e.target.value; show(); };
  $("#x-days").onchange = (e) => { S.x.days = e.target.value; show(); };
  $$("#x-sigma button").forEach((b) => b.onclick = () => {
    S.x.sigma = b.dataset.s; $$("#x-sigma button").forEach((x) => x.classList.toggle("on", x === b)); show();
  });
  $$("#x-view button").forEach((b) => b.onclick = () => {
    S.x.view = b.dataset.v; $$("#x-view button").forEach((x) => x.classList.toggle("on", x === b));
    buildTicks(); show();
  });
}

// ------------------------------------------------------------------ events (archive)
const EV = { list: [], sel: null, timer: null };
const fmtMB = (b) => (b > 1e9 ? (b / 1e9).toFixed(2) + " GB" : Math.round((b || 0) / 1e6) + " MB");
const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;");

async function showEvents() {
  try { EV.list = await api("/api/events"); } catch (e) { status("Erro: " + e.message, "err"); return; }
  if (!EV.sel && EV.list.length) EV.sel = EV.list[EV.list.length - 1].id;
  $("#ev-list").innerHTML = EV.list.length ? EV.list.map((e) => `<button class="ev-card ${e.id === EV.sel ? "on" : ""}" data-id="${e.id}">
      <b>${esc(e.name)}</b><small>${e.start_label} → ${e.end_label}</small></button>`).join("")
    : `<p class="hint">Ainda não há eventos.</p>`;
  $$("#ev-list .ev-card").forEach((b) => b.onclick = () => { EV.sel = b.dataset.id; showEvents(); });
  renderEventPanel();
  clearTimeout(EV.timer);
  if (S.tab === "events") EV.timer = setTimeout(showEvents, 8000);   // live progress
}

function renderEventPanel() {
  const e = EV.list.find((x) => x.id === EV.sel);
  const box = $("#ev-panel");
  if (!e) { box.innerHTML = `<h2>Eventos</h2><p class="sub">Crie um evento na barra lateral para arquivar uma situação meteorológica.</p>`; return; }
  if (e.historical) { renderHistoricalPanel(e, box); return; }
  const rows = (e.models || []).map((m) => {
    const p = (e.progress || {})[m];
    const name = `<td style="color:${MODEL_COLORS[m] || "#ccc"}"><b>${modelName(m)}</b></td>`;
    if (!p) return `<tr>${name}<td colspan="4">à espera…</td></tr>`;
    const pct = p.total ? Math.round(100 * p.done / p.total) : 0;
    return `<tr>${name}<td><span class="ev-bar"><div style="width:${pct}%"></div></span> ${p.done}/${p.total}</td>
      <td>${p.fields} campos</td><td>${fmtMB(p.bytes)}</td>
      <td>${p.unavailable ? `${p.unavailable} já não disponíveis no servidor` : ""}</td></tr>`;
  }).join("");
  box.innerHTML = `<h2>${esc(e.name)}</h2>
    <div class="sub">${e.start_label} → ${e.end_label} · região Atlântico NE e Ibéria · previsões desde ${e.lead_days} dia(s) antes ·
      ${e.radar_frames} imagens de radar · ${fmtMB(e.size || 0)} em disco</div>
    ${e.notes ? `<div class="ev-notes">${esc(e.notes)}</div>` : ""}
    <div class="ev-actions">
      <button class="primary" data-a="maps">Abrir mapas do evento</button>
      <button data-a="live">Satélite e radar do evento</button>
      <button data-a="archive">Atualizar arquivo agora</button>
      <button class="danger" data-a="delete">Apagar evento</button>
    </div>
    <table><thead><tr><th>Modelo</th><th>Runs/passos arquivados</th><th>Campos</th><th>Tamanho</th><th></th></tr></thead>
    <tbody>${rows}</tbody></table>
    <p class="hint">O arquivo guarda a sequência de análises (todas as runs iniciadas durante o evento, com todos os níveis
    para as sondagens) e as previsões das 00Z e 12Z feitas desde ${e.lead_days} dia(s) antes. Nos separadores Mapas,
    Sondagens, Meteogramas e Comparar, as runs arquivadas aparecem na lista de runs com a indicação «arquivo».</p>`;
  $$("#ev-panel .ev-actions button").forEach((b) => b.onclick = () => eventAction(e, b.dataset.a));
}

// historical events: filled once from archives (ERA5, NOAA GFS analyses, NHC ATCF, radiosondes)
function renderHistoricalPanel(e, box) {
  const prog = e.progress || {};
  const parts = [...(e.models || []).map((m) => [m, modelName(m), MODEL_COLORS[m]]),
    ...(e.atcf ? [["atcf", `NHC ${e.atcf} (trajetórias)`, "#f59e0b"]] : []), ["radiossondas", "Radiossondas", "#ddd"]];
  const rows = parts.map(([k, name, col]) => {
    const p = prog[k];
    const det = !p ? "" : p.campos != null ? `${p.campos} campos` : p.sondagens != null ? `${p.sondagens} sondagens`
      : p.total != null ? `${p.done}/${p.total} runs/passos · ${p.fields} campos · ${fmtMB(p.bytes)}` : "";
    const st = !p ? "à espera…" : p.estado || (p.total != null ? "previsões arquivadas (AWS)" : "");
    return `<tr><td style="color:${col || "#ccc"}"><b>${esc(name)}</b></td><td>${esc(st)}</td><td>${det}</td></tr>`;
  }).join("");
  box.innerHTML = `<h2>${esc(e.name)}</h2>
    <div class="sub">${e.start_label} → ${e.end_label} · evento histórico · ${fmtMB(e.size || 0)} em disco</div>
    ${e.notes ? `<div class="ev-notes">${esc(e.notes)}</div>` : ""}
    <div class="ev-actions">
      <button class="primary" data-a="cases">Analisar no separador Casos</button>
      <button data-a="maps">Abrir mapas do evento</button>
      <button data-a="archive">Importar / completar agora</button>
      <button class="danger" data-a="delete">Apagar evento</button>
    </div>
    <table><thead><tr><th>Fonte</th><th>Estado</th><th></th></tr></thead><tbody>${rows}</tbody></table>
    <p class="hint">Evento histórico: os servidores de previsão já não têm estes dados, por isso o arquivo usa a reanálise
    ERA5 do ECMWF (horária, 0,25°, 27 níveis), as análises do GFS guardadas pela NOAA (0,5°, 6 em 6 h; desde 2021, as
    previsões completas do GFS, e desde 2023–2025 as do ECMWF IFS e do AIFS, a partir dos arquivos na AWS), as radiossondas
    (Univ. Wyoming / IGRA) e, para ciclones tropicais, a trajetória observada e as previsões do NHC (arquivo ATCF).
    Não há radar nem satélite Meteosat no arquivo.</p>`;
  $$("#ev-panel .ev-actions button").forEach((b) => b.onclick = () => eventAction(e, b.dataset.a));
}

async function eventAction(e, a) {
  if (a === "cases") {
    S.cs.eid = e.id;
    await setTab("cases");
    return;
  }
  if (a === "archive") {
    await fetch(`/api/events/${e.id}/archive`, { method: "POST" });
    status("Arquivo a atualizar em segundo plano…", "ok");
    return;
  }
  if (a === "delete") {
    if (!confirm(`Apagar o evento «${e.name}» e os dados arquivados que só ele usa? Esta ação não pode ser desfeita.`)) return;
    await fetch(`/api/events/${e.id}`, { method: "DELETE" });
    EV.sel = null;
    showEvents();
    return;
  }
  if (a === "live") {
    await setTab("live");
    await live.refresh(true);
    await live.setEvent(e.id);
    return;
  }
  // maps: first model of the event, the run started closest to the event start, region "nea"
  const model = e.models[0] || "gfs";
  S.region = "nea";
  $("#region").value = "nea";
  await selectModel(model, false);
  const t0 = new Date(e.start + "Z").getTime();
  const best = S.runs.map((r) => r.run).sort((x, y) => Math.abs(parseRun(x) - t0) - Math.abs(parseRun(y) - t0))[0];
  if (best) await selectRun(best, new Date(t0));
  await setTab("maps");
}

function initEvents() {
  $("#ev-models").innerHTML = S.models.map((m) => `<label class="check" style="--c:${MODEL_COLORS[m.id]}">
    <input type="checkbox" value="${m.id}" checked><span class="sw"></span>${m.name}</label>`).join("");
  const now = Date.now();
  const iso = (t) => new Date(t).toISOString().slice(0, 16);
  $("#ev-start").value = iso(now - 24 * 3600e3);
  $("#ev-end").value = iso(now + 24 * 3600e3);
  $("#ev-create").onclick = async () => {
    const body = {
      name: $("#ev-name").value, start: $("#ev-start").value, end: $("#ev-end").value,
      models: $$("#ev-models input").filter((x) => x.checked).map((x) => x.value),
      lead_days: +$("#ev-lead").value, notes: $("#ev-notes").value, region: $("#ev-region").value || "nea", atcf: $("#ev-atcf").value,
    };
    const r = await fetch("/api/events", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) { status("Erro: " + (d.detail || r.statusText), "err"); return; }
    EV.sel = d.id;
    $("#ev-name").value = "";
    $("#ev-notes").value = "";
    status(`Evento «${d.name}» criado — a arquivar em segundo plano`, "ok");
    showEvents();
  };
}

// ------------------------------------------------------------------ case studies
async function showCases() {
  let evs = [];
  try { evs = await api("/api/events"); } catch (e) { status("Erro: " + e.message, "err"); }
  const sel = $("#case-event");
  sel.innerHTML = evs.length ? evs.map((e) => `<option value="${e.id}">${esc(e.name)} · ${e.start_label.slice(4, 10)} → ${e.end_label.slice(4, 10)}</option>`).join("")
    : `<option value="">(sem eventos — crie um no separador Eventos)</option>`;
  if (!S.cs.eid || !evs.some((e) => e.id === S.cs.eid)) S.cs.eid = evs.length ? evs[evs.length - 1].id : null;
  sel.value = S.cs.eid || "";
  S.cs.evs = Object.fromEntries(evs.map((e) => [e.id, e]));
  applyCaseEvent();
  status(S.cs.eid ? "Análise de caso a partir do arquivo do evento" : "Sem eventos arquivados", S.cs.eid ? "ok" : "err");
  setCaseProduct(S.cs.product);
}

// event-specific defaults: its own models, NHC products only for tropical cyclones
function applyCaseEvent() {
  const ev = (S.cs.evs || {})[S.cs.eid];
  if (!ev) return;
  const avail = S.models.map((m) => m.id);
  const ms = (ev.models || []).filter((m) => avail.includes(m));
  if (ms.length) {
    S.cs.models = ms;
    if (!ms.includes(S.cs.model)) { S.cs.model = ms[0]; $("#case-model").value = S.cs.model; }
    $$("#case-models input").forEach((x) => { x.checked = ms.includes(x.value); });
  }
  if (ev.historical) { S.cs.fc = ""; $("#case-fc").value = ""; }
  if (ev.region) { S.cs.region = ev.region; $("#case-region").value = ev.region; }
  $$("#case-products .atcf-only").forEach((b) => b.classList.toggle("hidden", !ev.atcf));
  if (!ev.atcf && ["atcf", "atcf_lf"].includes(S.cs.product)) S.cs.product = "track";
  S.cs.atcfBase = "all";
}
async function loadCaseAtcf() {
  const sel = $("#case-atcf-base");
  try {
    const d = await api(`/api/cases/${S.cs.eid}/atcf.json`);
    sel.innerHTML = `<option value="all">Evolução: NHC e GFS de 12 em 12 h</option>` +
      d.bases.filter((b) => +b.base.slice(8) % 6 === 0).reverse()
        .map((b) => `<option value="${b.base}">${b.label} (−${b.lead} h) · todos os modelos</option>`).join("");
    sel.value = S.cs.atcfBase || "all";
  } catch (e) { sel.innerHTML = ""; }
}

function setCaseProduct(p) {
  S.cs.product = p;
  $$("#case-products button").forEach((b) => b.classList.toggle("on", b.dataset.p === p));
  $$("#side-cases .case-only").forEach((el) => el.classList.toggle("hidden", !el.dataset.p.split(" ").includes(p)));
  if (p === "systems") { loadSystems(); $("#img").classList.add("hidden"); return; }
  $("#img").classList.remove("hidden");
  if (p === "sounding") { loadCaseObs().then(show); return; }
  if (p === "xsection") { loadCaseXs().then(show); return; }
  if (p === "atcf") { loadCaseAtcf().then(show); return; }
  show();
}

// ------------------------------------------------------------------ vertical cross-sections
function xsKinds(box) {
  box.innerHTML = XS_KINDS.map(([id, name]) => `<button data-k="${id}" class="${id === S.xs.kind ? "on" : ""}">${name}</button>`).join("");
  box.querySelectorAll("button").forEach((b) => b.onclick = () => {
    S.xs.kind = b.dataset.k;
    $$(".xs-kinds button").forEach((x) => x.classList.toggle("on", x.dataset.k === S.xs.kind));
    show();
  });
}
function applySndMode() {
  $$("#snd-mode button").forEach((b) => b.classList.toggle("on", b.dataset.m === S.sndMode));
  $$("#side-sounding .snd-point").forEach((el) => el.classList.toggle("hidden", S.sndMode !== "point"));
  $$("#side-sounding .snd-xsec").forEach((el) => el.classList.toggle("hidden", S.sndMode !== "xsec"));
  if (S.sndMode === "xsec") XS.fc.refresh();
}
function caseCentre(name, t) {
  // analysed centre of system `name` nearest to time t (within 3 h), from the case tracks
  const per = (S.cs.tracks || {})[name] || {};
  const tr = per[S.cs.model] || per.ecmwf || Object.values(per).find((x) => x && x.length) || [];
  const tt = new Date(t + "Z").getTime();
  let best = null, bd = Infinity;
  tr.forEach((p) => { const d = Math.abs(new Date(p.time + "Z").getTime() - tt); if (d < bd) { bd = d; best = p; } });
  return best && bd <= 3 * 3600e3 ? best : null;
}
function caseXsPresets() {
  if (S.tab !== "cases" || !S.cs.tracks) return [];
  const out = [];
  const ori = [["we", "O → E"], ["sn", "S → N"], ["swne", "SO → NE"], ["nwse", "NO → SE"]];
  for (const name of Object.keys(S.cs.tracks)) {
    for (const [o, lab] of ori) {
      out.push({ id: `C:${name}:${o}`, label: `Pelo centro de ${name} (${lab}, 1600 km)`, line: () => {
        const c = S.cs.xtime && caseCentre(name, S.cs.xtime);
        if (!c) return null;
        return { ...lineThrough(c.lat, c.lon, o, 1600), a: "", b: "" };
      } });
    }
  }
  return out;
}
async function loadCaseXs() {
  const c = S.cs;
  if (!c.eid) return;
  status("A preparar o corte (tempos e trajetórias do caso)…");
  const [times, tracks] = await Promise.all([
    api(`/api/cases/${c.eid}/xtimes?model=${c.model}`).catch(() => []),
    c.tracksFor === c.eid + c.model ? Promise.resolve(c.tracks)
      : api(`/api/cases/${c.eid}/tracks.json?models=${c.model === "ecmwf" ? "ecmwf" : "ecmwf," + c.model}`).catch(() => null),
  ]);
  c.tracks = tracks; c.tracksFor = c.eid + c.model;
  c.xtimes = times;
  $("#case-xtime").innerHTML = times.map((t) => `<option value="${t.time}">${fmt(new Date(t.time + "Z"))}${t.archived ? " ✓" : ""}</option>`).join("");
  if (!c.xtime || !times.some((t) => t.time === c.xtime)) {
    // default: deepest analysed point of the first system, else the middle of the event
    const tr = tracks && Object.values(tracks)[0];
    const pts = tr ? (tr[c.model] || tr.ecmwf || []) : [];
    const deep = pts.length ? pts.reduce((a, b) => (b.p < a.p ? b : a)) : null;
    let pick = times[Math.floor(times.length / 2)];
    if (deep) {
      const dt = new Date(deep.time + "Z").getTime();
      pick = times.reduce((a, b) => (Math.abs(new Date(b.time + "Z") - dt) < Math.abs(new Date(a.time + "Z") - dt) ? b : a), pick);
    }
    c.xtime = pick ? pick.time : null;
  }
  $("#case-xtime").value = c.xtime || "";
  if (!S.xs.preset || !S.xs.preset.startsWith("C:")) {
    const first = caseXsPresets()[0];
    if (first) XS.cs.applyPreset(first.id, false);
  } else XS.cs.applyPreset(S.xs.preset, false);
  XS.cs.refresh();
  status("Corte vertical da análise arquivada", "ok");
}
function caseXtime(t) {
  S.cs.xtime = t;
  $("#case-xtime").value = t;
  if (S.xs.preset && S.xs.preset.startsWith("C:")) XS.cs.applyPreset(S.xs.preset, false);
  show();
}
function initXs() {
  XS.fc = new LineBox($("#xs-line-fc"), S.xs, () => show());
  XS.cs = new LineBox($("#xs-line-case"), S.xs, () => show(), caseXsPresets);
  XS.cmp = new LineBox($("#xs-line-cmp"), S.xs, () => show());
  xsKinds($("#xs-kinds-fc")); xsKinds($("#xs-kinds-case")); xsKinds($("#xs-kinds-cmp"));
  $("#cmp-xdiff").onchange = (e) => { S.cmp.xdiff = e.target.checked; xdiffNote(); show(); };
  $$("#snd-mode button").forEach((b) => b.onclick = () => { S.sndMode = b.dataset.m; applySndMode(); show(); });
  $$("#click-mode button").forEach((b) => b.onclick = () => {
    S.clickMode = b.dataset.m; S.xsPending = null; $("#crosshair").classList.add("hidden");
    $$("#click-mode button").forEach((x) => x.classList.toggle("on", x === b));
    status(S.clickMode === "xsec" ? "Corte vertical: clique no ponto A e depois no ponto B" : "Clique no mapa para abrir a sondagem", "ok");
  });
  $("#case-xtime").onchange = (e) => caseXtime(e.target.value);
  const stepT = (d) => {
    const ts = (S.cs.xtimes || []).map((t) => t.time);
    const i = ts.indexOf(S.cs.xtime) + d;
    if (i >= 0 && i < ts.length) caseXtime(ts[i]);
  };
  $("#case-xprev").onclick = () => stepT(-1);
  $("#case-xnext").onclick = () => stepT(1);
}

async function loadCaseObs() {
  if (!S.cs.eid) return;
  const d = await api(`/api/cases/${S.cs.eid}/obs`);
  S.cs.obs = d;
  const arch = d.archived || {};
  $("#case-station").innerHTML = `<option value="">— ponto escolhido no mapa / local —</option>` +
    d.stations.map((st) => `<option value="${st.id}">${st.name} (${st.id})${arch[st.id] ? ` · ${arch[st.id].length} arquivada(s)` : ""}</option>`).join("");
  $("#case-station").value = S.cs.station;
  const fmtT = (iso) => fmt(new Date(iso + "Z"));
  const have = new Set(arch[S.cs.station] || []);
  $("#case-time").innerHTML = d.times.map((t) => `<option value="${t}">${fmtT(t)}${have.has(t) ? " · radiossonda ✓" : ""}</option>`).join("");
  if (!S.cs.time || !d.times.includes(S.cs.time)) {
    S.cs.time = (arch[S.cs.station] || [])[0] || d.times[Math.floor(d.times.length / 2)] || null;
  }
  $("#case-time").value = S.cs.time || "";
  $("#case-obs-note").textContent = "Radiossondas: Univ. Wyoming (BUFR) e NOAA IGRA v2 (1–2 dias de atraso). "
    + "As que forem encontradas ficam guardadas no arquivo do evento.";
}

async function loadSystems() {
  if (!S.cs.eid) return;
  S.cs.systems = await api(`/api/cases/${S.cs.eid}/systems`);
  renderSystems();
}
function renderSystems() {
  const box = $("#case-systems");
  box.innerHTML = S.cs.systems.map((s, i) => `<div class="case-sys" data-i="${i}">
      <input class="n" value="${esc(s.name)}" placeholder="nome">
      <input class="la" type="number" step="0.1" value="${s.lat}" title="latitude">
      <input class="lo" type="number" step="0.1" value="${s.lon}" title="longitude">
      <input class="t" type="datetime-local" value="${String(s.time).slice(0, 16)}" title="hora UTC">
      <input class="f" type="datetime-local" value="${s.focus ? String(s.focus).slice(0, 16) : ""}"
        title="momento de referência das previsões (opcional; por omissão, o de pressão mínima)">
      <button title="remover">✕</button></div>`).join("") || `<p class="hint">Sem sistemas definidos.</p>`;
  $$("#case-systems .case-sys button").forEach((b) => b.onclick = () => {
    readSystems(); S.cs.systems.splice(+b.parentElement.dataset.i, 1); renderSystems();
  });
}
function readSystems() {
  S.cs.systems = $$("#case-systems .case-sys").map((r) => ({
    name: $(".n", r).value, lat: +$(".la", r).value, lon: +$(".lo", r).value, time: $(".t", r).value,
    focus: $(".f", r).value || undefined }));
}

function initCases() {
  const opts = S.models.map((m) => `<option value="${m.id}">${m.name}</option>`).join("");
  $("#case-model").innerHTML = opts;
  $("#case-model").value = S.cs.model;
  $("#case-fc").innerHTML = `<option value="">— nenhuma —</option>` + opts;
  $("#case-fc").value = S.cs.fc;
  $("#case-models").innerHTML = S.models.map((m) => `<label class="check" style="--c:${MODEL_COLORS[m.id]}">
    <input type="checkbox" value="${m.id}" ${S.cs.models.includes(m.id) ? "checked" : ""}><span class="sw"></span>${m.name}</label>`).join("");
  $$("#case-models input").forEach((cb) => cb.onchange = () => {
    const sel = $$("#case-models input").filter((x) => x.checked).map((x) => x.value);
    if (!sel.length) { cb.checked = true; return; }
    S.cs.models = sel; show();
  });
  $("#case-event").onchange = (e) => { S.cs.eid = e.target.value; applyCaseEvent(); setCaseProduct(S.cs.product); };
  $("#case-atcf-base").onchange = (e) => { S.cs.atcfBase = e.target.value; show(); };
  $$("#case-products button").forEach((b) => b.onclick = () => setCaseProduct(b.dataset.p));
  $("#case-fc").onchange = (e) => { S.cs.fc = e.target.value; show(); };
  $("#case-model").onchange = async (e) => {
    S.cs.model = e.target.value;
    if (S.cs.product === "xsection") await loadCaseXs();
    show();
  };
  $("#case-region").value = S.cs.region;
  $("#case-region").onchange = (e) => { S.cs.region = e.target.value; show(); };
  $("#case-every").onchange = (e) => { S.cs.every = +e.target.value; show(); };
  $("#case-station").onchange = async (e) => { S.cs.station = e.target.value; await loadCaseObs(); show(); };
  $("#case-time").onchange = (e) => { S.cs.time = e.target.value; show(); };
  $("#case-sys-add").onclick = () => {
    readSystems();
    S.cs.systems.push({ name: "Novo", lat: 40, lon: -15, time: new Date().toISOString().slice(0, 16) });
    renderSystems();
  };
  $("#case-sys-save").onclick = async () => {
    readSystems();
    const r = await fetch(`/api/cases/${S.cs.eid}/systems`, { method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(S.cs.systems) });
    const d = await r.json();
    if (!r.ok) { status("Erro: " + (d.detail || r.statusText), "err"); return; }
    S.cs.systems = d; renderSystems();
    status("Sistemas guardados — trajetórias recalculadas", "ok");
  };
}

// ------------------------------------------------------------------ animation
let timer = null;
async function tick() {
  if (!S.playing) return;
  const t0 = performance.now();
  setIdx((curIdx() + 1) % tlLength());
  await show();
  const wait = Math.max(0, +$("#speed").value - (performance.now() - t0));
  timer = setTimeout(tick, wait);
}
function play() { if (!hasTimeline()) return; S.playing = true; $("#play").textContent = "❚❚"; tick(); }
function stop() { S.playing = false; $("#play").textContent = "▶"; clearTimeout(timer); }
function stepBy(d) { stop(); setIdx(Math.min(tlLength() - 1, Math.max(0, curIdx() + d))); show(); }

// ------------------------------------------------------------------ wiring
async function init() {
  S.models = await api("/api/models");
  $("#models").innerHTML = S.models.map((m) => `<button data-id="${m.id}" title="${m.description}">${m.name}</button>`).join("");
  $$("#models button").forEach((b) => b.onclick = () => { stop(); selectModel(b.dataset.id); });
  S.regions = await api("/api/regions");
  $$(".region-select").forEach((sel) => {
    sel.innerHTML = S.regions.map((r) => `<option value="${r.id}">${r.name}</option>`).join("");
    sel.value = "europe";
  });
  $("#region").onchange = (e) => { S.region = e.target.value; show(); };
  $("#cmp-region").onchange = (e) => { S.cmp.region = e.target.value; show(); };
  $("#cmp-preset").onchange = (e) => { S.cmp.preset = e.target.value; show(); };
  $("#run").onchange = (e) => { stop(); selectRun(e.target.value, null); };
  $("#apply").onclick = () => {
    S.layers = readBuilder();
    if (!S.layers.length) return;
    $$("#presets button").forEach((x) => x.classList.remove("on"));
    show();
  };
  $("#slider").oninput = (e) => { setIdx(+e.target.value); updateTime(); };
  $("#slider").onchange = () => show();
  $("#prev").onclick = () => stepBy(-1);
  $("#next").onclick = () => stepBy(1);
  $("#play").onclick = () => (S.playing ? stop() : play());
  $$(".tab").forEach((b) => b.onclick = () => setTab(b.dataset.tab));
  $$("#cmp-mode button").forEach((b) => b.onclick = () => setCmpMode(b.dataset.m));
  $("#mg-hours").onchange = () => show();
  $("#mg-profile").onchange = () => show();
  $("#cmg-hours").onchange = () => show();
  $$("#parcel button").forEach((b) => b.onclick = () => {
    S.parcel = b.dataset.p;
    $$("#parcel button").forEach((x) => x.classList.toggle("on", x === b));
    show();
  });
  initPlaces();
  initExtremes();
  initEvents();
  initCases();
  initXs();
  live.init({
    onPoint: (lat, lon, tab) => {
      S.lat = lat; S.lon = lon; S.place = ""; syncPlaces();
      if (tab === "extremes") S.x.product = "point";
      setTab(tab);
    },
    onFramesChanged: () => { buildTicks(); updateTime(); },
  });
  renderCmpModels();
  initDiffControls();

  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT"].includes(document.activeElement.tagName)) return;
    if (e.key === "ArrowRight") stepBy(1);
    else if (e.key === "ArrowLeft") stepBy(-1);
    else if (e.key === " ") { e.preventDefault(); S.playing ? stop() : play(); }
  });
  await selectModel("gfs", false);
}
init().catch((e) => status("Erro: " + e.message, "err"));
