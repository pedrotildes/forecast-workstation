// Forecast Workstation — front-end (vanilla ES module, no build step)
import * as live from "./live.js";
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
const MODEL_COLORS = { gfs: "#3b82f6", ecmwf: "#ef4444", aifs: "#22c55e" };

const S = {
  tab: "maps", model: "gfs", models: [], runs: [], run: null, steps: [], idx: 0,
  region: "europe", preset: "sfc_precip", layers: null, catalog: null, regions: [],
  lat: 38.72, lon: -9.14, place: "Lisboa", parcel: "ML", playing: false, axes: null,
  cmp: { mode: "maps", models: ["gfs", "ecmwf", "aifs"], preset: "z500_t500", region: "europe",
    times: [], idx: 0, a: "gfs", b: "ecmwf", v: "gh", level: 500, accum: 6, dregion: "europe",
    axes: {} },
  x: { product: "point", region: "iberia", days: "1-16", sigma: "1p5", view: "sum",
    steps: [], run: null, idx: 0 },
};
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
const hasTimeline = () => ["maps", "sounding", "live"].includes(S.tab) || isCompareTimed() || isXTimed();
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
function urlsAt(i) {
  if (isLive()) return [];
  if (S.tab === "extremes") return xUrls(i);
  if (S.tab === "maps") return [mapUrl(i)];
  if (S.tab === "sounding") return [soundingUrl(i)];
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
  const slow = S.tab === "meteogram" || (S.tab === "compare" && S.cmp.mode === "meteogram") ||
    (S.tab === "extremes" && !S.x.product.startsWith("ens"));
  const job = slow ? Math.random().toString(36).slice(2) : null;
  const cached = urls.every((u) => frames.has(u));
  let stopWatch = () => {};
  if (!cached) {
    $("#loader").classList.remove("hidden");
    $("#loadmsg").textContent = { maps: "A gerar mapa…", sounding: "A calcular sondagem…",
      meteogram: "A preparar meteograma…", compare: "A comparar modelos…",
      extremes: "A calcular Tmáx/Tmín diárias…" }[S.tab];
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
    $("#canvas").classList.toggle("clickable", S.tab === "maps" || (S.tab === "compare" && ["maps", "diff"].includes(S.cmp.mode)) ||
      (S.tab === "extremes" && S.x.product !== "point"));
    markCached();
    if (S.tab === "sounding") loadSummary();
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
    $("#tvalid").textContent = `${fmt(t).slice(0, 10)} · há ${ago < 120 ? ago + " min" : Math.round(ago / 60) + " h"}`;
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
    if (hourly ? d.getUTCMinutes() === 0 : (d.getUTCHours() === 0 && k++ % every === 0)) {
      const el = document.createElement("span");
      el.style.left = (100 * i / n) + "%";
      el.textContent = hourly ? `${String(d.getUTCHours()).padStart(2, "0")}h` : `${WEEK[d.getUTCDay()]} ${d.getUTCDate()}`;
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

$("#img").addEventListener("click", async (e) => {
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
    await loadCompareTimes(); show();
  });
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
  for (const t of ["maps", "sounding", "meteogram", "compare", "extremes", "live"]) $(`#side-${t}`).classList.toggle("hidden", t !== tab);
  $("#canvas").classList.toggle("hidden", tab === "live");
  $("#model-bar").classList.toggle("hidden", tab === "live");
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
