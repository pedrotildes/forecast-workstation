// Satellite / radar live viewer (Leaflet). Imagery is requested directly from
// EUMETSAT EUMETView (WMS) and RainViewer (radar tiles).
/* global L */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const VIEWS = {
  portugal: [[36.7, -10.2], [42.3, -6.0]], iberia: [[35.5, -10.5], [44.2, 4.5]],
  europe: [[33, -25], [68, 40]], natl: [[25, -60], [65, 15]],
  azores: [[36.6, -31.6], [40.0, -24.6]], madeira: [[27.3, -18.5], [33.3, -13.2]],
};
const MAX_PX = 2048;

const L_ = {
  map: null, cfg: null, sat: "ir105", overlays: new Set(["coast", "radar", "lightning"]),
  hours: 3, frames: [], idx: 0, satOverlay: null, ovlImages: {}, radarLayers: {},
  staticLayers: {}, radarOpacity: 0.8, onPoint: null, loaded: false, lastRefresh: 0,
};

const iso = (d) => d.toISOString().replace(/\.\d{3}Z$/, "Z");
const floorTo = (d, min) => new Date(Math.floor(d.getTime() / (min * 60e3)) * min * 60e3);

function wmsUrl(layer, style, time, transparent) {
  const m = L_.map;
  const b = m.getBounds();
  const crs = m.options.crs;
  const sw = crs.project(b.getSouthWest()), ne = crs.project(b.getNorthEast());
  const size = m.getSize();
  const k = Math.min(window.devicePixelRatio || 1, 2, MAX_PX / Math.max(size.x, size.y));
  const q = new URLSearchParams({
    service: "WMS", version: "1.3.0", request: "GetMap", layers: layer, styles: style || "",
    crs: "EPSG:3857", bbox: [sw.x, sw.y, ne.x, ne.y].join(","),
    width: Math.round(size.x * k), height: Math.round(size.y * k),
    format: "image/png", transparent: transparent ? "true" : "false",
  });
  if (time) q.set("time", time);
  return `${L_.cfg.wms}?${q}`;
}

// EUMETView allows ~20 requests/s and 8 concurrent per user: keep a small pool and
// retry rejected (HTTP 429) requests with a back-off.
const preload = new Map();
let active = 0;
const waiting = [];
async function slot() {
  if (active < 3) { active++; return; }
  await new Promise((r) => waiting.push(r));
  active++;
}
function release() { active--; const n = waiting.shift(); if (n) n(); }
function loadOnce(url) {
  return new Promise((res) => {
    const im = new Image();
    im.onload = () => res(true);
    im.onerror = () => res(false);
    im.src = url;
  });
}
function loadImg(url) {
  if (preload.has(url)) return preload.get(url);
  const p = (async () => {
    for (let attempt = 0; attempt < 4; attempt++) {
      await slot();
      const ok = await loadOnce(url);
      release();
      if (ok) return true;
      await new Promise((r) => setTimeout(r, 1200 * (attempt + 1)));
    }
    preload.delete(url);
    return false;
  })();
  preload.set(url, p);
  if (preload.size > 400) preload.delete(preload.keys().next().value);
  return p;
}

// ------------------------------------------------------------------ frames
function satCfg() { return L_.cfg.sat.find((s) => s.id === L_.sat); }

function buildFrames() {
  const sc = satCfg();
  let frames = [];
  if (sc) {
    const latest = new Date(sc.latest);
    const n = Math.round((L_.hours * 60) / sc.period) + 1;
    for (let k = n - 1; k >= 0; k--) frames.push(new Date(latest.getTime() - k * sc.period * 60e3));
  } else if (L_.cfg.radar) {
    frames = L_.cfg.radar.frames.map((f) => new Date(f.time * 1000));
  }
  L_.frames = frames;
  L_.idx = frames.length - 1;
}

function radarFrameFor(t) {
  const r = L_.cfg.radar;
  if (!r) return null;
  let best = null;
  for (const f of r.frames) {
    const ft = f.time * 1000;
    if (ft <= t.getTime() + 5 * 60e3 && t.getTime() - ft <= 10 * 60e3) best = f;
  }
  return best;
}
function overlayTime(o, t) {
  const latest = new Date(o.latest);
  const ft = floorTo(t, o.period);
  return iso(ft > latest ? latest : ft);
}

// ------------------------------------------------------------------ layers
// RainViewer allows ~300 tile requests per burst: 512-px tiles (4x fewer requests) and only
// the current frame plus the next two are kept on the map; tiles are browser-cached (48 h).
function radarLayer(f) {
  if (!L_.radarLayers[f.path]) {
    L_.radarLayers[f.path] = L.tileLayer(`${L_.cfg.radar.host}${f.path}/512/{z}/{x}/{y}/4/1_1.png`, {
      pane: "radar", opacity: 0, maxNativeZoom: 7, maxZoom: 12, tileSize: 512, zoomOffset: -1,
      attribution: 'Radar © <a href="https://www.rainviewer.com" target="_blank">RainViewer</a>',
    }).addTo(L_.map);
  }
  return L_.radarLayers[f.path];
}

// coastlines / borders: one image per view (far fewer requests than WMS tiles)
function syncStatic() {
  const bounds = L_.map.getBounds();
  for (const [k, v] of Object.entries(L_.cfg.static)) {
    const on = L_.overlays.has(k);
    if (!on) {
      if (L_.staticLayers[k]) { L_.map.removeLayer(L_.staticLayers[k]); delete L_.staticLayers[k]; }
      continue;
    }
    const url = wmsUrl(v.layer, v.style, null, true);
    loadImg(url).then(() => {
      if (!L_.overlays.has(k)) return;
      if (!L_.staticLayers[k]) {
        L_.staticLayers[k] = L.imageOverlay(url, bounds, { pane: "lines", interactive: false,
          opacity: k === "admin1" ? 0.6 : 0.9 }).addTo(L_.map);
      } else { L_.staticLayers[k].setUrl(url); L_.staticLayers[k].setBounds(bounds); }
    });
  }
}

// Render frame i; resolves when its images are loaded (so animation never shows gaps).
export async function render(i = L_.idx) {
  if (!L_.map || !L_.frames.length) return;
  L_.idx = i;
  const t = L_.frames[i];
  const bounds = L_.map.getBounds();
  const jobs = [];
  const sc = satCfg();
  if (sc) {
    const url = wmsUrl(sc.layer, sc.style, iso(t), false);
    jobs.push(loadImg(url).then((ok) => {
      if (L_.idx !== i || !ok) return;
      if (!L_.satOverlay) L_.satOverlay = L.imageOverlay(url, bounds, { pane: "sat", interactive: false }).addTo(L_.map);
      else { L_.satOverlay.setUrl(url); L_.satOverlay.setBounds(bounds); }
    }));
  } else if (L_.satOverlay) { L_.map.removeLayer(L_.satOverlay); L_.satOverlay = null; }

  for (const o of L_.cfg.overlays) {
    const on = L_.overlays.has(o.id);
    if (!on) {
      if (L_.ovlImages[o.id]) { L_.map.removeLayer(L_.ovlImages[o.id]); delete L_.ovlImages[o.id]; }
      continue;
    }
    const url = wmsUrl(o.layer, o.style, overlayTime(o, t), true);
    jobs.push(loadImg(url).then((ok) => {
      if (L_.idx !== i || !ok) return;
      if (!L_.ovlImages[o.id]) {
        L_.ovlImages[o.id] = L.imageOverlay(url, bounds, { pane: o.id === "lightning" || o.id === "rdt" ? "top" : "radar", interactive: false, opacity: 0.95 }).addTo(L_.map);
      } else { L_.ovlImages[o.id].setUrl(url); L_.ovlImages[o.id].setBounds(bounds); }
    }));
  }

  const rf = L_.overlays.has("radar") ? radarFrameFor(t) : null;
  const keep = new Set();
  if (rf) {
    keep.add(rf.path);
    for (let k = 1; k <= 2; k++) {
      const nf = L_.frames[(i + k) % L_.frames.length];
      const r2 = nf && radarFrameFor(nf);
      if (r2) keep.add(r2.path);
    }
  }
  for (const [path, lay] of Object.entries(L_.radarLayers)) {
    if (!keep.has(path)) { L_.map.removeLayer(lay); delete L_.radarLayers[path]; }
  }
  for (const f of L_.cfg.radar ? L_.cfg.radar.frames : []) {
    if (keep.has(f.path)) radarLayer(f).setOpacity(rf && f.path === rf.path ? L_.radarOpacity : 0);
  }
  $("#live-radar-time").textContent = L_.overlays.has("radar")
    ? (rf ? `radar ${new Date(rf.time * 1000).toISOString().slice(11, 16)} UTC` : "radar: sem imagem para esta hora") : "";
  await Promise.all(jobs);
}

// warm the browser cache with every frame of the current view
let preloadGen = 0;
async function preloadAll() {
  const gen = ++preloadGen;
  const sc = satCfg();
  const urls = [];
  for (const t of L_.frames) {
    if (sc) urls.push(wmsUrl(sc.layer, sc.style, iso(t), false));
    for (const o of L_.cfg.overlays) if (L_.overlays.has(o.id)) urls.push(wmsUrl(o.layer, o.style, overlayTime(o, t), true));
  }
  let done = 0;
  const prog = $("#live-preload");
  const queue = [...urls];
  const worker = async () => {
    while (queue.length && gen === preloadGen) {
      await loadImg(queue.shift());
      done++;
      prog.textContent = done < urls.length ? `a pré-carregar ${done}/${urls.length}` : "";
    }
  };
  await Promise.all([worker(), worker()]);
}

// ------------------------------------------------------------------ public API
export async function refresh(force = false) {
  if (!force && L_.cfg && Date.now() - L_.lastRefresh < 120e3) return;
  const r = await fetch("/api/live/layers");
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  L_.cfg = await r.json();
  L_.lastRefresh = Date.now();
  fillControls();
  buildFrames();
}

function fillControls() {
  const groups = {};
  L_.cfg.sat.forEach((s) => (groups[s.group] ??= []).push(s));
  $("#live-sat").innerHTML = `<option value="">— nenhum (mapa base) —</option>` +
    Object.entries(groups).map(([g, ls]) => `<optgroup label="${g}">` +
      ls.map((s) => `<option value="${s.id}">${s.name}</option>`).join("") + "</optgroup>").join("");
  $("#live-sat").value = L_.sat;
  const ovl = [["radar", "Radar de precipitação (RainViewer)"], ...L_.cfg.overlays.map((o) => [o.id, o.name]),
    ["coast", "Linhas de costa e fronteiras"], ["admin1", "Distritos / regiões"]];
  $("#live-ovl").innerHTML = ovl.map(([id, n]) => `<label class="check"><input type="checkbox" value="${id}" ${L_.overlays.has(id) ? "checked" : ""}> ${n}</label>`).join("");
  $$("#live-ovl input").forEach((cb) => cb.onchange = () => {
    cb.checked ? L_.overlays.add(cb.value) : L_.overlays.delete(cb.value);
    syncStatic(); render(); preloadAll();
  });
  const sc = satCfg();
  const lat = sc ? new Date(sc.latest) : null;
  $("#live-latest").textContent = lat ? `Última imagem: ${lat.toISOString().slice(0, 16).replace("T", " ")} UTC` : "";
}

export function init({ onPoint, onFramesChanged }) {
  L_.onPoint = onPoint;
  L_.onFramesChanged = onFramesChanged;
  $("#live-sat").onchange = (e) => { L_.sat = e.target.value; buildFrames(); fillControls(); onFramesChanged(); render(); preloadAll(); };
  $("#live-hours").onchange = (e) => { L_.hours = +e.target.value; buildFrames(); onFramesChanged(); render(); preloadAll(); };
  $("#live-opacity").oninput = (e) => { L_.radarOpacity = +e.target.value / 100; render(); };
  $$("#live-views button").forEach((b) => b.onclick = () => L_.map && L_.map.fitBounds(VIEWS[b.dataset.v]));
  $("#live-refresh").onclick = async () => { await refresh(true); onFramesChanged(); render(); preloadAll(); };
}

export async function show() {
  $("#livemap").classList.remove("hidden");
  if (!L_.map) {
    const m = L.map("livemap", { zoomControl: true, minZoom: 3, maxZoom: 12, worldCopyJump: false });
    for (const [name, z] of [["base", 200], ["sat", 350], ["radar", 420], ["top", 460], ["lines", 500]]) {
      m.createPane(name).style.zIndex = z;
    }
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      pane: "base", maxZoom: 12, subdomains: "abcd",
      attribution: '© <a href="https://www.openstreetmap.org/copyright">OSM</a> © <a href="https://carto.com/">CARTO</a> · Satélite © <a href="https://view.eumetsat.int" target="_blank">EUMETSAT</a>',
    }).addTo(m);
    m.fitBounds(VIEWS.iberia);
    L_.map = m;
    let timer = null;
    m.on("moveend", () => { clearTimeout(timer); timer = setTimeout(() => { syncStatic(); render(); preloadAll(); }, 300); });
    m.on("click", (e) => {
      const { lat, lng } = e.latlng;
      const html = `<div class="live-pop"><b>${lat.toFixed(2)}°, ${lng.toFixed(2)}°</b><br>
        <button data-go="sounding">Sondagem prevista</button>
        <button data-go="meteogram">Meteograma</button>
        <button data-go="extremes">Termograma calor/frio</button></div>`;
      const pop = L.popup().setLatLng(e.latlng).setContent(html).openOn(m);
      setTimeout(() => $$(".live-pop button").forEach((b) => b.onclick = () => {
        m.closePopup(pop);
        L_.onPoint(+lat.toFixed(3), +lng.toFixed(3), b.dataset.go);
      }), 0);
    });
  }
  setTimeout(() => L_.map.invalidateSize(), 0);
  await refresh();
  syncStatic();
  await render();
  preloadAll();
}

export function hide() { $("#livemap").classList.add("hidden"); }
export const frames = () => L_.frames;
export const idx = () => L_.idx;
export function setIdx(i) { L_.idx = i; }
