// Vertical cross-sections — line editor (preset list, A/B coordinates, mini map with
// draggable end points) shared by the Sondagens and Casos panels.
const $ = (s, el = document) => el.querySelector(s);

export const KINDS = [
  ["dyn", "Vento, θ e tropopausa (2 PVU)"],
  ["normal", "Vento normal ao corte e θ"],
  ["moist", "θe, humidade e subida (ω)"],
  ["omega", "Movimento vertical (ω)"],
  ["pv", "Vorticidade potencial"],
  ["temp", "Temperatura e 0 °C"],
];

// name, latA, lonA, latB, lonB, nameA, nameB
export const LINES = [
  ["Madeira → Lisboa", 32.65, -16.91, 38.72, -9.14, "Funchal", "Lisboa"],
  ["Açores → Lisboa", 37.74, -25.67, 38.72, -9.14, "Ponta Delgada", "Lisboa"],
  ["Lisboa → Madrid", 38.72, -9.14, 40.42, -3.70, "Lisboa", "Madrid"],
  ["Corunha → Gibraltar (N–S)", 43.36, -8.41, 36.14, -5.35, "A Corunha", "Gibraltar"],
  ["Porto → Faro (N–S)", 41.15, -8.61, 37.02, -7.93, "Porto", "Faro"],
  ["Canárias → Bordéus", 28.32, -16.38, 44.84, -0.58, "Tenerife", "Bordéus"],
  ["Paralelo 39°N (30°W → 0°)", 39, -30, 39, 0, "", ""],
  ["Meridiano 10°W (30°N → 46°N)", 30, -10, 46, -10, "", ""],
  ["Atlântico → Península (40°N, 25°W → 5°W)", 40, -25, 40, -5, "", ""],
];

const R = 6371;
const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;
const xyz = (la, lo) => [Math.cos(rad(la)) * Math.cos(rad(lo)), Math.cos(rad(la)) * Math.sin(rad(lo)), Math.sin(rad(la))];

export function lengthKm(l) {
  const a = xyz(l.lat0, l.lon0), b = xyz(l.lat1, l.lon1);
  return Math.acos(Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))) * R;
}

export function greatCircle(l, n = 48) {
  const a = xyz(l.lat0, l.lon0), b = xyz(l.lat1, l.lon1);
  const ang = Math.acos(Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));
  if (ang < 1e-9) return [[l.lat0, l.lon0], [l.lat1, l.lon1]];
  const out = [];
  for (let i = 0; i <= n; i++) {
    const f = i / n, s0 = Math.sin((1 - f) * ang) / Math.sin(ang), s1 = Math.sin(f * ang) / Math.sin(ang);
    const p = [0, 1, 2].map((k) => s0 * a[k] + s1 * b[k]);
    out.push([deg(Math.asin(p[2])), deg(Math.atan2(p[1], p[0]))]);
  }
  return out;
}

// destination point from (lat, lon) along bearing (deg) for km
export function dest(lat, lon, brg, km) {
  const d = km / R, b = rad(brg), p1 = rad(lat), l1 = rad(lon);
  const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(b));
  const l2 = l1 + Math.atan2(Math.sin(b) * Math.sin(d) * Math.cos(p1), Math.cos(d) - Math.sin(p1) * Math.sin(p2));
  return [+deg(p2).toFixed(2), +deg(l2).toFixed(2)];
}

// line through a centre: orientation "we", "sn", "swne", "nwse"; total length km
export function lineThrough(lat, lon, orient, km = 1600) {
  const brg = { we: 90, sn: 0, swne: 45, nwse: 135 }[orient];
  const [la0, lo0] = dest(lat, lon, brg + 180, km / 2);
  const [la1, lo1] = dest(lat, lon, brg, km / 2);
  return { lat0: la0, lon0: lo0, lat1: la1, lon1: lo1 };
}

const fmtLL = (la, lo) => `${Math.abs(la).toFixed(1)}°${la >= 0 ? "N" : "S"} ${Math.abs(lo).toFixed(1)}°${lo >= 0 ? "E" : "W"}`;

export class LineBox {
  /** el: container; line: shared state object; onChange(): redraw;
   *  extra(): additional presets [{id, label, line(): {lat0, lon0, lat1, lon1, a, b}}] */
  constructor(el, line, onChange, extra = () => []) {
    this.el = el; this.line = line; this.onChange = onChange; this.extra = extra;
    this.next = "A"; this.map = null;
    el.appendChild($("#line-tpl").content.cloneNode(true));
    this.sel = $(".line-preset", el);
    this.inp = ["la0", "lo0", "la1", "lo1"].map((c) => $("." + c, el));
    this.info = $(".line-info", el);
    this.sel.onchange = () => { if (this.sel.value) this.applyPreset(this.sel.value); };
    this.inp.forEach((i) => i.onchange = () => this.readInputs());
    $(".xs-go", el).onclick = () => { this.readInputs(false); this.onChange(); };
    $(".xs-swap", el).onclick = () => {
      const l = this.line;
      [l.lat0, l.lon0, l.lat1, l.lon1, l.a, l.b] = [l.lat1, l.lon1, l.lat0, l.lon0, l.b, l.a];
      l.preset = "";
      this.refresh(); this.onChange();
    };
  }

  presets() {
    const base = LINES.map((p) => ({ id: "L:" + p[0], label: p[0],
      line: () => ({ lat0: p[1], lon0: p[2], lat1: p[3], lon1: p[4], a: p[5], b: p[6] }) }));
    return [...this.extra(), ...base];
  }

  applyPreset(id, redraw = true) {
    const p = this.presets().find((x) => x.id === id);
    if (!p) return false;
    const l = p.line();
    if (!l) return false;
    Object.assign(this.line, { a: "", b: "" }, l, { preset: id });
    this.refresh();
    if (redraw) this.onChange();
    return true;
  }

  readInputs(redraw = true) {
    const [a, b, c, d] = this.inp.map((i) => +i.value);
    if ([a, b, c, d].some((x) => !isFinite(x))) return;
    const l = this.line;
    if (a === l.lat0 && b === l.lon0 && c === l.lat1 && d === l.lon1) return;
    if (a !== l.lat0 || b !== l.lon0) l.a = "";
    if (c !== l.lat1 || d !== l.lon1) l.b = "";
    Object.assign(l, { lat0: a, lon0: b, lat1: c, lon1: d, preset: "" });
    this.refresh();
    if (redraw) this.onChange();
  }

  setPoint(which, lat, lon) {
    const l = this.line;
    if (which === "A") Object.assign(l, { lat0: +lat.toFixed(2), lon0: +lon.toFixed(2), a: "" });
    else Object.assign(l, { lat1: +lat.toFixed(2), lon1: +lon.toFixed(2), b: "" });
    l.preset = "";
    this.refresh();
    this.onChange();
  }

  ensureMap() {
    if (this.map || !this.el.offsetParent) return;
    const m = L.map($(".line-map", this.el), { zoomControl: true, scrollWheelZoom: false,
      attributionControl: true, minZoom: 2, maxZoom: 9 });
    m.attributionControl.setPrefix(false);
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 9, attribution: "Fundo © Esri" }).addTo(m);
    const icon = (t) => L.divIcon({ className: "xs-pin", html: t, iconSize: [22, 22], iconAnchor: [11, 11] });
    this.poly = L.polyline([], { color: "#111", weight: 4, opacity: 0.85 }).addTo(m);
    this.poly2 = L.polyline([], { color: "#facc15", weight: 2 }).addTo(m);
    this.mA = L.marker([0, 0], { draggable: true, icon: icon("A") }).addTo(m);
    this.mB = L.marker([0, 0], { draggable: true, icon: icon("B") }).addTo(m);
    this.mA.on("dragend", () => { const p = this.mA.getLatLng(); this.setPoint("A", p.lat, p.lng); });
    this.mB.on("dragend", () => { const p = this.mB.getLatLng(); this.setPoint("B", p.lat, p.lng); });
    m.on("click", (e) => {
      this.setPoint(this.next, e.latlng.lat, e.latlng.lng);
      this.next = this.next === "A" ? "B" : "A";
    });
    this.map = m;
    this.fit = true;
  }

  refresh() {
    const l = this.line;
    const ps = this.presets();
    this.sel.innerHTML = `<option value="">— linha personalizada —</option>` +
      ps.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
    this.sel.value = ps.some((p) => p.id === l.preset) ? l.preset : "";
    [l.lat0, l.lon0, l.lat1, l.lon1].forEach((v, k) => { this.inp[k].value = v; });
    const km = lengthKm(l);
    this.info.textContent = `${l.a || fmtLL(l.lat0, l.lon0)} → ${l.b || fmtLL(l.lat1, l.lon1)} · ${km.toFixed(0)} km`;
    this.info.classList.toggle("err", km < 150 || km > 6000);
    this.ensureMap();
    if (!this.map) return;
    this.map.invalidateSize();
    const pts = greatCircle(l);
    this.poly.setLatLngs(pts); this.poly2.setLatLngs(pts);
    this.mA.setLatLng([l.lat0, l.lon0]); this.mB.setLatLng([l.lat1, l.lon1]);
    const b = this.poly.getBounds();
    if (this.fit || !this.map.getBounds().contains(b)) {
      this.map.fitBounds(b, { padding: [18, 18], maxZoom: 6 });
      this.fit = false;
    }
  }
}
