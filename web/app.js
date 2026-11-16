// Torricelli - pagina di consultazione (inglese con traduzione italiana)
"use strict";
const $ = s => document.querySelector(s);
const LAB = {
  tcwv_kgm2: ["Water vapour column [kg m⁻²]", "Vapore acqueo in colonna [kg m⁻²]"], w_gcm2: ["Water vapour [g cm⁻²]", "Vapore acqueo [g cm⁻²]"],
  aod550: ["AOD 550 nm", "AOD 550 nm"], dust_ugm3: ["Dust [µg m⁻³]", "Polveri [µg m⁻³]"], ps_hpa: ["Surface pressure [hPa]", "Pressione al suolo [hPa]"],
  t2m_c: ["Air temperature 2 m [°C]", "Temperatura dell'aria 2 m [°C]"], tskin_c: ["Surface (skin) temperature [°C]", "Temperatura della superficie [°C]"],
  rh2m: ["Relative humidity [%]", "Umidità relativa [%]"], cloud_pct: ["Cloud cover [%]", "Copertura nuvolosa [%]"],
  ghi_wm2: ["Global irradiance [W m⁻²]", "Irradianza globale [W m⁻²]"], dni_wm2: ["Direct normal irradiance [W m⁻²]", "Irradianza diretta normale [W m⁻²]"],
  dif_wm2: ["Diffuse irradiance [W m⁻²]", "Irradianza diffusa [W m⁻²]"], rho_sfc_kgm3: ["Air density [kg m⁻³]", "Densità dell'aria [kg m⁻³]"],
  sza_deg: ["Solar zenith [°]", "Zenit solare [°]"], tir_l8b10_tau: ["TIR L8/9 B10 transmittance", "TIR L8/9 B10 trasmittanza"],
  tir_l8b10_lu: ["TIR L8/9 B10 upwelling radiance", "TIR L8/9 B10 radianza in salita"], tir_l8b10_ld: ["TIR L8/9 B10 downwelling radiance", "TIR L8/9 B10 radianza in discesa"],
};
let lang = "en", D = null;
try { lang = localStorage.getItem("lang") || ((navigator.language || "").startsWith("it") ? "it" : "en"); } catch { /* */ }
const L = k => LAB[k] ? LAB[k][lang === "it" ? 1 : 0] : k;
function setLang(l) {
  lang = l; try { localStorage.setItem("lang", l); } catch { /* */ }
  document.documentElement.lang = l;
  document.querySelectorAll("[data-it]").forEach(e => { if (e._en == null) e._en = e.innerHTML; e.innerHTML = l === "it" ? e.dataset.it : e._en; });
  $("#en").classList.toggle("on", l === "en"); $("#it").classList.toggle("on", l === "it");
  if (D) { fillVars(); render(); }
}
function fillVars() {
  const v = $("#var").value || "tcwv_kgm2";
  $("#var").innerHTML = D.variables.map(k => `<option value="${k}">${L(k)}</option>`).join(""); $("#var").value = v;
}
function render() {
  const p = D.points[$("#pt").value]; if (!p) return;
  const days = {};
  p.time.forEach((t, i) => { (days[t.slice(0, 10)] ||= {})[+t.slice(11, 13)] = p[$("#var").value]?.[i]; });
  const hrs = [...Array(24).keys()];
  $("#tab").innerHTML = `<thead><tr><th>${lang === "it" ? "Giorno (UTC)" : "Day (UTC)"}</th>${hrs.map(h => `<th>${String(h).padStart(2, "0")}</th>`).join("")}</tr></thead><tbody>`
    + Object.entries(days).map(([d, r]) => `<tr><td>${d}</td>${hrs.map(h => `<td>${r[h] == null ? "–" : (+r[h]).toPrecision(4)}</td>`).join("")}</tr>`).join("") + "</tbody>";
}
(async () => {
  $("#en").onclick = () => setLang("en"); $("#it").onclick = () => setLang("it");
  D = await fetch("data/latest.json").then(r => r.json()).catch(() => ({ variables: [], points: {} }));
  const ids = Object.keys(D.points).sort((a, b) => (D.points[a].region > D.points[b].region ? -1 : 1));
  $("#pt").innerHTML = ids.map(k => `<option value="${k}">${D.points[k].region} · ${D.points[k].name}</option>`).join("");
  $("#pt").onchange = render; $("#var").onchange = render;
  setLang(lang);
})();
