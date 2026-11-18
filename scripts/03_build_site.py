# -*- coding: utf-8 -*-
"""03 - Sito pubblico di Torricelli (GitHub Pages): dati in JSON per i front end + pagina di consultazione.

build/site/
    index.html, app.js, style.css       pagina (inglese, con traduzione italiana)
    data/rt/lut_s2.json (lut_hyper.json) tabelle 6S lette da Prognostica e Columella
    data/points.json                     punti elaborati
    data/latest.json                     ultime 24 h + 3 giorni di previsione per ogni punto (grandezze principali)
    data/validation/rk_validation.json   regression kriging di vapore e AOD validato sulle misure AERONET
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from torricelli.common import DATA, SITE

shutil.rmtree(SITE, ignore_errors=True)
shutil.copytree(ROOT / "web", SITE)
(SITE / "data" / "rt").mkdir(parents=True, exist_ok=True)
for f in (DATA / "rt").glob("lut_*.json"):
    shutil.copy(f, SITE / "data" / "rt" / f.name)

# %% ultime ore e previsione (solo grandezze principali, per tenere il file leggero)
now = pd.Timestamp.now(tz="UTC").floor("h")
files = sorted((DATA / "atmos").glob("*.parquet"))[-2:]
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
KEEP = ["tcwv_kgm2", "w_gcm2", "aod550", "dust_ugm3", "ps_hpa", "t2m_c", "tskin_c", "rh2m", "cloud_pct", "ghi_wm2", "dni_wm2",
        "dif_wm2", "rho_sfc_kgm3", "sza_deg", "tir_l8b10_tau", "tir_l8b10_lu", "tir_l8b10_ld"]
out = {"built": now.isoformat(), "variables": KEEP, "points": {}}
if len(df):
    df = df[df["valid_time"] >= now - pd.Timedelta(hours=24)].sort_values("valid_time")
    for sid, g in df.groupby("station"):
        out["points"][sid] = {"name": g["name"].iloc[0], "region": g["region"].iloc[0], "lat": float(g["lat"].iloc[0]),
                              "lon": float(g["lon"].iloc[0]), "elev": None if pd.isna(g["elev"].iloc[0]) else float(g["elev"].iloc[0]),
                              "time": g["valid_time"].dt.strftime("%Y-%m-%dT%H:%MZ").tolist(),
                              **{v: [None if not np.isfinite(x) else round(float(x), 4) for x in g[v]] for v in KEEP if v in g}}
with open(SITE / "data" / "latest.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))
# validazione del regression kriging sulle misure AERONET (letta da Prognostica, scheda Atmosphere RK)
(SITE / "data" / "validation").mkdir(parents=True, exist_ok=True)
for val in (DATA / "validation").glob("rk_*.json"):            # metriche e modelli (rk_model_<var>.json) per le mappe nel browser
    shutil.copy(val, SITE / "data" / "validation" / val.name)
pts = DATA / "points.csv"
if pts.exists():
    pd.read_csv(pts, dtype={"station": str}).to_json(SITE / "data" / "points.json", orient="records")
print(f"Sito in {SITE}: {len(out['points'])} punti")
