# -*- coding: utf-8 -*-
"""05 - Valori dei modelli atmosferici nei siti AERONET (per il confronto con le misure).

Per ogni sito AERONET nell'area di studio e per ogni ora del periodo scarica da Open-Meteo:
  - ECMWF IFS 0.25 deg (archivio delle previsioni, "historical forecast"): vapore acqueo in colonna (TCWV),
    temperatura e umidita' a 2 m, pressione al suolo;
  - CAMS globale 0.4 deg (air quality): AOD a 550 nm e polveri al suolo.
Open-Meteo restituisce il valore della cella di griglia che contiene il sito (risoluzione del modello).

Input:  data/obs/aeronet/YYYY-MM.parquet (siti e ore con misure, da 04_fetch_aeronet.py)
Uscite: data/obs/model_at_aeronet/YYYY-MM.parquet  (site, time_utc, tcwv_model, t2m_c, rh2m, ps_hpa, aod_model, dust_ugm3)
        data/csv/obs/model_at_aeronet/YYYY-MM.csv

Uso:  python scripts/05_model_at_aeronet.py                     (ultimi 7 giorni)
      python scripts/05_model_at_aeronet.py --start 2026-01-01 --end 2026-03-31
"""
import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torricelli.common import CSV_DIR, DATA, http_get, om_params, om_url

# area di studio: Europa e Mediterraneo (siti RadCalNet / HYPERNETS fuori area si aggiungono qui)
BOX = {"lat": (30.0, 62.0), "lon": (-15.0, 40.0)}
ECMWF = {"total_column_integrated_water_vapour": "tcwv_model", "temperature_2m": "t2m_c",
         "relative_humidity_2m": "rh2m", "surface_pressure": "ps_hpa"}
CAMS = {"aerosol_optical_depth": "aod_model", "dust": "dust_ugm3"}
OUT = DATA / "obs" / "model_at_aeronet"
CSV_OUT = CSV_DIR / "obs" / "model_at_aeronet"
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--start")
ap.add_argument("--end")
args = ap.parse_args()
end = date.fromisoformat(args.end) if args.end else date.today()
start = date.fromisoformat(args.start) if args.start else end - timedelta(days=7)

# %% siti con misure nel periodo
months = pd.period_range(start, end, freq="M").strftime("%Y-%m")
obs = pd.concat([pd.read_parquet(f) for m in months if (f := DATA / "obs" / "aeronet" / f"{m}.parquet").exists()], ignore_index=True)
obs = obs[(obs["time_utc"].dt.date >= start) & (obs["time_utc"].dt.date <= end)
          & obs["lat"].between(*BOX["lat"]) & obs["lon"].between(*BOX["lon"])]
sites = obs.groupby("site")[["lat", "lon"]].first().reset_index()
print(f"{len(sites)} siti AERONET nell'area, {start} - {end}")


def fetch(url, variables, extra, chunk=20):
    """Richiesta multi-sito a Open-Meteo per il periodo; restituisce una tabella (site, time_utc, variabili)."""
    frames = []
    for i in range(0, len(sites), chunk):
        sub = sites.iloc[i:i + chunk]
        q = dict(extra, latitude=",".join(f"{v:.4f}" for v in sub["lat"]), longitude=",".join(f"{v:.4f}" for v in sub["lon"]),
                 hourly=",".join(variables), start_date=str(start), end_date=str(end), timezone="GMT")
        js = http_get(om_url(url), om_params(q), timeout=180).json()
        js = js if isinstance(js, list) else [js]
        for k, j in enumerate(js):
            h = j.get("hourly") or {}
            if not h:
                continue
            frames.append(pd.DataFrame({variables[v]: h.get(v) for v in variables})
                          .assign(site=sub["site"].iloc[k], time_utc=pd.to_datetime(h["time"])))
        time.sleep(2)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["site", "time_utc"])


met = fetch("https://historical-forecast-api.open-meteo.com/v1/forecast", ECMWF, {"models": "ecmwf_ifs025"})
aq = fetch("https://air-quality-api.open-meteo.com/v1/air-quality", CAMS, {"domains": "cams_global"})
df = met.merge(aq, on=["site", "time_utc"], how="outer")
print(f"ECMWF: {met['tcwv_model'].notna().sum()} ore, CAMS: {aq['aod_model'].notna().sum()} ore")

# %% archivio mensile (le righe nuove sostituiscono le vecchie)
for ym, part in df.groupby(df["time_utc"].dt.strftime("%Y-%m")):
    path = OUT / f"{ym}.parquet"
    if path.exists():
        part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
    part = part.drop_duplicates(["site", "time_utc"], keep="last").sort_values(["site", "time_utc"])
    part.to_parquet(path, index=False, compression="zstd")
    part.round({c: 4 for c in [*ECMWF.values(), *CAMS.values()]}).to_csv(CSV_OUT / f"{ym}.csv", index=False)
    print(f"  {path.name}: {part['site'].nunique()} siti, {len(part)} righe")
