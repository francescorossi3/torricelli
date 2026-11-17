# -*- coding: utf-8 -*-
"""02 - Database atmosferico orario per stazioni italiane e siti di calibrazione/validazione.

Per ogni punto (stazioni di verifica e siti di calibrazione RadCalNet e HYPERNETS, elenco letto da Prognostica) scarica da Open-Meteo:
  - superficie (ECMWF IFS 0.25°, riserva GFS): T2m, UR, punto di rugiada, pressione al suolo,
    nuvolosità, vapore acqueo in colonna (TCWV), radiazione globale/diretta/diffusa/DNI/al TOA;
  - profili sui livelli di pressione 1000-200 hPa: temperatura, UR, altezza geopotenziale, vento;
  - CAMS: spessore ottico degli aerosol a 550 nm (AOD) e polveri.
e calcola:
  - densità dell'aria al suolo e su ogni livello  rho = p / (Rd * Tv);
  - angolo zenitale solare;
  - trasferimento radiativo VNIR-SWIR per le bande Sentinel-2 dalla LUT 6S (data/rt/lut_s2.json):
    trasmittanze t_down, t_up, t_gas, path radiance, irradianza diretta e diffusa, albedo sferico;
  - TIR Landsat 8/9 banda 10 (Jiménez-Muñoz et al. 2014): trasmittanza, radianza in salita e in discesa.

Uscite: data/atmos/YYYY-MM.parquet (un valore per ora e punto, aggiornato all'ultima corsa)
        data/csv/atmos/YYYY-MM.csv  (stesso contenuto, leggibile in Excel)

Uso:  python scripts/02_fetch_atmos.py            (ultimi 2 giorni + 3 di previsione)
      python scripts/02_fetch_atmos.py --past 7
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from torricelli.common import CSV_DIR, DATA, http_get, om_params, om_url, points

ap = argparse.ArgumentParser()
ap.add_argument("--past", type=int, default=2)
ap.add_argument("--days", type=int, default=3)
args, _ = ap.parse_known_args()

LEVELS = [1000, 925, 850, 700, 500, 300, 200]
SURF = {"total_column_integrated_water_vapour": "tcwv_kgm2", "surface_pressure": "ps_hpa",
        "temperature_2m": "t2m_c", "relative_humidity_2m": "rh2m", "dew_point_2m": "td2m_c",
        "cloud_cover": "cloud_pct", "shortwave_radiation": "ghi_wm2", "direct_radiation": "dir_wm2",
        "diffuse_radiation": "dif_wm2", "direct_normal_irradiance": "dni_wm2",
        "terrestrial_radiation": "toa_wm2", "soil_temperature_0cm": "tskin_c",
        "visibility": "vis_m", "wind_speed_10m": "ws10_kmh"}
PROF = {}
for p in LEVELS:
    PROF[f"temperature_{p}hPa"] = f"t_{p}_c"
    PROF[f"relative_humidity_{p}hPa"] = f"rh_{p}"
    PROF[f"geopotential_height_{p}hPa"] = f"z_{p}_m"
    PROF[f"wind_speed_{p}hPa"] = f"ws_{p}_kmh"
OM_VARS = {**SURF, **PROF}
AQ_VARS = {"aerosol_optical_depth": "aod550", "dust": "dust_ugm3"}
RD = 287.05

# %% 1) punti: stazioni di verifica + siti di calibrazione
pts = points()
pts = pts.drop_duplicates("station").reset_index(drop=True)
print(f"{len(pts)} punti ({pts['region'].value_counts().to_dict()})")


ELEV = []


def om_multi(url, params, chunk=25):
    """Richiesta multi-punto a Open-Meteo; restituisce una lista di 'hourly' per punto."""
    out = []
    for i in range(0, len(pts), chunk):
        sub = pts.iloc[i:i + chunk]
        q = dict(params, latitude=",".join(f"{v:.4f}" for v in sub["lat"]),
                 longitude=",".join(f"{v:.4f}" for v in sub["lon"]), timezone="GMT",
                 past_days=args.past, forecast_days=args.days)
        js = http_get(om_url(url), om_params(q), timeout=150).json()
        js = js if isinstance(js, list) else [js]
        out += [j.get("hourly", {}) for j in js]
        ELEV.extend(j.get("elevation") for j in js)
        time.sleep(2)
    return out


# %% 2) download superficie + profili (ECMWF IFS, riserva GFS dove mancano valori)
frames = []
for model in ["ecmwf_ifs025", "gfs_global"]:
    try:
        hh = om_multi("https://api.open-meteo.com/v1/forecast",
                      {"hourly": ",".join(OM_VARS), "models": model})
    except Exception as ex:                                       # noqa: BLE001
        print(f"  ! {model}: {ex}")
        continue
    for k, h in enumerate(hh):
        if not h:
            continue
        d = pd.DataFrame({OM_VARS[v]: h.get(v) for v in OM_VARS if v in h})
        d["valid_time"] = pd.to_datetime(h["time"], utc=True)
        d["station"] = pts.loc[k, "station"]
        d["src"] = model
        frames.append(d)
    print(f"  {model}: {len(hh)} punti")
met = pd.concat(frames, ignore_index=True)
# quota mancante (siti di calibrazione senza quota): quella del DEM di Open-Meteo (90 m)
if len(ELEV) >= len(pts):
    pts["elev"] = pts["elev"].fillna(pd.Series(ELEV[:len(pts)], dtype=float))
# per ogni (ora, punto) si usa ECMWF e si riempiono i buchi con GFS
met = met.sort_values("src").groupby(["valid_time", "station"], as_index=False).first()

try:
    hh = om_multi("https://air-quality-api.open-meteo.com/v1/air-quality", {"hourly": ",".join(AQ_VARS)})
    aq = pd.concat([pd.DataFrame({AQ_VARS[v]: h.get(v) for v in AQ_VARS})
                    .assign(valid_time=pd.to_datetime(h["time"], utc=True), station=pts.loc[k, "station"])
                    for k, h in enumerate(hh) if h], ignore_index=True)
    met = met.merge(aq, on=["valid_time", "station"], how="left")
    print(f"  CAMS: {len(hh)} punti")
except Exception as ex:                                           # noqa: BLE001
    print(f"  ! CAMS: {ex}")
    met["aod550"] = np.nan
    met["dust_ugm3"] = np.nan

df = met.merge(pts, on="station", how="left")

# %% 3) grandezze derivate
def esat(tc):                                    # pressione di vapore saturo [hPa] (Bolton)
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def rho(p_hpa, tc, rh):                          # densità dell'aria umida [kg m-3]
    e = rh / 100 * esat(tc)
    q = 0.622 * e / (p_hpa - 0.378 * e)
    tv = (tc + 273.15) * (1 + 0.608 * q)
    return p_hpa * 100 / (RD * tv)


df["rho_sfc_kgm3"] = rho(df["ps_hpa"], df["t2m_c"], df["rh2m"])
for p in LEVELS:
    df[f"rho_{p}_kgm3"] = rho(p, df[f"t_{p}_c"], df[f"rh_{p}"])
    df.loc[df["ps_hpa"] < p, [f"t_{p}_c", f"rh_{p}", f"z_{p}_m", f"ws_{p}_kmh", f"rho_{p}_kgm3"]] = np.nan  # sotto terra

df["w_gcm2"] = df["tcwv_kgm2"] / 10.0

# angolo zenitale solare (formule NOAA), a metà dell'ora che termina a valid_time
t = df["valid_time"] - pd.Timedelta(minutes=30)
doy = t.dt.dayofyear.values
hr = (t.dt.hour + t.dt.minute / 60).values
g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
ha = np.radians((hr * 60 + eqt + 4 * df["lon"].values) / 4 - 180)
la = np.radians(df["lat"].values)
cz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
df["sza_deg"] = np.degrees(np.arccos(np.clip(cz, -1, 1))).round(2)

# TIR Landsat 8/9 TIRS banda 10 (Jiménez-Muñoz et al. 2014, GAPRI), w in g cm-2
w = df["w_gcm2"]
psi1 = 0.04019 * w ** 2 + 0.02916 * w + 1.01523
psi2 = -0.38333 * w ** 2 - 1.50294 * w + 0.20324
psi3 = 0.00918 * w ** 2 + 1.36072 * w - 0.27514
df["tir_l8b10_tau"] = 1 / psi1
df["tir_l8b10_ld"] = psi3                                       # W m-2 sr-1 um-1
df["tir_l8b10_lu"] = -(psi2 + psi3) / psi1

# VNIR-SWIR dalla LUT 6S (bande Sentinel-2), solo con il sole sopra l'orizzonte (sza < 80°)
lut_path = DATA / "rt" / "lut_s2.json"
if lut_path.exists():
    lut = json.load(open(lut_path))
    ax = lut["axes"]
    grid = (ax["w_gcm2"], ax["aot550"], ax["sza_deg"], ax["alt_km"])
    num = lambda c: pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy()   # colonne miste (None) -> float
    X = np.column_stack([np.clip(num("w_gcm2"), ax["w_gcm2"][0], ax["w_gcm2"][-1]),
                         np.clip(np.nan_to_num(num("aod550"), nan=0.1), ax["aot550"][0], ax["aot550"][-1]),
                         np.clip(num("sza_deg"), 0, ax["sza_deg"][-1]),
                         np.clip(np.nan_to_num(num("elev"), nan=0.0) / 1000, ax["alt_km"][0], ax["alt_km"][-1])])
    ok = (num("sza_deg") < 80) & np.isfinite(X).all(1)
    rtcols = {}
    for b, info in lut["bands"].items():
        V = np.array(info["values"], dtype=float)
        for j, key in enumerate(lut["outputs"]):
            col = np.full(len(df), np.nan)
            col[ok] = RegularGridInterpolator(grid, V[..., j])(X[ok])
            rtcols[f"{key}_{b}"] = col
    df = pd.concat([df, pd.DataFrame(rtcols, index=df.index)], axis=1)
    print(f"  RT: {len(lut['bands'])} bande x {len(lut['outputs'])} grandezze")
else:
    print("  ! LUT 6S assente (scripts/01_build_rt_lut.py): niente parametri VNIR-SWIR")

# %% 4) salvataggio (si sostituiscono le ore ricalcolate)
df["run_date"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
keep = ["valid_time", "station", "name", "region", "lat", "lon", "elev", "src", "run_date"]
df = df[keep + [c for c in df.columns if c not in keep]]
num = df.select_dtypes("number").columns.difference(["lat", "lon", "elev"])
df[num] = df[num].astype("float32")
out_dir = DATA / "atmos"
csv_dir = CSV_DIR / "atmos"
out_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)
for ym, g in df.groupby(df["valid_time"].dt.strftime("%Y-%m")):
    path = out_dir / f"{ym}.parquet"
    if path.exists():
        old = pd.read_parquet(path)
        key = pd.MultiIndex.from_frame(g[["valid_time", "station"]])
        old = old[~pd.MultiIndex.from_frame(old[["valid_time", "station"]]).isin(key)]
        g = pd.concat([old, g], ignore_index=True)
    g = g.sort_values(["valid_time", "station"])
    g.to_parquet(path, index=False)
    c = g.copy()
    c.insert(0, "valid_time_utc", c.pop("valid_time").dt.strftime("%Y-%m-%d %H:%M"))
    c.round(4).to_csv(csv_dir / f"{ym}.csv", index=False)
    print(f"  {path.name}: {len(g)} righe, {g.shape[1]} colonne")

print("Fatto.")
