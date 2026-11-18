# -*- coding: utf-8 -*-
"""06 - Covariate del terreno nei siti di misura (AERONET, GNSS) dal Copernicus DEM GLO-30.

Per ogni sito nuovo: quota del DEM, pendenza, TPI (quota del sito meno la media nel raggio) e rilievo
(deviazione standard della quota) a 500 e 2000 m. Le stesse grandezze sono calcolate nel browser
(Prognostica, Atmosphere RK) per ogni pixel della mappa.

Input:  data/obs/aeronet/ e data/obs/gnss/ YYYY-MM.parquet (elenco dei siti), data/obs/model_at_aeronet/ (area con il modello)
Uscita: data/obs/terrain_aeronet.csv  (site, dem_elev, slope_deg, tpi_500, relief_500, tpi_2000, relief_2000)

Uso:  python scripts/06_terrain.py
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torricelli.common import DATA, OBS_NETWORKS, load_obs

RADII = [500, 2000]                                      # m
RES = 20                                                 # m, griglia di lettura
TERRAIN = DATA / "obs" / "terrain_aeronet.csv"

# %% siti senza covariate del terreno
months = sorted({f.stem for n in OBS_NETWORKS for f in (DATA / "obs" / n).glob("*.parquet")})
sites = load_obs(months).groupby("site")[["lat", "lon"]].first().reset_index()
with_model = {s for f in (DATA / "obs" / "model_at_aeronet").glob("*.parquet") for s in pd.read_parquet(f, columns=["site"])["site"]}
sites = sites[sites["site"].isin(with_model)]                 # solo i siti dell'area con i valori del modello
ter = pd.read_csv(TERRAIN) if TERRAIN.exists() else pd.DataFrame(columns=["site"])
todo = sites[~sites["site"].isin(ter["site"])]
print(f"{len(sites)} siti, {len(todo)} senza terreno")

# maschere circolari sulla griglia a RES m
n0 = int(round(2 * RADII[-1] / RES)) + 1
yy, xx = np.mgrid[:n0, :n0]
dist = np.hypot(yy - n0 // 2, xx - n0 // 2) * RES
RING = {rad: dist <= rad for rad in RADII}


def terrain(s):
    la, lo = int(np.floor(s.lat)), int(np.floor(s.lon))
    name = f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
    try:
        with rasterio.open(f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif") as ds:
            xs, ys = warp_transform("EPSG:4326", ds.crs, [s.lon], [s.lat])
            r, c = ds.index(xs[0], ys[0])
            hx = int(np.ceil(RADII[-1] / (abs(ds.transform.a) * 111320 * np.cos(np.radians(s.lat)))))
            hy = int(np.ceil(RADII[-1] / (abs(ds.transform.e) * 111320)))
            z = ds.read(1, window=Window(c - hx, r - hy, 2 * hx + 1, 2 * hy + 1), boundless=True, fill_value=0,
                        out_shape=(n0, n0), resampling=Resampling.bilinear).astype("float64")
    except Exception as e:                                    # noqa: BLE001
        print(f"   ! DEM {s.site}: {e}")
        # mare aperto (nessun tassello): si registra senza terreno; errori di rete: si riprova la volta dopo
        return {"site": s.site} if "not exist" in str(e) or "404" in str(e) else None
    c = n0 // 2
    gy, gx = np.gradient(z, RES)
    out = {"site": s.site, "dem_elev": float(z[c, c]), "slope_deg": float(np.degrees(np.arctan(np.hypot(gx[c, c], gy[c, c]))))}
    for rad in RADII:
        out[f"tpi_{rad}"] = float(z[c, c] - z[RING[rad]].mean())
        out[f"relief_{rad}"] = float(z[RING[rad]].std())
    return out


# %% calcolo e salvataggio
if len(todo):
    with ThreadPoolExecutor(12) as ex:
        new = [t for t in ex.map(terrain, todo.itertuples()) if t]
    ter = pd.concat([ter, pd.DataFrame(new)], ignore_index=True).drop_duplicates("site", keep="last")
    ter.to_csv(TERRAIN, index=False)
print(f"terreno: {len(ter)} siti")
