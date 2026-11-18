# -*- coding: utf-8 -*-
"""06 - Covariate ad alta risoluzione nei siti di misura (AERONET, GNSS): Sentinel-2 L2A (contemporaneo) e terreno (Copernicus DEM).

Per ogni sito AERONET nell'area di studio e per ogni immagine Sentinel-2 L2A acquisita quando il sito ha una misura
(stessa ora UTC) legge una finestra intorno al sito e calcola le medie entro tre raggi (60, 500, 2000 m),
solo sui pixel validi secondo la classificazione della scena (SCL, senza nuvole, ombre e cirri):
  - AOT 550 nm e vapore acqueo (WVP) stimati da Sen2Cor   [aot, wvp in kg m-2]
  - riflettanze: blu, rosso, NIR, SWIR 1.6 e 2.2 um       [blue, red, nir, swir16, swir22]
  - indici: NDVI, NDBI (costruito), NDWI                   [ndvi, ndbi, ndwi]
  - frazioni della scena: acqua, vegetazione, suolo nudo, nuvole, pixel validi
  - geometria: elevazione solare, copertura nuvolosa della scena
Terreno (statico, calcolato una volta per sito, Copernicus DEM GLO-30):
  quota, pendenza, deviazione standard della quota e indice di posizione topografica (TPI) a 500 e 2000 m.

Fonti: Earth Search STAC (Element 84) collezione sentinel-2-c1-l2a, file COG pubblici su AWS;
       Copernicus DEM GLO-30 (COG pubblici su AWS, (c) DLR/Airbus, licenza Copernicus).
Input:  data/obs/aeronet/ e data/obs/gnss/ YYYY-MM.parquet
Uscite: data/obs/covariates_s2/YYYY-MM.parquet  (una riga per sito e immagine; le immagini gia' lette si saltano)
        data/obs/terrain_aeronet.csv           (covariate del terreno per sito)

Uso:  python scripts/06_covariates_s2.py                     (ultimi 7 giorni)
      python scripts/06_covariates_s2.py --start 2026-01-01 --end 2026-03-31
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torricelli.common import DATA, SESSION, load_obs

STAC = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-c1-l2a"
BOX = {"lat": (30.0, 62.0), "lon": (-15.0, 40.0)}
RADII = [60, 500, 2000]                                  # m
RES = 20                                                 # m, griglia comune di lettura
BANDS = ["aot", "wvp", "blue", "green", "red", "nir", "swir16", "swir22"]
SCL_OK = {2, 4, 5, 6, 7, 11}                             # scuro, vegetazione, suolo nudo, acqua, non classificato, neve
SCL_CLOUD = {3, 8, 9, 10}                                # ombre, nuvole medie/alte, cirri
OUT = DATA / "obs" / "covariates_s2"
OUT.mkdir(parents=True, exist_ok=True)
TERRAIN = DATA / "obs" / "terrain_aeronet.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--start")
ap.add_argument("--end")
ap.add_argument("--workers", type=int, default=12)
args = ap.parse_args()
end = date.fromisoformat(args.end) if args.end else date.today()
start = date.fromisoformat(args.start) if args.start else end - timedelta(days=7)

# %% misure AERONET nel periodo e nell'area
months = pd.period_range(start, end, freq="M").strftime("%Y-%m")
obs = load_obs(months)                                        # AERONET e GNSS
obs = obs[(obs["time_utc"].dt.date >= start) & (obs["time_utc"].dt.date <= end)
          & obs["lat"].between(*BOX["lat"]) & obs["lon"].between(*BOX["lon"])]
sites = obs.groupby("site")[["lat", "lon", "elev"]].first().reset_index()
sites = sites[sites["site"].isin(obs.loc[obs["time_utc"].dt.hour.between(8, 13), "site"])]   # solo siti con misure attorno al passaggio di S2
obs_hours = set(zip(obs["site"], obs["time_utc"]))
done = set()
for m in months:
    f = OUT / f"{m}.parquet"
    if f.exists():
        done |= set(zip(*pd.read_parquet(f, columns=["site", "item"]).values.T))
print(f"{len(sites)} siti, {len(obs)} ore con misure, {len(done)} immagini gia' lette")

# %% immagini Sentinel-2 sopra ogni sito, tenendo solo quelle nell'ora di una misura
tasks = []
for s in sites.itertuples():
    body = {"collections": [COLLECTION], "intersects": {"type": "Point", "coordinates": [s.lon, s.lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "query": {"eo:cloud_cover": {"lt": 90}}, "limit": 200}
    for k in range(3):
        try:
            r = SESSION.post(STAC, json=body, timeout=90)
            r.raise_for_status()
            break
        except Exception as e:                                # noqa: BLE001
            print(f"   ! STAC {s.site}: {e}")
            time.sleep(5 * (k + 1))
    else:
        continue
    for it in r.json().get("features", []):
        if s.site.startswith("GNSS_") and it["properties"]["eo:cloud_cover"] >= 50:   # GNSS misura anche con nuvole: solo scene abbastanza serene
            continue
        t = pd.Timestamp(it["properties"]["datetime"]).tz_convert(None)
        if (s.site, t.floor("h")) in obs_hours and (s.site, it["id"]) not in done:
            tasks.append((s, it, t))
print(f"{len(tasks)} coppie sito-immagine da leggere")


def asset_scale(a):
    rb = (a.get("raster:bands") or [{}])[0]
    return rb.get("scale", 1.0), rb.get("offset", 0.0)


def read_window(href, lon, lat, half_m, scale=1.0, offset=0.0, nodata0=True, resampling=Resampling.average):
    """Finestra quadrata di lato 2*half_m centrata sul sito, ricampionata a RES m; valori fisici (NaN = nessun dato)."""
    with rasterio.open(href) as ds:
        xs, ys = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
        r, c = ds.index(xs[0], ys[0])
        dx, dy = abs(ds.transform.a), abs(ds.transform.e)
        if ds.crs.is_geographic:                                      # gradi -> metri (DEM in EPSG:4326)
            dx, dy = dx * 111320 * np.cos(np.radians(lat)), dy * 111320
        hx, hy = int(np.ceil(half_m / dx)), int(np.ceil(half_m / dy))
        n = int(round(2 * half_m / RES))
        a = ds.read(1, window=Window(c - hx, r - hy, 2 * hx + 1, 2 * hy + 1), boundless=True, fill_value=0,
                    out_shape=(n + 1, n + 1), resampling=resampling).astype("float64")
    if nodata0:
        a[a == 0] = np.nan
    return a * scale + offset


# maschere circolari sulla griglia a RES m
n0 = int(round(2 * RADII[-1] / RES)) + 1
yy, xx = np.mgrid[:n0, :n0]
dist = np.hypot(yy - n0 // 2, xx - n0 // 2) * RES
RING = {rad: dist <= max(rad, RES) for rad in RADII}


def one(task):
    s, it, t = task
    a = it["assets"]
    row = {"site": s.site, "item": it["id"], "s2_time": t, "time_utc": t.floor("h"),
           "cloud_scene": it["properties"].get("eo:cloud_cover"), "sun_elev": it["properties"].get("view:sun_elevation")}
    scl = read_window(a["scl"]["href"], s.lon, s.lat, RADII[-1], nodata0=False, resampling=Resampling.nearest)
    ok = np.isin(scl, list(SCL_OK))
    v = {}
    for b in BANDS:
        if b not in a:
            continue
        sc, of = asset_scale(a[b])
        v[b] = read_window(a[b]["href"], s.lon, s.lat, RADII[-1], sc, of)
    if "wvp" in v:
        v["wvp"] = v["wvp"] * 10                                    # cm -> kg m-2
    with np.errstate(invalid="ignore", divide="ignore"):
        v["ndvi"] = (v["nir"] - v["red"]) / (v["nir"] + v["red"])
        v["ndbi"] = (v["swir16"] - v["nir"]) / (v["swir16"] + v["nir"])
        v["ndwi"] = (v["green"] - v["nir"]) / (v["green"] + v["nir"])
    for rad, m in RING.items():
        tot = m.sum()
        row[f"valid_frac_{rad}"] = (ok & m).sum() / tot
        row[f"cloud_frac_{rad}"] = (np.isin(scl, list(SCL_CLOUD)) & m).sum() / tot
        row[f"water_frac_{rad}"] = ((scl == 6) & m).sum() / tot
        row[f"veg_frac_{rad}"] = ((scl == 4) & m).sum() / tot
        row[f"bare_frac_{rad}"] = ((scl == 5) & m).sum() / tot
        for b, arr in v.items():
            x = arr[ok & m]
            x = x[np.isfinite(x)]
            row[f"{b}_{rad}"] = float(x.mean()) if x.size else np.nan
    return row


def safe(task):
    try:
        return one(task)
    except Exception as e:                                    # noqa: BLE001  (file non leggibile: si salta)
        print(f"   ! {task[0].site} {task[1]['id']}: {e}")
        return None


rows, t0 = [], time.time()
with ThreadPoolExecutor(args.workers) as ex:
    for k, res in enumerate(ex.map(safe, tasks)):
        if res:
            rows.append(res)
        if k % 50 == 49:
            print(f"   {k + 1}/{len(tasks)} ({time.time() - t0:.0f} s)")

# %% terreno (una volta per sito): Copernicus DEM GLO-30
ter = pd.read_csv(TERRAIN) if TERRAIN.exists() else pd.DataFrame(columns=["site"])
todo = sites[~sites["site"].isin(ter["site"])]


def dem_href(lat, lon):
    la, lo = int(np.floor(lat)), int(np.floor(lon))
    name = f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def terrain(s):
    try:
        z = read_window(dem_href(s.lat, s.lon), s.lon, s.lat, RADII[-1], nodata0=False, resampling=Resampling.bilinear)
    except Exception as e:                                    # noqa: BLE001  (mare aperto: nessun tassello)
        print(f"   ! DEM {s.site}: {e}")
        return {"site": s.site}
    c = z.shape[0] // 2
    gy, gx = np.gradient(z, RES)
    out = {"site": s.site, "dem_elev": float(z[c, c]), "slope_deg": float(np.degrees(np.arctan(np.hypot(gx[c, c], gy[c, c]))))}
    for rad in RADII[1:]:
        m = RING[rad]
        out[f"tpi_{rad}"] = float(z[c, c] - np.nanmean(z[m]))
        out[f"relief_{rad}"] = float(np.nanstd(z[m]))
    return out


if len(todo):
    with ThreadPoolExecutor(args.workers) as ex:
        new_t = list(ex.map(terrain, todo.itertuples()))
    ter = pd.concat([ter, pd.DataFrame(new_t)], ignore_index=True).drop_duplicates("site", keep="last")
    ter.to_csv(TERRAIN, index=False)
    print(f"terreno: {len(new_t)} siti nuovi, {len(ter)} in tutto")

# %% archivio mensile
if rows:
    df = pd.DataFrame(rows)
    for ym, part in df.groupby(df["time_utc"].dt.strftime("%Y-%m")):
        path = OUT / f"{ym}.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part.drop_duplicates(["site", "item"], keep="last").sort_values(["site", "s2_time"]).to_parquet(path, index=False, compression="zstd")
        print(f"  {path.name}: {len(part)} righe")
print(f"Fatto in {time.time() - t0:.0f} s")
