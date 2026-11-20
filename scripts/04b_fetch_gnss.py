# -*- coding: utf-8 -*-
"""04b - Vapore acqueo da stazioni GNSS (Nevada Geodetic Laboratory, NGL): misure per validare e correggere il vapore dei modelli.

Fonte: NGL, prodotti troposferici ogni 5 minuti (ritardo zenitale totale, vapore acqueo integrato allo zenit, temperatura media
       pesata), uno ZIP per stazione e anno: https://geodesy.unr.edu/gps_timeseries/IGS20/trop/<STA>/<STA>.<ANNO>.trop.zip
       (dentro: un file SINEX TRO compresso per giorno). Elenco stazioni: https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt
Scelta delle stazioni (per limitare i download e avere punti ben distribuiti):
  - Italia (lat 35.5-47.5, lon 6-19): una stazione per cella di 1 grado (quella con la serie piu' lunga);
  - Europa e Mediterraneo: la stazione piu' vicina a ogni sito AERONET, se entro 30 km.
Calcoli: medie orarie del vapore acqueo (mm = kg m-2) e del ritardo zenitale totale (si legge l'anno intero di ogni file).

Input:  data/obs/aeronet_sites.csv (per le stazioni vicine ai fotometri)
Uscite: data/obs/gnss/YYYY-MM.parquet   (site = "GNSS_<STA>", lat, lon, elev, time_utc, tcwv, ztd_mm, n)
        data/csv/obs/gnss/YYYY-MM.csv
        data/obs/gnss_sites.csv          (stazioni scelte)
        data/obs/gnss_done.csv           (stazione-anno gia' letti; l'anno in corso si rilegge sempre)

Uso:  python scripts/04b_fetch_gnss.py                     (ultimi 45 giorni: NGL pubblica con ~2 settimane di ritardo)
      python scripts/04b_fetch_gnss.py --start 2025-01-01 --end 2025-12-31
"""
import argparse
import gzip
import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torricelli.common import CSV_DIR, DATA, SESSION

LIST = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
URLS = ["https://geodesy.unr.edu/gps_timeseries/IGS20/trop/{s}/{s}.{y}.trop.zip",
        "https://geodesy.unr.edu/gps_timeseries/trop/{s}/{s}.{y}.trop.zip"]
BOX = {"lat": (30.0, 62.0), "lon": (-15.0, 40.0)}
ITALY = {"lat": (35.5, 47.5), "lon": (6.0, 19.0)}
OUT = DATA / "obs" / "gnss"
CSV_OUT = CSV_DIR / "obs" / "gnss"
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT.mkdir(parents=True, exist_ok=True)
DONE = DATA / "obs" / "gnss_done.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--start")
ap.add_argument("--end")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max-stations", type=int, default=0, help="solo per prove: limita il numero di stazioni")
args = ap.parse_args()
end = date.fromisoformat(args.end) if args.end else date.today()
start = date.fromisoformat(args.start) if args.start else end - timedelta(days=45)  # NGL ha ~2 settimane di ritardo

# %% elenco delle stazioni NGL attive nel periodo
txt = SESSION.get(LIST, timeout=120).text.splitlines()
rows = []
for line in txt[1:]:
    p = line.split()
    if len(p) < 9:
        continue
    try:
        lat, lon, h = float(p[1]), float(p[2]), float(p[3])
        beg, fin = pd.Timestamp(p[7]), pd.Timestamp(p[8])
    except ValueError:
        continue
    rows.append((p[0], lat, (lon + 180) % 360 - 180, h, beg, fin))
st = pd.DataFrame(rows, columns=["sta", "lat", "lon", "elev", "beg", "fin"])
st = st[(st["fin"] >= pd.Timestamp(start)) & (st["beg"] <= pd.Timestamp(end))
        & st["lat"].between(*BOX["lat"]) & st["lon"].between(*BOX["lon"])].copy()
st["years"] = (st["fin"] - st["beg"]).dt.days / 365.25
print(f"{len(st)} stazioni NGL attive nell'area")
if st.empty:
    print("   nessuna stazione con dati nel periodo (ritardo NGL?): niente da aggiornare")
    sys.exit(0)

# %% scelta: una per cella di 1 grado in Italia + la piu' vicina a ogni sito AERONET (entro 30 km)
it = st[st["lat"].between(*ITALY["lat"]) & st["lon"].between(*ITALY["lon"])].copy()
it["cell"] = np.floor(it["lat"]).astype(int).astype(str) + "_" + np.floor(it["lon"]).astype(int).astype(str)
pick = set(it.sort_values("years", ascending=False).drop_duplicates("cell")["sta"])
ae_path = DATA / "obs" / "aeronet_sites.csv"
if ae_path.exists():
    ae = pd.read_csv(ae_path)
    ae = ae[ae["lat"].between(*BOX["lat"]) & ae["lon"].between(*BOX["lon"])]
    la, lo = np.radians(st["lat"].to_numpy()), np.radians(st["lon"].to_numpy())
    for a in ae.itertuples():
        d = 6371 * np.arccos(np.clip(np.sin(la) * np.sin(np.radians(a.lat)) + np.cos(la) * np.cos(np.radians(a.lat)) * np.cos(lo - np.radians(a.lon)), -1, 1))
        k = int(np.argmin(d))
        if d[k] <= 30:
            pick.add(st["sta"].iloc[k])
sel = st[st["sta"].isin(pick)].reset_index(drop=True)
if args.max_stations:
    sel = sel.head(args.max_stations)
sel.drop(columns=["beg", "fin", "years"]).assign(site="GNSS_" + sel["sta"]).to_csv(DATA / "obs" / "gnss_sites.csv", index=False)
print(f"{len(sel)} stazioni scelte ({it['cell'].nunique()} celle in Italia)")

# %% download e lettura: un file ZIP per stazione e anno, un file SINEX TRO per giorno
done = pd.read_csv(DONE) if DONE.exists() else pd.DataFrame(columns=["sta", "year"])
done_set = set(zip(done["sta"], done["year"]))
this_year = date.today().year
tasks = [(s, y) for s in sel.itertuples() for y in range(start.year, end.year + 1)
         if (s.sta, y) not in done_set or y >= this_year - (1 if date.today().month <= 3 else 0)]
print(f"{len(tasks)} file stazione-anno da leggere")


def parse_trop(raw):
    """Blocco TROP/SOLUTION di un file SINEX TRO -> tabella (epoca UTC, ztd_mm, tcwv) usando i nomi delle colonne dell'intestazione."""
    lines = raw.decode("ascii", "ignore").splitlines()
    try:
        i0 = next(i for i, l in enumerate(lines) if l.startswith("+TROP/SOLUTION"))
    except StopIteration:
        return None
    head, data = None, []
    for l in lines[i0 + 1:]:
        if l.startswith("-TROP/SOLUTION"):
            break
        if l.startswith("*"):
            head = l[1:].split()
            continue
        data.append(l.split())
    if not head or not data:
        return None
    d = pd.DataFrame([r[:len(head)] for r in data if len(r) >= len(head)], columns=head)
    wv = next((c for c in head if c.upper() in ("PWV", "IWV", "WVAPOR", "WV", "PWVTOT", "IWVTOT")), None)
    ztd = next((c for c in head if c.upper().startswith("TROTOT")), None)
    ep = next((c for c in head if "EPOCH" in c.upper()), head[1])
    yy, doy, sec = zip(*(e.split(":") for e in d[ep]))
    t = (pd.to_datetime([f"{2000 + int(a) if int(a) < 80 else 1900 + int(a)}-{int(b):03d}" for a, b in zip(yy, doy)], format="%Y-%j")
         + pd.to_timedelta(np.array(sec, dtype=float), unit="s"))
    return pd.DataFrame({"time_utc": t, "ztd_mm": pd.to_numeric(d[ztd], errors="coerce") if ztd else np.nan,
                         "tcwv": pd.to_numeric(d[wv], errors="coerce") if wv else np.nan})


def one(task):
    s, y = task
    for u in URLS:
        try:
            r = SESSION.get(u.format(s=s.sta, y=y), timeout=180)
        except Exception:                                   # noqa: BLE001
            continue
        if r.status_code == 200:
            break
    else:
        return s.sta, y, None
    frames = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            raw = z.read(name)
            if name.endswith(".gz"):
                raw = gzip.decompress(raw)
            d = parse_trop(raw)
            if d is not None:
                frames.append(d)
    if not frames:
        return s.sta, y, None
    d = pd.concat(frames)                                     # tutto l'anno: il file e' annuale
    d["time_utc"] = d["time_utc"].dt.floor("h")
    g = d.groupby("time_utc").agg(tcwv=("tcwv", "mean"), ztd_mm=("ztd_mm", "mean"), n=("tcwv", "count")).reset_index()
    return s.sta, y, g.assign(site="GNSS_" + s.sta, lat=s.lat, lon=s.lon, elev=s.elev)


out, ok = [], []
with ThreadPoolExecutor(args.workers) as ex:
    for k, (sta, y, g) in enumerate(ex.map(one, tasks)):
        if g is not None and len(g):
            out.append(g)
            ok.append((sta, y))
        if k % 25 == 24:
            print(f"   {k + 1}/{len(tasks)}")
print(f"{len(ok)} file con dati su {len(tasks)}")
if out:
    if out[0]["tcwv"].isna().all():
        print("   ! nessuna colonna di vapore riconosciuta nei file SINEX TRO")
    new = pd.concat(out, ignore_index=True)
    for ym, part in new.groupby(new["time_utc"].dt.strftime("%Y-%m")):
        path = OUT / f"{ym}.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part = part.drop_duplicates(["site", "time_utc"], keep="last").sort_values(["site", "time_utc"])
        part.to_parquet(path, index=False, compression="zstd")
        part.round({"tcwv": 3, "ztd_mm": 2}).to_csv(CSV_OUT / f"{ym}.csv", index=False)
        print(f"  {path.name}: {part['site'].nunique()} stazioni, {len(part)} righe")
    pd.concat([done, pd.DataFrame(ok, columns=["sta", "year"])]).drop_duplicates().to_csv(DONE, index=False)
