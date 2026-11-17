# -*- coding: utf-8 -*-
"""04 - Misure AERONET (fotometri solari): AOD e vapore acqueo in colonna, tutti i siti del mondo.

Servono a validare (e correggere con il regression kriging) i dati atmosferici dei modelli
(vapore ECMWF IFS, AOD CAMS) che Torricelli archivia in data/atmos/.

Fonte: AERONET Version 3, livello 1.5 (screening delle nuvole, quasi in tempo reale),
       web service https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3 (tutti i siti, tutte le misure).
Calcoli: AOD 550 nm da AOD 500 nm con l'esponente di Angstrom 440-870 nm:  AOD550 = AOD500 * (550/500)^(-alfa);
         vapore acqueo: cm di acqua precipitabile -> kg m-2 (x 10), stessa unita' di TCWV di ECMWF;
         media oraria per sito (stessa ora UTC dei dati Open-Meteo), con il numero di misure.

Uscite: data/obs/aeronet/YYYY-MM.parquet  (site, lat, lon, elev, time_utc, aod550, aod500, angstrom, tcwv, n)
        data/csv/obs/aeronet/YYYY-MM.csv   (stesso contenuto, leggibile in Excel)
        data/obs/aeronet_sites.csv          (elenco dei siti con almeno una misura)

Uso:  python scripts/04_fetch_aeronet.py                       (ultimi 7 giorni)
      python scripts/04_fetch_aeronet.py --start 2026-01-01 --end 2026-03-31
"""
import argparse
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torricelli.common import CSV_DIR, DATA, http_get

URL = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
OUT = DATA / "obs" / "aeronet"
CSV_OUT = CSV_DIR / "obs" / "aeronet"
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--start", help="primo giorno AAAA-MM-GG (default: 7 giorni fa)")
ap.add_argument("--end", help="ultimo giorno AAAA-MM-GG (default: oggi)")
args = ap.parse_args()
end = date.fromisoformat(args.end) if args.end else date.today()
start = date.fromisoformat(args.start) if args.start else end - timedelta(days=7)

# colonne del web service che servono (i nomi sono quelli dell'intestazione AERONET v3)
COLS = {"AERONET_Site": "site", "Date(dd:mm:yyyy)": "date", "Time(hh:mm:ss)": "time", "AOD_500nm": "aod500",
        "440-870_Angstrom_Exponent": "angstrom", "Precipitable_Water(cm)": "pwv_cm",
        "Site_Latitude(Degrees)": "lat", "Site_Longitude(Degrees)": "lon", "Site_Elevation(m)": "elev"}

# un giorno per richiesta: tutte le misure di tutti i siti, poi medie orarie
hourly = []
day = start
while day <= end:
    params = {"year": day.year, "month": day.month, "day": day.day, "year2": day.year, "month2": day.month, "day2": day.day,
              "hour": 0, "hour2": 23, "AOD15": 1, "AVG": 10, "if_no_html": 1}
    txt = http_get(URL, params=params, timeout=300).text
    k = txt.find("AERONET_Site,")
    if k < 0:
        print(f"{day}: nessuna misura")
        day += timedelta(days=1)
        continue
    raw = pd.read_csv(io.StringIO(txt[k:]), usecols=lambda c: c in COLS, na_values=[-999, "-999", "-999.000000"])
    raw = raw.rename(columns=COLS)
    raw["time_utc"] = pd.to_datetime(raw["date"] + " " + raw["time"], format="%d:%m:%Y %H:%M:%S").dt.floor("h")
    raw["aod550"] = raw["aod500"] * (550 / 500) ** (-raw["angstrom"])
    raw["tcwv"] = raw["pwv_cm"] * 10
    g = (raw.groupby(["site", "time_utc"])
         .agg(lat=("lat", "first"), lon=("lon", "first"), elev=("elev", "first"), aod550=("aod550", "mean"),
              aod500=("aod500", "mean"), angstrom=("angstrom", "mean"), tcwv=("tcwv", "mean"), n=("aod500", "count"))
         .reset_index())
    print(f"{day}: {raw['site'].nunique()} siti, {len(raw)} misure -> {len(g)} medie orarie")
    hourly.append(g)
    day += timedelta(days=1)

if not hourly:
    sys.exit("Nessuna misura AERONET nel periodo")
new = pd.concat(hourly, ignore_index=True)

# archivio mensile: le nuove medie sostituiscono quelle vecchie dello stesso sito e ora
for ym, part in new.groupby(new["time_utc"].dt.strftime("%Y-%m")):
    path = OUT / f"{ym}.parquet"
    if path.exists():
        part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
    part = part.drop_duplicates(["site", "time_utc"], keep="last").sort_values(["site", "time_utc"])
    part.to_parquet(path, index=False, compression="zstd")
    part.round(4).to_csv(CSV_OUT / f"{ym}.csv", index=False)
    print(f"  {path.name}: {part['site'].nunique()} siti, {len(part)} righe")

# elenco dei siti (unione con quelli gia' noti)
sites = new.groupby("site")[["lat", "lon", "elev"]].first().reset_index()
sp = DATA / "obs" / "aeronet_sites.csv"
if sp.exists():
    sites = pd.concat([pd.read_csv(sp), sites]).drop_duplicates("site", keep="last")
sites.sort_values("site").to_csv(sp, index=False)
print(f"Siti AERONET noti: {len(sites)}  (con AOD valido: {int(np.isfinite(new['aod550']).sum())} medie orarie)")
