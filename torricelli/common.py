"""Percorsi, download HTTP e punti da elaborare (letti dal front end Prognostica)."""
import io
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSV_DIR = DATA / "csv"
SITE = ROOT / "build" / "site"

# Prognostica (front end meteo) pubblica l'elenco delle stazioni e dei siti di calibrazione:
# Torricelli li legge da li', cosi' i due progetti restano allineati senza copiare i dati.
PROGNOSTICA_RAW = os.environ.get("PROGNOSTICA_RAW", "https://raw.githubusercontent.com/francescorossi3/prognostica/main")

OM_KEY = os.environ.get("OPENMETEO_API_KEY", "")
SESSION = requests.Session()
SESSION.headers["User-Agent"] = os.environ.get("USER_AGENT", "torricelli/1.0 (github.com/francescorossi3/torricelli)")


def om_url(url):
    """Con chiave a pagamento si usano gli endpoint 'customer-' di Open-Meteo."""
    return url.replace("https://", "https://customer-", 1) if OM_KEY else url


def om_params(params):
    return dict(params, apikey=OM_KEY) if OM_KEY else params


def http_get(url, params=None, tries=4, timeout=120):
    """GET con nuovi tentativi e attesa sui limiti di frequenza (HTTP 429)."""
    for k in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:
            print(f"   ! rete: {e}; ritento")
            time.sleep(5 * (k + 1))
            continue
        if r.status_code == 429:
            print("   ! troppe richieste (429): attendo 65 s")
            time.sleep(65)
            continue
        if r.status_code >= 500:
            time.sleep(10 * (k + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"Richiesta fallita dopo {tries} tentativi: {url}")


def points():
    """Stazioni italiane di verifica + siti di calibrazione (RadCalNet, HYPERNETS, ...) da Prognostica.
    Colonne: station, name, lat, lon, elev, region. Copia locale in data/points.csv come riserva."""
    local = DATA / "points.csv"
    try:
        st = pd.read_csv(io.StringIO(http_get(f"{PROGNOSTICA_RAW}/data/stations.csv").text), dtype={"station": str})
        st = st.assign(region="italia")[["station", "name", "lat", "lon", "elev", "region"]]
        cfg = yaml.safe_load(http_get(f"{PROGNOSTICA_RAW}/config.yaml").text)
        rows = [{"station": s["id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"], "elev": s.get("elev"), "region": r["id"]}
                for r in cfg["regions"] if r.get("type") == "points" and r.get("enabled", True) for s in r["sites"]]
        pts = pd.concat([st, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("station").reset_index(drop=True)
        DATA.mkdir(exist_ok=True)
        pts.to_csv(local, index=False)
    except Exception as e:                                     # noqa: BLE001
        print(f"  ! elenco punti da Prognostica non disponibile ({e}): uso la copia locale")
        pts = pd.read_csv(local, dtype={"station": str})
    return pts
