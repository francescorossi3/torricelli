# -*- coding: utf-8 -*-
"""07 - Regression kriging dei dati atmosferici (vapore acqueo, AOD) validato sulle misure vere (AERONET, GNSS).

Obiettivo: portare il vapore acqueo (ECMWF IFS, ~25 km) e l'AOD (CAMS, ~45 km) alla scala del terreno (decine di metri)
e misurare il guadagno SOLO contro misure vere. Le covariate esistono per ogni pixel e per ogni ora, senza immagini satellitari.

Metodo (per ogni variabile):
  1. tabella: misura oraria + valori del modello (vapore, AOD, polveri, T2m, UR, pressione) + terreno
     (quota, pendenza, TPI e rilievo a 500/2000 m) + posizione, ora e giorno dell'anno;
  2. trend: HistGradientBoostingRegressor (scikit-learn) che impara la CORREZIONE del modello (misura - modello);
  3. kriging ordinario dei residui del trend tra i siti della stessa ora (variogramma esponenziale);
  4. validazione per zona: i siti sono raggruppati in celle di 0.5 gradi e ogni gruppo di celle viene escluso a turno (GroupKFold):
     il sito escluso non entra ne' nel trend ne' nel variogramma ne' nel kriging; i residui dei siti di
     addestramento usati nel kriging sono anch'essi fuori campione (GroupKFold interno).
  Metodi confrontati: model (dato grezzo), hgb (trend), hgb_rk (trend + kriging dei residui).

Input:  data/obs/aeronet, data/obs/gnss (solo vapore), data/obs/model_at_aeronet (YYYY-MM.parquet), data/obs/terrain_aeronet.csv
Uscite: data/validation/rk_validation.json        (metriche complessive e per sito, importanza delle covariate)
        data/validation/rk_model_<var>.json        (gradient boosting finale addestrato su tutti i dati, alberi in JSON per il browser:
                                                    correzione = baseline + somma delle foglie; NaN -> ramo "missing")
        data/csv/validation/rk_predictions_<var>.csv (ogni riga: sito, ora, misura, predizioni)

Uso:  python scripts/07_rk_validate.py                 (tutto l'archivio disponibile)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torricelli.common import CSV_DIR, DATA, OBS_NETWORKS, load_obs

VARS = {"tcwv": {"model": "tcwv_model", "unit": "kg m-2", "label": "Water vapour column"},
        "aod550": {"model": "aod_model", "unit": "", "label": "Aerosol optical depth 550 nm"}}
METHODS = ["model", "hgb", "hgb_rk"]
HOURS = (6, 17)                                          # ore UTC tenute (giorno in Europa)
MAX_KM = 1000.0                                          # vicini per il variogramma
OUT = DATA / "validation"
CSV_OUT = CSV_DIR / "validation"
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT.mkdir(parents=True, exist_ok=True)

# %% tabella: misure + modello + terreno
obs = load_obs(sorted({f.stem for n in OBS_NETWORKS for f in (DATA / "obs" / n).glob("*.parquet")}))
mod = pd.concat([pd.read_parquet(f) for f in sorted((DATA / "obs" / "model_at_aeronet").glob("*.parquet"))], ignore_index=True)
ter = pd.read_csv(DATA / "obs" / "terrain_aeronet.csv")
df = (obs[obs["time_utc"].dt.hour.between(*HOURS)].merge(mod, on=["site", "time_utc"], how="inner")
      .merge(ter, on="site", how="left"))
doy = df["time_utc"].dt.day_of_year
df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
df["hour"] = df["time_utc"].dt.hour + df["time_utc"].dt.minute / 60
# gruppi per la validazione: celle di 0.5 gradi, cosi' un sito di prova non ha vicini quasi identici (es. GNSS accanto ad AERONET) nell'addestramento
df["cell"] = (np.floor(df["lat"] / 0.5)).astype(int).astype(str) + "_" + (np.floor(df["lon"] / 0.5)).astype(int).astype(str)
FEAT = ["tcwv_model", "aod_model", "dust_ugm3", "t2m_c", "rh2m", "ps_hpa", "dem_elev", "slope_deg",
        "tpi_500", "relief_500", "tpi_2000", "relief_2000", "lat", "lon", "hour", "doy_sin", "doy_cos"]
print(f"{len(df)} righe (sito x ora), {df['site'].nunique()} siti, {len(FEAT)} covariate")


# %% kriging ordinario dei residui tra siti della stessa ora
def hav_km(la1, lo1, la2, lo2):
    p = np.pi / 180
    a = np.sin((la2 - la1) * p / 2) ** 2 + np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def expo(h, nug, sill, a):
    return nug + sill * (1 - np.exp(-h / a))


def fit_variogram(t):
    """Variogramma esponenziale dai residui (stessa ora, coppie entro MAX_KM). None se i dati non bastano."""
    h, g = [], []
    for _, d in t.groupby("time_utc"):
        if len(d) < 2:
            continue
        la, lo, r = d["lat"].to_numpy(), d["lon"].to_numpy(), d["res"].to_numpy()
        i, j = np.triu_indices(len(d), 1)
        dist = hav_km(la[i], lo[i], la[j], lo[j])
        keep = dist <= MAX_KM
        h.append(dist[keep]); g.append(0.5 * (r[i] - r[j])[keep] ** 2)
    if not h or sum(map(len, h)) < 30:
        return None
    h, g = np.concatenate(h), np.concatenate(g)
    k = np.digitize(h, np.linspace(0, MAX_KM, 11)) - 1
    hb = np.array([h[k == b].mean() for b in range(10) if (k == b).sum() >= 5])
    gb = np.array([g[k == b].mean() for b in range(10) if (k == b).sum() >= 5])
    if len(hb) < 3:
        return None
    try:
        p, _ = curve_fit(expo, hb, gb, p0=[gb.min(), max(gb.max() - gb.min(), 1e-9), 200], bounds=([0, 1e-12, 5], [np.inf, np.inf, 5000]))
        return p
    except RuntimeError:
        return None


def krige(train, test, vg):
    """Residuo stimato in ogni riga di test dai residui di train della stessa ora (0 se meno di 3 siti)."""
    out = np.zeros(len(test))
    if vg is None:
        return out
    nug, sill, a = vg
    cov = lambda h: np.where(h == 0, nug + sill, sill * np.exp(-h / a))   # noqa: E731
    test = test.reset_index(drop=True)
    by_hour = dict(tuple(train.groupby("time_utc")))
    for t, q in test.groupby("time_utc"):
        g = by_hour.get(t)
        if g is None:
            continue
        g = g.groupby(["lat", "lon"], as_index=False)["res"].mean()       # un residuo per sito: punti coincidenti renderebbero singolare il sistema
        n = len(g)
        if n < 3:
            continue
        la, lo = g["lat"].to_numpy(), g["lon"].to_numpy()
        A = np.ones((n + 1, n + 1)); A[n, n] = 0
        A[:n, :n] = cov(hav_km(la[:, None], lo[:, None], la[None, :], lo[None, :])) + np.eye(n) * 1e-6 * (nug + sill)
        B = np.ones((n + 1, len(q)))
        B[:n] = cov(hav_km(la[:, None], lo[:, None], q["lat"].to_numpy()[None, :], q["lon"].to_numpy()[None, :]))
        try:
            W = np.linalg.solve(A, B)[:n]
        except np.linalg.LinAlgError:
            continue
        ok = np.isfinite(W).all(0) & (np.abs(W).sum(0) <= 5)                 # sistema mal condizionato: nessuna correzione
        out[q.index.to_numpy()[ok]] = (W.T @ g["res"].to_numpy())[ok]
    return out


def learner():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=0)


def metrics(y, p):
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return {"n": int(len(y))}
    e = p - y
    sst = ((y - y.mean()) ** 2).sum()
    return {"n": int(len(y)), "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
            "bias": float(e.mean()), "r2": float(1 - (e ** 2).sum() / sst) if sst > 0 else None}


# %% validazione per gruppi di celle
result = {"built": pd.Timestamp.now(tz="UTC").isoformat(), "methods": METHODS, "variables": {}}
for var, info in VARS.items():
    d = df[df[var].notna() & df[info["model"]].notna()].reset_index(drop=True)
    nsite = d["site"].nunique()
    print(f"\n{var}: {len(d)} righe, {nsite} siti")
    if len(d) < 30 or nsite < 5:
        result["variables"][var] = {**info, "n_rows": int(len(d)), "n_sites": int(nsite), "note": "not enough data yet"}
        continue
    X, y, m0 = d[FEAT].to_numpy(float), d[var].to_numpy(float), d[info["model"]].to_numpy(float)
    corr = y - m0                                                     # il trend impara la correzione del modello
    pred = {k: np.full(len(d), np.nan) for k in METHODS}
    pred["model"] = m0
    for tr, te in GroupKFold(n_splits=min(10, d["cell"].nunique())).split(X, y, d["cell"]):
        pred["hgb"][te] = m0[te] + learner().fit(X[tr], corr[tr]).predict(X[te])
        # residui fuori campione dei siti di addestramento (GroupKFold interno) per variogramma e kriging
        inner = np.full(len(tr), np.nan)
        for itr, ite in GroupKFold(n_splits=min(5, d.iloc[tr]["cell"].nunique())).split(X[tr], corr[tr], d.iloc[tr]["cell"]):
            inner[ite] = m0[tr][ite] + learner().fit(X[tr][itr], corr[tr][itr]).predict(X[tr][ite])
        t_tr = d.iloc[tr][["time_utc", "lat", "lon"]].assign(res=y[tr] - inner)
        pred["hgb_rk"][te] = pred["hgb"][te] + krige(t_tr, d.iloc[te][["time_utc", "lat", "lon"]], fit_variogram(t_tr))
    overall = {k: metrics(y, pred[k]) for k in METHODS}
    for k in METHODS:
        print(f"  {k:7s} " + "  ".join(f"{a}={b:.4f}" for a, b in overall[k].items() if isinstance(b, float)))
    sites_out = []
    for site, g in d.assign(**{f"p_{k}": pred[k] for k in METHODS}).groupby("site"):
        yy = g[var].to_numpy(float)
        sites_out.append({"site": site, "network": g["network"].iloc[0], "lat": float(g["lat"].iloc[0]), "lon": float(g["lon"].iloc[0]),
                          "elev": None if pd.isna(g["dem_elev"].iloc[0]) else float(g["dem_elev"].iloc[0]),
                          **{k: metrics(yy, g[f"p_{k}"].to_numpy(float)) for k in METHODS}})
    # modello finale per le mappe: gradient boosting su tutti i dati, esportato albero per albero
    hgb = learner().fit(X, corr)
    trees = [{"f": n["feature_idx"].tolist(), "t": [float(v) for v in n["num_threshold"]], "m": n["missing_go_to_left"].astype(int).tolist(),
              "l": n["left"].tolist(), "r": n["right"].tolist(), "leaf": n["is_leaf"].astype(int).tolist(), "v": [float(v) for v in n["value"]]}
             for n in (it[0].nodes for it in hgb._predictors)]
    with open(OUT / f"rk_model_{var}.json", "w") as f:
        json.dump({"variable": var, "model_column": info["model"], "unit": info["unit"], "features": FEAT,
                   "baseline": float(np.ravel(hgb._baseline_prediction)[0]), "trees": trees,
                   "built": result["built"], "n_rows": int(len(d)), "rmse_validation": overall["hgb"]["rmse"]}, f, separators=(",", ":"))
    # importanza delle covariate: aumento dell'errore quando la covariata viene mescolata (su un campione)
    smp = np.random.default_rng(0).choice(len(d), min(len(d), 20000), replace=False)
    pi = permutation_importance(hgb, X[smp], corr[smp], n_repeats=3, random_state=0, scoring="neg_root_mean_squared_error")
    imp = sorted(zip(FEAT, pi.importances_mean), key=lambda x: -x[1])
    result["variables"][var] = {**info, "n_rows": int(len(d)), "n_sites": int(nsite),
                                "period": [str(d["time_utc"].min()), str(d["time_utc"].max())],
                                "overall": overall, "sites": sites_out,
                                "importance": [{"feature": f, "importance": float(v)} for f, v in imp]}
    d.assign(**{f"pred_{k}": pred[k] for k in METHODS})[["site", "time_utc", "lat", "lon", var, *[f"pred_{k}" for k in METHODS]]] \
        .round({var: 5, **{f"pred_{k}": 5 for k in METHODS}}).to_csv(CSV_OUT / f"rk_predictions_{var}.csv", index=False)

with open(OUT / "rk_validation.json", "w") as f:
    json.dump(result, f, separators=(",", ":"))
print(f"\nScritto {OUT / 'rk_validation.json'}")
