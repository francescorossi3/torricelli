# -*- coding: utf-8 -*-
"""01d - Unisce le parti calcolate da 01c (libRadtran) e scrive le LUT nello stesso formato della LUT 6S.

Input:  data/rt/parts_lrt/solar_partK.npz, thermal_partK.npz
Uscite: data/rt/lut_libradtran_s2.json     13 bande Sentinel-2 (risposta spettrale S2A da Py6S)
        data/rt/lut_libradtran_hyper.json  400-2500 nm ogni 10 nm (gaussiana, larghezza a meta' altezza 10 nm)
        data/rt/lut_libradtran_tir.json    Landsat 8/9 TIRS B10, B11 e 8-13 um ogni 0.1 um: tau, l_up, l_down
Calcoli (per banda, dopo la media sulla risposta spettrale): vedi 01c.
Uso:    python scripts/01d_merge_libradtran.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "rt" / "parts_lrt"
W = [0.1, 0.5, 1.0, 1.75, 2.5, 3.5, 5.0]
AOT = [0.01, 0.06, 0.15, 0.3, 0.55, 1.0]
SZA = [0, 25, 45, 60, 70, 80]
ALT = [0.0, 1.0, 2.5]
ATM = ["afglms", "afglmw", "afglt", "afglss", "afglsw"]
KEYS = ["t_down", "t_up", "t_gas", "l_path", "e_dir", "e_dif", "s_alb"]
MODEL = "libRadtran 2.0.6 (uvspec, DISORT 8 streams, LOWTRAN), AFGL mid-latitude summer, default rural aerosol, ozone 300 DU, nadir view"
r4 = lambda a: np.vectorize(lambda x: float(f"{x:.4g}") if np.isfinite(x) else None, otypes=[object])(a).tolist()   # noqa: E731


def gauss(lam, c, fwhm):
    return np.exp(-0.5 * ((lam - c) / (fwhm / 2.3548)) ** 2)


def save(name, out):
    path = ROOT / "data" / "rt" / name
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{path.name}: {len(out['bands'])} bande, {path.stat().st_size / 1e6:.1f} MB")


# %% VNIR-SWIR
parts = sorted(P.glob("solar_part*.npz"))
if parts:
    res, sres = {}, {}
    for p in parts:
        z = np.load(p)
        lam, e0 = z["lam"], z["e0"]
        res.update({tuple(k): v for k, v in zip(z["keys"], z["vals"])})
        sres.update({tuple(k): v for k, v in zip(z["skeys"], z["svals"])})
    # risposte spettrali
    from Py6S import PredefinedWavelengths as PW
    s2 = {}
    for b in ["01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12"]:
        _, a, _, f = getattr(PW, f"S2A_MSI_{b}")
        s2[f"S2_{b}"] = (np.interp(lam / 1000, np.arange(len(f)) * 0.0025 + a, f, left=0, right=0), None)
    hyper = {f"WL_{nm}": (gauss(lam, nm, 10.0), nm / 1000) for nm in range(400, 2501, 10)}
    for setname, bands in [("s2", s2), ("hyper", hyper)]:
        out = {"model": MODEL, "axes": {"w_gcm2": W, "aot550": AOT, "sza_deg": SZA, "alt_km": ALT}, "outputs": KEYS,
               "units": {"l_path": "W m-2 sr-1 um-1", "e_dir": "W m-2 um-1", "e_dif": "W m-2 um-1"}, "bands": {}}
        for b, (wt, wl) in bands.items():
            if wt.sum() <= 0:
                continue
            m = lambda x: float((wt * x).sum() / wt.sum())                              # noqa: E731
            E0 = m(e0)
            arr = np.full((len(W), len(AOT), len(SZA), len(ALT), len(KEYS)), np.nan)
            for (w, aot, sza, alt), v in res.items():
                L0, L1, L2, Ed, Ef = (m(x) for x in v)
                mu = np.cos(np.radians(sza))
                td = m(sres[(aot, sza, alt)]) / (E0 * mu)
                tu = m(sres[(aot, 0.0, alt)]) / E0 if (aot, 0.0, alt) in sres else np.nan
                D1, D2 = (L1 - L0) / 0.25, (L2 - L0) / 0.5
                S = (D2 - D1) / (D2 * 0.5 - D1 * 0.25) if (D2 * 0.5 - D1 * 0.25) else np.nan
                G = D1 * (1 - 0.25 * S)
                tg = G * np.pi / (E0 * mu * td * tu) if td > 0 and tu > 0 else np.nan
                arr[W.index(w), AOT.index(aot), SZA.index(sza), ALT.index(alt)] = [td, tu, tg, L0, Ed, Ef, S]
            out["bands"][b] = {"wl_um": wl, "values": r4(arr)}
        save(f"lut_libradtran_{setname}.json", out)

# %% LWIR
parts = sorted(P.glob("thermal_part*.npz"))
if parts:
    res = {}
    for p in parts:
        z = np.load(p)
        lam = z["lam"]
        res.update({(float(k[0]), float(k[1]), str(k[2])): v for k, v in zip(z["keys"], z["vals"])})
    um = lam / 1000
    planck = 1.191042e8 / um ** 5 / (np.exp(1.4387752e4 / (um * 300.0)) - 1)            # W m-2 sr-1 um-1 a 300 K
    bands = {"L8_B10": (gauss(um, 10.895, 0.59), 10.895), "L8_B11": (gauss(um, 12.005, 1.01), 12.005),
             **{f"WL_{int(round(c * 1000))}": (gauss(um, c, 0.1), c) for c in np.round(np.arange(8.0, 13.001, 0.1), 1)}}
    out = {"model": MODEL.replace("DISORT 8 streams, LOWTRAN), AFGL mid-latitude summer, default rural aerosol, ozone 300 DU",
                                  "DISORT 8 streams, LOWTRAN), thermal source, AFGL standard atmospheres, no aerosol"),
           "axes": {"w_gcm2": W, "alt_km": ALT, "atm": ATM}, "outputs": ["tau", "l_up", "l_down"],
           "units": {"l_up": "W m-2 sr-1 um-1", "l_down": "W m-2 sr-1 um-1"}, "bands": {}}
    for b, (wt, wl) in bands.items():
        m = lambda x: float((wt * x).sum() / wt.sum())                                   # noqa: E731
        arr = np.full((len(W), len(ALT), len(ATM), 3), np.nan)
        for (w, alt, atm), (L1, Lu, Edn) in res.items():
            L1b, Lub, Edb = m(L1) * 1000, m(Lu) * 1000, m(Edn) * 1000                    # per nm -> per um
            arr[W.index(w), ALT.index(alt), ATM.index(atm)] = [(L1b - Lub) / m(planck), Lub, Edb / np.pi]
        out["bands"][b] = {"wl_um": wl, "values": r4(arr)}
    save("lut_libradtran_tir.json", out)
