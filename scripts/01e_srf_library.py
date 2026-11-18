# -*- coding: utf-8 -*-
"""01e - Libreria delle funzioni di risposta spettrale (SRF) dei sensori, per convertire le grandezze di trasferimento
radiativo (LUT a 10 nm e a 0.1 um) alle bande dei satelliti in Prognostica (scheda Atmosphere RK).

VNIR-SWIR: curve misurate distribuite con Py6S (passo 2.5 nm; fonti delle agenzie: ESA, USGS/NASA, JAXA, VITO);
           dove Py6S da' solo i limiti della banda (Landsat 4-7, ASTER VNIR-SWIR, MERIS) risposta rettangolare.
LWIR:      curve gaussiane dal centro e dalla larghezza nominale delle bande (dati pubblicati delle missioni),
           perche' Py6S copre solo lo spettro solare.

Input:  Py6S (PredefinedWavelengths)
Uscita: data/rt/srf.json = {"sensors": {nome: {"bands": {banda: {"c": centro nm, "w0": primo nm, "dw": passo nm, "r": [...]}
                                                             oppure {"c", "lo", "hi"} (rettangolare) o {"c", "fwhm"} (gaussiana)}}}}

Uso:  python scripts/01e_srf_library.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from Py6S import PredefinedWavelengths as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torricelli.common import DATA

S2_BANDS = ["01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12"]
# sensore -> [(nome della banda, costante Py6S)]
MEASURED = {
    "Sentinel-2A MSI": [(f"B{b.lstrip('0')}", f"S2A_MSI_{b}") for b in S2_BANDS],
    "Sentinel-2B MSI": [(f"B{b.lstrip('0')}", f"S2B_MSI_{b}") for b in S2_BANDS],
    "Landsat 8/9 OLI/TIRS": [(f"B{i}", f"LANDSAT_OLI_B{i}") for i in range(1, 10)],
    "Landsat 7 ETM+": [(f"B{i}", f"LANDSAT_ETM_B{i}") for i in [1, 2, 3, 4, 5, 7]],
    "Landsat 4/5 TM": [(f"B{i}", f"LANDSAT_TM_B{i}") for i in [1, 2, 3, 4, 5, 7]],
    "Sentinel-3A OLCI": [(f"Oa{i:02d}", f"S3A_OLCI_{i:02d}") for i in range(1, 22)],
    "Sentinel-3B OLCI": [(f"Oa{i:02d}", f"S3B_OLCI_{i:02d}") for i in range(1, 22)],
    "Sentinel-3A SLSTR": [(f"S{i}", f"S3A_SLSTR_{i:02d}") for i in range(1, 7)],
    "MODIS Terra": [(f"B{i}", f"ACCURATE_MODIS_TERRA_{i}") for i in range(1, 17)],
    "MODIS Aqua": [(f"B{i}", f"ACCURATE_MODIS_AQUA_{i}") for i in range(1, 17)],
    "ASTER": [(b, f"ASTER_{b}") for b in ["B1", "B2", "B3N", "B4", "B5", "B6", "B7", "B8", "B9"]],
    "PROBA-V (centre camera)": [(f"B{i}", f"PROBAV_2_0{i}") for i in range(1, 5)],
    "MERIS": [(f"B{i}", f"MERIS_B{i}") for i in range(1, 16)],
}
# LWIR: (sensore, banda, centro um, larghezza um) dai limiti nominali delle bande
THERMAL = [("Landsat 8/9 OLI/TIRS", "B10", 10.895, 0.59), ("Landsat 8/9 OLI/TIRS", "B11", 12.005, 1.01),
           ("Landsat 7 ETM+", "B6", 11.45, 2.10), ("Landsat 4/5 TM", "B6", 11.45, 2.10),
           ("ASTER", "B10", 8.30, 0.35), ("ASTER", "B11", 8.65, 0.35), ("ASTER", "B12", 9.10, 0.35),
           ("ASTER", "B13", 10.60, 0.70), ("ASTER", "B14", 11.30, 0.70),
           ("MODIS Terra", "B29", 8.55, 0.30), ("MODIS Terra", "B31", 11.03, 0.50), ("MODIS Terra", "B32", 12.02, 0.50),
           ("MODIS Aqua", "B29", 8.55, 0.30), ("MODIS Aqua", "B31", 11.03, 0.50), ("MODIS Aqua", "B32", 12.02, 0.50),
           ("Sentinel-3A SLSTR", "S8", 10.854, 0.776), ("Sentinel-3A SLSTR", "S9", 12.0225, 0.905)]

# %% curve misurate (risposta ridotta all'intervallo non nullo, normalizzata a 1)
out = {}
for sensor, bands in MEASURED.items():
    out[sensor] = {"bands": {}}
    for name, const in bands:
        t = getattr(P, const, None)
        if t is None:
            continue
        if len(t) == 3:                                   # solo limiti della banda: risposta rettangolare
            out[sensor]["bands"][name] = {"c": round((t[1] + t[2]) * 500, 1), "lo": round(t[1] * 1000, 1), "hi": round(t[2] * 1000, 1)}
            continue
        r = np.asarray(t[3], float)
        w0 = t[1] * 1000
        nz = np.nonzero(r > 1e-4)[0]
        r = r[nz[0]:nz[-1] + 1] / r.max()
        wl = w0 + 2.5 * np.arange(nz[0], nz[-1] + 1)
        out[sensor]["bands"][name] = {"c": round(float((wl * r).sum() / r.sum()), 1), "w0": float(wl[0]), "dw": 2.5,
                                      "r": [round(float(v), 4) for v in r]}

# %% bande termiche gaussiane
for sensor, name, c, fwhm in THERMAL:
    out.setdefault(sensor, {"bands": {}})["bands"][name] = {"c": round(c * 1000, 1), "fwhm": round(fwhm * 1000, 1)}

(DATA / "rt").mkdir(parents=True, exist_ok=True)
with open(DATA / "rt" / "srf.json", "w") as f:
    json.dump({"source": "VNIR-SWIR: measured SRF distributed with Py6S (space agencies), rectangular where only band limits are given; LWIR: Gaussian from nominal band centre and width",
               "sensors": out}, f, separators=(",", ":"))
for s, v in out.items():
    print(f"{s}: {', '.join(v['bands'])}")
