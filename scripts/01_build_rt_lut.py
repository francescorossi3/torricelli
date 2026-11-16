# -*- coding: utf-8 -*-
"""
10 - Tabella di trasferimento radiativo (LUT) VNIR-SWIR con 6S.

Si esegue UNA volta (o quando si cambiano bande/griglia), non ogni giorno.
Per ogni banda/lunghezza d'onda e per ogni combinazione di
    w   vapore acqueo colonnare [g/cm2]
    aot spessore ottico aerosol a 550 nm (CAMS)
    sza angolo zenitale solare [gradi]
    alt quota del suolo [km]
6S (vista al nadir, sensore satellitare, aerosol continentale, ozono 0.30 cm-atm)
calcola:
    t_down, t_up   trasmittanza di scattering discendente / ascendente
    t_gas          trasmittanza gassosa totale (H2O, O3, O2, CO2, ...)
    l_path         path radiance (radianza intrinseca atmosferica) [W m-2 sr-1 um-1]
    e_dir, e_dif   irradianza solare diretta e diffusa al suolo [W m-2 um-1]
    s_alb          albedo sferica
Uso:
    python scripts/01_build_rt_lut.py --set s2       (Sentinel-2 MSI, 13 bande, con risposta spettrale)
    python scripts/01_build_rt_lut.py --set hyper    (400-2500 nm ogni 20 nm: PRISMA / EnMAP)
Serve l'eseguibile 6S (conda-forge: sixs). Percorso in SIXS_PATH o nel PATH.
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from Py6S import (AeroProfile, Altitudes, AtmosCorr, AtmosProfile, Geometry, GroundReflectance,
                  PredefinedWavelengths, SixS, Wavelength)

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="s2", choices=["s2", "hyper"])
ap.add_argument("--jobs", type=int, default=os.cpu_count())
args, _ = ap.parse_known_args()

SIXS = os.environ.get("SIXS_PATH", "sixs")
W = [0.1, 0.5, 1.0, 1.75, 2.5, 3.5, 5.0]           # g/cm2
AOT = [0.01, 0.06, 0.15, 0.3, 0.55, 1.0]           # 550 nm
SZA = [0, 25, 45, 60, 70, 80]                      # gradi
ALT = [0.0, 1.0, 2.5]                              # km
if args.set == "s2":
    BANDS = {f"S2_{b}": getattr(PredefinedWavelengths, f"S2A_MSI_{b}") for b in
             ["01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12"]}
else:
    BANDS = {f"WL_{int(nm)}": nm / 1000 for nm in np.arange(400, 2501, 20)}
KEYS = ["t_down", "t_up", "t_gas", "l_path", "e_dir", "e_dif", "s_alb"]


def run(job):
    band, w, aot, sza, alt = job
    s = SixS(SIXS)
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(w, 0.30)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.aot550 = aot
    s.geometry = Geometry.User()
    s.geometry.solar_z, s.geometry.solar_a, s.geometry.view_z, s.geometry.view_a = sza, 0, 0, 0
    s.altitudes = Altitudes()
    s.altitudes.set_target_custom_altitude(alt)
    s.altitudes.set_sensor_satellite_level()
    wl = BANDS[band]
    s.wavelength = Wavelength(wl) if not isinstance(wl, float) else Wavelength(wl)
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.2)
    s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(0.2)
    try:
        s.run()
        o = s.outputs
        sc = o.transmittance_total_scattering
        return (band, w, aot, sza, alt, [sc.downward, sc.upward, o.total_gaseous_transmittance,
                                         o.atmospheric_intrinsic_radiance, o.direct_solar_irradiance,
                                         o.diffuse_solar_irradiance, o.spherical_albedo.total])
    except Exception as e:                                      # noqa: BLE001
        return (band, w, aot, sza, alt, [np.nan] * len(KEYS))


combos = list(itertools.product(W, AOT, SZA, ALT))
print(f"{args.set}: {len(BANDS)} bande x {len(combos)} combinazioni = {len(BANDS) * len(combos)} esecuzioni 6S")
t0 = time.time()
# una banda alla volta, con salvataggio parziale: se il calcolo si interrompe riparte da dove era arrivato
part = ROOT / "data" / "rt" / f"_partial_{args.set}.json"
part.parent.mkdir(parents=True, exist_ok=True)
done = json.load(open(part)) if part.exists() else {}
arr = {b: np.array(done[b]) if b in done else np.full((len(W), len(AOT), len(SZA), len(ALT), len(KEYS)), np.nan) for b in BANDS}
with Pool(args.jobs) as pool:
    for b in BANDS:
        if b in done:
            continue
        for (band, w, aot, sza, alt, vals) in pool.imap_unordered(run, [(b,) + c for c in combos], chunksize=8):
            arr[band][W.index(w), AOT.index(aot), SZA.index(sza), ALT.index(alt)] = vals
        done[b] = np.round(arr[b], 6).tolist()
        json.dump(done, open(part, "w"))
        print(f"  {b}: {len(done)}/{len(BANDS)} bande  ({time.time() - t0:.0f}s)", flush=True)

out = {"model": "6S (Py6S), nadir view, continental aerosol, ozone 0.30 cm-atm, satellite-level sensor",
       "axes": {"w_gcm2": W, "aot550": AOT, "sza_deg": SZA, "alt_km": ALT}, "outputs": KEYS,
       "units": {"l_path": "W m-2 sr-1 um-1", "e_dir": "W m-2 um-1", "e_dif": "W m-2 um-1"},
       "bands": {b: {"wl_um": (float(BANDS[b]) if args.set == "hyper" else None),
                     "values": np.round(arr[b], 5).tolist()} for b in BANDS}}
path = ROOT / "data" / "rt" / f"lut_{args.set}.json"
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w") as f:
    json.dump(out, f, separators=(",", ":"))
part.unlink(missing_ok=True)
print(f"Salvato {path} ({path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
