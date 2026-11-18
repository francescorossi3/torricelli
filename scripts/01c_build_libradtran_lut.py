# -*- coding: utf-8 -*-
"""01c - Tabelle di trasferimento radiativo (LUT) con libRadtran (uvspec): VNIR-SWIR e LWIR.

Stessi assi, nomi e formato JSON della LUT 6S (01_build_rt_lut.py), cosi' il sito le usa allo stesso modo.
libRadtran e' un programma esterno (licenza GPL) compilato nel workflow libradtran_lut.yml: non e' incluso nel repository.
Impostazioni comuni: atmosfera AFGL di media latitudine, assorbimento molecolare LOWTRAN, solutore DISORT (8 flussi),
aerosol predefinito (rurale, Shettle) con AOD a 550 nm imposto, ozono 300 DU, vapore acqueo imposto, vista al nadir.

VNIR-SWIR (--kind solar), per ogni nodo (w, aot, sza, alt) uno spettro 400-2500 nm (risoluzione LOWTRAN, uscita ogni 1 nm):
  run A  albedo 0     -> L0 radianza al sensore (path radiance, con gas), E_dir ed E_dif al suolo
  run B  albedo 0.25  -> L1        run C  albedo 0.5 -> L2
         L(rho) = L0 + G rho / (1 - S rho)  ->  S (albedo sferica) e G = E0 cos(sza) Tg T_down T_up / pi
  run D  senza assorbimento dei gas, albedo 0, al suolo  -> T_down = (E_dir + E_dif) / (E0 cos sza)   (non dipende da w)
         con sza = 0 -> T_up (reciprocita', vista al nadir)
  Tg = G pi / (E0 cos sza T_down T_up)
  Le grandezze sono prima mediate sulla risposta spettrale di ogni banda (Sentinel-2 da Py6S; iperspettrale: gaussiana 10 nm),
  poi combinate. Uscite: t_down, t_up, t_gas, l_path [W m-2 sr-1 um-1], e_dir, e_dif [W m-2 um-1], s_alb.
LWIR (--kind thermal), per ogni nodo (w, alt, atmosfera standard):
  run T1 temperatura del suolo 300 K, emissivita' 1 -> L1 al sensore;  run T2 suolo a 1 K -> L_up (path radiance)
  tau = (L1 - L_up) / B(300 K);  L_down = irradianza discendente al suolo / pi
  Bande: Landsat 8/9 TIRS B10 e B11 (gaussiane) e 8-13 um ogni 0.1 um. Uscite: tau, l_up, l_down [W m-2 sr-1 um-1].

Uscite parziali: data/rt/parts_lrt/<kind>_partK.npz  (unite da 01d_merge_libradtran.py in lut_libradtran_{s2,hyper,tir}.json)
Uso:  python scripts/01c_build_libradtran_lut.py --kind solar --part 0 --nparts 12 --jobs 4
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="solar", choices=["solar", "thermal"])
ap.add_argument("--part", type=int, default=0)
ap.add_argument("--nparts", type=int, default=1)
ap.add_argument("--jobs", type=int, default=os.cpu_count())
ap.add_argument("--limit", type=int, default=0, help="solo per prove: numero massimo di nodi")
ap.add_argument("--lrt", default=os.environ.get("LIBRADTRAN", str(Path.home() / "libRadtran")))
args = ap.parse_args()
BIN = Path(args.lrt) / "bin"
DATAP = Path(args.lrt) / "data"
OUT = ROOT / "data" / "rt" / "parts_lrt"
OUT.mkdir(parents=True, exist_ok=True)

W = [0.1, 0.5, 1.0, 1.75, 2.5, 3.5, 5.0]           # g/cm2   (come la LUT 6S)
AOT = [0.01, 0.06, 0.15, 0.3, 0.55, 1.0]           # 550 nm
SZA = [0, 25, 45, 60, 70, 80]                      # gradi
ALT = [0.0, 1.0, 2.5]                              # km
ATM = ["afglms", "afglmw", "afglt", "afglss", "afglsw"]   # LWIR: estate/inverno medie latitudini, tropicale, subartica estate/inverno


def uvspec(lines):
    """Esegue uvspec con le righe di input; restituisce la matrice dei numeri in uscita."""
    inp = "\n".join([f"data_files_path {DATAP}", *lines, "quiet"]) + "\n"
    r = subprocess.run([str(BIN / "uvspec")], input=inp, capture_output=True, text=True, cwd=BIN, timeout=3600)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(r.stderr[-500:])
    return np.loadtxt(r.stdout.splitlines(), ndmin=2)


def common(atm, w, alt, extra=()):
    return [f"atmosphere_file {DATAP}/atmmod/{atm}.dat", "mol_abs_param lowtran", "rte_solver disort", "number_of_streams 8",
            f"mol_modify H2O {w * 10:.4f} MM", "mol_modify O3 300 DU", f"altitude {alt:.3f}", *extra]


def by_zout(a, nz):
    """Righe (lambda, zout, ...) -> array [nlambda, nz, colonne] (uvspec scrive i livelli uno dopo l'altro per ogni lambda)."""
    return a.reshape(-1, nz, a.shape[1])


# %% VNIR-SWIR
def solar_node(job):
    w, aot, sza, alt = job
    aer = ["aerosol_default", f"aerosol_set_tau_at_wvl 550 {aot}"]
    src = [f"source solar {DATAP}/solar_flux/kurudz_1.0nm.dat", "wavelength 400 2500", f"sza {sza}", "umu 1.0"]
    try:
        a = by_zout(uvspec(common("afglms", w, alt, src + aer + ["albedo 0", "zout sur toa", "output_user lambda zout edir edn uu"])), 2)
        L0, Edir, Edif, lam = a[:, 1, 4], a[:, 0, 2], a[:, 0, 3], a[:, 0, 0]
        L1 = uvspec(common("afglms", w, alt, src + aer + ["albedo 0.25", "zout toa", "output_user lambda uu"]))[:, 1]
        L2 = uvspec(common("afglms", w, alt, src + aer + ["albedo 0.5", "zout toa", "output_user lambda uu"]))[:, 1]
        return job, lam, np.stack([L0, L1, L2, Edir, Edif])
    except Exception as e:                                         # noqa: BLE001
        print(f"   ! {job}: {e}", flush=True)
        return job, None, None


def scatter_node(job):
    """Trasmittanza di scattering verso il basso senza assorbimento dei gas (dipende da aot, sza, alt)."""
    aot, sza, alt = job
    lines = common("afglms", W[0], alt, [f"source solar {DATAP}/solar_flux/kurudz_1.0nm.dat", "wavelength 400 2500", f"sza {sza}",
                                          "aerosol_default", f"aerosol_set_tau_at_wvl 550 {aot}", "no_absorption mol", "albedo 0",
                                          "zout sur", "output_user lambda edir edn"])
    a = uvspec(lines)
    return job, a[:, 1] + a[:, 2]


# %% LWIR
def thermal_node(job):
    w, alt, atm = job
    src = ["source thermal", "wavelength 7500 13500", "output_process per_nm", "umu 1.0", "albedo 0"]
    try:
        a = by_zout(uvspec(common(atm, w, alt, src + ["sur_temperature 300", "zout sur toa", "output_user lambda zout edn uu"])), 2)
        L1, Edn, lam = a[:, 1, 3], a[:, 0, 2], a[:, 0, 0]
        Lu = uvspec(common(atm, w, alt, src + ["sur_temperature 1", "zout toa", "output_user lambda uu"]))[:, 1]
        return job, lam, np.stack([L1, Lu, Edn])
    except Exception as e:                                         # noqa: BLE001
        print(f"   ! {job}: {e}", flush=True)
        return job, None, None


t0 = time.time()
if args.kind == "solar":
    nodes = list(itertools.product(W, AOT, SZA, ALT))[args.part::args.nparts]
    snodes = list(itertools.product(AOT, sorted(set(SZA)), ALT))[args.part::args.nparts]
    if args.limit:                                        # prova: pochi nodi e i loro nodi senza gas
        nodes = nodes[:args.limit]
        snodes = sorted({(a, z, h) for _, a, s_, h in nodes for z in (s_, 0)})
    print(f"VNIR-SWIR parte {args.part}/{args.nparts}: {len(nodes)} nodi + {len(snodes)} nodi senza gas", flush=True)
    res, sres, lam = {}, {}, None
    with Pool(args.jobs) as pool:
        for k, (job, lm, v) in enumerate(pool.imap_unordered(solar_node, nodes)):
            if v is not None:
                res[job], lam = v, lm
            if k % 10 == 9:
                print(f"   {k + 1}/{len(nodes)} ({time.time() - t0:.0f} s)", flush=True)
        for job, v in pool.imap_unordered(scatter_node, snodes):
            sres[job] = v
    # irradianza solare al limite dell'atmosfera sulla stessa griglia (una volta)
    e0 = uvspec([f"atmosphere_file {DATAP}/atmmod/afglms.dat", f"source solar {DATAP}/solar_flux/kurudz_1.0nm.dat",
                 "wavelength 400 2500", "sza 0", "zout toa", "rte_solver disort", "mol_abs_param lowtran", "output_user lambda edir"])
    np.savez_compressed(OUT / f"solar_part{args.part}.npz", lam=lam, e0=e0[:, 1],
                        keys=np.array(list(res)), vals=np.array(list(res.values())),
                        skeys=np.array(list(sres)), svals=np.array(list(sres.values())))
else:
    nodes = list(itertools.product(W, ALT, ATM))[args.part::args.nparts][:args.limit or None]
    print(f"LWIR parte {args.part}/{args.nparts}: {len(nodes)} nodi", flush=True)
    res, lam = {}, None
    with Pool(args.jobs) as pool:
        for k, (job, lm, v) in enumerate(pool.imap_unordered(thermal_node, nodes)):
            if v is not None:
                res[job], lam = v, lm
            if k % 5 == 4:
                print(f"   {k + 1}/{len(nodes)} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(OUT / f"thermal_part{args.part}.npz", lam=lam, keys=np.array([list(map(str, k)) for k in res]),
                        vals=np.array(list(res.values())))
print(f"Fatto in {time.time() - t0:.0f} s")
