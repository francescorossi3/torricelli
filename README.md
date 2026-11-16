# Torricelli

🇬🇧 English · 🇮🇹 [versione italiana](README.it.md)

**The name.** Evangelista **Torricelli** (1608–1647), Galileo's pupil, invented the barometer (1643) and showed that air has weight —
“we live submerged at the bottom of an ocean of air”. This project measures that ocean of air between the satellite and the ground.

**Role.** Atmospheric **back end** (Python) of a family of projects:
**Prognostica** (weather forecast, front end) · **Columella** (agriculture, front end) · **Torricelli** (this repository) · **Census** (usage monitoring, private).
Torricelli reads the list of stations and calibration sites (RadCalNet, HYPERNETS) from Prognostica. The front ends read from Torricelli's site
(https://francescorossi3.github.io/torricelli/) the 6S radiative-transfer tables and the atmospheric parameters.

## Contents

| Script | What it does | When |
|---|---|---|
| `scripts/01_build_rt_lut.py` | 6S look-up table: water vapour × AOD550 × solar zenith × altitude → T↓, T↑, T gas, path radiance, direct/diffuse irradiance, spherical albedo. `--set s2` (Sentinel-2) or `--set hyper` (400–2500 nm every 20 nm, PRISMA/EnMAP) | once (workflow “Tabella RT 6S”) |
| `scripts/02_fetch_atmos.py` | hourly database at every point: water vapour column, AOD and dust (CAMS), pressure, air and skin temperature, humidity, clouds, radiation, profiles 1000–200 hPa, air density, solar zenith, per-band 6S parameters, TIR Landsat 8/9 B10 τ / L↑ / L↓ | every day |
| `scripts/03_build_site.py` | public site and JSON for the front ends | every day |

Output: `data/atmos/YYYY-MM.parquet` and `data/csv/atmos/YYYY-MM.csv` (Excel-readable), `data/rt/lut_*.json`.
Scripts are flat and linear (Spyder-friendly).

## Licence

© 2026 Francesco Rossi. **Code: all rights reserved.** **Data: CC BY-NC 4.0.** See [LICENSE](LICENSE).
