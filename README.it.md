# Torricelli

🇬🇧 [English version](README.md) · 🇮🇹 versione italiana

**Il nome.** Evangelista **Torricelli** (1608–1647), allievo di Galileo, inventò il barometro (1643) e dimostrò che l'aria ha un peso:
«noi viviamo sommersi nel fondo d'un pelago d'aria». Questo progetto misura quel «mare d'aria» che sta tra il satellite e il suolo.

**Ruolo.** **Back end** atmosferico (Python) di una famiglia di progetti:
**Prognostica** (previsione meteo, front end) · **Columella** (agricoltura, front end) · **Torricelli** (questo repository) · **Census** (monitoraggio d'uso, privato).
Torricelli legge da Prognostica l'elenco di stazioni e siti di calibrazione (RadCalNet, HYPERNETS); i front end leggono dal sito di Torricelli
(https://francescorossi3.github.io/torricelli/) le tabelle di trasferimento radiativo 6S e i parametri atmosferici.

## Contenuto

| Script | Cosa fa | Quando |
|---|---|---|
| `scripts/01_build_rt_lut.py` | tabella 6S: vapore × AOD550 × zenit solare × quota → T↓, T↑, T gas, radianza di percorso, irradianza diretta/diffusa, albedo sferico. `--set s2` (Sentinel-2) o `--set hyper` (400–2500 nm ogni 20 nm, PRISMA/EnMAP) | una volta (workflow «Tabella RT 6S») |
| `scripts/02_fetch_atmos.py` | database orario in ogni punto: vapore in colonna, AOD e polveri (CAMS), pressione, temperatura dell'aria e della superficie, umidità, nuvole, radiazione, profili 1000–200 hPa, densità dell'aria, zenit solare, parametri 6S per banda, TIR Landsat 8/9 B10 τ / L↑ / L↓ | ogni giorno |
| `scripts/03_build_site.py` | sito pubblico e JSON per i front end | ogni giorno |

Uscite: `data/atmos/AAAA-MM.parquet` e `data/csv/atmos/AAAA-MM.csv` (apribili in Excel), `data/rt/lut_*.json`.
Gli script sono lineari, senza funzione main, comodi in Spyder.

## Licenza

© 2026 Francesco Rossi. **Codice: tutti i diritti riservati.** **Dati: CC BY-NC 4.0.** Vedi [LICENSE](LICENSE).
