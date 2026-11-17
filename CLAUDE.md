# Torricelli – istruzioni per Claude

Back end atmosferico (Python) per la correzione atmosferica: LUT 6S (S2 e 400–2500 nm ogni 10 nm), database orario (vapore, AOD, profili, densità, temperatura aria/superficie, RT per banda, TIR L8/9 B10).
Legge stazioni e siti da Prognostica (`torricelli/common.py: points()`); pubblica JSON su Pages per Prognostica e Columella.

## Mappa del codice
- `torricelli/common.py` – percorsi, http_get, Open-Meteo, elenco punti.
- `scripts/01_build_rt_lut.py` (+ `01b_merge_lut.py`) – LUT 6S; workflow `rt_lut.yml` (6 job paralleli).
- `scripts/02_fetch_atmos.py` – database orario → `data/atmos/*.parquet`, `data/csv/atmos/*.csv`.
- `scripts/03_build_site.py` + `web/` – sito e `data/latest.json`.
- Workflow `daily.yml` 07:41 UTC.

## Regole comuni (famiglia Prognostica · Columella · Torricelli · Census)
- Proprietario: Francesco Rossi (CNR-IMIOT). Rispondere in italiano.
- Python (back end): script **lineari stile Spyder**, poche funzioni, **niente `main()` / `if __name__`**; file numerati `NN_nome.py`.
- JavaScript (front end): sito statico GitHub Pages, calcolo nel browser, niente build step.
- Lingua dei siti: **inglese** principale + traduzione italiana (pulsante EN/IT). Nomi di file/CSV in inglese.
- Licenza: codice tutti i diritti riservati, dati CC BY-NC 4.0 (`LICENSE`). Non aggiungere dipendenze con licenze copyleft senza chiedere.
- Mai mettere chiavi/token nel codice: solo GitHub Secrets. Open-Meteo gratuito = uso non commerciale.
- Commit piccoli e descrittivi in italiano; dopo modifiche al sito lanciare il workflow di pubblicazione.
- Non aprire file sul PC di Francesco senza permesso; niente controllo di mouse/schermo.
- Risparmio token: leggere solo i file necessari (usare grep), non rileggere file appena modificati, risposte brevi.
