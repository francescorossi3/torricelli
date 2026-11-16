# -*- coding: utf-8 -*-
"""01b - Unisce le parti della LUT calcolate in parallelo (data/rt/lut_<set>_partK.json) in data/rt/lut_<set>.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SET = sys.argv[1] if len(sys.argv) > 1 else "hyper"
parts = sorted((ROOT / "data" / "rt").rglob(f"lut_{SET}_part*.json"))
out = None
for p in parts:
    d = json.load(open(p))
    if out is None:
        out = d
    else:
        out["bands"].update(d["bands"])
out["bands"] = dict(sorted(out["bands"].items(), key=lambda kv: kv[1]["wl_um"] or 0))
with open(ROOT / "data" / "rt" / f"lut_{SET}.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))
for p in parts:
    p.unlink()
print(f"{len(parts)} parti -> {len(out['bands'])} bande")
