#!/usr/bin/env python
"""Ceiling check for the pipeline architecture.

Feeds `render()` the *gold* patch -- the exact per-field edits the reference
report made -- and scores it. If this is not close to 0 then the assembly
(field order, label spelling, untouched-field copying) is broken, and no
amount of prompt engineering will help.

    python scripts/analysis/ceiling_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from natoe.config import TRAIN_CSV
from natoe.pipeline import gold_patch, guard, render
from natoe.evaluate import score_predictions, per_case

train = pd.read_csv(TRAIN_CSV)

for level in ("off", "numbers", "strict"):
    outs, drops = [], []
    for _, r in train.iterrows():
        case = {"template_content": r.template_content, "dictation": r.dictation}
        p = guard(gold_patch(r), case, level)
        drops += p["drops"]
        outs.append(render(case, p))
    res = score_predictions(train, outs)
    print(f"guard={level:8s}  RES={res['RES']:.4f}  F={res['F']:.4f}  "
          f"I={res['I']:.4f}  dropped={len(drops)}")

# Diagnose the residual at the default setting.
outs = []
for _, r in train.iterrows():
    case = {"template_content": r.template_content, "dictation": r.dictation}
    outs.append(render(case, guard(gold_patch(r), case, "numbers")))
res = score_predictions(train, outs)
per = per_case(res, train).sort_values(ascending=False)

print(f"\nresidual comes from {int((per > 0.05).sum())} rows. Worst 5:")
for i in per.index[:5]:
    r = train.loc[i]
    print(f"\n--- RES={per[i]:.3f}  {r.body_part}/{r.modality}")
    print("  dictation:", str(r.dictation)[:130].replace("\n", " "))
    for label, d in res["rows"][list(train.index).index(i)]["fields"].items():
        if d["edit"] > 0.05:
            flag = ("MISSING" if d.get("missing")
                    else "UNEXPECTED" if d.get("unexpected") else "")
            print(f"   {flag} {d.get('ref_display', label)} "
                  f"w={d['w']} e={d['edit']} ref={d.get('ref', '')[:70]!r}")
