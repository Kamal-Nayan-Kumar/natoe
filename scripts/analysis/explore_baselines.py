#!/usr/bin/env python
"""Dataset reconnaissance: shape, label inventory, and the style statistics
that drove the prompt and renderer design.

    python scripts/analysis/explore_baselines.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from natoe.config import TRAIN_CSV, TEST_CSV
from natoe.res_scorer import (split_sections, parse_fields, normalize,
                              tokenize)
from natoe.evaluate import baselines, diagnosis_table

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print(f"train {train.shape} | test {test.shape}\n")
for c in ("modality", "body_part", "patient_age_band", "patient_sex"):
    a = train[c].value_counts().rename("train")
    b = test[c].value_counts().rename("test")
    print(f"== {c}")
    print(pd.concat([a, b], axis=1).fillna(0).astype(int).head(10).to_string())
    print()

print("== baselines (full train set)")
print(baselines(train).to_string(index=False))

print("\n== style statistics")
diag = diagnosis_table(train)
print(f"fields per reference report          : {diag.n_fields.mean():.2f}")
print(f"fields that DIFFER from the template : {diag.n_changed.mean():.2f} "
      f"({100 * diag.frac_changed.mean():.0f}%)")
print(f"reports with zero changed fields     : {(diag.n_changed == 0).sum()}")
print(f"IMPRESSION >80% dictation vocabulary : "
      f"{(diag.impression_overlap_with_dictation > 0.8).sum()} / {len(diag)}")
print(f"IMPRESSION identical to the template : "
      f"{diag.impression_equals_template.sum()} / {len(diag)}")

print("\n== label inventory (top 25)")
labels = Counter()
for text in train.report:
    _, display, _ = parse_fields(split_sections(text)[0])
    labels.update(display.values())
for k, v in labels.most_common(25):
    print(f"  {v:>4}  {k}")

print("\n== field-label set: reference vs template")
same = new = gone = 0
for _, r in train.iterrows():
    ref, _, _ = parse_fields(split_sections(r.report)[0])
    tpl, _, _ = parse_fields(split_sections(r.template_content)[0])
    if set(ref) == set(tpl):
        same += 1
    if set(ref) - set(tpl):
        new += 1
    if set(tpl) - set(ref):
        gone += 1
print(f"identical label sets : {same}/{len(train)}")
print(f"reference adds labels: {new}   template drops labels: {gone}")
print("=> the renderer can always emit every reference field, so the ceiling "
      "is limited only by the LLM, not by the assembly.")

print("\n== dictation length")
lengths = sorted(train.dictation.str.len())
print(f"median {lengths[len(lengths)//2]}  p90 {lengths[int(.9*len(lengths))]}  "
      f"max {lengths[-1]}")
print(f"dictations that already contain an IMPRESSION section: "
      f"{train.dictation.str.contains('IMPRESSION', case=False).sum()}")
print(f"template chars median {train.template_content.str.len().median():.0f} | "
      f"report chars median {train.report.str.len().median():.0f}")
