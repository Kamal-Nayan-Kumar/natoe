"""Where is the remaining loss concentrated?

Replays cached dev responses (no API calls) and breaks the error down by
dictation length, modality, and field label, so the next change targets the
biggest bucket rather than the most intuitive one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from natoe.config import CACHE_PATH, TRAIN_CSV
from natoe.evaluate import build_retriever, dev_split, per_case
from natoe.impression import dict_sentences
from natoe.pipeline import (SYSTEM_PROMPT, Cache, build_shots,
                            build_user_prompt, guard, parse_json, render)
from natoe.res_scorer import split_sections, parse_fields

N_SHOTS, GUARD, MODEL, EFFORT = 3, "numbers", "space-bunny-free", "low"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1

train = pd.read_csv(TRAIN_CSV)
dev, pool = dev_split(train, n_dev=60)
retriever = build_retriever(pool)
cache = Cache(str(CACHE_PATH))

outs = []
for _, r in dev.iterrows():
    case = {"template_content": r.template_content, "dictation": r.dictation,
            "modality": r.modality, "body_part": r.body_part,
            "study_description": r.study_description,
            "patient_age_band": r.patient_age_band, "patient_sex": r.patient_sex}
    prompt = build_user_prompt(case, build_shots(retriever, pool, case, N_SHOTS))
    k0 = Cache.key(MODEL, EFFORT, GUARD, N_SHOTS, 0.0, SYSTEM_PROMPT, prompt)
    raw = cache.get(k0)
    if not raw:
        continue
    try:
        patch = parse_json(raw)
    except Exception:
        patch = {}
    outs.append((r, render(case, guard(patch, case, GUARD)), guard(patch, case, GUARD)))

from natoe.res_scorer import res_case
rows = []
for r, out, clean in outs:
    rc = res_case(r.report, out, r.template_content)
    rows.append({
        "RES": rc["RES"], "F": rc["F"], "I": rc["I"],
        "n_sents": len(dict_sentences(r.dictation)),
        "dict_chars": len(r.dictation),
        "body_part": r.body_part, "modality": r.modality,
        "n_fields": len(parse_fields(split_sections(r.template_content)[0])[0]),
        "edited": len(clean["fields"]),
        "impression_empty": not (clean["impression"] or "").strip(),
    })
D = pd.DataFrame(rows)
print(f"n = {len(D)}   overall RES {D.RES.mean():.4f}  F {D.F.mean():.4f}  "
      f"I {D.I.mean():.4f}\n")

D["bucket"] = pd.cut(D.n_sents, [0, 1, 3, 7, 15, 999],
                     labels=["1", "2-3", "4-7", "8-15", "16+"])
print("BY DICTATION LENGTH")
g = D.groupby("bucket", observed=True).agg(
    n=("RES", "size"), RES=("RES", "mean"), F=("F", "mean"), I=("I", "mean"),
    fields=("n_fields", "mean"), edited=("edited", "mean"),
    share_of_total=("RES", lambda s: s.sum() / D.RES.sum()))
print(g.round(4).to_string())

print("\nBY BODY PART (>=3 rows)")
g = D.groupby("body_part").agg(n=("RES", "size"), RES=("RES", "mean"),
                               F=("F", "mean"), I=("I", "mean"))
g = g[g.n >= 3].sort_values("RES", ascending=False)
g["share"] = (g.n * g.RES) / (D.RES.sum())
print(g.round(4).head(14).to_string())

print("\nFIELDS EDITED vs TEMPLATE FIELD COUNT")
print(D.groupby("n_fields")[["RES", "F", "I"]].agg(["size", "mean"]).round(4).to_string())

print(f"\nrows where the model edited 0 fields: {(D.edited == 0).sum()}")
print(f"rows with an empty impression   : {D.impression_empty.sum()}")
print(f"rows rendering == the template  : "
      f"{sum(1 for r, o, c in outs if o.split('IMPRESSION')[0].strip() == r.template_content.split('IMPRESSION')[0].strip())}")

print("\nWORST 5")
for r, out, clean in sorted(outs, key=lambda t: res_case(t[0].report, t[1], t[0].template_content)["RES"])[-5:]:
    rc = res_case(r.report, out, r.template_content)
    print(f"\n--- RES={rc['RES']:.3f} F={rc['F']:.3f} I={rc['I']:.3f}  "
          f"{r.body_part}/{r.modality}  sents={len(dict_sentences(r.dictation))} "
          f"fields={len(clean['fields'])}")
    print("  dict:", str(r.dictation)[:200].replace("\n", " "))
    print("  pred:", out[:260].replace("\n", " | "))
    print("  gold:", str(r.report)[:260].replace("\n", " | "))
