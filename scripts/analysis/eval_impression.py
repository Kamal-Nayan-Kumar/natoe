"""Evaluate IMPRESSION strategies offline against cached LLM responses.

No API calls: the dev split is deterministic, so the cached responses replay
exactly. Reports the IMPRESSION component only, plus the implied RES at the
current F.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from natoe.config import CACHE_PATH, TRAIN_CSV
from natoe.evaluate import build_retriever, dev_split
from natoe.impression import (append_template_closer, heuristic_impression,
                              has_impression_header, last_sentence)
from natoe.pipeline import (SYSTEM_PROMPT, Cache, build_shots,
                            build_user_prompt, guard, parse_json, render)
from natoe.res_scorer import split_sections, weighted_word_edit

N_SHOTS, GUARD = 0, "numbers"
MODEL, EFFORT = "space-bunny-free", "low"

train = pd.read_csv(TRAIN_CSV)
dev, pool = dev_split(train, n_dev=60)
retriever = build_retriever(pool)
cache = Cache(str(CACHE_PATH))

rows, missing = [], 0
for _, r in dev.iterrows():
    case = {"template_content": r.template_content, "dictation": r.dictation,
            "modality": r.modality, "body_part": r.body_part,
            "study_description": r.study_description,
            "patient_age_band": r.patient_age_band, "patient_sex": r.patient_sex}
    prompt = build_user_prompt(case, build_shots(retriever, pool, case, N_SHOTS))
    key = Cache.key(MODEL, EFFORT, GUARD, N_SHOTS, SYSTEM_PROMPT, prompt)
    raw = cache.get(key)
    if not raw:
        missing += 1
        continue
    try:
        patch = parse_json(raw)
    except Exception:
        continue
    llm_imp = str(patch.get("impression") or "").strip()
    heur = heuristic_impression(r.template_content, r.dictation)
    tpl = split_sections(r.template_content)[1].strip()
    tail = last_sentence(tpl)
    gold = split_sections(r.report)[1]

    cands = {
        "A LLM as-is (current)": llm_imp,
        "B LLM + template closer": append_template_closer(llm_imp, r.template_content),
        "C deterministic heuristic": heur,
        "D template verbatim": tpl,
        "E LLM, else heuristic": llm_imp or heur,
        "F header, else LLM+closer": (llm_imp + " " + tail) if has_impression_header(
            r.dictation) else append_template_closer(llm_imp, r.template_content),
        "G header, else heuristic": heur if has_impression_header(
            r.dictation) else append_template_closer(heur, r.template_content),
    }
    # length-conditional variants: only append the closer when the dictated
    # summary is short, on the theory that a long summary already covers it
    for lim in (15, 25, 40):
        cands[f"H closer only if <{lim}w"] = (
            append_template_closer(llm_imp, r.template_content)
            if len(llm_imp.split()) < lim else llm_imp)
    row = {"body_part": r.body_part, "n_dict_sents": len(
        [x for x in r.dictation.split("\n") if x.strip()]),
        "llm_len": len(llm_imp.split()), "gold_len": len(gold.split())}
    for k, v in cands.items():
        row[k] = weighted_word_edit(gold, v)
    rows.append(row)

print(f"evaluated {len(rows)} rows from cache ({missing} not cached)\n")
D = pd.DataFrame(rows)
cols = [c for c in D.columns if c[0] in "ABCDEFG" and " " in c[2:]]
order = ["A LLM as-is (current)", "B LLM + template closer",
         "C deterministic heuristic", "D template verbatim",
         "E LLM, else heuristic", "F header, else LLM+closer",
         "G header, else heuristic",
         "H closer only if <15w", "H closer only if <25w",
         "H closer only if <40w"]
res = pd.DataFrame([{"strategy": c, "mean_I": D[c].mean(), "median_I": D[c].median(),
                     "RES at F=0.333": 0.65 * 0.333 + 0.35 * D[c].mean()}
                    for c in order if c in D])
print(res.sort_values("mean_I").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nmean LLM impression length {D.llm_len.mean():.1f} words | "
      f"gold {D.gold_len.mean():.1f} words")
print("\nby dictation size (mean I):")
D["bucket"] = pd.cut(D.n_dict_sents, [0, 1, 3, 7, 15, 999],
                     labels=["1", "2-3", "4-7", "8-15", "16+"])
print(D.groupby("bucket", observed=True)[[c for c in order if c in D]]
      .mean().round(4).to_string())
