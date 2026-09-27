"""Hybrid hypothesis: selection vs generation split by dictation length.

The copy-ceiling measurement says:
  * long dictations (8+ sentences) -> 97-98% of the gold field text is
    already present in the dictation+template, so it is a SELECTION problem
  * 1-sentence dictations        -> only 73% is findable, so 27% must be
    GENERATED

Yet the 8-15 bucket has the worst F of any bucket. If selection really is the
whole task there, a deterministic copy-assembler should beat the LLM on it.

This tests, per bucket: the LLM, the trained router doing verbatim
copy-assembly, and the best hybrid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from natoe.config import CACHE_PATH, TRAIN_CSV
from natoe.evaluate import build_retriever, dev_split
from natoe.field_router import train as train_router
from natoe.impression import append_template_closer, dict_sentences
from natoe.pipeline import (SYSTEM_PROMPT, Cache, build_shots,
                            build_user_prompt, guard, parse_json, render)
from natoe.res_scorer import (parse_fields, split_sections, tokenize,
                              weighted_word_edit)

N_SHOTS, GUARD, MODEL, EFFORT = 3, "numbers", "space-bunny-free", "low"
MIN_SENTS_FOR_ROUTER = 8          # tune this

train_df = pd.read_csv(TRAIN_CSV)
dev, pool = dev_split(train_df, n_dev=60)
retriever = build_retriever(pool)
cache = Cache(str(CACHE_PATH))

print("training the router on the few-shot pool (excludes dev)...", flush=True)
ROUTER = train_router(pool)
print("router trained\n", flush=True)


def bucket_of(n):
    return "1" if n == 1 else ("2-3" if n <= 3 else ("4-7" if n <= 7 else "8-15+"))


def llm_patch(r):
    case = {"template_content": r.template_content, "dictation": r.dictation,
            "modality": r.modality, "body_part": r.body_part,
            "study_description": r.study_description,
            "patient_age_band": r.patient_age_band,
            "patient_sex": r.patient_sex}
    prompt = build_user_prompt(case, build_shots(retriever, pool, case, N_SHOTS))
    key = Cache.key(MODEL, EFFORT, GUARD, N_SHOTS, 0.0, SYSTEM_PROMPT, prompt)
    raw = cache.get(key)
    if not raw:
        return None
    try:
        return parse_json(raw)
    except Exception:
        return {}


def router_patch(r):
    """Verbatim copy-assembly: routed dictation sentences prepended to each
    field's template text, template sentences kept."""
    routed = ROUTER.route_document(r.template_content, r.dictation, r.body_part)
    fields, display, _ = parse_fields(split_sections(r.template_content)[0])
    out = {}
    for key, original in fields.items():
        sents = routed.get(key) or []
        if sents:
            out[display[key]] = (" ".join(sents) + " " + original).strip()
    imp = " ".join(dict_sentences(r.dictation)[-6:])
    return {"fields": out, "impression": imp}


rows = []
for _, r in dev.iterrows():
    n = len(dict_sentences(r.dictation))
    b = bucket_of(n)
    lp = llm_patch(r)
    rp = router_patch(r)
    if lp is None:
        continue
    case = {"template_content": r.template_content, "dictation": r.dictation}
    llm_txt = render(case, guard(lp, case, GUARD))
    rt_txt = render(case, guard(rp, case, GUARD))
    # hybrid: router for long dictations, LLM for short
    use_router = n >= MIN_SENTS_FOR_ROUTER
    hyb_txt = rt_txt if use_router else llm_txt
    # hybrid on the IMPRESSION only: always take the LLM's
    hyb2_fields = rp["fields"] if use_router else lp.get("fields", {})
    hyb2 = render(case, guard({"fields": hyb2_fields,
                               "impression": lp.get("impression", "")},
                              case, GUARD))
    rows.append({"bucket": b, "n_sents": n,
                 "llm": weighted_word_edit(r.report, llm_txt),
                 "router": weighted_word_edit(r.report, rt_txt),
                 "hybrid": weighted_word_edit(r.report, hyb_txt),
                 "hyb2": weighted_word_edit(r.report, hyb2)})

D = pd.DataFrame(rows)
print(f"n = {len(D)}")
print("\nMEAN RES BY BUCKET  (per-row weighted edit of the whole report;")
print(" this is a proxy for RES, not RES itself -- use for RELATIVE ranking)\n")
g = D.groupby("bucket").agg(n=("llm", "size"), llm=("llm", "mean"),
                            router=("router", "mean"),
                            hybrid=("hybrid", "mean"), hyb2=("hyb2", "mean"))
g["best"] = g[["llm", "router", "hybrid", "hyb2"]].idxmin(axis=1)
g["router_minus_llm"] = (g.router - g.llm).round(4)
print(g.round(4).to_string())
print("\noverall:", {c: round(D[c].mean(), 4) for c in
                    ("llm", "router", "hybrid", "hyb2")})
print("\nthreshold sweep (use router when dictation >= K sentences):")
for K in (4, 6, 8, 10, 12, 99):
    m = D[D.n_sents >= K]
    print(f"  K={K:3d}  hybrid={weighted_mean(m, D):.4f}")


def weighted_mean(m, D):
    """Mean RES over ALL rows, where rows below K use the LLM."""
    if m.empty:
        return D.llm.mean()
    tot = D.copy()
    tot["sel"] = tot.router.where(tot.n_sents >= 8, tot.llm)
    # recompute properly for this K
    sel = tot.router.where(tot.n_sents >= K, tot.llm)
    return sel.mean()
