#!/usr/bin/env python
"""Fill the response cache for the test set, then write a submission.

Rate limits make a single pass unreliable: rows that 429 fall back to the
template, which is the worst possible answer for them. So this script loops in
small batches until every row either has a real cached response or has been
declared unreachable, and reports coverage honestly rather than silently
degrading.

    python scripts/fill_and_submit.py --passes 8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from natoe.config import (CACHE_PATH, DEFAULT_GUARD_LEVEL, DEFAULT_N_SHOTS,
                          OUTPUT_DIR, TEST_CSV, TRAIN_CSV, have_key,
                          provider_name)
from natoe.evaluate import build_retriever, validate_submission
from natoe.pipeline import (SYSTEM_PROMPT, Cache, LLM, MockLLM, Retriever,
                            build_shots, build_user_prompt, guard, parse_json,
                            render, _sentences)
from natoe.res_scorer import split_sections, parse_fields, tokenize

# Anatomy keywords -> template field. Used ONLY by the offline fallback so
# that a rate-limited row still receives the dictated finding instead of
# nothing. Deliberately conservative.
ANATOMY = {
    "LUNGS": "lung lungs pulmonary parenchyma airspace consolidation infiltrate "
             "opacity nodule mass emphysema bronchic atelectasis interstitial",
    "LUNGS/AIRWAYS": "lung lungs pulmonary airway trachea bronchus",
    "PLEURA": "pleural pleura effusion pneumothorax",
    "PLEURALSPACES": "pleural pleura effusion pneumothorax",
    "HEART": "heart cardiac cardiomegaly pericardium",
    "CARDIOVASCULAR": "heart cardiac aorta vascular calcification",
    "HEARTANDMEDIASTINUM": "heart cardiac mediastinum pericardium aorta great vessels",
    "MEDIASTINUM": "mediastinum mediastinal hilar hilum adenopathy lymph node",
    "MEDIASTINALHILARLYMPHNODES": "mediastinum mediastinal hilar hilum adenopathy",
    "MEDIASTINUMHILA": "mediastinum mediastinal hilar hilum",
    "LYMPHNODES": "lymph node adenopathy nodal",
    "BONES": "bone osseous bony fracture fracture cortic trabecular spondylosis",
    "OSSEOUSSTRUCTURES": "bone osseous bony fracture cortic",
    "CHESTWALLMUSCULOSKELETAL": "rib chest wall osseous muscle scapula clavicle",
    "JOINTS": "joint joint space arthritis arthrosis degenerative subluxation "
              "effusion dislocation alignment",
    "DISCSPACES": "disc space disc vertebral spondylosis",
    "DISCSDEGENERATIVECHANGES": "disc space disc vertebral spondylosis facet",
    "VERTEBRAE": "vertebral vertebra spine spondylosis compression body height",
    "VERTEBRALBODIES": "vertebral vertebra compression body height marrow",
    "ALIGNMENT": "alignment lordosis kyphosis scoliosis curvature straightening",
    "SOFTTISSUES": "soft tissue swelling soft tissues subcutaneous edema",
    "PARAVERTEBRALSOFTTISSUES": "paraspinal paravertebral soft tissue",
    "SPINALCORD": "cord thecal conus spinal canal stenosis",
    "BRAIN": "brain cerebral hemispheric parenchyma infarct mass",
    "ORBITS": "orbit orbital globe eyeball",
    "SINUSESANDMASTOIDS": "sinus sinusitis mastoid mastoiditis",
    "TENDONS": "tendon tendinosis tear rotator cuff supraspinatus infraspinatus "
               "subscapularis biceps",
    "MUSCLES": "muscle muscular edema atrophy",
    "LIGAMENTS": "ligament ligamentous acl pcl mcl lcl sprain",
    "CARTILAGE": "cartilage chondral chondropathy meniscus meniscal",
    "BURSAE": "bursa bursal bursitis",
    "PATELLA": "patella patellar",
    "OTHERFINDINGS": "",
}

# Words that assert an abnormality. Used to decide whether a sentence belongs
# in a field at all, and whether it belongs in the IMPRESSION.
FINDING = {
    "fracture", "effusion", "fractures", "opacity", "consolidation", "mass",
    "nodule", "nodules", "spondylosis", "arthrosis", "arthritis", "stenosis",
    "edema", "swelling", "tear", "torn", "sprain", "fracture", "dislocation",
    "subluxation", "collapse", "infarct", "hemorrhage", "hemorrhagic",
    "contusion", "cyst", "abscess", "lesion", "lesions", "abnormality",
    "abnormal", "abnormalities", "degenerative", "deformity", "deformities",
    "atrophy", "calcification", "calcifications", "ossification", "spurring",
    "osteophyte", "osteophytes", "narrowing", "irregularity", "irregularities",
    "tendinosis", "tenosynovitis", "bursitis", "synovitis", "chondropathy",
    "chondral", "meniscal", "herniation", "bulge", "compression", "fractures",
    "destructive", "lytic", "blastic", "metastasis", "lyses", "ulcer",
    "stricture", "dilatation", "dilation", "enlarged", "prominence",
    "hyperintensity", "hyperintense", "demineralization", "destructive",
    "infiltrate", "infiltrates", "opacities", "pneumonia", "bronchiectasis",
    "atelectasis", "effusions", "pneumothorax", "fracturing", "angulation",
    "displacement", "callus", "pseudoarthrosis", "nonunion", "osteomyelitis",
}


def fallback_patch(case: dict) -> dict:
    """Offline, deterministic edit, used only when the API is unreachable.

    Routes each dictation sentence to the template field whose anatomy
    vocabulary it matches best, and PREPENDS it to that field's existing
    template text, which is the pattern the reference reports use most often.

    CAUTION -- the local RES re-implementation and the real Kaggle metric
    DISAGREE on this component. Three variants of the IMPRESSION fallback were
    submitted; the local ranking and the real ranking disagree, and the local
    "best" was the real worst:
      drop sentences with <=2 tokens (kept here) : Kaggle 0.62616
      keep only sentences asserting a finding     : Kaggle 0.63136
      keep every sentence, truncate to 600 chars  : Kaggle 0.64123
    Locally these measured 0.5695 / 0.5632 / (worse). So the local metric is a
    coarse guide, not a proxy -- validate real changes against the leaderboard
    and do not tune the last decimal against it.
    """
    fields, display, _ = parse_fields(split_sections(case["template_content"])[0])
    sents = _sentences(case["dictation"])
    out: dict[str, str] = {}
    for key, original in fields.items():
        vocab = set(ANATOMY.get(key, "").split())
        if not vocab or not sents:
            continue
        best, best_hits = "", 0
        for s in sents:
            hits = len(vocab & set(tokenize(s)))
            if hits > best_hits:
                best, best_hits = s, hits
        if best_hits >= 1:
            out[key] = f"{best} {original}".strip() if original else best
    imp = "\n".join(s for s in sents if len(tokenize(s)) > 2)[:600]
    return {"fields": out, "impression": imp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, choices=["groq", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    ap.add_argument("--guard", default=DEFAULT_GUARD_LEVEL)
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds to wait between individual calls")
    ap.add_argument("--sleep", type=float, default=45.0)
    ap.add_argument("--out", default=str(OUTPUT_DIR / "submission.csv"))
    args = ap.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    retriever = build_retriever(train)

    name = (args.provider or provider_name()).lower()
    llm = MockLLM() if not have_key(name) else LLM(provider=name,
                                                   model=args.model or None)
    print(f"provider {llm.provider} | model {getattr(llm, 'model', 'mock')} "
          f"| n_shots={args.n_shots} guard={args.guard} | {len(test)} test rows\n",
          flush=True)

    # Pre-compute prompts and cache keys once so coverage is exact.
    cache = Cache(str(CACHE_PATH))
    cases, keys = [], []
    for _, r in test.iterrows():
        case = {"template_content": r.template_content, "dictation": r.dictation,
                "modality": r.modality, "body_part": r.body_part,
                "study_description": r.study_description,
                "patient_age_band": r.patient_age_band,
                "patient_sex": r.patient_sex}
        cases.append(case)
        prompt = build_user_prompt(case, build_shots(retriever, train, case,
                                                      args.n_shots))
        keys.append(Cache.key(getattr(llm, "model", "mock"),
                              getattr(llm, "reasoning_effort", ""),
                              args.guard, args.n_shots, SYSTEM_PROMPT, prompt))

    def coverage() -> int:
        return sum(1 for k in keys if cache.get(k))

    print(f"initial cache coverage: {coverage()}/{len(keys)}", flush=True)

    for attempt in range(1, args.passes + 1):
        missing = [i for i, k in enumerate(keys) if not cache.get(k)]
        if not missing:
            break
        if attempt > 1:
            print(f"\n-- pass {attempt}: waiting {args.sleep:.0f}s before "
                  f"retrying {len(missing)} row(s)", flush=True)
            time.sleep(args.sleep)
        print(f"\n-- pass {attempt}: fetching {len(missing)} row(s) in batches "
              f"of {args.batch}", flush=True)
        for start in range(0, len(missing), args.batch):
            idxs = missing[start:start + args.batch]
            for i in idxs:
                prompt = build_user_prompt(cases[i], build_shots(
                    retriever, train, cases[i], args.n_shots))
                try:
                    cache.put(keys[i], llm.complete(prompt))
                except Exception as e:                    # noqa: BLE001
                    print(f"   row {i}: {type(e).__name__}: {str(e)[:90]}",
                          flush=True)
                if args.pace:
                    time.sleep(args.pace)
            print(f"   batch {start // args.batch + 1}/"
                  f"{(len(missing) + args.batch - 1) // args.batch} -> "
                  f"coverage {coverage()}/{len(keys)}", flush=True)

    cov = coverage()
    print(f"\nfinal cache coverage: {cov}/{len(keys)} "
          f"({100 * cov / len(keys):.0f}%)", flush=True)

    preds, n_fallback = [], 0
    for i, case in enumerate(cases):
        raw = cache.get(keys[i])
        if raw:
            try:
                patch = parse_json(raw)
            except Exception:                              # noqa: BLE001
                patch = fallback_patch(case)
                n_fallback += 1
        else:
            patch = fallback_patch(case)
            n_fallback += 1
        preds.append(render(case, guard(patch, case, args.guard)))

    if n_fallback:
        print(f"WARNING: {n_fallback} row(s) used the offline rule-based "
              f"fallback because the API did not return a usable response.")

    sub = pd.DataFrame({"case_id": test.case_id, "report": preds})
    errs = validate_submission(sub, test)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, quoting=csv.QUOTE_ALL)
    print(f"wrote {out}  {sub.shape}")
    if errs:
        print(f"{len(errs)} PROBLEM(S):")
        for e in errs[:20]:
            print("  -", e)
        return 1
    print("PASS - submission is structurally valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
