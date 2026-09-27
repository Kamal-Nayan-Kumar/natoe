#!/usr/bin/env python
"""Score the pipeline on a held-out dev split of train.csv.

    python scripts/run_dev.py --n-shots 3 --guard numbers
    python scripts/run_dev.py --provider openrouter --model google/gemma-4-31b-it:free
    python scripts/run_dev.py --n-dev 160 --variants
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from natoe.config import (CACHE_PATH, DEFAULT_DEV_SIZE, DEFAULT_GUARD_LEVEL, PROVIDERS,
                          DEFAULT_N_SHOTS, OUTPUT_DIR, TRAIN_CSV, have_key,
                          model_name, provider_name)
from natoe.evaluate import (baselines, build_retriever, dev_split,
                            diagnosis_table, field_loss_table, per_case,
                            run_and_score, score_predictions)
from natoe.pipeline import LLM, MockLLM


def get_llm(provider: str | None, model: str | None):
    name = (provider or provider_name()).lower()
    if not have_key(name):
        print(f"No key for provider {name!r} -> falling back to MockLLM "
              f"(plumbing check only; the score is meaningless).")
        return MockLLM()
    return LLM(provider=name, model=model or None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, choices=sorted(PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-dev", type=int, default=DEFAULT_DEV_SIZE)
    ap.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    ap.add_argument("--guard", default=DEFAULT_GUARD_LEVEL,
                    choices=["off", "numbers", "strict"])
    ap.add_argument("--n-samples", type=int, default=1,
                    help="candidates per row; >1 enables the aggressiveness "
                         "selector (no gold needed)")
    ap.add_argument("--variants", action="store_true",
                    help="sweep n_shots x guard_level")
    ap.add_argument("--n-errors", type=int, default=3)
    ap.add_argument("--out", default=str(OUTPUT_DIR / "dev_results.json"))
    args = ap.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    dev, pool = dev_split(train, n_dev=args.n_dev)
    retriever = build_retriever(pool)
    llm = get_llm(args.provider, args.model)

    print(f"provider : {llm.provider}  model: {llm.model}")
    print(f"dev rows : {len(dev)}   few-shot pool: {len(pool)}")
    print(f"config   : n_shots={args.n_shots}  guard={args.guard}\n")

    print("=" * 78)
    print("REFERENCE POINTS")
    print("=" * 78)
    print(baselines(dev).to_string(index=False))

    print("\n" + "=" * 78)
    print("STYLE STATISTICS OF THE REFERENCE REPORTS (why the prompt looks like this)")
    print("=" * 78)
    diag = diagnosis_table(train)
    print(f"fields per reference report          : {diag.n_fields.mean():.2f}")
    print(f"fields that DIFFER from the template : {diag.n_changed.mean():.2f}"
          f"  ({100 * diag.frac_changed.mean():.0f}% of all fields)")
    print(f"reports with zero changed fields     : {(diag.n_changed == 0).sum()}")
    print(f"IMPRESSION >80% dictation vocabulary : "
          f"{(diag.impression_overlap_with_dictation > 0.8).sum()}")
    print(f"IMPRESSION identical to template     : "
          f"{diag.impression_equals_template.sum()}")

    combos = ([(s, g) for s in (0, 1, 3) for g in ("off", "numbers", "strict")]
              if args.variants else [(args.n_shots, args.guard)])

    print("\n" + "=" * 78)
    print("PIPELINE")
    print("=" * 78)
    results, best = [], None
    for n_shots, guard in combos:
        if n_shots > len(pool):
            continue
        preds, res = run_and_score(llm, dev, retriever, pool, n_shots=n_shots,
                                   guard_level=guard, cache_path=CACHE_PATH,
                                   progress=len(combos) == 1,
                                   n_samples=args.n_samples)
        row = {"n_shots": n_shots, "guard": guard, "n_samples": args.n_samples,
               "RES": res["RES"], "F": res["F"], "I": res["I"],
               "seconds": round(res["seconds"], 1)}
        results.append(row)
        print(f"n_shots={n_shots}  guard={guard:8s}  N={args.n_samples} -> "
              f"RES={res['RES']:.4f}  F={res['F']:.4f}  I={res['I']:.4f}  "
              f"({res['seconds']:.0f}s)")
        if best is None or res["RES"] < best[0]:
            best = (res["RES"], n_shots, guard, preds, res)

    if len(results) > 1:
        print("\n" + pd.DataFrame(results).sort_values("RES").to_string(index=False))

    assert best is not None
    res_best, n_shots, guard, preds, res = best
    oracle = score_predictions(dev, dev.report)["RES"]
    base = baselines(dev).iloc[1]["RES"]
    print(f"\nBEST: n_shots={n_shots} guard={guard}")
    print(f"  RES {res_best:.4f}   vs template-unedited {base:.4f}   "
          f"vs oracle {oracle:.4f}")
    print(f"  closed {100 * (base - res_best) / max(1e-9, base - oracle):.0f}% "
          f"of the gap to the oracle")

    print("\n" + "=" * 78)
    print("WHERE THE REMAINING FINDINGS LOSS IS")
    print("=" * 78)
    loss = field_loss_table(res)
    print(loss.head(12).round(4).to_string())
    if len(loss):
        print(f"\ntop 10 labels = {loss.share.head(10).sum():.0%} of total F loss")

    if args.n_errors:
        print("\n" + "=" * 78)
        print("WORST CASES")
        print("=" * 78)
        per = per_case(res, dev)
        for i in per.sort_values(ascending=False).index[: args.n_errors]:
            r = dev.loc[i]
            print("=" * 90)
            print(f"RES={per[i]:.3f}  {r.body_part}/{r.modality}  "
                  f"{r.study_description}  age {r.patient_age_band} {r.patient_sex}")
            print("--- DICTATION ---\n" + str(r.dictation)[:500])
            print("--- PREDICTION ---\n" + preds[i][:700])
            print("--- REFERENCE ---\n" + str(r.report)[:700])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"provider": llm.provider, "model": llm.model,
                   "n_dev": args.n_dev, "variants": results,
                   "best": {"RES": res_best, "n_shots": n_shots,
                            "guard": guard}}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
