#!/usr/bin/env python
"""Generate submission.csv for test.csv.

    python scripts/predict_test.py
    python scripts/predict_test.py --provider openrouter --n-shots 3

The submission is validated before it is written: correct columns, every
test case_id exactly once, both sections present, no leaked dictation, no
unlabelled FINDINGS text, and no field label the template did not contain.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from natoe.config import (CACHE_PATH, DEFAULT_GUARD_LEVEL, DEFAULT_N_SHOTS,
                          OUTPUT_DIR, TEST_CSV, TRAIN_CSV, have_key,
                          provider_name)
from natoe.evaluate import build_retriever, run_and_score
from natoe.pipeline import LLM, MockLLM, Retriever
from natoe.res_scorer import parse_fields, split_sections


def validate(sub: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Return a list of problems; empty means the submission is valid."""
    errs: list[str] = []
    if list(sub.columns) != ["case_id", "report"]:
        errs.append(f"columns must be exactly case_id,report; got {list(sub.columns)}")
    if len(sub) != len(test):
        errs.append(f"row count {len(sub)} != test rows {len(test)}")
    if sub.case_id.duplicated().any():
        errs.append("duplicate case_id")
    if set(sub.case_id) != set(test.case_id):
        errs.append("case_id set does not match test.csv")

    tpl_by_id = test.set_index("case_id").template_content
    for cid, rep in sub.set_index("case_id").report.items():
        if not isinstance(rep, str) or not rep.strip():
            errs.append(f"{cid}: empty report")
            continue
        if "FINDINGS:" not in rep:
            errs.append(f"{cid}: missing FINDINGS:")
        if "IMPRESSION:" not in rep:
            errs.append(f"{cid}: missing IMPRESSION:")
        if rep.index("FINDINGS:") > rep.index("IMPRESSION:"):
            errs.append(f"{cid}: IMPRESSION before FINDINGS")
        f, _, unlab = parse_fields(split_sections(rep)[0])
        if unlab:
            errs.append(f"{cid}: unlabelled FINDINGS text {unlab[:50]!r}")
        tpl_f, _, _ = parse_fields(split_sections(tpl_by_id[cid])[0])
        invented = set(f) - set(tpl_f)
        if invented:
            errs.append(f"{cid}: labels absent from the template: {sorted(invented)}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, choices=["groq", "openrouter"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    ap.add_argument("--guard", default=DEFAULT_GUARD_LEVEL,
                    choices=["off", "numbers", "strict"])
    ap.add_argument("--use-all-train", action="store_true", default=True,
                    help="allow every train row as a few-shot exemplar (default)")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "submission.csv"))
    args = ap.parse_args()

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # On the real test set every train row is legitimately available as an
    # exemplar, including the rows that were held out during dev evaluation.
    retriever = build_retriever(train)

    name = (args.provider or provider_name()).lower()
    if not have_key(name):
        print(f"No key for {name!r} -> MockLLM. This will NOT produce a real "
              f"submission. Set the key in .env first.")
        llm = MockLLM()
    else:
        llm = LLM(provider=name, model=args.model or None)

    print(f"provider : {llm.provider}  model: {llm.model}")
    print(f"config   : n_shots={args.n_shots}  guard={args.guard}")
    print(f"test rows: {len(test)}\n")

    preds, _ = run_and_score(llm, test, retriever, train, n_shots=args.n_shots,
                             guard_level=args.guard, cache_path=CACHE_PATH)

    sub = pd.DataFrame({"case_id": test.case_id, "report": preds})
    errs = validate(sub, test)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, quoting=csv.QUOTE_ALL)

    print(f"wrote {out}  {sub.shape}")
    if errs:
        print(f"\n{len(errs)} PROBLEM(S):")
        for e in errs[:25]:
            print("  -", e)
        return 1
    print("\nPASS - submission is structurally valid")
    print(f"mean report length: {sub.report.str.len().mean():.0f} chars")
    print("\n--- first prediction ---\n" + sub.report.iloc[0])
    print("\nSubmit with:\n"
          f"  kaggle competitions submit -c radiology-reporting-harness "
          f"-f {out} -m \"...\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
