"""Shared evaluation helpers: dev split, scoring, error analysis.

Both the CLI scripts and the notebook call these, so a number reported in the
notebook is produced by exactly the code that produced the submission.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import (DEFAULT_DEV_SIZE, DEFAULT_GUARD_LEVEL, DEFAULT_N_SHOTS,
                     DEV_SEED)
from .pipeline import Retriever, build_shots, run_dataset
from .res_scorer import (res_dataset, split_sections, parse_fields, normalize,
                         tokenize)


# --------------------------------------------------------------------- splits

def dev_split(train: pd.DataFrame, n_dev: int = DEFAULT_DEV_SIZE,
              seed: int = DEV_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `n_dev` rows for scoring; the rest becomes the few-shot pool.

    Few-shot exemplars must never include the row being scored, otherwise the
    dev number is meaningless.
    """
    shuffled = list(pd.Series(train.index).sample(frac=1.0, random_state=seed))
    dev_idx = set(shuffled[:n_dev])
    dev = train.loc[sorted(dev_idx)].reset_index(drop=True)
    pool = train.loc[[i for i in train.index if i not in dev_idx]].reset_index(drop=True)
    return dev, pool


def retrieval_text(frame: pd.DataFrame) -> pd.Series:
    return (frame.body_part + " " + frame.modality + " " + frame.study_description
            + " " + frame.template_content + " " + frame.dictation)


def build_retriever(pool: pd.DataFrame) -> Retriever:
    return Retriever(retrieval_text(pool).tolist())


# ------------------------------------------------------------------ scoring

def score_predictions(frame: pd.DataFrame, preds) -> dict:
    return res_dataset(frame["report"], pd.Series(list(preds), index=frame.index),
                       frame["template_content"])


def baselines(frame: pd.DataFrame) -> pd.DataFrame:
    """Dumb reference points, always worth quoting alongside a real run."""
    rows = [("ORACLE: the reference report", score_predictions(frame, frame.report)["RES"]),
            ("B1: template submitted unedited",
             score_predictions(frame, frame.template_content)["RES"])]

    raw = []
    for _, r in frame.iterrows():
        fb, _, _ = split_sections(r.template_content)
        raw.append(f"FINDINGS:\n{fb.strip()}\n\nIMPRESSION:\n{r.dictation.strip()}")
    rows.append(("B2: template FINDINGS + raw dictation as IMPRESSION",
                 score_predictions(frame, raw)["RES"]))
    return pd.DataFrame(rows, columns=["submission", "RES"]).sort_values("RES")


# ------------------------------------------------------------------ analysis

def per_case(res: dict, frame: pd.DataFrame) -> pd.Series:
    return pd.Series([x["RES"] for x in res["rows"]], index=frame.index)


def field_loss_table(res: dict) -> pd.DataFrame:
    """Where the FINDINGS loss actually is, aggregated by field label."""
    agg: dict[str, dict] = {}
    for case in res["rows"]:
        for label, d in case["fields"].items():
            if d["edit"] <= 0.02:
                continue
            slot = agg.setdefault(d.get("ref_display", label),
                                  {"weighted_loss": 0.0, "missing": 0, "cases": 0})
            slot["weighted_loss"] += d["w"] * d["edit"]
            slot["missing"] += int(bool(d.get("missing")))
            slot["cases"] += 1
    if not agg:
        return pd.DataFrame(columns=["weighted_loss", "share", "missing", "cases"])
    t = pd.DataFrame(agg).T
    t["share"] = t.weighted_loss / t.weighted_loss.sum()
    return t.sort_values("weighted_loss", ascending=False)


def diagnosis_table(train: pd.DataFrame) -> pd.DataFrame:
    """The style statistics that justify the prompt and the renderer."""
    rows = []
    for _, r in train.iterrows():
        ref_f, _, _ = parse_fields(split_sections(r.report)[0])
        tpl_f, _, _ = parse_fields(split_sections(r.template_content)[0])
        changed = [k for k, v in ref_f.items()
                   if normalize(v).strip() != normalize(tpl_f.get(k, "")).strip()]
        ref_im = split_sections(r.report)[1]
        overlap = len(set(tokenize(ref_im)) & set(tokenize(r.dictation))) / \
            max(1, len(set(tokenize(ref_im))))
        rows.append({
            "n_fields": len(ref_f),
            "n_changed": len(changed),
            "frac_changed": len(changed) / max(1, len(ref_f)),
            "impression_overlap_with_dictation": overlap,
            "impression_equals_template":
                ref_im.strip() == split_sections(r.template_content)[1].strip(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- runner

def run_and_score(llm, frame: pd.DataFrame, retriever: Retriever,
                  pool: pd.DataFrame, n_shots: int = DEFAULT_N_SHOTS,
                  guard_level: str = DEFAULT_GUARD_LEVEL,
                  cache_path: str | Path | None = None,
                  progress: bool = True) -> tuple[list[str], dict]:
    t0 = time.time()
    preds = run_dataset(llm, frame, retriever, pool, n_shots=n_shots,
                        guard_level=guard_level,
                        cache_path=str(cache_path) if cache_path else None,
                        progress=progress)
    res = score_predictions(frame, preds)
    res["seconds"] = time.time() - t0
    return preds, res


# ---------------------------------------------------------------- submission

def validate_submission(sub: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Structural checks a submission must pass. Empty list means valid.

    Catches the failure modes that would silently cost leaderboard score or
    break the hiring review: wrong columns, a mismatched case_id set, a
    missing section, leaked dictation text, unlabelled FINDINGS content, and
    any field label the template did not contain.
    """
    errs: list[str] = []
    if list(sub.columns) != ["case_id", "report"]:
        errs.append(f"columns must be exactly case_id,report; "
                    f"got {list(sub.columns)}")
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
            errs.append(f"{cid}: labels absent from the template: "
                        f"{sorted(invented)}")
    return errs
