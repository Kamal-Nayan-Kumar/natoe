"""
Regenerates notebooks/natoe_pipeline.ipynb.

Every module under src/natoe/ is embedded as a `%%writefile` cell, so the
notebook is a single self-contained artefact: running it rewrites the package
on disk and then imports it. That is what makes it reviewable on Kaggle
without attaching files, and it keeps the notebook and the library from
drifting apart.

    python scripts/build_notebook.py
"""
from pathlib import Path


import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "natoe"
OUT = ROOT / "notebooks" / "natoe_pipeline.ipynb"

MODULES = ["__init__.py", "res_scorer.py", "pipeline.py", "config.py",
           "evaluate.py", "impression.py"]

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


def dcode(src: str, why: str = "") -> None:
    """A code cell that needs the competition CSVs.

    Wraps the body in a guard so that running the notebook on Kaggle *without*
    the data attached prints one clear message per cell instead of cascading
    into `NameError: name 'TRAIN' is not defined` a dozen times. A notebook
    that is a review artefact should stay readable even when it cannot do the
    work.
    """
    body = src.strip("\n")
    indented = "\n".join(("    " + ln) if ln.strip() else ln
                         for ln in body.split("\n"))
    msg = why or ("SKIPPED: the competition CSVs are not available. "
                  "Attach the competition under Add Input, or upload "
                  "train.csv / test.csv and set NATOE_DATA_DIR.")
    code(f"if not HAVE_DATA:\n    print({msg!r})\nelse:\n{indented}")


def writefile(rel: str, src: str) -> None:
    code(f"%%writefile {rel}\n{src.strip()}")


# ------------------------------------------------------------------ 0. title
md(r"""
# Natoe Radiology Edit — dictation → structured report

Turn a radiologist's telegraphic **dictation** into a complete structured
report by **editing a supplied normal template**.

| | |
|---|---|
| train | 636 rows, 9 cols (8 inputs + `report`) |
| test | 132 rows, 8 cols |
| metric | `RES_case = 0.65·F + 0.35·I`, lower is better |

## The one idea behind the design

RES is a **template-edit** metric, not a semantic-similarity metric. It
compares each FINDINGS field of the submission against the same field of the
reference, and weights fields that had to change **3×** more than fields that
stayed normal. So the winning move is: *change what the dictation requires,
copy everything else byte-for-byte.*

That argues against asking a model to "write the report". Instead:

```
 retrieve few-shot  ->  LLM emits a JSON *patch*  ->  guard  ->  render
  (house style)        (only the fields that         (drop        (Python rebuilds
                        change + IMPRESSION)          invented      in the template's
                                                       measurements)   field order)
```

**The model never writes the report. It writes a patch. Python renders it.**
Field order, label spelling and untouched-field preservation then become
structurally impossible to get wrong, not merely unlikely.
""")

# ------------------------------------------------------------------ 1. setup
md(r"""
## 1. Setup — secrets

API keys come from the environment, **never** from this notebook.
""")

code(r"""
import os, sys, json, re, time, statistics
from pathlib import Path

import pandas as pd          # every data cell below relies on this

HERE = Path.cwd()
DATA = HERE / "data" if (HERE / "data").exists() else HERE

# --- secrets: local .env, or Kaggle Secrets ----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# On Kaggle: attach the key under Settings -> Secrets, named exactly
# OPENCODE_API_KEY (the provider that produced the shipped submission).
# It is picked up here automatically. If no secret is attached the notebook
# still runs end to end on the offline MockLLM, which exercises the whole
# pipeline with no API call -- useful for a reviewer who has no key.
try:
    from kaggle_secrets import UserSecretsClient
    _ks = UserSecretsClient()
    for _k in ("OPENCODE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        if not os.environ.get(_k):
            try:
                os.environ[_k] = _ks.get_secret(_k)
            except Exception:
                pass
except Exception:
    pass

print("provider:", os.environ.get("LLM_PROVIDER", "opencode"))
for k in ("OPENCODE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
    v = os.getenv(k)
    print(f"{k:20s} {'set' if v else '-- not set --'}"
          + (f"  ({v[:6]}...)" if v else ""))
print("data dir:", DATA)
print("csvs:", sorted(p.name for p in DATA.glob('*.csv')))
""")

# ---------------------------------------------------------- 2. the pipeline
md(r"""
## 2. The pipeline code

Four modules, embedded here in full so the notebook stands alone. Running this
section rewrites them onto disk and imports them.

| module | role |
|---|---|
| `__init__.py` | package marker (required, or `import natoe` is a namespace package) |
| `res_scorer.py` | local re-implementation of the RES metric |
| `pipeline.py` | retriever, prompt, JSON patch, guard, renderer |
| `config.py` | paths, provider table, defaults |
| `evaluate.py` | dev split, scoring, error analysis |
| `impression.py` | IMPRESSION strategy (35% of the metric) |
""")

code(r'''
# IPython's %%writefile does NOT create parent directories, and on Kaggle
# there is no src/ tree yet. Create it first or the next cells die with
# FileNotFoundError.
import os
os.makedirs("src/natoe", exist_ok=True)
print("wrote to:", os.path.abspath("src/natoe"))
''')

for name in MODULES:
    writefile(f"src/natoe/{name}", (PKG / name).read_text())

code(r"""
import importlib, sys
from pathlib import Path

sys.path.insert(0, "src")
for m in ("natoe", "natoe.res_scorer", "natoe.config", "natoe.impression",
          "natoe.pipeline", "natoe.evaluate"):
    if m in sys.modules:
        importlib.reload(sys.modules[m])
    else:
        importlib.import_module(m)

import natoe
from natoe import config as C
from natoe.res_scorer import (res_dataset, res_case, split_sections,
                              parse_fields, normalize, tokenize,
                              weighted_word_edit)
from natoe.pipeline import (LLM, MockLLM, Retriever, build_shots,
                            build_user_prompt, gold_patch, guard, render,
                            parse_json, gold_json, run_dataset)
from natoe.evaluate import (dev_split, build_retriever, baselines,
                            score_predictions, run_and_score, per_case,
                            field_loss_table, diagnosis_table)
print("natoe", getattr(natoe, "__version__", "?"), "| provider:",
      C.provider_name(), "| model:", C.model_name())
""")

# ------------------------------------------------------------------ 3. data
md(r"""
## 3. The data
""")

code(r'''
# The competition CSVs are not in the repository. On Kaggle attach the
# competition to this notebook, or upload them to a dataset and point
# natoe.config.DATA_DIR at it. Without them the data-dependent cells below
# are skipped rather than crashing the run.
HAVE_DATA = C.TRAIN_CSV.exists() and C.TEST_CSV.exists()
print("data available:", HAVE_DATA, "|", C.DATA_DIR)
if not HAVE_DATA:
    print("SKIPPING: expected", C.TRAIN_CSV, "and", C.TEST_CSV)
else:
    TRAIN = pd.read_csv(C.TRAIN_CSV)
    TEST  = pd.read_csv(C.TEST_CSV)
    SAMPLE = pd.read_csv(C.SAMPLE_SUBMISSION_CSV)
    print("train", TRAIN.shape, "| test", TEST.shape, "| sample", SAMPLE.shape)
    print("train cols:", list(TRAIN.columns))
    print("test  cols:", list(TEST.columns))
    display(TRAIN["modality"].value_counts().to_frame("train")
            .join(TEST["modality"].value_counts().to_frame("test"))
            .fillna(0).astype(int))
''')

md(r"""
### One worked example, so the shape of the data is concrete
""")

dcode(r"""
_ex = TRAIN[(TRAIN.body_part == "Chest") & (TRAIN.modality == "XRAY")].iloc[0]
print(f"STUDY: {_ex.study_description} | age {_ex.patient_age_band} | {_ex.patient_sex}")
print("\n--- DICTATION ---\n" + _ex.dictation)
print("\n--- TEMPLATE (the normal report we must edit) ---\n" + _ex.template_content)
print("\n--- REFERENCE REPORT ---\n" + _ex.report)
""")

md(r"""
Note the house style: the dictated abnormality (`Mild thoracic spondylosis`) is
**prepended** to the still-true template sentence inside `BONES`, the untouched
fields are copied verbatim, and the IMPRESSION became a numbered list. This is
not guessable from the rules alone — which is why few-shot exemplars are
retrieved from `train.csv` rather than hand-written.
""")

# ---------------------------------------------------------------- 4. metric
md(r"""
## 4. The metric, re-implemented

`RES_case = 0.65·F + 0.35·I`

* **`I`** — weighted word edit distance over the whole IMPRESSION section.
* **`F`** — field-aware FINDINGS score. For each reference field, first ask
  *did this field change from the template?* changed → weight **3**,
  unchanged → weight **1**. Then `F = Σ(w × word_edit) / Σ(w)`.

Token weights: clinically critical words / numbers / units = **4.0**, other
content words = **2.0**, function words = **0.25**. Each field's cost is divided
by the larger total token weight of reference vs submission, so it lands in
`[0, 1]`.

**What it rewards**

1. Leaving normal fields untouched — weight 1 *and* free to match exactly.
2. Copying the radiologist's words instead of paraphrasing — a synonym is a
   weight-2 substitution; a changed measurement is weight-4.
3. Correct field **routing** — a finding under the wrong label counts as
   missing from the right field *and* as extra content under the wrong one.
4. Not inventing findings. Unsupported content is pure penalty.

> The exact private scorer may differ in the unspecified corners (the
> function-word list, how "unexpected field" penalties are quantified). Use
> this for **relative** comparison between variants.
""")

code(r'''
_ref = """FINDINGS:
LUNGS: Mild right basilar airspace opacity. No pulmonary edema.
PLEURA: Small right pleural effusion. No pneumothorax.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
Mild right basilar airspace opacity and small right pleural effusion."""

_tpl = """FINDINGS:
LUNGS: No focal airspace opacity or pulmonary edema.
PLEURA: No pleural effusion or pneumothorax.
OSSEOUS STRUCTURES: No acute osseous abnormality identified on these views.

IMPRESSION:
No acute cardiopulmonary abnormality."""

for name, sub in [
    ("perfect submission", _ref),
    ("template left unedited", _tpl),
    ("effusion filed under LUNGS",
     _ref.replace("PLEURA: Small right pleural effusion. No pneumothorax.",
                  "LUNGS: Small right pleural effusion. No pneumothorax.")),
]:
    r = res_case(_ref, sub, _tpl)
    print(f"{name:28s} RES={r['RES']:.3f}  (F={r['F']:.3f}, I={r['I']:.3f})")

print()
print("air-space == airspace :", tokenize("air-space") == tokenize("airspace"))
print("8 mm == 8 millimetres :",
      weighted_word_edit("8 mm nodule", "8 millimetres nodule") == 0.0)
print("L5-S1 == l5s1         :", tokenize("L5-S1") == tokenize("l5s1"))
''')

# ------------------------------------------------------------- 5. baselines
md(r"""
## 5. Baselines

Measure the dumb options before trying to be clever.
""")

dcode(r"""
print("ON THE FULL TRAIN SET")
print(baselines(TRAIN).to_string(index=False))
print("\n'B1' is the number every real submission has to beat.")
""")

# ---------------------------------------------------------------- 6. design
md(r"""
## 6. What the reference reports actually look like

These measurements determined the design of the prompt and the renderer.
""")

dcode(r"""
diag = diagnosis_table(TRAIN)
print(f"fields per reference report          : {diag.n_fields.mean():.2f}")
print(f"fields that DIFFER from the template : {diag.n_changed.mean():.2f}"
      f"  ({100 * diag.frac_changed.mean():.0f}% of all fields)")
print(f"reports with zero changed fields     : {(diag.n_changed == 0).sum()}")
print(f"IMPRESSION >80% dictation vocabulary : "
      f"{(diag.impression_overlap_with_dictation > 0.8).sum()} / {len(diag)}")
print(f"IMPRESSION 40-80%                    : "
      f"{((diag.impression_overlap_with_dictation > 0.4) & (diag.impression_overlap_with_dictation <= 0.8)).sum()}")
print(f"IMPRESSION identical to the template : "
      f"{diag.impression_equals_template.sum()} / {len(diag)}")
""")

dcode(r"""
# Where does a CHANGED field's wording come from: the dictation or the template?
from collections import Counter
src = Counter()
for _, r in TRAIN.iterrows():
    ref_f, _, _ = parse_fields(split_sections(r.report)[0])
    tpl_f, _, _ = parse_fields(split_sections(r.template_content)[0])
    d_tok, t_tok = set(tokenize(r.dictation)), set(tokenize(r.template_content))
    for k, v in ref_f.items():
        if normalize(v).strip() == normalize(tpl_f.get(k, "")).strip():
            continue
        v_tok = set(tokenize(v))
        nd, nt = len(v_tok & d_tok), len(v_tok & t_tok)
        src["from_dictation" if nd > nt else
            ("from_template" if nt > nd else "mixed")] += 1
tot = sum(src.values())
for k, v in src.most_common():
    print(f"  {k:15s} {v:>5}  ({100 * v / tot:.0f}%)")
""")

md(r"""
### Conclusions that shape the design

1. **~63% of fields change.** A "minimal edit" instinct is wrong. The
   instruction is *change every field the dictation speaks to, and nothing
   else* — not *touch as little as possible*.
2. **Changed fields copy the radiologist's words** (76% reuse dictation
   vocabulary). So: **copy verbatim, never paraphrase.**
3. **IMPRESSION is lifted from the dictation** in 60% of cases. The dictation
   usually already contains the radiologist's own summary; we should be
   *selecting* it, not writing it.
4. **Field order follows the template.** The renderer must always emit the
   template's labels in the template's order.
""")

# ---------------------------------------------------------------- 7. ceiling
md(r"""
## 7. Does the architecture even work? (ceiling test)

Before spending a single API call: feed the renderer the **gold** patch — the
exact per-field edits the reference made — and score it. If that is not ≈0,
the assembly is broken and no amount of prompting will save us.
""")

dcode(r"""
rows = []
for level in ("off", "numbers", "strict"):
    outs = []
    for _, r in TRAIN.iterrows():
        case = {"template_content": r.template_content, "dictation": r.dictation}
        outs.append(render(case, guard(gold_patch(r), case, level)))
    res = score_predictions(TRAIN, outs)
    rows.append({"guard level": level, "RES": res["RES"],
                 "F": res["F"], "I": res["I"]})
display(pd.DataFrame(rows))
""")

md(r"""
Reading:

* **`off` → 0.0023.** The renderer reproduces the reference almost exactly, so
  field order, label spelling and untouched-field copying are all correct. The
  architecture is sound and the LLM only has to get the *content* right.
* **`numbers` → 0.0030.** Dropping sentences that introduce a measurement found
  in neither the dictation nor the template costs 0.001. Cheap insurance.
* **`strict` → 0.0154.** Also policing laterality/severity starts deleting text
  the reference legitimately contains.

=> ship **`guard="numbers"`**.

An earlier version of this guard also policed laterality and severity and cost
**0.19 RES**: the reference reports themselves add laterality the dictation
never stated, so "unsupported laterality" is not a safe thing to strip.
Negation is never a trigger either — templates legitimately say
"No pneumothorax." when the dictation never mentioned pneumothorax.
""")

# ------------------------------------------------------------- 8. few-shot
md(r"""
## 8. Few-shot exemplars, retrieved

`train.csv` is the style manual. For each case we retrieve the most similar
training rows (body part + modality + template + dictation) and show the model
their template → dictation → **gold patch** triple.

Similarity is TF-IDF cosine over normalised tokens, implemented in
`pipeline.Retriever` so the notebook has no extra dependency.
""")

dcode(r"""
# Sized as a demonstration, not a benchmark: the TF-IDF retriever and the
# offline MockLLM are pure Python and slow down over a few hundred rows. The
# full 300-row evaluation behind the numbers quoted in the README is
#   python scripts/run_dev.py --n-dev 300 --n-shots 3 --guard numbers
NB_DEV_N = int(os.getenv("NATOE_NB_DEV", "60"))
DEV, POOL = dev_split(TRAIN, n_dev=NB_DEV_N)
print(f"dev {len(DEV)} rows (scored) | few-shot pool {len(POOL)} rows")
print("exemplars never include the row being scored, so the dev number is honest.")

RETRIEVER = build_retriever(POOL)
_probe = dict(DEV.iloc[0])
_probe_query = (f"{_probe['body_part']} {_probe['modality']} "
                f"{_probe['study_description']} {_probe['template_content']} "
                f"{_probe['dictation']}")
print(f"\nprobe case: {_probe['body_part']} / {_probe['modality']} | "
      f"{_probe['study_description']}")
for i in RETRIEVER.query(_probe_query, top_k=3):
    print(f"  -> {POOL.body_part[i]:14s} {POOL.modality[i]:5s} | "
          f"{str(POOL.dictation[i])[:70]!r}")
""")

dcode(r"""
_shots = build_shots(RETRIEVER, POOL, _probe, n_shots=1)
_prompt = build_user_prompt(_probe, _shots)
print(f"prompt with 1 shot: ~{len(_prompt) // 4} tokens\n")
print(_prompt[-2200:])
""")

# ------------------------------------------------------------------ 9. LLM
md(r"""
## 9. The model

Both providers speak the OpenAI chat API, so one client covers both and
switching is a one-line change. Free tiers are enough — 132 test rows.

| provider | endpoint | default model |
|---|---|---|
| `groq` | `api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| `openrouter` | `openrouter.ai/api/v1` | `google/gemma-4-31b-it:free` |

Two operational details that mattered a lot in practice:

* **`reasoning_effort=low`** — reasoning models spend most of their output
  budget on chain-of-thought. This task does not need it, and it is a ~4×
  speed difference (13.6s → 3.3s per call).
* **low concurrency** — free tiers rate-limit hard. `workers=4` with
  exponential backoff; failures are surfaced, never cached.
""")

code(r"""
if C.have_key():
    LLM_RUNNER = LLM(provider=C.provider_name(), model=os.getenv("LLM_MODEL") or None)
    print("provider:", LLM_RUNNER.provider, "| model:", LLM_RUNNER.model)
    print("chain   :", LLM_RUNNER.chain)
else:
    LLM_RUNNER = MockLLM()
    print("No API key found -> MockLLM.")
    print("The plumbing runs, but the score below is MEANINGLESS.")
    print("Add GROQ_API_KEY or OPENROUTER_API_KEY to .env and re-run.")
""")

dcode(r'''
# one row, end to end, so the contract is visible
_row = DEV.iloc[1].to_dict()          # .to_dict(), not dict(): attribute access
_case = {"template_content": _row["template_content"],
         "dictation": _row["dictation"],
         "modality": _row["modality"], "body_part": _row["body_part"],
         "study_description": _row["study_description"],
         "patient_age_band": _row["patient_age_band"],
         "patient_sex": _row["patient_sex"]}
_raw = LLM_RUNNER.complete(build_user_prompt(_case, build_shots(RETRIEVER, POOL, _case, 2)))
print("RAW MODEL OUTPUT (truncated):\n" + _raw[:600])
print("\n--- RENDERED REPORT ---\n" + render(_case, guard(parse_json(_raw), _case, "numbers")))
print("\n--- REFERENCE ---\n" + _row["report"])
''')

# ----------------------------------------------------------------- 10. run
md(r"""
## 10. Run on the dev split and score
""")

dcode(r"""
dev_preds, dev_res = run_and_score(LLM_RUNNER, DEV, RETRIEVER, POOL,
                                   n_shots=C.DEFAULT_N_SHOTS,
                                   guard_level=C.DEFAULT_GUARD_LEVEL,
                                   cache_path=C.CACHE_PATH)
print(f"\nDEV RES = {dev_res['RES']:.4f}   F = {dev_res['F']:.4f}   "
      f"I = {dev_res['I']:.4f}   ({dev_res['seconds']:.0f}s)")

_b = baselines(DEV)
print("\n" + _b.to_string(index=False))
print(f"\n" + pd.DataFrame([{
    "submission": f"pipeline ({C.DEFAULT_N_SHOTS} shots, "
                  f"guard={C.DEFAULT_GUARD_LEVEL})",
    "RES": dev_res["RES"]}]).to_string(index=False))
""")

md(r"""
The headline number to watch is **how much of the gap between "template
unedited" and the oracle we closed**. That, not the absolute RES, is what tells
us whether a change helped.
""")

dcode(r"""
_b = baselines(DEV)
base, oracle = float(_b.iloc[1]["RES"]), 0.0
print(f"template unedited : {base:.4f}")
print(f"oracle            : {oracle:.4f}")
print(f"pipeline          : {dev_res['RES']:.4f}")
print(f"\nclosed {100 * (base - dev_res['RES']) / max(1e-9, base - oracle):.0f}% "
      f"of the gap to the oracle")
""")

# -------------------------------------------------------------- 11. errors
md(r"""
## 11. Error analysis

RES is an average, so the mean hides which cases fail. The worst cases are
where the remaining work is.
""")

dcode(r"""
per = per_case(dev_res, DEV)
print(per.describe(percentiles=[.25, .5, .75, .9, .95]).round(3).to_string())
""")

dcode(r"""
loss = field_loss_table(dev_res)
display(loss.head(15).round(4))
if len(loss):
    print(f"top 10 field labels = {loss.share.head(10).sum():.0%} of total FINDINGS loss")
""")

dcode(r"""
for i in per.sort_values(ascending=False).index[:3]:
    r = DEV.loc[i]
    print("=" * 96)
    print(f"RES={per[i]:.3f}  {r.body_part}/{r.modality}  {r.study_description}"
          f"  age {r.patient_age_band} {r.patient_sex}")
    print("--- DICTATION ---\n" + str(r.dictation)[:420])
    print("--- PREDICTION ---\n" + dev_preds[i][:620])
    print("--- REFERENCE ---\n" + str(r.report)[:620])
""")

# ------------------------------------------------------------- 12. iterate
md(r"""
## 12. Iterate

Three knobs. All cheap, because responses are cached on disk by
`(model, guard_level, n_shots, prompt)` — only genuinely new prompts cost a call.

| knob | trades |
|---|---|
| `n_shots` 0 → 1 → 3 | prompt tokens vs. house-style fidelity |
| `guard_level` off / numbers / strict | measured in §7; `numbers` is nearly free |
| provider / model | instruction-following vs. cost and rate limits |

The knob that usually matters most is the **prompt**: if error analysis shows
routing mistakes, the fix belongs in the rules of the system prompt, not in a
bigger model.
""")

dcode(r'''
# A compact sweep so the notebook stays a readable demonstration. The full
# 300-row evaluation behind these numbers lives in
#   python scripts/run_dev.py --n-dev 300 --n-shots 3 --guard numbers
variants = []
SUBSET = DEV.head(60)
for n_shots, level in ((0, "numbers"), (1, "numbers"), (3, "numbers"),
                       (3, "strict")):
    _, res = run_and_score(LLM_RUNNER, SUBSET, RETRIEVER, POOL,
                           n_shots=n_shots, guard_level=level,
                           cache_path=C.CACHE_PATH, progress=False)
    variants.append({"n_shots": n_shots, "guard": level,
                     "RES": res["RES"], "F": res["F"], "I": res["I"]})
    print(f"n_shots={n_shots} guard={level:8s} -> RES={res['RES']:.4f} "
          f"F={res['F']:.4f} I={res['I']:.4f}")

VARIANTS = pd.DataFrame(variants).sort_values("RES")
display(VARIANTS)
print("\nExpected ordering: more shots help; guard level is near-neutral.")
print("This is a 60-row subset, so only differences above ~0.02 mean anything.")
BEST = VARIANTS.iloc[0]
print(f"\nbest: n_shots={int(BEST.n_shots)} guard={BEST.guard}")
''')

# ---------------------------------------------------------------- 13. test
md(r"""
## 13. Predict the test set and write `submission.csv`

For the real test set *every* train row is legitimately available as a
few-shot exemplar, including the ones held out during dev evaluation.
""")

dcode(r"""
N_SHOTS, LEVEL = int(BEST.n_shots), BEST.guard
TEST_RETRIEVER = build_retriever(TRAIN)      # full train set as the pool

# run_dataset, not run_and_score: test.csv has no `report` column, so there is
# nothing to score against.
test_preds = run_dataset(LLM_RUNNER, TEST, TEST_RETRIEVER, TRAIN,
                         n_shots=N_SHOTS, guard_level=LEVEL,
                         cache_path=C.CACHE_PATH)

SUB = pd.DataFrame({"case_id": TEST.case_id, "report": test_preds})
import csv as _csv
OUT_PATH = C.OUTPUT_DIR / "submission.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SUB.to_csv(OUT_PATH, index=False, quoting=_csv.QUOTE_ALL)
print(f"wrote {OUT_PATH}  {SUB.shape}   (n_shots={N_SHOTS}, guard={LEVEL})")
""")

dcode(r"""
import csv

def validate(sub, test):
    errs = []
    if list(sub.columns) != ["case_id", "report"]:
        errs.append(f"columns must be exactly case_id,report; got {list(sub.columns)}")
    if len(sub) != len(test):
        errs.append(f"row count {len(sub)} != {len(test)}")
    if sub.case_id.duplicated().any():
        errs.append("duplicate case_id")
    if set(sub.case_id) != set(test.case_id):
        errs.append("case_id set does not match test.csv")

    tpl_by_id = test.set_index("case_id").template_content
    for cid, rep in sub.set_index("case_id").report.items():
        if not isinstance(rep, str) or not rep.strip():
            errs.append(f"{cid}: empty report"); continue
        if "FINDINGS:" not in rep:   errs.append(f"{cid}: missing FINDINGS:")
        if "IMPRESSION:" not in rep: errs.append(f"{cid}: missing IMPRESSION:")
        if rep.index("FINDINGS:") > rep.index("IMPRESSION:"):
            errs.append(f"{cid}: IMPRESSION before FINDINGS")
        f, _, unlab = parse_fields(split_sections(rep)[0])
        if unlab:
            errs.append(f"{cid}: unlabelled FINDINGS text {unlab[:50]!r}")
        tpl_f, _, _ = parse_fields(split_sections(tpl_by_id[cid])[0])
        if set(f) - set(tpl_f):
            errs.append(f"{cid}: labels absent from the template: "
                        f"{sorted(set(f) - set(tpl_f))}")
    return errs

errs = validate(SUB, TEST)
print("PASS - submission is structurally valid" if not errs
      else f"{len(errs)} PROBLEM(S):")
for e in errs[:20]:
    print("  -", e)

print(f"\nmean report length: {SUB.report.str.len().mean():.0f} chars")
print("\n--- first prediction ---\n" + SUB.report.iloc[0])
""")

dcode(r"""
print("Submit:\n"
      "  kaggle competitions submit -c radiology-reporting-harness \\\n"
      f"      -f {OUT_PATH} -m \"template-edit pipeline: LLM JSON patch + deterministic render\"\n\n"
      "With the pipeline notebook attached (paste the notebook URL into the\n"
      "submission description on the Kaggle page):\n"
      "  kaggle competitions submit -c radiology-reporting-harness \\\n"
      f"      -f {OUT_PATH} -k kamalnayan90/<NOTEBOOK> -v <VERSION> -m \"...\"")
""")

# ---------------------------------------------------------------- 14. notes
md(r"""
## 14. What is left on the table, and the fine-tuning path

The prompt pipeline is the right **first** system: cheap, auditable, and every
failure has a named cause. But the metric rewards verbatim copying, and no
amount of prompting fully removes paraphrasing. That is the natural place for
stage 2.

**Stage 2 — synthetic-data alignment**

1. From every train row, *derive* a dictation from the reference report (or
   forward-generate one), producing `(dictation, template, report)` triples.
2. LoRA fine-tune a small open model (e.g. Qwen 2.5 7B/14B) on the JSON-patch
   format. The output space is tiny and highly templated, so a 7B model learns
   it quickly and inference gets far cheaper than the API.
3. Keep the **identical** guard and render code, so the fine-tuned model is a
   drop-in swap and the guardrails still apply.

**Where Jev fits**

Jev (`typesafe/jev-1.13`, ~$0.042/M input, output free) is a *decision* model,
not a text generator — it returns calibrated probabilities for questions you
define. That suits this task precisely, because two of the remaining decisions
are not generation problems at all:

* *"does this dictated finding belong in field X?"* — a `choice` question over
  the template's field labels. This is the routing error we keep seeing.
* *"is this generated sentence supported by the dictation?"* — a `noul`
  question, which is exactly what the current regex guard approximates.

So the natural end state is a **generative model for the wording, Jev for the
decisions**, with the same renderer in front.

**Design decisions worth defending in review**

* *The model never writes the report.* It writes a patch; Python renders. This
  makes field order, label spelling and untouched-field preservation
  structurally impossible to get wrong.
* *The guard is deliberately narrow.* Unsupported **measurements** are the
  defensible thing to strip (cost 0.001). Unsupported laterality is not, because
  the reference itself adds it (cost 0.19).
* *No clinical inference from the context columns.* `modality`, `body_part`,
  `study_description`, `patient_age_band`, `patient_sex` are used only to select
  few-shot exemplars, never as a source of findings.
* *Reproducible.* Temperature 0, on-disk response cache, a single `.env`, and a
  submission validator that must pass before the file is handed over.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT.relative_to(ROOT)}  ({len(cells)} cells)")
