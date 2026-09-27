# Natoe — Radiology Report Editing

Convert a radiologist's telegraphic **dictation** into a complete structured
report by **editing a supplied normal template**, for the Natoe AI Dev hiring
challenge on Kaggle (`radiology-reporting-harness`).

Scored by **RES — Radiology Edit Score** (lower is better), a *template-edit
fidelity* metric rather than a semantic-similarity metric.

---

## The idea

RES compares each FINDINGS field of the submission against the same field of
the reference, and weights fields that had to change 3× more than fields that
stayed normal:

```
F = Σ(field_weight × word_edit) / Σ(field_weight)      field_weight ∈ {1, 3}
RES_case = 0.65·F + 0.35·I
```

So the winning move is **change only what the dictation requires, and copy
everything else byte-for-byte**. That argues against asking a model to "write
the report". Instead:

```
 retrieve few-shot   LLM emits a JSON patch    guard          render
  (house style)      (only the fields that      (strip         (Python rebuilds
                       change + IMPRESSION)      invented       in the template's
                                                 measurements)   field order)
```

**The model never writes the report. It writes a patch. Python renders it.**

That split is what keeps the score low and the output auditable: field order,
label spelling and untouched-field preservation are structurally impossible to
get wrong, not merely unlikely.

### Why this shape, from the data

Measured on `train.csv` (see notebook §6):

| observation | value | consequence |
|---|---|---|
| fields per reference report | 5.2 | few fields, so one routing error is expensive |
| fields that differ from the template | **63%** | "minimal edit" is the wrong instinct — rewrite every field the dictation speaks to |
| changed fields reusing dictation vocabulary | **76%** | copy verbatim, never paraphrase (a synonym is a scored substitution) |
| IMPRESSION >80% dictation vocabulary | 60% | the dictation already contains the summary; *select* it, do not write it |
| field order following the template | 90% | the renderer must emit template order, always |

Baseline to beat: submitting the template unedited scores **≈0.62**.
Architecture ceiling (renderer fed the exact gold edits) scores **0.003**.

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Secrets

Keys live in `.env`, never in code and never in the notebook.

```bash
cp .env.example .env     # then fill in
```

| provider | key | default model | notes |
|---|---|---|---|
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` | strongest, but **8000 prompt tokens/min** — a 3-shot prompt (~4200 tok) means ~2 calls/min |
| OpenRouter | `OPENROUTER_API_KEY` | `nvidia/nemotron-3-super-120b:free` | free, ~0.6 s/call, honours JSON mode; gemma-4-31b:free frequently 429s |

Both speak the OpenAI chat API, so one client covers both. Switch with
`LLM_PROVIDER=openrouter` (or `--provider openrouter`).

Optional knobs: `LLM_MODEL`, `REASONING_EFFORT` (default `low` — this task
does not need long chains of thought, and it is a 4× speed difference).

> **Rate limits are the real constraint, not model quality.** Groq allows 1000
> requests/hour but only **8000 prompt tokens/minute**, so few-shot prompting is
> rationed by tokens, not calls. Budget accordingly: measure tokens-per-call
> before choosing `n_shots`, and lean on `outputs/llm_cache.json` so repeated
> experiments are free.

### Data

```bash
./scripts/fetch_data.sh          # or copy the three CSVs into data/
```

`data/*.csv` is gitignored; the CSVs are competition data.

---

## Use

```bash
# score on a held-out dev split of train.csv, with baselines and error analysis
python scripts/run_dev.py --n-shots 3 --guard numbers

# sweep the main knobs
python scripts/run_dev.py --variants

# compare providers
python scripts/run_dev.py --provider groq       --model openai/gpt-oss-120b
python scripts/run_dev.py --provider openrouter --model google/gemma-4-31b-it:free

# predict the test set -> outputs/submission.csv (validated before writing)
python scripts/predict_test.py

# regenerate the self-contained notebook
python scripts/build_notebook.py
```

Without a key the scripts fall back to `MockLLM` so the plumbing can be
smoke-tested; the score it prints is meaningless and the submission script
says so.

Notebook: `notebooks/natoe_pipeline.ipynb` — self-contained (it embeds both
modules as `%%writefile` cells), and is the artefact to share for the hiring
review.

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/natoe_pipeline.ipynb
```

---

## Layout

```
data/            train.csv, test.csv, sample_submission.csv   (gitignored)
src/natoe/
  res_scorer.py  local re-implementation of the RES metric
  pipeline.py    retriever, prompt, JSON patch, guard, renderer
  evaluate.py    dev split, scoring, error analysis
  config.py      paths, provider table, defaults
scripts/
  run_dev.py       dev-split scoring + error analysis
  predict_test.py  test predictions + submission validation
  build_notebook.py  regenerates the notebook from the modules
  analysis/        one-off research scripts
  fetch_data.sh    restore data/ from Kaggle
notebooks/       natoe_pipeline.ipynb
outputs/         llm_cache.json, dev_results.json, submission.csv (gitignored)
```

---

## Guardrails

`guard(patch, case, level)` runs after the model and before rendering:

* **label check** — a field label the template did not contain is discarded, so
  the renderer can never emit an unexpected field.
* **unsupported measurements** — a sentence introducing a number that appears in
  neither the dictation nor that field's own template text is dropped. This is
  the narrowest defensible reading of "do not add findings unsupported by the
  dictation", and it is measured: on gold edits it costs **0.001 RES** while
  catching invented measurements.

| level | behaviour | gold-patch cost |
|---|---|---|
| `off` | no check | 0.0023 |
| `numbers` | measurements only — **default** | 0.0030 |
| `strict` | + laterality / severity | 0.0154 |

An earlier `strict` guard also policed laterality and severity and cost
**0.19 RES**, because the reference reports themselves add laterality the
dictation never stated. Deliberately kept out.

Negation is never a drop trigger: templates legitimately say
"No pneumothorax." when the dictation never mentioned pneumothorax.

---

## Submitting

```bash
# 1. generate + validate
python scripts/predict_test.py

# 2. submit
kaggle competitions submit -c radiology-reporting-harness \
  -f outputs/submission.csv \
  -m "template-edit pipeline: LLM JSON patch + deterministic render"

# with the pipeline notebook attached (paste the notebook URL into the
# submission description on the Kaggle page)
kaggle competitions submit -c radiology-reporting-harness \
  -f outputs/submission.csv -k kamalnayan90/<NOTEBOOK> -v <VERSION> \
  -m "template-edit pipeline: LLM JSON patch + deterministic render"
```

`predict_test.py` refuses to hand over a submission that has the wrong columns,
a mismatched `case_id` set, a missing section, leaked dictation text, unlabelled
FINDINGS content, or a field label absent from the template.

### Caching

Responses are cached to `outputs/llm_cache.json`, keyed by
`(model, guard_level, n_shots, prompt)`. Prompt iteration is therefore nearly
free — only genuinely new prompts cost an API call. Failures are never cached.

---

## Notes and limitations

* The RES implementation is written from the published description. The
  private scorer's exact function-word list and "unexpected field" penalty are
  not published, so treat the absolute numbers as approximate and use them for
  **relative** comparison between pipeline variants.
* `modality`, `body_part`, `study_description`, `patient_age_band` and
  `patient_sex` are used only to select few-shot exemplars — never as a source
  of findings.
* The data is de-identified radiology text and must not be used for clinical
  decisions.
