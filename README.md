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

## Calibration against the real leaderboard

The local scorer is a re-implementation, so it was checked against real Kaggle
submissions. Its accuracy turned out to depend sharply on *what* is being
scored.

| # | submission | local RES | Kaggle public | Kaggle private |
|---|---|---|---|---|
| — | rule-based fallback, no sentence filter | (worse) | 0.64123 | 0.64932 |
| — | rule-based fallback, finding-filtered IMPRESSION | **0.5632** | 0.63136 | 0.65007 |
| — | rule-based fallback, ≤2-token filter | 0.5695 | 0.62616 | 0.64567 |
| 5 | LLM JSON-patch pipeline, 0-shot | 0.4026 | 0.42523 | 0.40149 |
| 6 | + append template IMPRESSION closer | 0.3721 | 0.42283 | 0.39879 |
| 7 | **+ 3-shot, prompt asymmetry, robust JSON parse** | **0.2842** | **0.36066** | **0.33012** |

Three clear conclusions:

1. **The local scorer tracks the real one on FINDINGS-dominated submissions, but
   is consistently optimistic, and the bias grows as the pipeline improves:**
   +0.000 on #5, +0.027 on #6, **+0.046** on #7.
2. **It over-credits IMPRESSION changes by ~10×** — 0.031 rated locally versus
   0.0027 actually delivered.
3. For the hand-written fallback it was off by ~0.08 and **mis-ranked three
   variants in the wrong order**. The disagreement was concentrated entirely in
   the IMPRESSION section, which the fallback fills with raw dictation text.

Practical rule: **iterate locally, submit to confirm, and add ~0.05 when
interpreting a local number.** Local scores are reliable for *ranking* changes —
every change that improved the local score did improve the leaderboard — but
never quote one as a predicted score.

## Status

| | |
|---|---|
| Best leaderboard score | **0.36066** public / **0.33012** private (submission 7) |
| Improvement | 0.64567 → 0.33012 private vs the no-LLM baseline, a **49% reduction** |
| Local dev RES | 0.2842 (60-row dev split; optimistic by ~0.046) |
| Same config, 300-row dev split | 0.3097 — the more trustworthy figure |
| Model | `space-bunny-free` on OpenCode Zen, ~0.43 rows/s, no rate limiting |
| Shipped config | 3 retrieved few-shot exemplars, temperature 0, `guard=numbers` |

### Providers

| provider | availability | notes |
|---|---|---|
| **opencode** | `space-bunny-free` only | Zen's paid models return 402 with no credits; other `-free` ids return 403 "free tier can only be used from within OpenCode" |
| groq | heavily throttled | gpt-oss 429s on ~half of prompt-sized calls; 8000 prompt tokens/min |
| openrouter | `:free` exhausted | daily free-model cap; adding 10 credits raises it |

### Reproducing the submission

```bash
python scripts/predict_test.py --provider opencode --model space-bunny-free \
    --n-shots 3 --guard numbers
```

> **`--n-shots 3` is required.** The script's built-in default is `0`, which does
> **not** reproduce the submitted CSV — the shipped submission used 3 retrieved
> few-shot exemplars, and 3-shot is what took the private score from 0.40149 to
> 0.33012. Pass it explicitly.

Full step-by-step, including data setup, is in
[SOLUTION_DETAILS.md](SOLUTION_DETAILS.md).

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
| **opencode** (default) | `OPENCODE_API_KEY` | `space-bunny-free` | **the provider the shipped submission used.** No rate limiting, ~0.43 rows/s. Zen ids carry no `opencode/` prefix on the wire |
| groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` | **8000 prompt tokens/min** — a 3-shot prompt (~4200 tok) means ~2 calls/min |
| openrouter | `OPENROUTER_API_KEY` | `nvidia/nemotron-3-super-120b-a12b:free` | ~0.6 s/call, honours JSON mode; daily `:free` cap |

All three speak the OpenAI chat API, so one client covers them. Switch with
`LLM_PROVIDER` (or `--provider`).

To reproduce the submission you need **`OPENCODE_API_KEY`** (an `oc_sk_...` key
from your OpenCode Zen account). Note `.env.example` still ships the older
`LLM_PROVIDER=groq` default — set `LLM_PROVIDER=opencode` yourself.

Optional knobs: `LLM_MODEL`, `REASONING_EFFORT` (default `low` — this task
does not need long chains of thought, and it is a 4× speed difference. It is
also a correctness fix: reasoning models otherwise spend the whole output budget
on chain-of-thought and return no `content`.)

> **Rate limits, not model quality, decided the provider.** All three are free
> tiers and rationed differently, and the *shape* of the limit is what matters:
>
> | limit | value | consequence |
> |---|---|---|
> | **opencode Zen** | **none observed** | ~0.43 rows/s, no throttling — this is why the submission used it |
> | Groq requests | 1000 / hour | comfortable |
> | Groq **prompt tokens** | **8000 / minute** | ~7 calls/min at 0-shot, only ~2 at 3-shot — tokens, not calls, are the budget |
> | OpenRouter `:free` | **free-model requests per day** | a hard daily cap; adding 10 credits raises it to 1000/day |
>
> Practical consequences, all of which are in the code:
> * **`n_shots` is a token-budget decision, not a quality one — and which way it
>   cuts depends on the provider.** On Groq, a 3-shot prompt is ~4200 tokens and
>   at 1 shot 6 of 40 dev rows were rate-limited into a silent template
>   fallback, so fewer shots won *there*. On an unlimited provider that
>   argument disappears and the exemplars help: on `space-bunny-free`, 3-shot
>   scored **0.284** against 0-shot's **0.403** on the same 60-row split. The
>   shipped submission is 3-shot. The `n_shots=0` default in `config.py` is
>   still the Groq-era value and should be passed explicitly.
> * `REASONING_EFFORT=low` by default — reasoning models otherwise spend the
>   whole output budget on chain-of-thought and return no `content`.
> * `workers=2` plus a `retries=2` pass over the rows that failed. A 429 storm
>   silently drags the score back toward "template unedited", so `run_dataset`
>   reports every row that exhausted its retries rather than swallowing it.
> * `outputs/llm_cache.json` is keyed by
>   `(model, reasoning_effort, guard_level, n_shots, temperature, system_prompt,
>   prompt)` — the system prompt *must* be in the key, or editing the prompt
>   rules would silently reuse every stale response. Repeated experiments
>   therefore cost nothing. Failures are never cached.

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
python scripts/run_dev.py --provider opencode  --model space-bunny-free
python scripts/run_dev.py --provider groq      --model openai/gpt-oss-120b
python scripts/run_dev.py --provider openrouter --model nvidia/nemotron-3-super-120b-a12b:free

# predict the test set -> outputs/submission.csv (validated before writing).
# --n-shots 3 is what the shipped submission used; the default of 0 does not
# reproduce it.
python scripts/predict_test.py --provider opencode --model space-bunny-free \
    --n-shots 3 --guard numbers

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
  run_dev.py         dev-split scoring + error analysis
  predict_test.py    test predictions + submission validation
  fill_and_submit.py resilient cache-filling + submission (safe to re-run)
  build_notebook.py  regenerates the notebook from the modules
  analysis/          one-off research scripts
  fetch_data.sh      restore data/ from Kaggle
notebooks/       natoe_pipeline.ipynb
results/         committed dev-split scores (see RESULTS.md)
outputs/         llm_cache.json, dev_results.json, submission.csv (gitignored)
SOLUTION_DETAILS.md  step-by-step reproduction, provider, packages, submission
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
`(model, reasoning_effort, guard_level, n_shots, temperature, system_prompt,
prompt)`. Prompt iteration is therefore nearly free — only genuinely new prompts
cost an API call. Failures are never cached.

---

## Measured results

Every dev-split score behind the numbers quoted above is committed as raw JSON
in `results/`, with the caveats (split sizes, cache replays, where the local
scorer disagrees with the private one) written up in [RESULTS.md](RESULTS.md).

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
