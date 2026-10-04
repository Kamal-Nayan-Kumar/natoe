# Natoe — Radiology Report Editing: Solution Details

**Submission:** Natoe AI Dev hiring challenge, Kaggle competition `radiology-reporting-harness`
**Best leaderboard score:** 0.36066 public / **0.33012** private (lower is better)
**Code:** https://github.com/Kamal-Nayan-Kumar/natoe
**Author:** Kamal Nayan Kumar

---

## 1. The problem, and the approach

The task is to turn a radiologist's teleographic **dictation** into a complete
structured radiology report, editing a supplied **normal template**.

Scoring is by **RES (Radiology Edit Score)**, lower is better. RES is not a
semantic-similarity metric — it is a **template-edit fidelity** metric. It
compares each FINDINGS field of the submission against the same field of the
reference, and weights fields that had to change **3× more heavily** than fields
that stayed normal:

```
F        = Σ(field_weight × word_edit) / Σ(field_weight)     field_weight ∈ {1, 3}
RES_case = 0.65·F + 0.35·I
```

This metric shape dictates the architecture. Because unchanged fields dominate
the denominator and untouched content is scored too, the winning move is to
**change only what the dictation requires and copy everything else
byte-for-byte**. That argues *against* asking a model to "write the report".

So the model never writes the report. It writes a **patch**, and Python renders it:

```
retrieve few-shot  →  LLM emits a JSON patch  →  guard  →  render
   (house style)       (only the fields that        (strip      (Python rebuilds in
                         change + IMPRESSION)        invented   the template's field
                                                     measurements)  order)
```

The benefit is that correctness becomes structural rather than probabilistic.
Field order, label spelling, and preservation of untouched fields are
*impossible to get wrong*, not merely unlikely. The output is also fully
auditable: every change traces back to a field in the patch.

### What the data said (measured on `train.csv`)

| observation | value | consequence |
|---|---|---|
| fields per reference report | 5.2 | few fields, so one routing error is expensive |
| fields that differ from the template | **63%** | "minimal edit" is the wrong instinct — rewrite every field the dictation speaks to |
| changed fields reusing dictation vocabulary | **76%** | copy verbatim, never paraphrase (a synonym is a scored substitution) |
| IMPRESSION with >80% dictation vocabulary | 60% | the dictation already contains the summary; *select* it, do not write it |
| field order following the template | 90% | the renderer must emit template order, always |

Baselines: submitting the template unedited scores **≈0.62**. An architecture
ceiling — the renderer fed the exact gold edits — scores **0.003**.

---

## 2. AI provider and exact model name

| | |
|---|---|
| **Provider** | **OpenCode Zen** (`https://opencode.ai/zen/v1`, OpenAI-compatible `/chat/completions`) |
| **Exact model id** | **`space-bunny-free`** |
| Auth | `OPENCODE_API_KEY` (an `oc_sk_...` key) |
| Reasoning effort | `low` (via `REASONING_EFFORT`) |
| Temperature | 0 |
| Few-shot | **3 retrieved exemplars** |

This is the configuration that produced the submitted CSV. Model ids on Zen carry
**no `opencode/` prefix** on the wire.

**Why this model, stated honestly.** The account had no credits, so the paid Zen
ids (`gpt-5.5`, `claude-sonnet-5`, `gemini-3.8-flash`) return *402 Insufficient
account balance*, and the other `-free` ids return *403 "free tier can only be
used from within OpenCode"*. `space-bunny-free` was the one id that both worked
from a script and was not rate-limited (0.43 rows/s). So the choice was
availability-driven, not a claim that it is the strongest model available.

Two consequences that are **baked into the code as a result**:

* `REASONING_EFFORT=low` is not a tuning choice, it is a correctness fix.
  Reasoning models otherwise spend the entire output budget on
  chain-of-thought and return no `content` at all — with high effort the model
  returned a bare `{`.
* Low concurrency (`workers=2`) plus a retry pass. A 429 storm silently drags
  the score back toward "template unedited", so rows that exhaust their retries
  are **reported** rather than quietly degraded.

`LLM_PROVIDER` also accepts `groq` (default `openai/gpt-oss-120b`) and
`openrouter` (default `nvidia/nemotron-3-super-120b-a12b:free`), each with a
fallback chain, but neither was rate-limit-free during this work: Groq allows
only **8000 prompt tokens/min**, and OpenRouter's `:free` daily cap was
exhausted.

---

## 3. Libraries and packages used

Runtime requirements are deliberately minimal.

| package | version used | purpose |
|---|---|---|
| Python | 3.12.13 | — |
| pandas | 2.3.3 | CSV I/O |
| numpy | 2.5.3 | the trained router only |
| openai | 3.19.2 | OpenAI-compatible chat client (covers all three providers) |
| python-dotenv | 1.2.3 | loads `.env` |

Notebook only:

| package | version used |
|---|---|
| notebook | 7.6.3 |
| ipykernel | 7.3.0 |
| nbformat | 5.11.1 |
| nbconvert | 7.17.1 |

Development only: `pytest` 9.1.1 (114 tests, all passing).

Submission (optional, CLI only): `kaggle>=1.6`.

**No third-party ML stack.** There is no torch, no transformers, no sklearn, no
sentence-transformers. The few-shot retriever is a hand-rolled dependency-free
TF-IDF cosine similarity, and the field router is a hand-rolled naive Bayes in
numpy. The corpus is a few hundred rows, so a sparse dot product is fast enough
and keeping it dependency-free is what lets the notebook be self-contained.

### Source layout

```
src/natoe/
  res_scorer.py  local re-implementation of the RES metric
  pipeline.py    retriever, prompt, JSON patch, guard, renderer
  impression.py  IMPRESSION strategy
  field_router.py  naive-Bayes sentence -> field router
  evaluate.py    dev split, scoring, error analysis
  config.py      paths, provider table, defaults
scripts/
  predict_test.py    test predictions + submission validation
  run_dev.py         dev-split scoring + error analysis
  fill_and_submit.py resilient cache-filling + submission
  build_notebook.py  regenerates the notebook from the modules
  analysis/          one-off research scripts
  fetch_data.sh      restore data/ from Kaggle
notebooks/
  natoe_pipeline.ipynb   the submitted notebook
```

---

## 4. Steps to run the code and reproduce the CSV

### Prerequisites

Python 3.10+ (developed on 3.12) and an OpenCode Zen API key.

### Step 1 — clone and install

```bash
git clone https://github.com/Kamal-Nayan-Kumar/natoe.git
cd natoe

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Step 2 — add the API key

```bash
cp .env.example .env
```

Then edit `.env` and set:

```
LLM_PROVIDER=opencode
LLM_MODEL=space-bunny-free
OPENCODE_API_KEY=oc_sk_...
REASONING_EFFORT=low
```

The key is read from the environment only. It is never written into code, never
into the notebook, and `.env` is gitignored. (A test asserts the notebook
contains no secrets.)

### Step 3 — get the competition data

```bash
./scripts/fetch_data.sh
```

This places `train.csv`, `test.csv`, and `sample_submission.csv` in `data/`.
The CSVs are competition data and are gitignored; restore them with this script.

### Step 4 — generate the submission CSV

```bash
python scripts/predict_test.py --provider opencode --model space-bunny-free \
    --n-shots 3 --guard numbers
```

This writes `outputs/submission.csv`.

> **`--n-shots 3` matters.** The script's built-in default is `0`, which does
> **not** reproduce the submitted CSV. The shipped submission used 3 retrieved
> few-shot exemplars; passing `--n-shots 3` explicitly is required.

Roughly 6 s per row on the free tier, so expect ~10–15 minutes for the 132 test
rows.

### Step 5 — (optional) reproduce the dev-split score first

```bash
python scripts/run_dev.py --provider opencode --n-shots 3 --guard numbers
```

### Step 6 — (optional) submit from the CLI

```bash
kaggle competitions submit -c radiology-reporting-harness \
  -f outputs/submission.csv \
  -m "Template-edit pipeline. LLM returns a JSON patch of only the fields that change; Python renders the report from the template, preserving field order and untouched normals. Model: space-bunny-free (OpenCode Zen), 3-shot retrieved exemplars, n=0 temp, unsupported-measurement guard."
```

### About exact bit-for-bit reproduction

The CSV cannot be guaranteed byte-identical, and it is worth being straight about
why. `space-bunny-free` is a free, non-deterministic tier; a different draw can
produce a different patch. Two things make the run as reproducible as it can be:

* **Temperature 0** and a fixed dev seed (`DEV_SEED=42`).
* **A response cache** at `outputs/llm_cache.json`, keyed by
  `(model, reasoning_effort, guard_level, n_shots, system_prompt, prompt)`.
  Re-running an unchanged configuration costs nothing and returns the identical
  result. Prompt iteration is therefore nearly free.

The **deterministic** half of the pipeline — the renderer, the guard, and the
submission validator — has no model in it at all, so given the same patch the
rendered report is always byte-identical.

The submitted `outputs/submission.csv` itself is in the repository, so the exact
artifact can be inspected without re-running anything.

### The notebook runs with no key at all

`notebooks/natoe_pipeline.ipynb` is self-contained (both modules are embedded as
`%%writefile` cells). With `OPENCODE_API_KEY` attached as a Kaggle Secret it
calls `space-bunny-free` and reproduces the real pipeline. **Without a key it
still runs end to end** on a built-in offline `MockLLM` — no API call, no cost —
and exercises the whole retrieve → prompt → guard → render path. The setup cell
auto-detects Kaggle Secrets, so there is nothing to edit and no key is pasted
into any cell.

---

## 5. Submission file details

| property | value |
|---|---|
| File | `outputs/submission.csv` |
| Rows | 132 (one per test case) |
| Columns | `case_id,report` |
| `case_id` values | match `test.csv` and `sample_submission.csv` exactly |
| Blank reports | 0 |
| Mean report length | ~1019 characters |

`predict_test.py` **refuses to hand over a submission** that has the wrong
columns, a mismatched `case_id` set, a missing section, leaked dictation text,
unlabelled FINDINGS content, or a field label absent from the template. The
uploaded file was re-verified after generation: 132 rows, correct columns,
`case_id` set identical to both `test.csv` and `sample_submission.csv`, no empty
reports.

---

## 6. Guardrails

A guard runs after the model and before rendering.

* **Label check** — a field label the template did not contain is discarded, so
  the renderer can never emit an unexpected field.
* **Unsupported measurements** — a sentence introducing a number that appears in
  neither the dictation nor that field's own template text is dropped. This is
  the narrowest defensible reading of "do not add findings unsupported by the
  dictation", and it is measured rather than assumed.

| guard level | behaviour | cost on gold patches |
|---|---|---|
| `off` | no check | 0.0023 |
| `numbers` | measurements only — **used for the submission** | 0.0030 |
| `strict` | + laterality / severity | 0.0154 |

An earlier `strict` guard also policed laterality and severity and cost
**0.19 RES**, because the reference reports themselves add laterality the
dictation never stated. It was deliberately removed.

Negation is never a drop trigger: templates legitimately say "No pneumothorax."
when the dictation never mentioned pneumothorax.

---

## 7. Results

| submission | local RES | public | private |
|---|---|---|---|
| Oracle (the reference itself) | 0.0000 | — | — |
| Architecture ceiling (renderer fed gold edits) | 0.0030 | — | — |
| Rule-based fallback (no LLM) | 0.5695 | 0.62616 | 0.64567 |
| LLM pipeline, 0-shot | 0.4026 | 0.42523 | 0.40149 |
| + append template IMPRESSION closer | 0.3721 | 0.42283 | 0.39879 |
| **+ 3-shot, prompt asymmetry, robust JSON parse** | **0.2842** | **0.36066** | **0.33012** |

**0.64567 → 0.33012 private, a 49% reduction.**

All 15 raw dev-split score files are committed under `results/`, and the
caveats around them are written up in `RESULTS.md` in the repository.

### An honest note on the local scorer

The RES implementation here is written from the published description; the
private scorer's exact function-word list and "unexpected field" penalty are not
published. Checked against three real submissions, the local scorer:

1. tracks the real one closely for FINDINGS-dominated submissions but is
   **consistently optimistic, and the bias grows as the pipeline improves**
   (+0.000, then +0.027, then **+0.046**);
2. **over-credits IMPRESSION changes by roughly 10×** (0.031 rated locally
   versus 0.0027 actually delivered);
3. earlier, for the hand-written fallback, mis-ranked three variants in the
   wrong order.

So local numbers are reliable for **ranking** changes — every change that
improved the local score did improve the leaderboard — but a local figure should
never be quoted as a predicted score. **Rule followed throughout: iterate
locally, submit to confirm, and add ~0.05 when interpreting a local number.**

---

## 8. Known limitations

* Absolute RES figures are approximate for the reason above; use them for
  relative comparison between pipeline variants.
* The free model tier is non-deterministic, so the CSV is not guaranteed
  byte-identical across runs (see §4).
* `OTHER FINDINGS` is a known trap. Every routing rule tried for it had
  precision ~0.07: the reference uses it as filler for findings matching no
  named field, and 56 of 84 such sentences *also* match a specific field. It is
  best left alone.
* 13.7% of dictation sentences are technique/history noise with no field at all
  (e.g. "Right Hip Radiographs, 2 Views"). The router still has to place them
  somewhere; a `DROP` class would be the fix and is not implemented.
* The trained router reaches 83% top-1 accuracy, below what the LLM already
  achieves, so it is intended as a *prior handed to the model*, not a constraint.
  Its training labels come from the same token-overlap matching used to build
  them, so its top-1 score partly measures agreement with the labeller.
* `modality`, `body_part`, `study_description`, `patient_age_band` and
  `patient_sex` are used only to select few-shot exemplars — never as a source
  of findings.
* The data is de-identified radiology text and must not be used for clinical
  decisions.

---

## 9. Reproducing the notebook

The notebook is generated from the modules, so it can be rebuilt rather than
hand-edited:

```bash
pip install -e ".[notebook]"
python scripts/build_notebook.py
jupyter lab notebooks/natoe_pipeline.ipynb
```

`tests/test_notebook.py` (part of the 114-test suite) asserts the generated
notebook still runs and still contains no secrets.
