# Natoe Radiology Edit — working notes

Everything learned building this, in one place. Read this before touching the
code. Newest findings at the bottom.

---

## 1. The task and the metric

Convert a radiologist's **dictation** into a completed structured report by
**editing a supplied normal `template_content`**. Output must contain exactly
`FINDINGS:` and `IMPRESSION:`.

**RES — Radiology Edit Score, lower is better.**

```
F   = Σ(field_weight × field_word_edit) / Σ(field_weight)
I   = weighted_word_edit over the whole IMPRESSION section
RES_case = 0.65·F + 0.35·I
```

* field_weight = **3** if the reference field differs from the template field,
  **1** if unchanged.
* Token weights: critical words / numbers / units **4.0**, other content
  **2.0**, function words **0.25**. Normalised by the larger total weight, so
  each field's score lands in `[0, 1]`.
* A finding under the wrong label = missing from the right field **and** extra
  content in the wrong one. **Routing errors are doubly penalised.**
* Everything normalises to lowercase, so field labels must be matched
  **case-insensitively** (real templates mix `BONES:` and `Bones:`).

**The metric rewards template-edit fidelity, not semantic correctness.**
Paraphrasing a normal statement is a scored edit.

---

## 2. Data facts that drove every design decision

Measured on `train.csv` (636 rows) with `scripts/analysis/explore_baselines.py`.

| observation | value | consequence |
|---|---|---|
| fields per reference report | 5.16 | few fields, so one routing error is expensive |
| fields that **differ** from the template | **63%** | "minimal edit" is the *wrong* instinct — rewrite every field the dictation speaks to |
| reports with **zero** changed fields | 54 | a minority really do need no edit |
| changed fields reusing **dictation** vocabulary | **76%** | copy verbatim, never paraphrase |
| IMPRESSION >80% dictation vocabulary | 382/636 | the dictation already contains the summary — *select* it |
| IMPRESSION identical to the template's | 44/636 | leave it alone that often |
| field order follows the template | 572/636 | always emit the template's order |
| reference uses a label **outside** the template | **0 / 4037** | candidate-mask to the template's labels; free, always valid |
| gold field tokens present in the dictation — changed fields | **0.90** | copy has a high ceiling where it matters |
| gold field tokens present in the dictation — unchanged fields | **0.24** | unchanged fields must be copied from the **template**, not the dictation |
| spinal-level shorthand → canonical `C5-C6` | **298/344 (87%)** | normalise spinal levels |
| ordinary misspellings kept verbatim | **139/139 (100%)** | **do not** fix spelling errors — the reference keeps the typo |

The last two are the subtle ones and they pull in opposite directions.

### Dictation shape

Median 579 chars, p90 1666, max 3270. Only 189/636 are short. 22 contain a
literal `IMPRESSION` section. Most are long structured mini-reports.

---

## 3. Architecture

```
 retrieve few-shot  →  LLM emits a JSON *patch*  →  guard  →  render
   (house style)      (only the fields that change   (strip     (Python rebuilds
                       + the IMPRESSION text)         invented    in the template's
                                                     numbers)     field order)
```

**The model never writes the report. It writes a patch. Python renders it.**
This makes field order, label spelling and untouched-field preservation
*structurally impossible* to get wrong rather than merely unlikely.

### The guard is deliberately narrow

| level | behaviour | cost on gold patches |
|---|---|---|
| `off` | no check | 0.0023 |
| `numbers` | measurements only — **default** | 0.0030 |
| `strict` | + laterality / severity | 0.0154 |

An earlier `strict` guard also policed laterality/severity and cost **0.19
RES**: the reference reports themselves add laterality the dictation never
stated, so "unsupported laterality" is not safe to strip. Negation is never a
trigger — templates legitimately say "No pneumothorax." when the dictation
never mentioned pneumothorax.

### Ceiling of the architecture

Fed the **gold** per-field edits, `render()` scores **0.0023**. The assembly is
correct; all remaining error is content quality.

---

## 4. Baselines and scores

| submission | local RES | public | private |
|---|---|---|---|
| ORACLE (the reference itself) | 0.0000 | — | — |
| rule-based fallback (best variant) | 0.5695 | 0.62616 | 0.64567 |
| rule-based fallback, finding-filtered IMPRESSION | **0.5632** | 0.63136 | 0.65007 |
| rule-based fallback, no sentence filter | (worse) | 0.64123 | 0.64932 |
| **LLM JSON-patch pipeline** | **0.4026** | **0.42523** | **0.40149** |

Current split: **F = 0.333, I = 0.532**.

### Calibration — read this before trusting the local scorer

The local RES implementation is a re-implementation; the real scorer is
private. Three submissions, three behaviours:

| # | change | local RES | public | private | local Δ | real Δ |
|---|---|---|---|---|---|---|
| 5 | LLM pipeline baseline | 0.4026 | 0.42523 | 0.40149 | — | — |
| 6 | + append template IMPRESSION closer | 0.3721 | 0.42283 | **0.39879** | −0.0305 | **−0.0027** |

Earlier, for the hand-written fallback, local read 0.5695 against a private
0.64567 and mis-ranked three variants in the wrong order.

**Two consistent findings:**

1. For a **FINDINGS-dominated** submission the local scorer is accurate to
   ~0.001 (0.4026 vs 0.40149).
2. The local scorer **systematically over-credits IMPRESSION changes, by
   roughly 10×.** It rated the closer-append as worth 0.031 RES; it was worth
   0.0027. Backing out the arithmetic, the real IMPRESSION component moved only
   ~0.532 → ~0.524 where local said → 0.444.

**Rule: iterate locally on FINDINGS. Treat local IMPRESSION numbers as an
upper bound on the benefit and confirm with a real submission.** Stop spending
effort on IMPRESSION micro-optimisation — the metric barely rewards it.

---

## 5. API providers — what actually works

| provider | status | detail |
|---|---|---|
| **opencode (Zen)** | `space-bunny-free` only | ~6s/call, **0.43 rows/s, no rate limiting**. Paid ids → 402 "Insufficient account balance". Other `-free` ids → 403 "free tier can only be used from within OpenCode" |
| groq | heavily throttled | `gpt-oss-120b`/`20b` 429 on ~half of prompt-sized calls. 1000 req/hr but only **8000 prompt tokens/min**. `qwen3.8-27b` ~50% success. Account headers show quota free while calls still 429 — the throttle is per-model and invisible |
| openrouter | exhausted | `:free` daily cap hit. `nvidia/nemotron-3-super-120b-a12b:free` honours JSON mode at 0.6s; `gemma-4-31b-it:free` frequently 429s |

### Operational lessons

* **`reasoning_effort=low` is essential.** Reasoning models otherwise spend the
  whole output budget on chain-of-thought and return no `content`.
  `space-bunny-free` with `low`: 4.6s / 236 reasoning tokens. Without it: the
  model returns a bare `{`.
* **`n_shots` is a token-budget decision, not a quality one.** A 3-shot prompt
  is ~4200 tokens; at 8000 tokens/min that is ~2 calls/min. The earlier
  "1-shot is worse" finding was rate limiting, not quality — it must be
  re-measured now that the API is fast.
* Never cache failures, and report rows that exhausted retries instead of
  silently degrading them to the template.
* Low concurrency (`workers=2`) plus a retry pass. A 429 storm silently drags
  the score back toward "template unedited".

---

## 6. Key prompt rules, and why each exists

1. **Copy verbatim**, never paraphrase. → 76% of changed fields reuse dictation
   vocabulary.
2. **Expand spinal-level shorthand only** (`c56` → `C5-C6`), **keep ordinary
   misspellings** (`andd`, `ostearth`). → 87% vs 100% respectively.
3. **Only include fields you are changing.** → the renderer keeps the rest
   byte-for-byte.
4. **Use only the template's labels.** → 0/4037 gold cases leave the template's
   label set.
5. **Route each finding to exactly one field.** → routing is the top FINDINGS
   loss; it is scored twice.
6. **Delete the contradicting template normal, keep still-true ones.**
7. **IMPRESSION:** if the dictation has no abnormality, keep the template's
   unchanged and never put a technique note there.
8. Never add a finding, measurement or diagnosis not in the dictation.

---

## 7. Error analysis so far

* **Routing dominates** FINDINGS loss. Top confusion pairs: BONES↔JOINTS (8.7%
  of errors), BONES↔SOFTTISSUES, OTHERFINDINGS↔SOFTTISSUES,
  BONES↔CARTILAGE, BONES↔OTHERFINDINGS, ARTICULARCARTILAGE↔OSSEOUSSTRUCTURES.
* **OTHER FINDINGS is a trap.** Every rule for routing there has precision
  ~0.07. The reference uses it as filler for findings that match no named
  field, and 56 of 84 such sentences *also* match a specific field. Leave it
  alone.
* **Facet arthropathy → `VERTEBRAE`**, or `DISCS/DEGENERATIVE CHANGES` when the
  template has that label. Never OTHER FINDINGS.
* 13.7% of dictation sentences are technique/history noise with no field
  ("Right Hip Radiographs, 2 Views"), and the router currently has to place
  them somewhere. It needs a **DROP** class.
* Severity and laterality adjectives are **pure routing noise** — "mild" maps
  to 14 different labels.

### Trained router (hand-rolled naive Bayes, numpy only)

4037 labelled sentence→field examples, 128 labels, 5-fold group CV by case.

| model | top-1 | top-2 | acc on top-20 labels |
|---|---|---|---|
| unigrams, unconstrained | 0.599 | 0.747 | 0.666 |
| + body_part | 0.644 | 0.803 | 0.709 |
| **+ candidate mask to template** | **0.838** | **0.909** | **0.896** |
| + bigrams | 0.848 | 0.910 | 0.904 |

**Candidate masking is worth +19pp and is free.** The deciding factor for where
a finding goes is *which fields the template happens to have*, not the finding
itself — that is exactly what the mask exploits.

Slope: **~0.035 F per 10pp of sentence-routing accuracy** (~0.027 RES per
10pp). 83% → 95% would be worth ~0.042 F.

Caveat: the router's training labels come from the same token-overlap matching
used to build them, so top-1 partly measures agreement-with-the-labeller. The
confusion analysis and the OTHER FINDINGS precision figures are not subject to
that.

**Use the router as a prior handed to the LLM, not as a constraint.** It is
83% accurate; the LLM already achieves F=0.333, so constraining the model to
the router would cap us below where we are.

---

## 8. Important negative result

**Oracle routing + pure copy-append scores F = 0.4595. The LLM pipeline
already achieves F = 0.3332.**

So the "select and order existing sentences, never write new text" architecture
— which was the original plan — would be a **regression**. The model is already
writing better than perfect-routing copying. Any route to a much lower score
has to come from *better generation*, not cleverer assembly.

---

## 9. Where we are, and the target

Current: **0.39879 private**. Target: **< 0.23750**.

Split: **F = 0.333, I = 0.444**. F carries 65% of the metric, so that is where
the score actually lives.

Levers, in expected-value order:

1. **F — better field content.** 0.333 × 0.65 = 0.217 of the 0.372 total. The
   LLM already beats oracle-routing-plus-copying (F 0.333 vs 0.4595), so this
   needs better *generation*, not better assembly. **LoRA fine-tuning on the
   636 labelled rows is the highest-expected-value remaining move.**
2. **Few-shot, re-measured** (agent running). The earlier negative was rate
   limiting, not quality.
3. **Best-of-N with a self-scoring proxy** (agent running) — reject a patch
   that leaves dictated findings unrouted or invents a measurement. A
   selection signal available at inference time, with no gold.
4. **Router as a prior** (agent running). 0.83 top-1; worth ~0.027 RES per
   10pp, but only if it beats the LLM on the rows where they disagree.
5. **IMPRESSION — deprioritised.** It is 35% of the metric, but the real scorer
   barely rewards improvements to it (see the calibration table: local rated
   the closer-append at 0.031 RES, reality delivered 0.0027). The copy-only
   ceiling is 0.296 so there is headroom on paper, but measurement says effort
   here converts poorly. Stop micro-optimising it.
6. **Jev** (`jev-1.13` is on Zen, ~$0.042/M input) as the *decision* layer:
   "does this finding belong in field X?" is a `choice` question, "is this
   sentence supported by the dictation?" a `noul`.

### Honest assessment of the 0.2375 target

Reaching it needs roughly F ≈ 0.26 **and** I ≈ 0.20 simultaneously. Measured
ceilings say a copy-based approach cannot get there — oracle routing plus
copy-append is F 0.4595, worse than what the prompted model already achieves.
So the number is reachable only through materially better generation, which in
practice means fine-tuning on the 636 labelled rows (or a stronger model, which
this account cannot currently buy). Prompt engineering is close to exhausted:
the three levers above are worth a few hundredths between them, not the ~0.16
that is needed.

---

## 10. File map

```
data/            train.csv, test.csv, sample_submission.csv   (gitignored)
src/natoe/
  res_scorer.py  local RES implementation; normaliser, weighted
                 edit distance, section/field parsing
  pipeline.py    retriever, system prompt, JSON patch, guard, renderer,
                 LLM client, on-disk cache, run_dataset
  evaluate.py    dev split, scoring, error analysis, submission validator
  config.py      paths, provider table, defaults
scripts/
  run_dev.py         dev-split scoring + error analysis
  predict_test.py    test predictions + validation
  fill_and_submit.py resilient cache-filling (safe to re-run)
  build_notebook.py  regenerates notebooks/natoe_pipeline.ipynb
  analysis/          one-off research scripts
  fetch_data.sh      restore data/ from Kaggle
notebooks/       natoe_pipeline.ipynb (self-contained, 46 cells)
tests/           50 tests, no API key required
outputs/         llm_cache.json, dev_results.json, submission.csv (gitignored)
```

### Gotchas encoded in the code

* Field labels must be matched **case-insensitively** (`canonical()`).
  Missing this silently drops whole fields.
* `parse_fields` takes the FINDINGS **block**, without the `FINDINGS:` header.
* Lines after a label are a **continuation** of that field, not stray text.
  Text before the first label is the unlabelled remainder.
* List markers must stay glued to their sentence (`"1. foo"`), or
  `1. Mild spondylosis.` splits and the IMPRESSION is reflowed.
* The cache key **must** include the system prompt, or editing the prompt
  rules silently reuses every stale response.
* Never cache a failure.
* `%%writefile` the modules into the notebook; `tests/test_notebook.py` fails
  if the notebook drifts from `src/`.

---

## 11. Score history on Kaggle

| # | submission | public | private |
|---|---|---|---|
| 1 | rule-based fallback | 0.62616 | 0.64567 |
| 2 | + abnormality-filtered IMPRESSION | 0.63136 | 0.65007 |
| 3 | no sentence filter | 0.64123 | 0.64932 |
| 4 | restore of #1 | 0.62616 | 0.64567 |
| 5 | LLM JSON-patch pipeline | 0.42523 | 0.40149 |
| 6 | + append template IMPRESSION closer | **0.42283** | **0.39879** |

Submission 5 reproduced deterministically and validates cleanly
(`validate_submission`: columns, case_id set, both sections, no leaked
dictation, no unlabelled FINDINGS text, no labels outside the template).

### Still outstanding for the hiring review

Share `notebooks/natoe_pipeline.ipynb` privately with `natoeaidev` and paste
its URL into the submission description. Needs Kaggle account access.
