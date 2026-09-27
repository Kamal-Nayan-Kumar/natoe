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
| 6 | + append template IMPRESSION closer | 0.3721 | 0.42283 | 0.39879 | −0.0305 | **−0.0027** |
| 7 | + 3-shot, prompt asymmetry, robust JSON parse | 0.2842 | 0.36066 | **0.33012** | −0.0879 | **−0.0687** |

Earlier, for the hand-written fallback, local read 0.5695 against a private
0.64567 and mis-ranked three variants in the wrong order.

**Three consistent findings:**

1. For a **FINDINGS-dominated** submission the local scorer tracks the real
   one closely (0.4026 vs 0.40149), but it is consistently **optimistic** and
   the bias grows as the pipeline improves: +0.000 on submission 5, +0.027 on
   6, **+0.046 on 7**. Treat a local figure as a *lower bound* on the real
   score, and expect roughly +0.05 at the current level.
2. The local scorer **over-credits IMPRESSION changes by ~10×** (0.031 rated vs
   0.0027 delivered).
3. Because the bias is *optimistic* and grows with quality, local numbers are
   still useful for **ranking** changes — every change that improved local has
   improved the leaderboard — but never quote one as a predicted score.

**Rule: iterate locally, submit to confirm, and add ~0.05 when interpreting a
local number.**

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

### CORRECTION — the guard comparison was confounded

An earlier version of these notes claimed `strict` costs 0.0154 RES. **That
number is not supported.** `guard_level` is part of the cache key, so each
guard level triggered its own *generation*, and the observed spread was almost
entirely sampling noise. Re-applying all three levels to one fixed set of
responses moves RES by ≤0.0008 (at 3 shots: off 0.2966 / numbers 0.2967 /
strict 0.2967).

**The three guard levels are near-equivalent at ≥2 shots.** Keep `numbers`
because it is free and never hurt, but treat the choice as *unresolved* rather
than confirmed. Testing it properly means dropping `guard_level` from the cache
key so the levels share one generation.

### Few-shot DOES help — the earlier negative result was rate limiting

Re-measured properly with the fast API, 0 failed rows in every config, same 60
rows (verified three ways including a byte-identical baseline fingerprint):

| n_shots | RES | F | I | prompt tokens | vs 0-shot |
|---|---|---|---|---|---|
| 0 | 0.3721 | 0.3332 | 0.4444 | 330 | — |
| 1 | 0.3287 | 0.2959 | 0.3896 | 751 | −0.0434 |
| 2 | 0.3053 | 0.2691 | 0.3726 | 1151 | −0.067 |
| **3** | **0.2951** | **0.2560** | **0.3676** | 1590 | **−0.0770** |

0-shot → 3-shot paired Δ = **−0.077** (95% CI 0.048–0.110; 42 wins vs 6
losses). **But do not overstate the 2→3 step:** Δ=0.010 with a CI that crosses
zero, and pure model noise for a *fixed* config was measured at 0.009–0.019.
The defensible claim is "shots help, 0→1→2 is solid, 2→3 is inside the noise."

Zero-shot also produces **malformed JSON on 2/60 rows** (a stray `}`) which
`parse_json` used to reject, silently degrading those rows to the template.
That is a real quality difference favouring few-shot, independent of rate
limiting. Fixed — see the parser tests.

### The router must NOT go in the prompt

Decisive measurement, no API calls needed (60/60 cache coverage): router and
LLM agree on 85.7% of comparable sentences and are **94.5% correct when they
agree**. On the 60 disagreements: **router 15, LLM 37, both wrong 8 →
28.8%**. Substituting the router's routing into the LLM's own content, content
held fixed, costs **+0.043 F**. Three runs bracketed it: 0.263 / 0.308 / 0.288.

Router-only submission scores RES 0.5839 (F 0.5047) against the LLM's 0.3327
(F 0.2960) — a real 0.25 RES behind. The router beats the naive baselines and
loses to the model.

**The one component that does beat the LLM is the facet rule (2/2 across runs)
— and that is a prompt line, not a router.** Note the correct ordering is the
opposite of the intuitive one: when the template has *both* `VERTEBRAE` and
`DISCS/DEGENERATIVE CHANGES`, the reference routes facet arthropathy to
**DISCS** (16/16), because that template's own DISCS normal literally reads "No
significant facet arthropathy". VERTEBRAE-first scores 42/99, DISCS-first
58/99.

Also: the router's `margin` is a genuine confidence signal (out-of-fold accuracy
0.458 at margin<2 rising to 0.967 at margin≥40), but applying it selectively
moves 1–25 sentences and lands within ±0.0016 F, **flipping sign between
runs** — inside the noise, and consistent with every other content-based
selection proxy having failed.

### The metric punishes under-editing 3× more than over-editing

This is forced by the scorer, not by radiology, and it is the single most
important thing to know when writing the prompt:

* a field the **reference changed** is charged `field_weight = 3`
* a field the **reference left alone** is charged `field_weight = 1`

So an unnecessary edit costs ~1/3 of what a **missed** edit costs. Every
intuition about "be conservative, don't invent" points the wrong way here.

A best-of-N experiment (5 samples/row, 80 rows, no gold used for selection)
confirmed it from the other direction. The best selector found was two terms
with no fitted parameters:

```python
aggressiveness(clean) = len(clean["fields"]) - 2.0 * (not clean["impression"].strip())
```

Pick the candidate that edits the most fields and always writes an impression.
It recovered **41% of the oracle headroom** (0.3912 random → 0.3595, oracle
0.3133), and it works out-of-sample (the rule was found on set A and did
*better* on set B, +54%).

Measured headroom, for reference: oracle best-of-5 = **0.078 RES (19.9%)** of
the score, ~2/3 of it in FINDINGS.

### The copy ceiling splits the problem in two

Fraction of each **changed** gold field's tokens that are findable, by dictation
length (636 train rows, 2153 changed fields):

| dictation length | from dictation | from template | **from either** | must be **generated** |
|---|---|---|---|---|
| **1 sentence** (215 f) | 0.296 | 0.504 | **0.733** | **27%** |
| 2–3 (86) | 0.573 | 0.478 | 0.878 | 12% |
| 4–7 (160) | 0.849 | 0.368 | 0.939 | 6% |
| 8–15 (676) | 0.906 | 0.472 | **0.975** | 2.5% |
| 16+ (1016) | 0.944 | 0.538 | **0.980** | 2% |

So long dictations are almost entirely a **selection** problem and 1-sentence
dictations are a **generation** problem. Combined with the error concentration
(1-sentence = 34% of total loss, 8–15 = 33%), that says where the difficulty
lives.

### Two hypotheses tested and REFUTED

**1. "Long dictations are 98% copyable, so a deterministic copy-assembler
should beat the LLM there."** It does not. Per-bucket mean per-row edit:

| bucket | LLM | router copy-assembly |
|---|---|---|
| 1 | **0.286** | 0.334 |
| 2–3 | **0.366** | 0.495 |
| 4–7 | **0.323** | 0.514 |
| 8–15+ | **0.360** | 0.527 |

The router loses in **every** bucket, including the one where copying is
essentially the whole task. Its ~23% sentence-level routing error rate costs
more than the LLM's paraphrasing does — and routing is scored twice by the
metric. No hybrid threshold helps. This closes the router line.

**2. "Best-of-N with the aggressiveness selector is worth ~0.03."** It is worth
**−0.020**: N=4 scores 0.3040 against N=1's 0.2842.

The reason is instructive. The selector's proxy ("edit as many fields as
possible, always write an impression") was fitted under the *conservative*
prompt, where the model under-edited. Once the prompt itself is liberal the
selector has nothing left to correct and actively over-edits. **The two fixes
are substitutes, not complements** — and the prompt fix is free. Ship N=1.

Generalisable lesson: a selection rule tuned against one generator can invert
once that generator is fixed. Re-measure the selector after changing the thing
it was selecting over.

### The noise floor — why further prompt tweaking is pointless

Adding `temperature` to the response-cache key (a correctness fix) invalidated
the cache, so the *identical* config was regenerated from scratch:

| run of the same config | local RES |
|---|---|
| first generation | 0.2842 |
| second generation | **0.2956** |

**Δ = 0.011 for the same prompt, same model, same temperature.** That matches
the independently measured 0.009–0.019 noise band for a fixed config. Only
9–12 of 60 responses were byte-identical between the two runs.

**Consequence: any local improvement smaller than ~0.02 is unmeasurable on a
60-row dev split.** Three recent changes fall inside that band and are recorded
as no-gain, not as wins:

| change | local RES | verdict |
|---|---|---|
| validated config (3-shot, liberal prompt) | 0.2842 / 0.2956 | baseline |
| + facet-arthropathy routing rule in the prompt | 0.3004 | no gain — reverted |
| 4 shots | 0.2989 | no gain |
| 5 shots | 0.2936 | no gain |

A 60-row split cannot resolve these, and neither can 132 test rows. Shipping
any of them would be selecting on noise — the same overfitting trap that the
local-vs-real disagreement already demonstrated once.

### Things that were tried and did not work

* **Every content-based proxy failed.** Dictation coverage, sentence drop rate,
  unsupported content, verbatim-copy fraction, cross-candidate consensus — all
  within noise of random on one of the two evaluation sets. Pooled Spearman
  correlation is actively misleading here: it is dominated by row difficulty,
  and gave the *wrong sign* for coverage.
* **Unsupported-content proxy is dead on arrival**, not merely weak: the
  `numbers` guard already strips every unsupported numeric token, so the
  feature is identically zero for all candidates. Consistent with `strict`
  costing 0.19 RES — the references *do* add laterality the dictation never
  stated.
* **Majority vote is actively harmful**: RES 0.5145, i.e. −111% of headroom,
  worse than every individual candidate. It unions the fields any candidate
  touched, so a single hallucinating candidate poisons the report.
* **A 16-feature ridge over all these signals ties the 2-term rule.** Do not
  build learned machinery here.
* **temperature = 0 is not the best single sample** — on one 40-row set it was
  the *worst* of five. Roughly 0.02 RES of the current score is pure sampling
  noise, recoverable for free. The temperature=0 call can also be dropped
  entirely: four T=0.7 samples select as well as five including it.

### Levers, in expected-value order

1. **F — better field content.** 0.333 × 0.65 = 0.217 of the 0.372 total. The
   LLM already beats oracle-routing-plus-copying (F 0.333 vs 0.4595), so this
   needs better *generation*, not better assembly. **LoRA fine-tuning on the
   636 labelled rows is the highest-expected-value remaining move.**
2. **Prompt asymmetry** (done): the original rule 3 told the model to be
   conservative, which is actively harmful given the 3:1 penalty. Rewritten to
   be explicitly liberal, plus "never return an empty impression". This is the
   best-of-N finding applied at N=1 for zero extra API cost.
3. **Few-shot, re-measured.** The earlier negative was rate limiting, not
   quality.
4. **Best-of-N**, if the above does not capture it. Worth ~0.03 at 5× API
   cost; the two-term selector is trivial to implement.
5. **Router as a prior.** 0.83 top-1; worth ~0.027 RES per 10pp, but only if it
   beats the LLM on the rows where they disagree.
6. **IMPRESSION — deprioritised.** 35% of the metric, but the real scorer barely
   rewards improvements to it (local rated the closer-append at 0.031 RES,
   reality delivered 0.0027).
7. **Jev** (`jev-1.13` is on Zen, ~$0.042/M input) as the *decision* layer:
   "does this finding belong in field X?" is a `choice` question, "is this
   sentence supported by the dictation?" a `noul`.

### Honest assessment of the 0.2375 target

Reaching it needs roughly **F ≈ 0.26 and I ≈ 0.20** simultaneously. Where we
are after the prompt-asymmetry and few-shot fixes: **F ≈ 0.256, I ≈ 0.368**
(3-shot), i.e. RES ≈ 0.295 locally.

That is a much better place than the 0.40149 we started from today, and F is
now close to the 0.26 needed. But **I = 0.368 against a required 0.20 is the
binding constraint**, and the IMPRESSION is the component where (a) the copy
ceiling is 0.296, and (b) the real leaderboard rewards improvements roughly 10×
less than the local scorer predicts. Note also that the unreachable-looking part
of the IMPRESSION is concentrated in the 1-sentence dictation bucket (185/636
rows), where the gold requires *rewriting telegraphic shorthand into clinical
prose* — a generation problem, not a selection one.

So the remaining honest path to 0.2375 is a better generator: **LoRA fine-tune
on the 636 labelled rows**, reusing the identical guard and render code so it is
a drop-in swap. Prompt engineering has now delivered most of what it can
(0.4026 → ~0.295); the remaining ~0.06 is unlikely to come from more prompt
tweaks.

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
| 5 | LLM JSON-patch pipeline, 0-shot | 0.42523 | 0.40149 |
| 6 | + append template IMPRESSION closer | 0.42283 | 0.39879 |
| 7 | **+ 3-shot, prompt asymmetry, robust JSON parse** | **0.36066** | **0.33012** |

Every submission validates cleanly (`validate_submission`: columns, case_id
set, both sections, no leaked dictation, no unlabelled FINDINGS text, no labels
outside the template). Submissions 1 and 4 reproduced identically, confirming
the pipeline is deterministic.

**Progress: 0.64567 → 0.33012 private, a 49% reduction**, all since the API was
unblocked.

## 12. Reproduce the current best

```bash
python scripts/predict_test.py --provider opencode --n-shots 3 --guard numbers
```

`LLM_PROVIDER`, `LLM_MODEL` and `REASONING_EFFORT` can override the defaults in
`src/natoe/config.py`. Responses cache to `outputs/llm_cache.json`, keyed by
`(model, reasoning_effort, guard_level, n_shots, temperature, system_prompt,
prompt)`, so re-runs are free.

### Still outstanding for the hiring review

Share `notebooks/natoe_pipeline.ipynb` privately with `natoeaidev` and paste
its URL into the submission description. Needs Kaggle account access.
