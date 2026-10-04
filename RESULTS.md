# Measured results

Every dev-split score this project produced, committed as raw JSON under
`results/`. The README quotes a few of these; this file is the full set, so a
reviewer can check the claims instead of taking them on trust.

Reproduce any row with:

```bash
python scripts/run_dev.py --provider opencode --n-shots 3 --guard numbers
```

Each file is the output of one `run_dev.py` invocation.

## What the numbers mean

`RES` is lower-is-better. `F` is the FINDINGS edit component and `I` is the
IMPRESSION component; `RES_case = 0.65·F + 0.35·I`, so `F` dominates.

**The splits are not all the same size.** `n_dev` in each file is the number of
held-out rows, and it varies (40, 60, 300). Scores are only comparable *within*
a file's split, never across files. The 300-row split is the most trustworthy
of the set.

**Some rows are cache replays, not fresh API calls.** `run_dev.py` caches LLM
responses in `outputs/llm_cache.json`, so re-running a config that has already
been run costs nothing and returns the original score. Those files show
`seconds` near zero. Treat their RES as the original measurement and ignore
their `seconds`:

| file | what it is |
|---|---|
| `dev300_final.json` | replay of `dev300_base.json` (identical RES, `seconds` 1.8 vs 717.2) |
| `dev_closer.json` | replay (`seconds` 0.2) |

## opencode / `space-bunny-free`

| file | n_dev | n_shots | n_samples | RES | F | I | seconds |
|---|---|---|---|---|---|---|---|
| `dev_lib3.json` | 60 | 3 | – | **0.2842** | 0.2470 | 0.3533 | 64.8 |
| `dev_f5.json` | 60 | 5 | 1 | 0.2936 | 0.2543 | 0.3666 | 222.9 |
| `dev_final.json` | 60 | 3 | 1 | 0.2956 | 0.2578 | 0.3658 | 121.4 |
| `dev_f4.json` | 60 | 4 | 1 | 0.2989 | 0.2580 | 0.3748 | 122.8 |
| `dev_f3.json` | 60 | 3 | 1 | 0.3004 | 0.2597 | 0.3761 | 127.5 |
| `dev_bon4.json` | 60 | 3 | 4 | 0.3040 | 0.2660 | 0.3746 | 323.6 |
| `dev300_keep.json` | 300 | 3 | 1 | 0.3196 | 0.2836 | 0.3865 | 843.9 |
| `dev300_base.json` | 300 | 3 | 1 | **0.3097** | 0.2673 | 0.3884 | 717.2 |
| `dev_liberal.json` | 60 | 0 | – | 0.3327 | 0.2960 | 0.4008 | 141.8 |
| `dev_closer.json` | 60 | 0 | – | 0.3721 | 0.3332 | 0.4444 | 0.2 (replay) |
| `dev_sb.json` | 60 | 0 | – | 0.4026 | 0.3332 | 0.5317 | 139.6 |

## groq / `openai/gpt-oss-120b`

| file | n_dev | n_shots | RES | F | I | seconds |
|---|---|---|---|---|---|---|
| `dev_ns1.json` | 40 | 1 | 0.4114 | 0.3544 | 0.5171 | 1065.3 |
| `dev_ns0.json` | 40 | 0 | 0.4366 | 0.3706 | 0.5592 | 680.5 |
| `dev_results.json` | 60 | 3 | 0.4693 | 0.4026 | 0.5933 | 890.1 |

The Groq rows are the slow ones: `dev_ns1` took 17.8 minutes for 40 rows, which
is the 8000 prompt-tokens/minute limit, not the model being slow.

## Two things worth flagging

**1. More shots won here, not fewer.** The README argues for `n_shots=0` as the
default because on Groq a 3-shot prompt burns the token budget and 429s. That
reasoning is sound for Groq. But on `space-bunny-free`, which is not rate
limited, 3-shot beat 0-shot on the same 60-row split: 0.2842–0.3004 versus
0.3327–0.4026. On a rate-unlimited provider the token-budget argument does not
apply, and few-shot exemplars help. The `n_shots=0` default looks like it was
tuned against the wrong constraint.

**2. The 300-row split is worse than the 60-row split.** Same config, same
model: 0.3097 on 300 rows versus 0.2956 on 60. A 60-row split is small enough
that this gap can be split noise. The 300-row number is the one to trust, which
means the README's headline "local dev RES 0.4026" is measured on the weaker
0-shot config, not the best one.

## Caveat carried over from the README

The RES implementation here is written from the published description, so the
private scorer's exact function-word list and "unexpected field" penalty are
unknown. Checked against real submissions, the local scorer matched the private
leaderboard to ~0.001 for LLM-generated text (0.4026 local vs 0.40149 private)
but was off by ~0.08 and mis-ranked variants for the hand-written fallback.
Local numbers are reliable for **relative** comparison of LLM-pipeline
variants; anything IMPRESSION-shaped needs a real submission to confirm.
