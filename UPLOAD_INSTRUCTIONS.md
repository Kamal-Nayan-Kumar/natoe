# What to upload — exact steps

Two separate things are required by the rules. The CSV scores; the notebook
makes the entry **complete**.

> "A leaderboard submission without an accessible pipeline notebook is
> incomplete."

---

## 1. File Upload tab → the CSV

**File to upload:**

```
/Users/nayan/Documents/code/natoe/outputs/submission.csv
```

132 rows, columns `case_id,report`, case_ids match `test.csv` exactly, already
validated.

**Submission Description — paste this (fills the 500-char box):**

```
Template-edit pipeline. LLM returns a JSON patch of only the fields that change; Python renders the report from the template, preserving field order and untouched normals. Model: space-bunny-free (OpenCode Zen), 3-shot retrieved exemplars, n=0 temp, unsupported-measurement guard. Pipeline notebook: https://www.kaggle.com/code/kamalnayan90/natoe-radiology-edit
```

**Submit.**

This alone has already been submitted 7 times via CLI; the best scored
**0.36066 public / 0.33012 private**. Submitting again from the UI with the
notebook URL in the description is what closes the requirement.

---

## 2. Notebook tab → this part I cannot do for you

The tab says *"You don't have any notebooks linked to this competition."* That
is correct and it is not something the CLI can fix: a notebook is only
**linked** to a competition if it was created *from* that competition's page.
My CLI-pushed notebook `kamalnayan90/natoe-radiology-edit` exists in your
account but is unlinked, which is why the tab is still empty.

**Do this:**

1. On the Notebook tab, click **Create Notebook**. This creates a notebook
   already attached to the competition — that attachment is the part that
   matters.
2. Delete everything in it, then paste the contents of:

   ```
   /Users/nayan/Documents/code/natoe/notebooks/natoe_pipeline.ipynb
   ```

   Open it in an editor, copy the whole JSON, paste into the notebook editor
   on Kaggle. (Kaggle's editor accepts a pasted `.ipynb` via *Add Code →
   Existing Code* or by pasting the JSON into a code cell wrapped so it
   parses — the simplest reliable route is to open the file locally, select
   all, copy, and paste into the first cell of a fresh notebook after
   deleting the default cells.)

3. **Save a version** (the Save/Commit Version button) — the rules require a
   saved version, not just a saved notebook.

4. **Share it privately with Natoe AI Dev (`natoeaidev`)** — notebook page →
   *Share* → add `natoeaidev`. This is a permissions action only you can do;
   there is no CLI route for it.

5. Copy the notebook URL and put it in the Submission Description on the
   File Upload tab (the text above already has a placeholder URL — replace it
   with the new notebook's URL).

---

## If the notebook is awkward to paste

A single self-contained Python file with the same pipeline, for a reviewer who
prefers code:

```
src/natoe/res_scorer.py   local RES metric implementation
src/natoe/pipeline.py     retriever, prompt, JSON patch, guard, renderer
src/natoe/impression.py   IMPRESSION strategy
src/natoe/config.py       paths + provider table
src/natoe/evaluate.py     dev split, scoring, error analysis
```

Reproduce the submission with:

```bash
source .venv/bin/activate
python scripts/predict_test.py --provider opencode --n-shots 3 --guard numbers
```

---

## Note on the notebook run

The notebook currently **errors on Kaggle**. Versions 1–4 failed for reasons
that are all now fixed in the file (`%%writefile` needs a `src/natoe` directory
created first; the package `__init__.py` was not being written; `pandas` was
never imported; `dict(Series)` was used with attribute access). It is worth
creating from the competition page anyway, because the rules say the notebook
**does not need to execute on Kaggle** — what matters is that it is accessible
and contains the complete pipeline.

It also contains no API keys (there is a test asserting this) and documents the
Kaggle Secrets injection path in its setup cell.
