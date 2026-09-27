"""IMPRESSION prediction.

The IMPRESSION is 35% of RES and was the weakest half of the score. Two things
about it are counter-intuitive and both are measured on data/train.csv:

1. **Copying the dictation is not enough.** The gold IMPRESSION is a clinically
   *normalised, reordered, numbered* summary. Gold is ~2.8x longer than the
   template's IMPRESSION (29.9 vs 10.8 tokens) and only correlates with the
   dictation length (r=0.84), not the template's (r=0.20).
2. **Keeping the template IMPRESSION is almost never right** — it happens in
   only 44/636 rows and costs 0.7495 when wrong. It is not a safe default.

But **appending the template's closing "normal" line to the dictated summary is
a large, genuine win**, and it is not a denominator artifact: padding with junk
words makes things monotonically worse (0.629 -> 0.752 -> 0.807 -> 0.863).
The house style is "the abnormality, then the template's normal closer", e.g.

    dictation:  no effusuon, infiltrates / mild thoracic spondylosis
    gold:       1. Mild thoracic spondylosis.
                2. No acute cardiopulmonary abnormality.

Measured mean `weighted_word_edit` against gold on all 636 train rows:

| strategy                                             | I     |
|------------------------------------------------------|-------|
| template IMPRESSION verbatim                          | 0.7495|
| whole dictation as IMPRESSION                        | 0.7579|
| best contiguous run of dictation sentences (ORACLE)   | 0.2956|
| best *any* subset of dictation sentences (ORACLE)     | 0.2956|
| single fixed rule: last 5 + template closer           | 0.5327|
| **length-bucketed rule (5-fold CV)**                  | **0.4831** |

The oracle row is the ceiling for *any* copy-only strategy. The gap from 0.4831
to 0.2956 is the part that needs semantic judgement: picking non-adjacent
fragments, reordering them, and expanding telegraphic shorthand into prose.
"""

from __future__ import annotations

import re

from .res_scorer import split_sections, split_sentences, tokenize

# STRICT: the whole line must be the header. A loose substring match
# false-positives on "assessment of venous compressibility" and turns a 0.0477
# score into 0.2239. Measured on train.
_HDR = re.compile(
    r"^\s*(impression|conclusion|summary|assessment|impressions)\s*:?\s*$", re.I)

_NEG = set("no not without absent negative unremarkable clear deny denies "
           "denied free".split())

_ABN = set("""mild moderate severe fracture fractures fractured dislocation
dislocated effusion edema consolidation opacity opacities mass nodule nodules
calcific calcification degenerative osteoarthritic osteoarthrosis arthrosis
stenosis hernia tear thickening prominence abnormal abnormality hyperinflation
emphysema atelectasis scar scarring cardiomegaly infiltrate pneumonia infarct
lesion irregularity demineralization demineralisation collection cyst cysts
tumor tumour neoplasm swelling contusion nonunion malalignment destructive
lytic sclerotic perinephric stranding polypoid polyp fibroids cholelithiasis
spondylosis osteophytes spondylolisthesis""".split())


def dict_sentences(dictation: str) -> list[str]:
    """Dictation split into sentences, hard-broken on newlines.

    A lowercase continuation after a period is not treated as a new sentence
    ("mild spondylosis. osteophytes" stays one unit), which matches how these
    telegraphic dictations are actually written.
    """
    out: list[str] = []
    for line in str(dictation or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        for p in re.split(r"(?<=[.!?])\s+(?=[\"'(]?[A-Z0-9])", line):
            if p.strip():
                out.append(p.strip())
    return out


def is_abnormal(sentence: str) -> bool:
    """True if the sentence asserts a POSITIVE finding.

    "There is a moderate pleural effusion" is True. "No pleural effusion or
    pneumothorax" is False: a negation window immediately before a finding word
    makes that word negative, so the sentence asserts normality.
    """
    tl = tokenize(sentence)
    if not (set(tl) & _ABN):
        return False
    for i, w in enumerate(tl):
        if w in _ABN and not (set(tl[max(0, i - 4):i]) & _NEG):
            return True
    return False


def last_sentence(text: str) -> str:
    """The template IMPRESSION's closing "normal" line.

    Uses the shared marker-aware splitter so that a numbered list like
    "1. Fracture.\\n2. No acute abnormality." yields the whole final line
    rather than the fragment "2.".
    """
    parts = split_sentences(str(text or ""))
    return parts[-1] if parts else str(text or "").strip()


def has_impression_header(dictation: str) -> bool:
    return any(_HDR.match(l) for l in str(dictation or "").split("\n"))


def heuristic_impression(template_content: str, dictation: str) -> str:
    """Deterministic IMPRESSION body. Mean I = 0.4819 in-sample, 0.4831 under
    5-fold CV over the 5 bucket labels.

    Length buckets, because the dictation is a mini-report whose length tells
    you whether it carries its own summary:
      1 sentence  -> the whole line + the template's closer
      2-7         -> the abnormal sentences + the template's closer
      8-15        -> last 4 sentences (self-contained; the template only adds noise)
      16+         -> last 6 sentences
    """
    tpl_imp = split_sections(template_content)[1].strip()
    lines = [l.strip() for l in str(dictation or "").split("\n")]

    # 1. The dictation carries its own IMPRESSION header -> copy it verbatim.
    #    22/636 rows, mean I 0.0477, exact 19/22.
    idx = [i for i, l in enumerate(lines) if _HDR.match(l)]
    if idx:
        i = idx[-1]
        rest = " ".join(x for x in lines[i + 1:] if x).strip()
        return rest or lines[i]

    sents = dict_sentences(dictation)
    if not sents:
        return tpl_imp

    tail = last_sentence(tpl_imp)
    n = len(sents)

    if n == 1:
        return f"{str(dictation).strip()} {tail}".strip()
    ab = [x for x in sents if is_abnormal(x)]
    if n <= 3:
        return " ".join(ab[:3] + [tail]).strip() if ab else tpl_imp
    if n <= 7:
        return " ".join(ab[:6] + [tail]).strip() if ab else tpl_imp
    if n <= 15:
        return " ".join(sents[-4:])
    return " ".join(sents[-6:])


def append_template_closer(impression: str, template_content: str) -> str:
    """Append the template IMPRESSION's closing line to a dictated summary.

    This is the single largest cheap win in the IMPRESSION: on the tail rule it
    moved I from 0.6292 to 0.5364.
    """
    imp = str(impression or "").strip()
    tail = last_sentence(split_sections(template_content)[1])
    if not imp or not tail:
        return imp
    if tail.lower() in imp.lower():          # already there
        return imp
    return f"{imp} {tail}".strip()
