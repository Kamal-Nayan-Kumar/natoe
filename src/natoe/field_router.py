"""
Trained sentence -> FINDINGS-field router.

    "Which FINDINGS field does this dictation sentence belong in?"

Routing is the single largest source of FINDINGS loss on this task and the
metric scores it **twice** -- once as content missing from the right field, once
as extra content in the wrong one. This module turns routing into a cheap,
reproducible, dependency-light (numpy-only) classifier.

--------------------------------------------------------------------------
How it works
--------------------------------------------------------------------------
1. `build_training_set` turns `train.csv` into labelled sentence->field pairs.
   A sentence is labelled with the *changed* reference field whose content it
   covers best, scored by what fraction of the sentence's own weighted token
   mass appears in that field. Sentences that match no field confidently are
   labelled `DROP`: 15.1% of dictation sentences at the default settings
   ("Right Hip Radiographs, 2 Views", "Most likely differential diagnoses ...")
   are technique/history noise that own no field at all.

2. `FieldRouter` is a hand-rolled multinomial naive Bayes over normalised
   tokens + bigrams + the study's `body_part`, idf-weighted, with a **uniform**
   class prior. (scikit-learn is not a dependency; the model is three numpy
   arrays.) The uniform prior and the candidate mask below are what make it
   work: the empirical prior makes the frequent `BONES` field swallow sentences
   belonging to rarer fields, and flattening it is worth ~+2pp.

3. **Candidate masking is the single biggest win (+17.3pp measured).** A
   prediction is only ever allowed to be one of the labels present in *that
   row's own template*. Gold never uses a label outside the template in
   0 / 4037 cases, so the mask is free and always valid. It works because where
   a finding goes is decided by which fields the template happens to have, not
   by the finding alone -- information a pure sentence->label model cannot see.

4. Three hard rules sit on top of the model, each backed by a measurement:

   * `facet` / `arthropathy` -> the template's `DISCS/DEGENERATIVE CHANGES`,
     falling back to `VERTEBRAE` when the template has no such field. Never
     OTHER FINDINGS: 146 labelled "facet" sentences went to OTHER FINDINGS
     **0** times, against a 15% base rate. See `facet_prefers` for the
     measured ordering, which is the *reverse* of the obvious one.
   * `OTHER FINDINGS` is **never** a top-1 prediction. Every rule that routes
     there has measured precision ~0.07; the reference uses it as filler for
     findings that match no named field, and 56 of 84 such sentences *also*
     match a specific field. It remains reachable only when the template offers
     no other candidate at all.
   * If a sentence's content tokens are a subset of some template field's text,
     the router is skipped and that field is used directly. On that slice the
     direct lookup beats the model 0.913 vs 0.772.

--------------------------------------------------------------------------
Measured accuracy (5-fold group CV by case, `train.csv`, numpy only)
--------------------------------------------------------------------------
`build_training_set(min_overlap=0.6, min_margin=0.4)` keeps 4188 labelled
sentences (14.4% DROP); the model below gets **0.839 top-1 / 0.922 top-2** on
that confidence-filtered set -- the 4037-example / 0.838 operating point
recorded in NOTES.md. On the *unfiltered* set -- every non-DROP sentence,
including the ones the labeller itself is unsure about -- top-1 is **0.773** and
top-2 **0.881**. The gap is label noise in the labeller, not model error:
readings like "The urinary bladder is partially distended" and bare header lines
such as "Lateral Meniscus:" are counted as routing errors whatever the model
says.

The candidate mask is worth **+17.3pp top-1** (0.600 -> 0.773), the subset
lookup **+14pp on the 1% of sentences it handles** (0.913 vs 0.772), and
flipping the class prior from empirical to uniform **+2pp**.

The learned `DROP` class is a *weak* signal on its own -- precision 0.642,
recall 0.485, F1 0.553 against a 15.1% base rate. The `drop_threshold` margin is
the better lever: 0.809 of the sentences it *keeps* are correctly routed with
the threshold off, 0.936 with it at the 65th percentile. (Read that as "trust in
what you keep", not as "the dropped ones were really noise": the share of
dropped sentences that are genuine gold-DROP *falls* from 0.68 to 0.37 as the
threshold rises, because the extra drops are mostly real findings. A low margin
is a reason to distrust a routing, not a reason to delete the finding.)

The `margin` is also a well-calibrated confidence estimate, monotonically:

    margin < 2      n= 406   top-1 0.458        margin 20..40   n=1329   0.934
    margin 2..5     n= 472        0.536        margin >= 40    n= 976   0.967
    margin 5..10    n= 641        0.680
    margin 10..20   n=1215        0.854

That is the one plausible *use* for this module: as a checker on a candidate
patch ("this patch routes three sentences with margin < 3") rather than as a
generator of the routing. It is **unproven** -- see the caveat below.

--------------------------------------------------------------------------
The number that decides whether to ship this: the LLM is better
--------------------------------------------------------------------------
Against the cached LLM patch on the full 60-row `dev_split(n_dev=60)` (zero new
API calls), the router and the LLM agree on 85.7% of the 421 sentences both of
them route, and are **94.5%** correct when they agree. On the 14.3% where they
disagree -- 60 sentences across 23 of the 60 rows -- the **LLM wins 37 to 15, a
router win rate of 0.288**. Swapping the router's routing into the LLM's own
content, holding the content fixed, costs **+0.043 F** (0.296 -> 0.339): 17 rows
worsened, 3 improved, 40 unchanged.

Two earlier runs bracket it: 0.263 (60 rows, previous system prompt) and 0.308
(35 rows, while the cache was being rewritten concurrently). All three land in
0.26-0.31.

Decomposed by which rule produced the router's answer, the deterministic `facet`
rule is the only component that beats the model -- 2/2, against 11/42 for the
naive Bayes and 2/8 for the OTHER-FINDINGS exclusion. So: **do not hand this
router to the LLM as a prior.** If any of it ships, ship the facet rule as a
prompt line.

The router is *not* uniformly bad, though. Its `margin` separates the two
regimes cleanly, and in the top quartile the router beats the LLM **11/13**:

    bottom quartile  margin  0..1.2   router  0 / LLM 13   win rate  0%
    2nd quartile     margin  1.4..5.5 router  2 / LLM 13            13%
    3rd quartile     margin  6.3..12  router  2 / LLM  9            18%
    top quartile     margin 13..inf   router 11 / LLM  2            85%

That is the one plausible *use* for this module: as a checker on a candidate
patch ("this patch routes three sentences with margin < 3") rather than as a
generator of the routing. It is **unproven**, though. Applying the router
selectively at the margin thresholds above moves 1-25 sentences per run and
lands within +-0.0016 F, i.e. inside the noise floor, and it flips sign between
runs. NOTES.md also records that every other content-based selection proxy tried
on this task -- dictation coverage, sentence drop rate, unsupported content,
verbatim-copy fraction, cross-candidate consensus -- landed within noise of
random, and that a 16-feature ridge over them merely ties a 2-term rule. Read
the prior odds off that result, not off the calibration table above.

See NOTES.md section 7.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

import numpy as np

from .res_scorer import (normalize, parse_fields, split_sections,
                         token_weight, tokenize)

__all__ = [
    "DROP", "OTHER_FINDINGS", "VERTEBRAE", "DISCS_DEGENERATIVE",
    "FACET_RE", "FACET_PREFERS_DISCS_FIRST", "FACET_PREFERS_VERTEBRAE_FIRST",
    "RouteResult", "template_fields", "split_dictation", "FieldRouter",
    "build_training_set", "train",
]

# --------------------------------------------------------------- vocabulary

DROP = "__DROP__"
"""Pseudo-label for a sentence that owns no field (technique / history noise)."""

OTHER_FINDINGS = "OTHERFINDINGS"
VERTEBRAE = "VERTEBRAE"
DISCS_DEGENERATIVE = "DISCSDEGENERATIVECHANGES"

FACET_RE = re.compile(r"\b(?:facet|facets|arthropath(?:y|ic))\b")

#: Preference order for the facet rule, **measured**, and the reverse of the
#: obvious one. NOTES.md records "facet arthropathy -> VERTEBRAE, or
#: DISCS/DEGENERATIVE CHANGES when the template has that label", with a quoted
#: 50-sentence split of VERTEBRAE 56% / DISCS 22%. Re-measured over all 636
#: rows, that split does not hold:
#:
#:     template has VERTEBRAE only  (n=44):  reference -> VERTEBRAE 42, DISCS 0
#:     template has DISCS only     (n= 0):  -
#:     template has BOTH           (n=55):  reference -> VERTEBRAE  0, DISCS 16
#:
#: When both labels are present the reference uses VERTEBRAE **zero** times,
#: because these are MRI-spine templates whose own `DISCS/DEGENERATIVE CHANGES`
#: normal already reads "No significant facet arthropathy". VERTEBRAE-first
#: therefore scores 42/99 = 0.424 and DISCS-first 58/99 = 0.586.
#:
#: Either way the rule's real value is the ban: 146 facet sentences went to
#: OTHER FINDINGS 0 times.
FACET_PREFERS_DISCS_FIRST = (DISCS_DEGENERATIVE, VERTEBRAE)
FACET_PREFERS_VERTEBRAE_FIRST = (VERTEBRAE, DISCS_DEGENERATIVE)

#: Function words carry no anatomy, so they are excluded when deciding whether
#: a sentence is a subset of a template field's text. Kept separate from
#: `res_scorer.FUNCTION_WORDS` because that list is about edit *weight*.
ROUTING_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "these", "this", "those", "to", "was", "were", "which",
    "with", "within", "there", "here", "notably", "again", "also", "no", "not",
    "seen", "identified", "noted", "present", "appear", "appears", "appeared",
    "is", "are", "of", "on", "in", "as", "at", "or", "and", "with",
}


# ------------------------------------------------------------------ helpers

def template_fields(template_content: str) -> dict[str, str]:
    """FINDINGS fields of a template as ``{canonical_label: text}``.

    Labels are canonicalised with `res_scorer.canonical` so a template's
    ``BONES:`` and ``Bones:`` collapse to one key. Callers that need the label
    exactly as written should use `res_scorer.parse_fields` and take its
    `display` dict.
    """
    fields, _display, _unlabelled = parse_fields(split_sections(template_content)[0])
    return fields


def split_dictation(dictation: str) -> list[str]:
    """Dictation -> sentence list.

    Re-uses the pipeline's splitter so that list markers stay glued to their
    first sentence ("1. Mild spondylosis.") and the router sees exactly the
    units the LLM prompt and renderer do.
    """
    from .pipeline import _sentences

    return _sentences(dictation)


def _content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in ROUTING_STOP]


def _weight(tokens) -> float:
    return sum(token_weight(t) for t in tokens)


# ----------------------------------------------------------------- training

def build_training_set(frame, min_overlap: float = 0.6, min_margin: float = 0.4
                       ) -> tuple[list[str], list[str], list[str], list[str]]:
    """Label every dictation sentence with the reference field it belongs in.

    Returns ``(sentences, labels, body_parts, groups)`` where `groups[i]` is the
    position of the source row, so cross-validation can split by case (two
    sentences from the same study are not independent).

    Labelling rule: for each sentence, compute the fraction of its own
    *weighted* token mass that also appears in each **changed** reference field
    (unchanged fields are template normals, so a sentence matching one of those
    is a retained normal, not a new finding). The best-scoring field wins if it
    clears `min_overlap` and beats the runner-up by `min_margin`; otherwise the
    sentence is labelled `DROP`.

    `OTHER FINDINGS` is excluded from the argmax, matching the routing rule: it
    is filler, and preferring a specific field is right 56 of 84 times.
    """
    sentences: list[str] = []
    labels: list[str] = []
    body_parts: list[str] = []
    groups: list[str] = []

    for gi, row in frame.iterrows():
        tpl = template_fields(row.template_content)
        ref, _d, _u = parse_fields(split_sections(row.report)[0])
        changed = {k: v for k, v in ref.items()
                   if normalize(v).strip() != normalize(tpl.get(k, "")).strip()}
        changed_tok = {k: set(_content_tokens(v)) for k, v in changed.items()}
        bp = str(getattr(row, "body_part", "") or "")
        for sent in split_dictation(row.dictation):
            toks = _content_tokens(sent)
            if not toks:
                sentences.append(sent)
                labels.append(DROP)
                body_parts.append(bp)
                groups.append(str(gi))
                continue
            total = _weight(toks) or 1.0
            scored = []
            for key, ftok in changed_tok.items():
                if not ftok or key == OTHER_FINDINGS:
                    continue
                scored.append((_weight(set(toks) & ftok) / total, key))
            scored.sort(reverse=True)
            best_score, best_key = scored[0] if scored else (0.0, DROP)
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            if best_score < min_overlap or best_score - runner_up < min_margin:
                labels.append(DROP)
            else:
                labels.append(best_key)
            sentences.append(sent)
            body_parts.append(bp)
            groups.append(str(gi))
    return sentences, labels, body_parts, groups


# ------------------------------------------------------------------ routing

@dataclass
class RouteResult:
    """One sentence's routing decision."""

    label: str
    """Canonical FINDINGS label, or `DROP` for technique/history noise."""

    score: float
    """Log posterior-ish score of `label` (unnormalised NB score)."""

    margin: float
    """`score` minus the runner-up's score, over the restricted label set."""

    source: str
    """Which rule decided: `subset`, `facet`, `other-excluded`, or `nb`."""

    alternatives: tuple[str, ...] = ()
    """Masked candidate labels, best first (for handing the model options)."""

    @property
    def dropped(self) -> bool:
        return self.label == DROP


class FieldRouter:
    """Naive Bayes sentence -> field router, masked to one row's template.

    Parameters
    ----------
    alpha
        Additive Laplace smoothing on the token counts. Flat (0.1-0.3) is best;
        1.5 costs 2.6pp top-1.
    use_bigrams, use_body_part
        Feature groups. Both are worth ~+0.3pp; on by default.
    idf_power
        Token counts are weighted by ``idf ** idf_power`` before the log.
        Worth ~+0.8pp top-1; 0 disables it.
    min_token_count
        Features rarer than this are dropped from the vocabulary. Keep it at 1:
        pruning a token because it occurred twice removes the *only* evidence a
        rare field has, and it costs 0.8pp top-1 at 2 and 3.6pp at 10. The dense
        matrix is 17 MB at the default, which is not a problem.
    drop_threshold
        A sentence is left unrouted when its best-vs-runner-up `margin` falls
        below this. ``-inf`` (the default) disables the margin test, leaving
        only the learned `DROP` class in charge. Raise it to drop more. The
        measured curve, out-of-fold on `train.csv`: ``-inf`` leaves 12% of
        sentences unrouted at 0.809 keep-precision; 22.8 leaves 65% unrouted at
        0.936. Use a high threshold when you only want high-confidence hints.
        Do NOT read it as a noise detector: the share of dropped sentences that
        are genuine technique noise *falls* as the threshold rises.
    allow_other_findings
        When False (default) `OTHER FINDINGS` is removed from the candidate set
        unless it is the only candidate.
    facet_rule, subset_rule
        The two measured hard rules. Both are on by default. Both are
        *deterministic*: they pick the label outright rather than nudging the
        model, because a rule that can be outvoted is not a rule.
    facet_prefers
        Preference order for the facet rule, best first. The default
        ``(DISCS_DEGENERATIVE, VERTEBRAE)`` is what the data says; see
        `FACET_PREFERS_VERTEBRAE_FIRST` for the numbers and why.
    subset_min_tokens
        A subset lookup needs at least this many content tokens, otherwise
        two-token sentences match half the template.
    """

    def __init__(self, *, alpha: float = 0.2, use_bigrams: bool = True,
                 use_body_part: bool = True, idf_power: float = 1.0,
                 min_token_count: int = 1,
                 drop_threshold: float = float("-inf"),
                 allow_other_findings: bool = False,
                 facet_rule: bool = True, subset_rule: bool = True,
                 facet_prefers: tuple[str, ...] = FACET_PREFERS_DISCS_FIRST,
                 subset_min_tokens: int = 4):
        self.alpha = float(alpha)
        self.use_bigrams = bool(use_bigrams)
        self.use_body_part = bool(use_body_part)
        self.idf_power = float(idf_power)
        self.min_token_count = int(min_token_count)
        self.drop_threshold = float(drop_threshold)
        self.allow_other_findings = bool(allow_other_findings)
        self.facet_rule = bool(facet_rule)
        self.facet_prefers = tuple(facet_prefers)
        self.subset_rule = bool(subset_rule)
        self.subset_min_tokens = int(subset_min_tokens)
        self.vocab: dict[str, int] = {}
        self.classes: list[str] = []
        self._index: dict[str, int] = {}
        self._ll: np.ndarray | None = None       # (n_classes, n_features)
        self._log_prior: np.ndarray | None = None
        self._drop_score = -1e9

    # -- features ---------------------------------------------------------
    def _features(self, sentence: str, body_part: str | None) -> list[str]:
        toks = tokenize(sentence)
        feats = list(toks)
        if self.use_bigrams:
            feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        if self.use_body_part and body_part:
            feats.append("bp=" + str(body_part).strip().lower())
        return feats

    # -- fit --------------------------------------------------------------
    def fit(self, sentences, labels, body_parts=None) -> "FieldRouter":
        sentences = list(sentences)
        labels = [str(l) for l in labels]
        if len(sentences) != len(labels):
            raise ValueError("sentences and labels must be the same length")
        if body_parts is None:
            body_parts = [""] * len(sentences)
        if len(body_parts) != len(sentences):
            raise ValueError("body_parts must match sentences")

        feats = [self._features(s, bp) for s, bp in zip(sentences, body_parts)]
        counts: dict[tuple[str, str], int] = {}
        for fs, lab in zip(feats, labels):
            for t in fs:
                counts[(t, lab)] = counts.get((t, lab), 0) + 1

        # Prune rare features: they cannot generalise, and the model is a dense
        # (n_classes x n_features) float64 matrix.
        totals: dict[str, int] = {}
        for (t, _lab), n in counts.items():
            totals[t] = totals.get(t, 0) + n
        self.vocab = {t: i for i, t in enumerate(
            sorted(t for t, n in totals.items() if n >= self.min_token_count))}
        v = len(self.vocab)
        if not v:
            raise ValueError("no features survived min_token_count")

        self.classes = sorted(set(labels))
        self._index = {c: i for i, c in enumerate(self.classes)}
        matrix = np.zeros((len(self.classes), v), dtype=np.float64)
        for (t, lab), n in counts.items():
            j = self.vocab.get(t)
            if j is not None:
                matrix[self._index[lab], j] = n

        # idf weighting, then Laplace-smoothed log P(token | class).
        doc_freq = (matrix > 0).sum(axis=0)
        idf = np.log((1.0 + len(sentences)) / (1.0 + doc_freq)) + 1.0
        weighted = matrix * (idf ** self.idf_power)[None, :]
        self._ll = np.log(weighted + self.alpha) - np.log(
            weighted.sum(axis=1, keepdims=True) + self.alpha * v)

        # A uniform prior on purpose: the empirical prior makes frequent
        # classes (BONES appears in 398 of 636 templates) swallow sentences
        # that belong to rarer fields. Flattening it measured +2pp top-1.
        self._log_prior = np.zeros(len(self.classes), dtype=np.float64)
        self._drop_score = -1e9
        return self

    # -- scoring ----------------------------------------------------------
    def _scores(self, sentence: str, body_part: str | None) -> np.ndarray:
        feats = self._features(sentence, body_part)
        out = self._log_prior.copy()
        for t in feats:
            j = self.vocab.get(t)
            if j is not None:
                out += self._ll[:, j]
        return out

    def _subset_lookup(self, sentence: str, candidates, field_text):
        """The direct-lookup rule: a sentence whose content tokens are a subset
        of exactly one candidate field's text goes to that field, no model."""
        if not self.subset_rule or not field_text:
            return None
        toks = set(_content_tokens(sentence))
        if len(toks) < self.subset_min_tokens:
            return None
        hits = []
        for label in candidates:
            ftok = field_text.get(label)
            if ftok is not None and toks <= set(_content_tokens(ftok)):
                hits.append(label)
        if len(hits) == 1:
            return hits[0]
        return None

    def route(self, sentence: str, candidates, body_part: str | None = None,
              field_text: dict | None = None) -> RouteResult:
        """Route one sentence to one of `candidates` (or to `DROP`).

        `candidates` is the row's own template labels, canonicalised. Nothing
        outside it can ever be returned, which is what makes the mask safe:
        gold never leaves the template's label set.
        """
        allowed = [str(c) for c in dict.fromkeys(candidates)]
        if not allowed:
            return RouteResult(DROP, 0.0, 0.0, "nb", ())

        # Rule 1 -- direct subset lookup beats the model on its slice. Reported
        # with an infinite margin: it is a deterministic decision, so a
        # `drop_threshold` must not be able to discard it.
        hit = self._subset_lookup(sentence, allowed, field_text)
        if hit is not None:
            return RouteResult(hit, 0.0, float("inf"), "subset", tuple(allowed))

        # Rule 2 -- facet / arthropathy is a disc or bone finding, never
        # OTHER FINDINGS. 146 labelled sentences went to OTHER FINDINGS 0
        # times, against a 15% base rate. Picked outright rather than merely
        # preferred: the NB scores for a real facet sentence sit within ~10-25
        # log units of the DROP class, so a soft version of this rule flips
        # with training-set size.
        if self.facet_rule and FACET_RE.search(sentence.lower()):
            pick = next((c for c in self.facet_prefers if c in allowed), None)
            if pick is not None:
                return RouteResult(pick, 0.0, float("inf"), "facet",
                                   tuple(allowed))

        source = "nb"
        # Rule 3 -- OTHER FINDINGS is a last resort only. Every rule that
        # routes there has measured precision ~0.07.
        if not self.allow_other_findings and OTHER_FINDINGS in allowed \
                and len(allowed) > 1:
            allowed = [c for c in allowed if c != OTHER_FINDINGS]
            source = "other-excluded"

        scores = self._scores(sentence, body_part)
        # DROP always competes: it is a real class trained on technique noise.
        pooled = list(allowed)
        if DROP in self._index:
            pooled.append(DROP)
        ranked = sorted(((float(scores[self._index[c]]), c) for c in pooled
                         if c in self._index), reverse=True)
        if not ranked:                                     # unknown labels only
            return RouteResult(DROP, self._drop_score, 0.0, source, ())
        top_score, top_label = ranked[0]
        margin = top_score - ranked[1][0] if len(ranked) > 1 else float("inf")
        return RouteResult(top_label, top_score, margin, source,
                           tuple(c for _, c in ranked[:3]))

    def route_document(self, template_content: str, dictation: str,
                       body_part: str | None = None,
                       drop_threshold: float | None = None) -> dict[str, list[str]]:
        """Route a whole dictation into ``{canonical_label: [sentences]}``.

        Only template labels appear as keys. Sentences the router refuses
        (technique/history noise, or a margin under `drop_threshold`) are left
        out entirely rather than being forced into some field.
        """
        fields = template_fields(template_content)
        thr = self.drop_threshold if drop_threshold is None else float(drop_threshold)
        out: dict[str, list[str]] = {}
        for sent in split_dictation(dictation):
            res = self.route(sent, fields.keys(), body_part, fields)
            if res.dropped or res.margin < thr:
                continue
            out.setdefault(res.label, []).append(sent)
        return out


def train(frame, **kwargs) -> FieldRouter:
    """Build a training set from a `train.csv`-shaped DataFrame and fit."""
    accepted = set(inspect.signature(FieldRouter).parameters)
    accepted |= {"min_overlap", "min_margin"}
    unknown = set(kwargs) - accepted
    if unknown:
        raise TypeError(f"train() got unexpected kwargs: {sorted(unknown)}")
    data_kwargs = {k: kwargs[k] for k in ("min_overlap", "min_margin")
                   if k in kwargs}
    sentences, labels, body_parts, _groups = build_training_set(
        frame, **data_kwargs)
    return FieldRouter(**{k: v for k, v in kwargs.items()
                          if k in inspect.signature(FieldRouter).parameters}
                       ).fit(sentences, labels, body_parts)
