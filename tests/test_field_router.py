"""Tests for the trained sentence -> FINDINGS-field router.

The router's value is entirely in the *constraints* it imposes, so that is what
these tests pin down: the candidate mask, the DROP class, the OTHER FINDINGS
ban, the facet rule, and the guarantee that `route_document` can only ever emit
a label the template actually contains.

No network and no API key: the fit is built from `train.csv`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natoe.config import TRAIN_CSV                              # noqa: E402
from natoe.field_router import (DROP, DISCS_DEGENERATIVE,        # noqa: E402
                                OTHER_FINDINGS, VERTEBRAE, FieldRouter,
                                build_training_set, split_dictation,
                                template_fields, train)
from natoe.res_scorer import parse_fields, split_sections       # noqa: E402

TPL_SPINE = """FINDINGS:
VERTEBRAE: No acute osseous abnormality identified on these views.
DISCS/DEGENERATIVE CHANGES: No significant degenerative change.
SOFT TISSUES: The soft tissues are unremarkable.

IMPRESSION:
No acute abnormality.
"""

TPL_BODY = """FINDINGS:
BONES: No acute fracture or focal osseous lesion identified on these views.
SOFT TISSUES: The visualized soft tissues are unremarkable without mass or
pneumothorax.

IMPRESSION:
No acute abnormality.
"""

CAND = ["OTHER FINDINGS", "BONES", "LUNGS", "VERTEBRAE", "LIVER",
        "SOFTTISSUES"]


@pytest.fixture(scope="module")
def slice_() -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        pytest.skip("train.csv not present")
    return pd.read_csv(TRAIN_CSV).head(80)


@pytest.fixture(scope="module")
def full() -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        pytest.skip("train.csv not present")
    return pd.read_csv(TRAIN_CSV)


@pytest.fixture(scope="module")
def router(full) -> FieldRouter:
    """A real fit on the real training data.

    The full set rather than a slice: on 80 rows the learned DROP class is
    over-eager (it wins on 43% of sentences against a 26% gold share) and the
    facet-versus-DROP margin flips sign, so a small fit would be testing the
    slice instead of the router.
    """
    return train(full)


# ---------------------------------------------------------------- the mask

def test_route_is_masked_to_the_given_candidates(router):
    candidates = ["BONES", "SOFTTISSUES"]
    for sent in ("acute fracture of the distal radius",
                 "mild degenerative changes of the lumbar spine",
                 "there is a small pleural effusion"):
        res = router.route(sent, candidates)
        assert res.label in set(candidates) | {DROP}


def test_route_never_invents_a_label_the_caller_did_not_offer(router):
    for sent in ("no acute abnormality", "mild rotator cuff tear",
                 "the lungs are clear"):
        assert router.route(sent, ["BONES"]).label in {"BONES", DROP}


def test_mask_binds_on_a_real_case(router):
    """The mask is worth ~+19pp, so prove it actually changes the answer."""
    frame = full_or_skip()
    checked = 0
    for _, row in frame.head(120).iterrows():
        tpl = template_fields(row.template_content)
        if len(tpl) < 2:
            continue
        for sent in split_dictation(row.dictation):
            masked = router.route(sent, tpl.keys(), row.body_part, tpl)
            free = router.route(sent, router.classes, row.body_part, tpl)
            if masked.label != free.label:
                checked += 1
    assert checked > 0


def full_or_skip() -> pd.DataFrame:
    if not TRAIN_CSV.exists():
        pytest.skip("train.csv not present")
    return pd.read_csv(TRAIN_CSV)


def test_route_document_only_emits_template_labels(router):
    checked = 0
    for _, row in full_or_skip().head(60).iterrows():
        tpl = set(template_fields(row.template_content))
        out = router.route_document(row.template_content, row.dictation,
                                    row.body_part)
        assert set(out) <= tpl, f"invented {set(out) - tpl}"
        checked += len(out)
    assert checked > 0


def test_route_document_covers_most_templates(router):
    """A router that returned nothing would pass every test above. Pin coverage."""
    n_sent = n_routed = 0
    for _, row in full_or_skip().head(60).iterrows():
        out = router.route_document(row.template_content, row.dictation,
                                    row.body_part)
        n_sent += len(split_dictation(row.dictation))
        n_routed += sum(len(v) for v in out.values())
    assert n_routed / n_sent > 0.6, f"only routed {n_routed}/{n_sent}"


# ------------------------------------------------------------------- DROP

def test_route_document_never_routes_outside_the_template(router):
    for _, row in full_or_skip().head(60).iterrows():
        tpl = set(template_fields(row.template_content))
        out = router.route_document(row.template_content, row.dictation,
                                    row.body_part)
        assert set(out) <= tpl


def test_technique_noise_lands_nowhere_or_in_a_real_field(router):
    noise = ("Right Hip Radiographs, 2 Views. Right hip pain. "
             "Most likely differential diagnoses. ")
    out = router.route_document(TPL_BODY, noise)
    assert set(out) <= {"BONES", "SOFTTISSUES"}


def test_findings_are_routed_more_often_than_technique_noise(router):
    noise = ("Right Hip Radiographs, 2 Views. Most likely differential "
             "diagnoses. Clinical correlation recommended.")
    finding = "acute comminuted fracture of the distal radius"
    n_noise = sum(len(v) for v in router.route_document(TPL_BODY, noise).values())
    n_find = sum(len(v) for v in
                 router.route_document(TPL_BODY, finding).values())
    assert n_find > n_noise


def test_drop_class_exists_and_is_reachable(slice_):
    sentences, labels, body_parts, _groups = build_training_set(slice_)
    assert DROP in labels, "the training set must contain DROP examples"
    r = FieldRouter().fit(sentences, labels, body_parts)
    assert DROP in r.classes


TOY = [
    ("acute comminuted fracture of the distal radius", "BONES"),
    ("there is a small right pleural effusion", "BONES"),
    ("the lungs are clear without infiltrate", "BONES"),
    ("mild degenerative change of the lumbar spine", "VERTEBRAE"),
    ("mild thoracic spondylosis is present", "VERTEBRAE"),
    ("no compression fracture identified", "VERTEBRAE"),
    ("right hip radiographs 2 views", DROP),
    ("most likely differential diagnoses", DROP),
    ("clinical correlation is recommended", DROP),
    ("comparison is made with the prior examination", DROP),
]


def _toy(**kw) -> FieldRouter:
    return FieldRouter(**kw).fit([s for s, _ in TOY], [y for _, y in TOY],
                                ["XR"] * len(TOY))


def test_margin_is_monotone_in_accuracy(router, full):
    """The margin is a real confidence estimate, which is the one thing here
    worth wiring into a best-of-N selector. Out-of-fold on the full set the
    lowest-margin bucket is ~0.46 accurate and the highest ~0.93; this pins the
    ordering (the fit is in-sample, so only the gap, not the levels, is
    meaningful)."""
    _sents, labels, _bps, groups = build_training_set(
        full, min_overlap=0.4, min_margin=0.0)
    hit = {"low": [0, 0], "high": [0, 0]}
    for s, y, g in zip(_sents, labels, groups):
        if y == DROP:
            continue
        tpl = template_fields(full.template_content.iloc[int(g)])
        r = router.route(s, tpl.keys(), None, tpl)
        if r.dropped:
            continue
        k = "high" if r.margin >= 20 else ("low" if r.margin < 5 else None)
        if k is None:
            continue
        hit[k][0] += r.label == y
        hit[k][1] += 1
    assert hit["low"][1] > 100 and hit["high"][1] > 100, hit
    acc_lo = hit["low"][0] / hit["low"][1]
    acc_hi = hit["high"][0] / hit["high"][1]
    assert acc_hi > acc_lo + 0.20, f"low-margin {acc_lo:.3f} vs high {acc_hi:.3f}"


def test_deterministic_rules_survive_a_drop_threshold(router):
    """A `drop_threshold` must not be able to throw away a decision that a
    hard rule already made. Both deterministic rules report an infinite margin."""
    tpl = template_fields(TPL_BODY)
    exact = tpl["SOFTTISSUES"]
    facet = "mild facet arthropathy at L4-L5"
    for sent in (exact, facet):
        res = router.route(sent, ["BONES", "SOFTTISSUES", VERTEBRAE,
                                  DISCS_DEGENERATIVE], "XR SPINE", tpl)
        assert res.source in {"subset", "facet"}
        assert res.margin == float("inf")
    out = router.route_document(TPL_SPINE, facet, "XR SPINE",
                                drop_threshold=1e6)
    assert out, "a high drop_threshold discarded a facet-rule decision"


def test_drop_threshold_controls_how_much_is_left_unrouted():
    d = "acute comminuted fracture of the distal radius"
    assert sum(len(v) for v in _toy().route_document(TPL_BODY, d, "XR HAND").values()) == 1
    assert _toy(drop_threshold=1e9).route_document(TPL_BODY, d, "XR HAND") == {}, \
        "an unreachable margin must drop everything"


def test_drop_rate_is_calibrated_to_the_measured_noise_share(router):
    """The learned DROP class is a weak but real signal.

    Measured out-of-fold on the full set it reaches only precision 0.639 /
    recall 0.483, so it is NOT a clean separator -- the `drop_threshold` margin
    is the better lever (0.798 precision at 12.5% unrouted, 0.950 at 74%). What
    the class *can* do is drop roughly the right *number* of sentences: 15.1%
    of train.csv sentences are technique/history noise.
    """
    frame = full_or_skip()
    base = fired = 0
    for g in range(60):
        row = frame.iloc[g]
        tpl = template_fields(row.template_content)
        for sent in split_dictation(row.dictation):
            base += 1
            if router.route(sent, tpl.keys(), row.body_part, tpl).label == DROP:
                fired += 1
    rate = fired / max(1, base)
    assert 0.07 < rate < 0.22, f"DROP fires on {rate:.1%} of sentences, 15.1% expected"


def test_drop_class_separates_noise_from_findings_when_told_to():
    """Mechanism test on a hand-built corpus, independent of the real data."""
    r = _toy()

    def gap(sent):
        cands = ["BONES", "VERTEBRAE"]
        sc = r._scores(sent, "XR")
        return float(sc[r._index[DROP]]) - max(
            float(sc[r._index[c]]) for c in cands if c in r._index)

    noise = ["right hip radiographs 2 views",
             "most likely differential diagnoses",
             "clinical correlation is recommended"]
    finding = ["acute comminuted fracture of the distal radius",
               "there is a small right pleural effusion",
               "mild degenerative change of the lumbar spine"]
    assert min(gap(s) for s in noise) > max(gap(s) for s in finding)


# ------------------------------------------------- OTHER FINDINGS is a trap

def test_other_findings_is_never_top1(router):
    for sent in ("mild degenerative change of the lumbar spine",
                 "small right pleural effusion",
                 "the lungs are clear",
                 "there is a 1 cm hepatic cyst",
                 "surgical clips over the pelvis",
                 "atherosclerosis of the intracranial arteries",
                 "nonobstructing renal calculus"):
        res = router.route(sent, CAND)
        assert res.label != OTHER_FINDINGS, f"{sent!r} -> OTHER FINDINGS"
        assert res.label in set(CAND) | {DROP}


def test_other_findings_is_never_top1_over_real_templates(router):
    n = 0
    for _, row in full_or_skip().head(300).iterrows():
        tpl = template_fields(row.template_content)
        if OTHER_FINDINGS not in tpl or len(tpl) < 2:
            continue
        for sent in split_dictation(row.dictation):
            res = router.route(sent, tpl.keys(), row.body_part, tpl)
            assert res.label != OTHER_FINDINGS
            n += 1
    assert n > 200, f"only {n} sentences exercised the OTHER FINDINGS ban"


def test_other_findings_still_reachable_when_it_is_the_only_option():
    corpus = [("a small nonspecific density over the pelvis", OTHER_FINDINGS),
              ("surgical clips project over the pelvis", OTHER_FINDINGS),
              ("acute comminuted fracture of the distal radius", "BONES"),
              ("there is a small right pleural effusion", "BONES"),
              ("right hip radiographs 2 views", DROP),
              ("most likely differential diagnoses", DROP)]
    r = FieldRouter().fit([s for s, _ in corpus], [y for _, y in corpus],
                          ["XR"] * len(corpus))
    assert r.route("a small nonspecific density", [OTHER_FINDINGS]).label \
        == OTHER_FINDINGS
    # ... but only when it is genuinely the sole option.
    assert r.route("a small nonspecific density",
                   [OTHER_FINDINGS, "BONES"]).label != OTHER_FINDINGS

# ------------------------------------------------------------- the facet rule

def test_facet_goes_to_vertebrae_or_discs_never_other_findings(router):
    for sent in ("mild facet arthropathy at L4-L5",
                 "facet arthropathy at the lumbar spine",
                 "hypertrophic arthropathy of the facet joints",
                 "there is mild facet joint hypertrophy"):
        cands = [OTHER_FINDINGS, VERTEBRAE, DISCS_DEGENERATIVE, "SOFTTISSUES",
                 "BONES"]
        res = router.route(sent, cands)
        assert res.label in {VERTEBRAE, DISCS_DEGENERATIVE}, \
            f"{sent!r} -> {res.label}"
        assert res.source == "facet"


def test_facet_prefers_discs_when_the_template_has_both(router):
    """MEASURED, and the reverse of the obvious ordering.

    Over all 636 rows: template has VERTEBRAE only (n=44) -> reference uses
    VERTEBRAE 42 times, DISCS 0. Template has BOTH (n=55) -> reference uses
    VERTEBRAE **0** times, DISCS 16. These are MRI-spine templates whose own
    `DISCS/DEGENERATIVE CHANGES` normal already reads "No significant facet
    arthropathy". DISCS-first scores 58/99, VERTEBRAE-first 42/99.
    """
    res = router.route("mild facet arthropathy at L4-L5",
                       [VERTEBRAE, DISCS_DEGENERATIVE])
    assert res.label == DISCS_DEGENERATIVE
    # candidate order must not matter
    res = router.route("mild facet arthropathy at L4-L5",
                       [DISCS_DEGENERATIVE, VERTEBRAE])
    assert res.label == DISCS_DEGENERATIVE


def test_facet_uses_vertebrae_when_the_template_has_no_discs_field(router):
    res = router.route("mild facet arthropathy at L4-L5",
                       [VERTEBRAE, "SOFTTISSUES"])
    assert res.label == VERTEBRAE
    res = router.route("mild facet arthropathy at L4-L5",
                       ["SOFTTISSUES", VERTEBRAE])
    assert res.label == VERTEBRAE


def test_the_vertebrae_first_ordering_is_reproducible(router):
    """The ordering NOTES.md describes is still available, one kwarg away."""
    from natoe.field_router import FACET_PREFERS_VERTEBRAE_FIRST

    alt = FieldRouter(facet_prefers=FACET_PREFERS_VERTEBRAE_FIRST).fit(
        [s for s, _ in TOY], [y for _, y in TOY], ["XR"] * len(TOY))
    assert alt.route("mild facet arthropathy at L4-L5",
                     [VERTEBRAE, DISCS_DEGENERATIVE]).label == VERTEBRAE


def test_facet_falls_back_when_the_template_has_neither(router):
    """No VERTEBRAE and no DISCS -> the rule must not invent a label."""
    res = router.route("mild facet arthropathy at L4-L5",
                       [OTHER_FINDINGS, "CARTILAGE", "SOFTTISSUES"])
    assert res.label in {"CARTILAGE", "SOFTTISSUES", DROP}


# --------------------------------------------------------- the subset shortcut

def test_subset_lookup_beats_the_model_on_its_slice(router):
    tpl = template_fields(TPL_BODY)
    exact = tpl["SOFTTISSUES"]
    res = router.route(exact, tpl.keys(), "XR HIP", tpl)
    assert res.source == "subset"
    assert res.label == "SOFTTISSUES"


def test_subset_lookup_does_not_fire_on_short_sentences(router):
    tpl = template_fields(TPL_BODY)
    assert router.route("bones", tpl.keys(), "XR HIP", tpl).source != "subset"


def test_subset_lookup_needs_the_field_text(router):
    tpl = template_fields(TPL_BODY)
    exact = tpl["SOFTTISSUES"]
    assert router.route(exact, tpl.keys(), "XR HIP").source != "subset"


# ----------------------------------------------------------------- training

def test_training_on_a_small_slice_runs_and_predicts(slice_):
    sentences, labels, body_parts, groups = build_training_set(slice_)
    assert len(sentences) == len(labels) == len(body_parts) == len(groups)
    assert len(sentences) > 200
    r = FieldRouter().fit(sentences, labels, body_parts)
    assert r.vocab and r._ll is not None
    res = r.route("mild degenerative changes of the lumbar spine",
                  ["VERTEBRAE", "DISCSDEGENERATIVECHANGES"], "XR SPINE")
    assert res.label in {"VERTEBRAE", "DISCSDEGENERATIVECHANGES", DROP}
    assert res.alternatives


def test_build_training_set_output_is_well_formed(slice_):
    sentences, labels, body_parts, groups = build_training_set(slice_)
    known = set()
    for tpl in slice_.template_content:
        known |= set(template_fields(tpl))
    unknown = {l for l in labels if l != DROP} - known
    assert not unknown, f"labels absent from every template: {unknown}"
    assert all(isinstance(s, str) and s for s in sentences)
    assert all(g in {str(i) for i in range(len(slice_))} for g in groups)
    # The DROP class must not be a rounding error: 13.7% of sentences are
    # technique/history noise, so a plausible labelled set keeps a chunk of them.
    assert 0.05 < sum(1 for l in labels if l == DROP) / len(labels) < 0.6


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        FieldRouter().fit(["a", "b"], ["BONES"])
    with pytest.raises(ValueError):
        FieldRouter().fit(["a"], ["BONES"], ["XR", "CT"])


def test_train_rejects_unknown_kwargs(slice_):
    with pytest.raises(TypeError):
        train(slice_, nonsense=1)


# ------------------------------------------------------------------ plumbing

def test_template_fields_canonicalises_case():
    tpl = "FINDINGS:\nBONES: a\nBones: b\nIMPRESSION:\nc"
    fields = template_fields(tpl)
    assert "BONES" in fields
    assert len(fields) == 1, "BONES and Bones must collapse to one key"


def test_template_fields_ignores_the_impression_section():
    assert set(template_fields(TPL_SPINE)) == {"VERTEBRAE",
                                                "DISCSDEGENERATIVECHANGES",
                                                "SOFTTISSUES"}


def test_route_document_keys_are_scorer_ready(router):
    """The keys must be directly usable by the RES scorer."""
    fields, display, unlabelled = parse_fields(split_sections(TPL_SPINE)[0])
    assert not unlabelled
    assert set(display) == {"VERTEBRAE", "DISCSDEGENERATIVECHANGES",
                            "SOFTTISSUES"}
    out = router.route_document(TPL_SPINE, "mild facet arthropathy at L4-L5",
                                "XR SPINE")
    assert set(out) <= set(fields)


def test_split_dictation_keeps_list_markers_glued():
    out = split_dictation("1. Mild spondylosis.\n2. No effusion.")
    assert out[0].startswith("1.")
    assert len(out) == 2


def test_route_with_no_candidates_returns_drop(router):
    assert router.route("anything", []).label == DROP


def test_route_document_is_empty_for_an_empty_dictation(router):
    assert router.route_document(TPL_BODY, "") == {}


def test_route_document_respects_a_per_call_drop_threshold():
    d = "acute comminuted fracture of the distal radius"
    assert sum(len(v) for v in
               _toy().route_document(TPL_BODY, d, "XR HAND").values()) == 1
    assert _toy().route_document(TPL_BODY, d, "XR HAND",
                                 drop_threshold=1e9) == {}
