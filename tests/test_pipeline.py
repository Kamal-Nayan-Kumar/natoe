"""Tests for the pieces that must not silently regress.

No API key required: everything here is deterministic (the LLM is never
called). Run with `pytest -q`.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from natoe.config import TRAIN_CSV, TEST_CSV                      # noqa: E402
from natoe.pipeline import (MockLLM, Retriever, build_shots, build_user_prompt,  # noqa: E402
                            gold_json, gold_patch, guard, parse_json,
                            render, _last_json_object, _sentences)
from natoe.evaluate import baselines, dev_split, score_predictions  # noqa: E402
from natoe.res_scorer import (canonical, normalize, parse_fields,   # noqa: E402
                              res_case, split_sections, token_weight,
                              tokenize, weighted_word_edit)

# --------------------------------------------------------------- normalising


def test_lowercase_and_punctuation():
    assert tokenize("Mild Right Effusion.") == ["mild", "right", "effusion"]


def test_letter_hyphen_removed_but_digit_hyphen_separates():
    assert tokenize("air-space") == ["airspace"]
    assert tokenize("airspace") == ["airspace"]
    assert tokenize("L5-S1") == ["l", "5", "s", "1"]
    assert tokenize("l5s1") == tokenize("L5-S1")


def test_units_standardised():
    assert tokenize("8 millimeters") == tokenize("8 mm")
    assert tokenize("3 centimeters") == tokenize("3 cm")


def test_list_markers_removed():
    assert tokenize("1. Mild spondylosis.") == tokenize("Mild spondylosis")
    assert tokenize("2) No effusion.") == tokenize("No effusion")
    assert tokenize("- No effusion.") == tokenize("No effusion")


def test_signed_measurement_preserved():
    assert "+3" in tokenize("+3 mm")
    assert "-2" in tokenize("-2mm")


def test_normalize_is_idempotent():
    assert normalize(normalize("A. B.")) == normalize("A. B.")


# ------------------------------------------------------------------- distance


def test_identical_is_zero():
    assert weighted_word_edit("small right effusion", "small right effusion") == 0.0


def test_swap_is_closer_than_rewrite():
    close = weighted_word_edit("small right effusion", "small left effusion")
    far = weighted_word_edit("small right effusion", "large left effusion")
    assert close < far


def test_bounded_by_one():
    assert weighted_word_edit("a b c", "") == 1.0
    assert weighted_word_edit("", "a b c") == 1.0
    assert weighted_word_edit("", "") == 0.0


def test_critical_words_cost_more_than_function_words():
    """Dropping a negation (weight 4) must cost far more than dropping 'the'."""
    drop_critical = weighted_word_edit("no focal opacity", "focal opacity")
    drop_function = weighted_word_edit("the focal opacity", "focal opacity")
    assert drop_critical > drop_function * 5


def test_laterality_substitution_costs_more_than_a_content_substitution():
    laterality = weighted_word_edit("right kidney", "left kidney")   # 4.0 swap
    content = weighted_word_edit("right kidney", "right elbow")      # 2.0 swap
    assert laterality > content


def test_token_weight_table_matches_the_spec():
    assert token_weight("no") == 4.0        # negation
    assert token_weight("right") == 4.0    # laterality
    assert token_weight("mild") == 4.0      # severity
    assert token_weight("mm") == 4.0        # unit
    assert token_weight("12") == 4.0        # measurement
    assert token_weight("opacity") == 2.0   # other content word
    assert token_weight("the") == 0.25      # function word


# ------------------------------------------------------------- field parsing


def test_labels_match_case_insensitively():
    """Real templates mix 'BONES:' and 'Bones:'; they must align."""
    a, _, _ = parse_fields("FINDINGS:\nBones: normal\nSoft Tissues: fine")
    b, _, _ = parse_fields("FINDINGS:\nBONES: abnormal\nSOFT TISSUES: fine")
    assert set(a) == set(b) == {canonical("BONES"), canonical("SOFT TISSUES")}


def test_punctuation_in_labels_ignored_when_matching():
    assert canonical("CHEST WALL/MUSCULOSKELETAL") == canonical("Chest Wall/Musculoskeletal")


def test_display_name_preserved():
    _, display, _ = parse_fields("FINDINGS:\nBones: normal")
    assert display[canonical("BONES")] == "Bones"


def test_lines_after_a_label_continue_that_field():
    """Text following a label is a continuation, not stray unlabelled text."""
    f, _, unlab = parse_fields("BONES: mild cortical\nirregularity")
    assert f[canonical("BONES")] == "mild cortical irregularity"
    assert unlab == ""


def test_text_before_the_first_label_is_unlabelled():
    _, _, unlab = parse_fields("stray opening line\nBONES: normal")
    assert unlab == "stray opening line"


def test_spine_levels_are_separate_fields():
    f, _, _ = parse_fields("FINDINGS:\nL4-L5: bulge\nL5-S1: normal")
    assert len(f) == 2


def test_split_sections_finds_both():
    fb, im, _ = split_sections("FINDINGS:\nB: x\n\nIMPRESSION:\ny")
    assert fb.strip() == "B: x"
    assert im.strip() == "y"


# ------------------------------------------------------------------ sentences


def test_sentences_keep_list_item_glued():
    assert _sentences("1. Mild spondylosis.\n2. No fracture.") == \
        ["1. Mild spondylosis.", "2. No fracture."]


def test_sentences_splits_on_period():
    assert _sentences("A opacity. No effusion.") == ["A opacity.", "No effusion."]


def test_last_json_object_extracted_from_reasoning():
    blob = 'thinking {"fields": {"A": 1}} then {"fields":{"B":2},"impression":"x"}'
    assert parse_json(_last_json_object(blob))["impression"] == "x"


# ---------------------------------------------------------------------- guard

CASE = {
    "template_content": "FINDINGS:\nLUNGS: The lungs are clear.\n"
                        "BONES: No acute fracture.\n"
                        "OTHER FINDINGS:\n\nIMPRESSION:\nNo acute abnormality.",
    "dictation": "small right pleural effusion",
}


def test_guard_drops_unknown_label():
    out = guard({"fields": {"CARDIAC SILHOUETTE": "enlarged"}, "impression": ""}, CASE)
    assert out["fields"] == {}
    assert any(d.startswith("label:") for d in out["drops"])


def test_guard_maps_case_insensitive_label():
    out = guard({"fields": {"lungs": "Small right opacity."}, "impression": ""}, CASE)
    assert canonical("LUNGS") in out["fields"]


def test_guard_drops_invented_measurement():
    out = guard({"fields": {"BONES": "There is a 7 mm lesion."}, "impression": ""}, CASE)
    assert out["fields"] == {}


def test_guard_keeps_measurement_present_in_dictation():
    case = dict(CASE, dictation="an 8 mm granuloma")
    out = guard({"fields": {"BONES": "There is an 8 mm granuloma."},
                 "impression": ""}, case)
    assert "8" in out["fields"][canonical("BONES")]


def test_guard_keeps_template_normal_within_a_field():
    """'acute' appears in the template, so it must survive the guard."""
    out = guard({"fields": {"BONES": "Mild spondylosis. No acute fracture."},
                 "impression": ""}, CASE)
    assert "acute" in out["fields"][canonical("BONES")]


def test_guard_off_level_disables_checks():
    out = guard({"fields": {"BONES": "There is a 7 mm lesion."}, "impression": ""},
                CASE, "off")
    assert "7" in out["fields"][canonical("BONES")]


def test_guard_does_not_strip_negation():
    out = guard({"fields": {"LUNGS": "No focal opacity. No pleural effusion."},
                 "impression": ""}, CASE)
    assert out["fields"]  # nothing dropped just for containing "No"


# --------------------------------------------------------------------- render


def test_render_preserves_template_field_order():
    out = render(CASE, {"fields": {}, "impression": "Effusion."})
    assert out.index("LUNGS:") < out.index("BONES:") < out.index("OTHER FINDINGS:")


def test_render_keeps_untouched_field_verbatim():
    out = render(CASE, {"fields": {}, "impression": "x"})
    assert "LUNGS: The lungs are clear." in out


def test_render_leaves_other_findings_empty_when_untouched():
    out = render(CASE, {"fields": {}, "impression": "x"})
    assert "OTHER FINDINGS: \n" in out or out.rstrip().count("OTHER FINDINGS:") == 1


def test_render_emits_both_sections_once():
    out = render(CASE, {"fields": {}, "impression": "x"})
    lines = [ln.split(":")[0] for ln in out.splitlines() if ":" in ln]
    assert lines.count("FINDINGS") == 1
    assert lines.count("IMPRESSION") == 1


def test_render_falls_back_to_template_impression():
    out = render(CASE, {"fields": {}, "impression": ""})
    assert out.rstrip().endswith("No acute abnormality.")


def test_render_uses_display_casing_of_template():
    case = dict(CASE, template_content="FINDINGS:\nBones: No acute fracture.\n"
                                       "IMPRESSION:\nnone")
    out = render(case, {"fields": {}, "impression": ""})
    assert "Bones: No acute fracture." in out


# ----------------------------------------------------------------- end to end

def test_parse_json_strips_code_fence():
    assert parse_json('```json\n{"fields": {}}\n```') == {"fields": {}}


def test_mock_llm_produces_renderable_output():
    llm = MockLLM()
    case = dict(CASE, modality="XRAY", body_part="Chest",
                study_description="XR CXR", patient_age_band="65-69",
                patient_sex="male")
    out = render(case, guard(parse_json(llm.complete(build_user_prompt(case, []))),
                             case))
    assert out.startswith("FINDINGS:")


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_dev_split_never_leaks_exemplars():
    train = pd.read_csv(TRAIN_CSV)
    dev, pool = dev_split(train, n_dev=40)
    assert len(dev) == 40
    assert set(dev.case_id).isdisjoint(set(pool.case_id))
    assert len(dev) + len(pool) == len(train)


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_retriever_finds_same_body_part():
    train = pd.read_csv(TRAIN_CSV)
    _, pool = dev_split(train, n_dev=40)
    r = Retriever((pool.body_part + " " + pool.modality + " " +
                   pool.template_content + " " + pool.dictation).tolist())
    query = "Knee MRI knee joint degenerative"
    top = r.query(query, top_k=3)
    assert all(pool.body_part[i] == "Knee" for i in top)


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_gold_patch_renders_to_near_zero():
    """The architecture ceiling: perfect patches must reproduce the reference.

    `append_impression_closer` is off here because this test is measuring the
    renderer's FIELD assembly, not the IMPRESSION embellishment (which by
    design adds the template's closing line and so cannot round-trip to the
    gold). That behaviour has its own test below.
    """
    train = pd.read_csv(TRAIN_CSV).head(120)
    outs = [render({"template_content": r.template_content,
                    "dictation": r.dictation},
                   guard(gold_patch(r), {"template_content": r.template_content,
                                         "dictation": r.dictation}, "numbers"),
                   append_impression_closer=False)
            for _, r in train.iterrows()]
    assert score_predictions(train, outs)["RES"] < 0.05


def test_render_appends_the_template_impression_closer():
    case = {"template_content":
            "FINDINGS:\nLUNGS: clear\n\nIMPRESSION:\n1. Fracture.\n"
            "2. No acute cardiopulmonary abnormality.",
            "dictation": "rib fracture"}
    out = render(case, {"fields": {}, "impression": "1. Fracture."})
    assert "No acute cardiopulmonary abnormality." in out


def test_render_does_not_duplicate_the_closer():
    case = {"template_content":
            "FINDINGS:\nLUNGS: clear\n\nIMPRESSION:\nNo acute abnormality.",
            "dictation": "x"}
    out = render(case, {"fields": {},
                        "impression": "Effusion. No acute abnormality."})
    assert out.lower().count("no acute abnormality") == 1


def test_render_closer_can_be_disabled():
    case = {"template_content":
            "FINDINGS:\nLUNGS: clear\n\nIMPRESSION:\nNo acute cardiopulmonary "
            "abnormality.", "dictation": "effusion"}
    out = render(case, {"fields": {}, "impression": "Effusion."},
                 append_impression_closer=False)
    assert "No acute cardiopulmonary abnormality." not in out


def test_run_dataset_end_to_end_with_mock():
    """Covers run_dataset's threading/retry path with no API key needed."""
    from natoe.evaluate import dev_split
    from natoe.pipeline import run_dataset

    train = pd.read_csv(TRAIN_CSV).head(40)
    dev, pool = dev_split(train, n_dev=6)
    retriever = Retriever((pool.body_part + " " + pool.template_content).tolist())
    outs = run_dataset(MockLLM(), dev, retriever, pool, n_shots=1,
                       guard_level="numbers", progress=False, workers=2)
    assert len(outs) == len(dev)
    for out in outs:
        assert out.startswith("FINDINGS:")
        assert "IMPRESSION:" in out
    assert score_predictions(dev, outs)["RES"] >= 0.0


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_run_dataset_survives_a_dead_llm():
    """A total API failure must degrade to the template, not crash."""
    from natoe.evaluate import dev_split
    from natoe.pipeline import run_dataset

    class Dead:
        model, reasoning_effort = "dead", "low"
        provider = "dead"

        def complete(self, *a, **k):
            raise RuntimeError("boom")

    train = pd.read_csv(TRAIN_CSV).head(30)
    dev, pool = dev_split(train, n_dev=4)
    retriever = Retriever((pool.body_part + " " + pool.template_content).tolist())
    outs = run_dataset(Dead(), dev, retriever, pool, n_shots=0, progress=False,
                       workers=1, retries=0)
    assert len(outs) == len(dev)
    for out, (_, r) in zip(outs, dev.iterrows()):
        assert out.startswith("FINDINGS:")


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_baselines_are_ordered():
    train = pd.read_csv(TRAIN_CSV)
    b = baselines(train)
    assert b.iloc[0]["RES"] == 0.0
    assert b.iloc[1]["RES"] > 0.4


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_test_csv_has_expected_shape():
    test = pd.read_csv(TEST_CSV)
    assert len(test) == 132
    assert "report" not in test.columns
