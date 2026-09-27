"""Tests for the IMPRESSION strategy.

The measurements these encode came from scripts/analysis/eval_impression.py
replaying cached LLM responses, and from the IMPRESSION analysis pass. They
guard the two findings that are easy to undo by accident:

  * keeping the template IMPRESSION is almost never right (0.7495 mean edit)
  * appending the template's closing line to a dictated summary is a large,
    genuine win (0.5370 -> 0.4492)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from natoe.config import TRAIN_CSV
from natoe.impression import (append_template_closer, dict_sentences,
                              has_impression_header, heuristic_impression,
                              is_abnormal, last_sentence)
from natoe.res_scorer import split_sections, weighted_word_edit


def test_dict_sentences_breaks_on_newline_and_capitalised_period():
    got = dict_sentences("no effusion.\n\nmild spondylosis. Osteophytes noted.")
    assert got == ["no effusion.", "mild spondylosis.", "Osteophytes noted."]


def test_dict_sentences_keeps_lowercase_continuations_joined():
    """Telegraphic dictations run sentences together; do not over-split them."""
    assert dict_sentences("mild spondylosis. osteophytes") == \
        ["mild spondylosis. osteophytes"]


def test_last_sentence_keeps_the_list_marker():
    """Regression: the marker must not be split off, or the closer is lost."""
    assert last_sentence("1. Fracture.\n2. No acute abnormality.") == \
        "2. No acute abnormality."


def test_is_abnormal_understands_negation():
    assert is_abnormal("There is a moderate pleural effusion.")
    assert not is_abnormal("No pleural effusion or pneumothorax.")
    assert not is_abnormal("The lungs are clear.")


def test_impression_header_must_be_the_whole_line():
    """A loose substring match false-positives on clinical prose."""
    assert has_impression_header("FINDINGS\nIMPRESSION\n1. Effusion.")
    assert has_impression_header("CONCLUSION:\nacute diverticulitis")
    assert not has_impression_header("assessment of venous compressibility")
    assert not has_impression_header("Lungs:\n  no consolidation")


def test_header_dictation_is_copied_verbatim():
    tpl = "FINDINGS:\nLUNGS: clear\n\nIMPRESSION:\nNo acute abnormality."
    got = heuristic_impression(tpl, "Lungs: clear\nIMPRESSION\n1. Acute PE.")
    assert got.strip() == "1. Acute PE."


def test_short_dictation_gets_the_template_closer():
    tpl = ("FINDINGS:\nVERTEBRAE: normal\n\nIMPRESSION:\n"
           "Degenerative changes of the lumbar spine.")
    got = heuristic_impression(tpl, "lumbar spondylosis , osteophytes")
    assert "osteophytes" in got
    assert "Degenerative changes of the lumbar spine." in got


def test_long_dictation_does_not_get_the_template_closer():
    """A long dictation is a self-contained mini-report; the template adds noise."""
    tpl = ("FINDINGS:\nVERTEBRAE: normal\n\nIMPRESSION:\n"
           "Degenerative changes of the lumbar spine.")
    dictation = "\n".join(f"finding number {i} is present" for i in range(20))
    got = heuristic_impression(tpl, dictation)
    assert "Degenerative changes of the lumbar spine." not in got


def test_append_closer_is_idempotent():
    tpl = "FINDINGS:\nX: y\n\nIMPRESSION:\nNo acute cardiopulmonary abnormality."
    once = append_template_closer("Mild effusion.", tpl)
    twice = append_template_closer(once, tpl)
    assert once == twice


def test_append_closer_handles_empty_input():
    tpl = "FINDINGS:\nX: y\n\nIMPRESSION:\nNo acute abnormality."
    assert append_template_closer("", tpl) == ""
    assert append_template_closer("Effusion.", "FINDINGS:\nX: y\n\nIMPRESSION:") \
        == "Effusion."


def test_last_sentence_picks_the_closing_normal_line():
    assert last_sentence("Mild fracture.\nNo acute abnormality.") == \
        "No acute abnormality."


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="train.csv not present")
def test_heuristic_beats_template_verbatim():
    """The headline: keeping the template IMPRESSION is a losing default."""
    train = pd.read_csv(TRAIN_CSV)
    heur = [weighted_word_edit(split_sections(r.report)[1],
                               heuristic_impression(r.template_content,
                                                    r.dictation))
            for _, r in train.iterrows()]
    tpl = [weighted_word_edit(split_sections(r.report)[1],
                              split_sections(r.template_content)[1])
           for _, r in train.iterrows()]
    assert sum(heur) / len(heur) < sum(tpl) / len(tpl)
