import json
from pathlib import Path

import pytest

from signvaani.sentence import RULES_PATH, SentenceBuilder

b = SentenceBuilder()


def test_isl_word_order_to_english():
    assert b.build(["ME", "STOMACH", "PAIN"]) == ("My stomach hurts.", True)
    assert b.build(["ME", "PAIN", "HEAD"])[0] == "My head hurts."
    assert b.build(["ME", "FEVER", "YESTERDAY"])[0] == "I have had a fever since yesterday."
    assert b.build(["TOILET", "WHERE"])[0] == "Where is the toilet?"
    assert b.build(["DOCTOR", "CALL"])[0] == "Please call the doctor."
    assert b.build(["ME", "NEED", "WATER"])[0] == "I need water."


def test_hindi():
    assert b.build(["ME", "STOMACH", "PAIN"], "hi")[0] == "मेरे पेट में दर्द है।"
    assert b.build(["ME", "NEED", "WATER"], "hi")[0] == "मुझे पानी चाहिए।"


def test_repeats_and_idle_ignored():
    assert b.build(["HELP", "HELP", "IDLE"])[0] == "Please help me."


def test_fallback():
    text, matched = b.build(["WATER", "TODAY"])
    assert not matched and text == "Water today."


def test_every_vocabulary_sign_is_in_lexicon():
    vocab = json.loads((Path(__file__).parents[1] / "configs" / "vocabulary.json").read_text())["signs"]
    lexicon = json.loads(RULES_PATH.read_text(encoding="utf-8"))["lexicon"]
    assert [s for s in vocab if s != "IDLE" and s not in lexicon] == []


DEMO = json.loads((Path(__file__).parents[1] / "configs" / "sentences.json").read_text(encoding="utf-8"))["sentences"]


@pytest.mark.parametrize("s", DEMO, ids=[s["id"] for s in DEMO])
def test_demo_sentences_s1_to_s10(s):
    """TC-07 / TC-08 offline: every frozen demo sentence comes out exactly, in English and Hindi."""
    assert b.build(s["glosses"], "en") == (s["en"], True)
    assert b.build(s["glosses"], "hi") == (s["hi"], True)


@pytest.mark.parametrize("s", DEMO, ids=[s["id"] for s in DEMO])
def test_demo_sentences_survive_live_noise(s):
    """The spotter can repeat a sign or let IDLE through between signs; the sentence must not change."""
    noisy = ["IDLE"] + [g for gloss in s["glosses"] for g in (gloss, gloss, "IDLE")]
    assert b.build(noisy, "en")[0] == s["en"] and b.build(noisy, "hi")[0] == s["hi"]
