"""Day 1 freeze (1-week blueprint, Section 2): 15 hospital signs + IDLE and the demo sentences S1-S10."""
import json
from pathlib import Path

from signvaani.data import vocabulary

SENTENCES = Path(__file__).resolve().parents[1] / "configs" / "sentences.json"


def test_vocabulary_is_frozen_at_16_classes():
    signs = vocabulary()
    assert len(signs) == len(set(signs)) == 16 and "IDLE" in signs


def test_demo_sentences_use_exactly_the_vocabulary():
    sentences = json.loads(SENTENCES.read_text(encoding="utf-8"))["sentences"]
    used = {g for s in sentences for g in s["glosses"]}
    assert used == set(vocabulary()) - {"IDLE"}  # no unknown gloss, and every sign is demonstrated
    assert len({s["id"] for s in sentences}) == len(sentences)
    assert all(s["en"] and s["hi"] for s in sentences)
