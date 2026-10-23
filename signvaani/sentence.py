"""Turn a sequence of recognised ISL glosses into a sentence (English / Hindi).

ISL drops articles and tense markers and often puts the topic first, e.g.
"ME STOMACH PAIN". Patterns in rules.json map such gloss sequences to fluent
templates. Unmatched input falls back to a plain word-by-word rendering.
"""
import json
from pathlib import Path

RULES_PATH = Path(__file__).with_name("rules.json")


class SentenceBuilder:
    def __init__(self, rules_path=RULES_PATH):
        rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        self.lexicon = rules["lexicon"]
        self.patterns = rules["patterns"]

    @staticmethod
    def dedupe(glosses):
        out = []
        for g in glosses:
            if not out or out[-1] != g:
                out.append(g)
        return out

    def _matches(self, pattern, glosses):
        if len(pattern) != len(glosses):
            return False
        for p, g in zip(pattern, glosses):
            entry = self.lexicon.get(g)
            if p != g and (entry is None or entry["cat"] != p):
                return False
        return True

    def build(self, glosses, lang="en"):
        """Return (sentence, matched) where matched is False for the fallback."""
        glosses = [g for g in self.dedupe(glosses) if g != "IDLE"]
        if not glosses:
            return "", False
        for pat in self.patterns:
            if lang in pat and self._matches(pat["match"], glosses):
                words = [self.lexicon[g].get(lang, self.lexicon[g]["en"]) for g in glosses]
                return pat[lang].format(*words), True
        words = [self.lexicon.get(g, {}).get(lang) or g.lower().replace("-", " ") for g in glosses]
        text = " ".join(words)
        if lang == "en":
            text = text[0].upper() + text[1:] + "."
        return text, False
