"""Speech without speakers or internet: fake engines record what would be said or downloaded."""
import sys
import threading
import time
import types

import pytest

from signvaani import speech


def test_sentences_are_spoken_one_at_a_time(monkeypatch):
    log, busy = [], threading.Event()

    def engine(text):
        assert not busy.is_set(), "two sentences spoken at once"
        busy.set()
        time.sleep(0.05)
        log.append(text)
        busy.clear()

    monkeypatch.setattr(speech, "_speak_english", engine)
    speech.speak("I need water.")
    speech.speak("Please help me.")  # pressed SPACE again while the first sentence is playing
    speech.speak("Yes.", block=True)
    deadline = time.time() + 2
    while len(log) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert sorted(log) == sorted(["I need water.", "Please help me.", "Yes."])


def test_audio_errors_never_reach_the_app(monkeypatch, capsys):
    monkeypatch.setattr(speech, "_speak_indic", lambda text, lang: 1 / 0)
    speech.speak("मुझे पानी चाहिए।", "hi", block=True)
    assert "[tts] could not speak" in capsys.readouterr().out


def test_cache_saves_each_hindi_sentence_once(monkeypatch, tmp_path, capsys):
    downloads = []

    class FakeGTTS:
        def __init__(self, text, lang):
            downloads.append((text, lang))

        def save(self, path):
            open(path, "wb").write(b"mp3")

    monkeypatch.setitem(sys.modules, "gtts", types.SimpleNamespace(gTTS=FakeGTTS))
    monkeypatch.setattr(speech, "CACHE", tmp_path / "tts")
    sentences = speech.demo_sentences()
    assert speech.main(["--cache"]) == 0
    assert len(downloads) == len(sentences) and {lang for _, lang in downloads} == {"hi"}
    assert len(list((tmp_path / "tts").glob("hi_*.mp3"))) == len(sentences)
    assert "Hindi now works offline" in capsys.readouterr().out
    assert speech.main(["--cache"]) == 0 and len(downloads) == len(sentences)  # cached: nothing downloaded again


def test_cache_reports_no_internet(monkeypatch, tmp_path, capsys):
    class Offline:
        def __init__(self, text, lang):
            raise ConnectionError("no internet")

    monkeypatch.setitem(sys.modules, "gtts", types.SimpleNamespace(gTTS=Offline))
    monkeypatch.setattr(speech, "CACHE", tmp_path / "tts")
    assert speech.main(["--cache"]) == 1
    assert "connect to the internet and run --cache again" in capsys.readouterr().out


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_demo_speaks_s1_to_s10_in_order(monkeypatch, lang):
    said = []
    monkeypatch.setattr(speech, "speak", lambda text, lang, block=False: said.append((text, lang, block)))
    assert speech.main(["--demo", "--lang", lang]) == 0
    assert said == [(s[lang], lang, True) for s in speech.demo_sentences()]
