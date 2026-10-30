"""Text-to-speech. English works offline via pyttsx3; Indian languages use gTTS
(needs internet) until an offline Indic TTS model is plugged into `_speak_indic`.

    python -m signvaani.speech --demo --lang en     # speak S1-S10, one after the other (Day 5, TC-07)
    python -m signvaani.speech --demo --lang hi     # the Hindi sentences (TC-08)
    python -m signvaani.speech --cache              # save the Hindi audio of S1-S10 for offline use (F7)

gTTS audio is cached in ~/.cache/signvaani_tts, so a sentence spoken (or cached) once plays again without
internet. Run --cache on the presentation laptop the day before the review.
"""
import argparse
import hashlib
import json
import threading
from pathlib import Path

CACHE = Path.home() / ".cache" / "signvaani_tts"
SENTENCES = Path(__file__).resolve().parents[1] / "configs" / "sentences.json"
_lock = threading.Lock()  # one sentence at a time: two SPACE presses must not start two speech engines


def cache_file(text, lang):
    return CACHE / f"{lang}_{hashlib.md5(text.encode()).hexdigest()}.mp3"


def save_indic(text, lang):
    """Download the gTTS audio for text into the cache (needs internet) unless it is there already."""
    f = cache_file(text, lang)
    if not f.exists():
        from gtts import gTTS
        CACHE.mkdir(parents=True, exist_ok=True)
        gTTS(text, lang=lang).save(str(f))
    return f


def _speak_english(text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def _speak_indic(text, lang):
    # Swap this for AI4Bharat Indic-TTS to run fully offline.
    import pygame
    f = save_indic(text, lang)
    pygame.mixer.init()
    pygame.mixer.music.load(str(f))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(50)


def speak(text, lang="en", block=False):
    """Speak `text`; runs in a background thread unless block=True."""
    if not text:
        return

    def run():
        with _lock:
            try:
                _speak_english(text) if lang == "en" else _speak_indic(text, lang)
            except Exception as e:  # never crash the live app because of audio
                print(f"[tts] could not speak ({e})")

    if block:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()


def demo_sentences(path=SENTENCES):
    return json.loads(Path(path).read_text(encoding="utf-8"))["sentences"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="speak the demo sentences S1-S10")
    p.add_argument("--lang", default="en", choices=["en", "hi"])
    p.add_argument("--cache", action="store_true", help="save the Hindi audio of S1-S10 for offline use")
    args = p.parse_args(argv)
    if not (args.demo or args.cache):
        p.error("give --demo and/or --cache")

    sentences = demo_sentences()
    if args.cache:
        failed = 0
        for s in sentences:
            try:
                print(f"{s['id']:<5} cached  {save_indic(s['hi'], 'hi')}")
            except Exception as e:  # usually no internet
                failed += 1
                print(f"{s['id']:<5} FAILED  {s['hi']}  ({e})")
        print(f"{len(sentences) - failed} of {len(sentences)} Hindi sentences cached in {CACHE}"
              + ("; connect to the internet and run --cache again" if failed else "; Hindi now works offline"))
        if failed:
            return 1
    if args.demo:
        for s in sentences:
            text = s[args.lang]
            print(f"{s['id']:<5} {' '.join(s['glosses']):<18} -> {text}")
            speak(text, args.lang, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
