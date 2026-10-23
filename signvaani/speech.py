"""Text-to-speech. English works offline via pyttsx3; Indian languages use gTTS
(needs internet) until an offline Indic TTS model is plugged into `_speak_indic`.
"""
import hashlib
import threading
from pathlib import Path

CACHE = Path.home() / ".cache" / "signvaani_tts"


def _speak_english(text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def _speak_indic(text, lang):
    # Swap this for AI4Bharat Indic-TTS to run fully offline.
    from gtts import gTTS
    import pygame
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{lang}_{hashlib.md5(text.encode()).hexdigest()}.mp3"
    if not f.exists():
        gTTS(text, lang=lang).save(str(f))
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
        try:
            _speak_english(text) if lang == "en" else _speak_indic(text, lang)
        except Exception as e:  # never crash the live app because of audio
            print(f"[tts] could not speak ({e})")

    if block:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()
