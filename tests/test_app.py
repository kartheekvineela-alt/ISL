"""Live app control flow with a fake camera and a fake MediaPipe (no webcam, no model)."""
import importlib
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from signvaani.data import vocabulary

Point = types.SimpleNamespace
NORMAL, KEEPRATIO, FULLSCREEN = 0, 0, 1  # the real cv2 values (WINDOW_AUTOSIZE, the old default, is 1)
CHAR = 16.8  # pixels per character at scale 1, about what cv2's FONT_HERSHEY_SIMPLEX measures
W = 640  # the camera frame width


def text_size(text, font, scale, thickness):
    return (int(len(text) * CHAR * scale), int(22 * scale)), 5


def landmarks(points):
    return Point(landmark=[Point(x=x, y=y, z=0.0) for x, y in points])


class MovingBody:
    """Holistic stand-in: shoulders still, left wrist and hand following `xs`, one value per frame."""

    def __init__(self, xs):
        self.xs = list(xs)

    def __call__(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def process(self, rgb):
        x = self.xs.pop(0) if self.xs else 0.5
        pose = [(0.5, 0.3)] * 33
        pose[11], pose[12], pose[15] = (0.6, 0.6), (0.4, 0.6), (x, 0.5)
        return Point(pose_landmarks=landmarks(pose), face_landmarks=landmarks([(0.5, 0.3)] * 468),
                     left_hand_landmarks=landmarks([(x, 0.5)] * 21), right_hand_landmarks=None)


class Screen:
    """Fake cv2 window: keeps every frame's text, presses keys[n] after frame n, quits after `frames`."""

    def __init__(self, frames, keys=()):
        self.frames, self.keys, self.text, self.shown, self.n = frames, dict(keys), "", [], 0
        self.windows, self.fullscreen = {}, []

    def namedWindow(self, name, flags):
        self.windows[name] = flags

    def setWindowProperty(self, name, prop, value):
        self.fullscreen.append(value == FULLSCREEN)

    def putText(self, img, text, *args):
        self.text += text + "|"

    def imshow(self, name, frame):
        self.shown.append(self.text)
        self.text, self.n = "", self.n + 1

    def waitKey(self, ms):
        return ord("q") if self.n >= self.frames else ord(self.keys.get(self.n, "\0"))


@pytest.fixture
def app(monkeypatch):
    def run(argv, xs=(), frames=80, keys=(), classifier=None, patches=None):
        screen = Screen(frames, keys)
        camera = Point(read=lambda: (True, np.zeros((480, W, 3), np.uint8)), release=lambda: None)
        cv2 = Point(FONT_HERSHEY_SIMPLEX=0, LINE_AA=0, COLOR_BGR2RGB=0, VideoCapture=lambda index: camera,
                    flip=lambda frame, code: frame, cvtColor=lambda frame, code: frame, putText=screen.putText,
                    imshow=screen.imshow, waitKey=screen.waitKey, destroyAllWindows=lambda: None,
                    namedWindow=screen.namedWindow, setWindowProperty=screen.setWindowProperty,
                    WINDOW_NORMAL=NORMAL, WINDOW_KEEPRATIO=KEEPRATIO, WND_PROP_FULLSCREEN=0,
                    WINDOW_FULLSCREEN=FULLSCREEN, getTextSize=text_size, rectangle=lambda *a: None)
        holistic = Point(Holistic=MovingBody(xs), POSE_CONNECTIONS=None, HAND_CONNECTIONS=None)
        mp = Point(solutions=Point(holistic=holistic, drawing_utils=Point(draw_landmarks=lambda *a: None)))
        monkeypatch.setitem(sys.modules, "cv2", cv2)
        monkeypatch.setitem(sys.modules, "mediapipe", mp)
        sys.modules.pop("signvaani.app", None)
        module = importlib.import_module("signvaani.app")
        spoken = []
        monkeypatch.setattr(module, "speak", lambda text, lang: spoken.append(text))
        if classifier:
            monkeypatch.setattr(module, "SignClassifier", lambda path: classifier)
        for name, value in (patches or {}).items():
            monkeypatch.setattr(module, name, value)
        module.main(argv)
        return screen, spoken

    yield run
    sys.modules.pop("signvaani.app", None)
    vars(sys.modules["signvaani"]).pop("app", None)  # no stale fake-camera module for later imports


def one_sign():
    return [0.5] * 10 + list(0.5 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, 25))) + [0.5] * 30


def test_preview_runs_without_a_model_and_counts_signs(app):
    screen, spoken = app(["--preview", "--model", "missing.onnx"], one_sign(), keys={70: " "})
    assert all("PREVIEW" in t for t in screen.shown)
    seen = [t for t in screen.shown if "1 signs detected (preview, no model)" in t]
    assert seen and "Signs: SIGN|" in seen[0]
    assert not any("2 signs detected" in t for t in screen.shown)
    assert "0 signs detected" in screen.shown[-1] and spoken == []  # SPACE clears, never speaks


def test_preview_stays_quiet_without_motion(app):
    screen, _ = app(["--preview"], [0.5] * 80)
    assert all("0 signs detected" in t and "ready" in t for t in screen.shown)


def test_missing_model_points_to_preview(app, capsys):
    with pytest.raises(SystemExit):
        app(["--model", "missing.onnx"])
    assert "python -m signvaani.app --preview" in capsys.readouterr().err


def test_window_is_resizable_and_f_toggles_full_screen(app):
    screen, _ = app(["--preview"], frames=10, keys={3: "f", 6: "f"})
    assert screen.windows == {"SignVaani": NORMAL | KEEPRATIO} and screen.fullscreen == [True, False]
    screen, _ = app(["--preview", "--fullscreen"], frames=5, keys={2: "f"})
    assert screen.fullscreen == [True, False]


class TakeTurns:
    """Model stand-in: answers each window with the next probabilities in `answers`, round and round."""
    labels, seq_len = ["IDLE", "HAPPY", "SAD"], 32

    def __init__(self, answers):
        self.answers, self.n = answers, 0

    def __call__(self, features):
        self.n += 1
        return np.array(self.answers[(self.n - 1) % len(self.answers)])


def test_newest_signs_stay_on_screen_and_rejected_signs_say_why(app, tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"")
    answers = [[0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.3, 0.1, 0.6], [0.9, 0.05, 0.05]]  # HAPPY, SAD, weak, IDLE
    clf = TakeTurns(answers)
    screen, _ = app(["--model", str(model)], one_sign() * 30, frames=30 * 65, classifier=clf)
    assert clf.n >= 24  # (nearly) every movement was classified

    texts = [t.split("|")[:-1][::2] for t in screen.shown]  # put() draws each text twice: outline, then colour
    verdicts = {re.sub(r" \(\d+ ms\)$", "", line) for lines in texts for line in lines  # drop the latency
                if line.startswith(("added", "not added"))}
    assert verdicts == {"added: HAPPY", "added: SAD", "not added: below 80%", "not added: IDLE (no sign)"}

    *_, signs, sentence = texts[-1]
    assert signs.startswith("Signs: ... ") and signs.endswith("SAD")  # older signs dropped, newest kept
    assert len(signs) * CHAR * 0.6 <= W - 15 and len(sentence) * CHAR * 0.8 <= W - 15  # both fit the frame
    assert sentence.startswith("... ") and sentence.endswith("sad.")


def test_fit_keeps_the_end_of_the_text(app):
    app(["--preview"], frames=1)
    fit = sys.modules["signvaani.app"].fit
    assert fit("HAPPY SAD", 1000, 0.6) == "HAPPY SAD"
    assert fit("ONE TWO THREE FOUR FIVE", int(15 * CHAR * 0.6), 0.6) == "... FOUR FIVE"
    assert fit("AVERYLONGSIGNNAME", 10, 0.6) == "AVERYLONGSIGNNAME"  # one word is never dropped


def test_latency_log_and_summary(app, tmp_path, capsys):
    import csv
    model, log = tmp_path / "m.pt", tmp_path / "runs" / "live.csv"
    model.write_bytes(b"")
    clf = TakeTurns([[0.05, 0.9, 0.05], [0.3, 0.1, 0.6]])  # one sign added, the next one too weak
    app(["--model", str(model), "--log", str(log)], one_sign() * 4, frames=4 * 65, classifier=clf)
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    assert [r["sign"] for r in rows] == ["HAPPY", "HAPPY"]
    assert all(0 <= float(r["latency_ms"]) < 1000 for r in rows)
    out = capsys.readouterr().out
    assert "Frame rate:" in out and "FPS on average over 260 frames" in out
    assert "Latency, hands stop -> sign accepted: median" in out and "over 2 signs" in out


DEMO = json.loads((Path(__file__).parents[1] / "configs" / "sentences.json").read_text(encoding="utf-8"))["sentences"]


class Signs:
    """Model stand-in that 'recognises' the given glosses one after the other, at 95% confidence."""
    labels, seq_len = vocabulary(), 32

    def __init__(self, glosses):
        self.glosses, self.n = list(glosses), 0

    def __call__(self, features):
        p = np.full(len(self.labels), 0.05 / (len(self.labels) - 1))
        p[self.labels.index(self.glosses[min(self.n, len(self.glosses) - 1)])] = 0.95
        self.n += 1
        return p


def sign_sentence(app, tmp_path, glosses, argv, **kw):
    """Sign `glosses` one movement each, then press SPACE: returns the screen, what was spoken, the model."""
    model = tmp_path / "m.pt"
    model.write_bytes(b"")
    clf, n = Signs(glosses), len(glosses)
    screen, spoken = app(["--model", str(model), *argv], one_sign() * n, frames=65 * n + 2,
                         keys={65 * n: " "}, classifier=clf, **kw)
    return screen, spoken, clf


@pytest.mark.parametrize("lang", ["en", "hi"])
@pytest.mark.parametrize("s", DEMO, ids=[s["id"] for s in DEMO])
def test_signing_s1_to_s10_speaks_the_exact_sentence(app, tmp_path, s, lang):
    """TC-07 / TC-08 end to end: signs from the spotter, sentence rules, SPACE, speech."""
    _, spoken, clf = sign_sentence(app, tmp_path, s["glosses"], ["--lang", lang],
                                   patches={"hindi_font": lambda: None})
    assert clf.n == len(s["glosses"]) and spoken == [s[lang]]


def test_auto_speak_after_the_pause(app, tmp_path):
    model = tmp_path / "m.pt"
    model.write_bytes(b"")
    _, spoken = app(["--model", str(model), "--auto-speak", "--pause", "0"], one_sign(), frames=65,
                    classifier=Signs(["HELP"]))
    assert spoken == ["Please help me."]


def test_hindi_caption_is_drawn_with_a_devanagari_font(app, tmp_path):
    from signvaani.app import HINDI_FONTS
    pil = pytest.importorskip("PIL.ImageFont")
    font_path = next((f for f in HINDI_FONTS if Path(f).is_file()), None)
    if font_path is None:
        pytest.skip("no Devanagari font on this computer")
    drawn = []
    font = pil.truetype(font_path, 26)
    patches = {"hindi_font": lambda: font, "put_unicode": lambda img, text, org, f, color=None: drawn.append(text)}
    sign_sentence(app, tmp_path, ["ME", "NEED", "WATER"], ["--lang", "hi"], patches=patches)
    assert "मुझे पानी चाहिए।" in drawn  # the caption shows the Hindi sentence, not the English one
