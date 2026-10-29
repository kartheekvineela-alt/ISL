"""Live app control flow with a fake camera and a fake MediaPipe (no webcam, no model)."""
import importlib
import sys
import types

import numpy as np
import pytest

Point = types.SimpleNamespace


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

    def putText(self, img, text, *args):
        self.text += text + "|"

    def imshow(self, name, frame):
        self.shown.append(self.text)
        self.text, self.n = "", self.n + 1

    def waitKey(self, ms):
        return ord("q") if self.n >= self.frames else ord(self.keys.get(self.n, "\0"))


@pytest.fixture
def app(monkeypatch):
    def run(argv, xs=(), frames=80, keys=()):
        screen = Screen(frames, keys)
        camera = Point(read=lambda: (True, np.zeros((48, 64, 3), np.uint8)), release=lambda: None)
        cv2 = Point(FONT_HERSHEY_SIMPLEX=0, LINE_AA=0, COLOR_BGR2RGB=0, VideoCapture=lambda index: camera,
                    flip=lambda frame, code: frame, cvtColor=lambda frame, code: frame, putText=screen.putText,
                    imshow=screen.imshow, waitKey=screen.waitKey, destroyAllWindows=lambda: None)
        holistic = Point(Holistic=MovingBody(xs), POSE_CONNECTIONS=None, HAND_CONNECTIONS=None)
        mp = Point(solutions=Point(holistic=holistic, drawing_utils=Point(draw_landmarks=lambda *a: None)))
        monkeypatch.setitem(sys.modules, "cv2", cv2)
        monkeypatch.setitem(sys.modules, "mediapipe", mp)
        sys.modules.pop("signvaani.app", None)
        module = importlib.import_module("signvaani.app")
        spoken = []
        monkeypatch.setattr(module, "speak", lambda text, lang: spoken.append(text))
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
    seen = [t for t in screen.shown if "1 signs detected (preview: no model, nothing is recognised)" in t]
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
