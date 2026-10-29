"""Recorder control flow with a fake camera, a fake MediaPipe and a fast clock (no webcam needed)."""
import importlib
import sys
import types

import numpy as np
import pytest

Point = types.SimpleNamespace


class Landmarks:
    def __init__(self, n):
        self.landmark = [Point(x=0.5, y=0.5, z=0.0) for _ in range(n)]


BODY = Point(pose_landmarks=Landmarks(33), face_landmarks=Landmarks(468),
             left_hand_landmarks=Landmarks(21), right_hand_landmarks=Landmarks(21))


class FakeHolistic:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def process(self, rgb):
        return BODY


class Screen:
    """Fake cv2 window: presses `key` the first time a frame shows `trigger`."""

    def __init__(self):
        self.text, self.shown, self.presses = "", "", {}

    def putText(self, img, text, *args):
        self.text += text + "|"

    def imshow(self, name, frame):
        self.shown, self.text = self.text, ""

    def waitKey(self, ms):
        for trigger, key in list(self.presses.items()):
            if trigger in self.shown:
                del self.presses[trigger]
                return ord(key)
        return -1


class Clock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        self.now += 0.05
        return self.now


@pytest.fixture
def recorder(monkeypatch):
    screen = Screen()
    camera = Point(read=lambda: (True, np.zeros((48, 64, 3), np.uint8)), release=lambda: None)
    cv2 = Point(FONT_HERSHEY_SIMPLEX=0, LINE_AA=0, COLOR_BGR2RGB=0, VideoCapture=lambda index: camera,
                flip=lambda frame, code: frame, cvtColor=lambda frame, code: frame, putText=screen.putText,
                imshow=screen.imshow, waitKey=screen.waitKey, destroyAllWindows=lambda: None,
                namedWindow=lambda name, flags: None, WINDOW_NORMAL=0, WINDOW_KEEPRATIO=0)
    holistic = Point(Holistic=FakeHolistic, POSE_CONNECTIONS=None, HAND_CONNECTIONS=None)
    mp = Point(solutions=Point(holistic=holistic, drawing_utils=Point(draw_landmarks=lambda *a: None)))
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    sys.modules.pop("signvaani.record", None)
    record = importlib.import_module("signvaani.record")
    monkeypatch.setattr(record, "time", Clock())
    yield record, screen
    sys.modules.pop("signvaani.record", None)
    vars(sys.modules["signvaani"]).pop("record", None)  # no stale fake-camera module for later imports


def names(folder):
    return sorted(p.name for p in folder.glob("*.npy"))


def test_resumes_without_overwriting_and_redo_deletes(recorder, tmp_path, capsys):
    record, screen = recorder
    (tmp_path / "PAIN").mkdir()
    for take in (0, 1):
        np.save(tmp_path / "PAIN" / f"s01_{take}.npy", np.full((3, 63, 3), take, np.float32))
    screen.presses = {"Saved take 1/3": "r"}  # throw away the first IDLE take

    record.main(["--signer", "s01", "--signs", "PAIN", "IDLE", "--takes", "3", "--out", str(tmp_path)])

    assert names(tmp_path / "PAIN") == ["s01_0.npy", "s01_1.npy", "s01_2.npy"]
    assert np.load(tmp_path / "PAIN" / "s01_1.npy")[0, 0, 0] == 1  # earlier take untouched
    assert names(tmp_path / "IDLE") == ["s01_0.npy", "s01_1.npy", "s01_2.npy"]
    clip = np.load(tmp_path / "IDLE" / "s01_0.npy")
    assert clip.dtype == np.float32 and clip.shape[1:] == (63, 3) and not np.isnan(clip).any()
    out = capsys.readouterr().out
    assert "4 takes to record" in out and out.count("(redo)") == 1

    record.main(["--signer", "s01", "--signs", "PAIN", "IDLE", "--takes", "3", "--out", str(tmp_path)])
    assert "0 takes to record" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--signer", "rahul"], ["--signer", "s01_2"], ["--signer", "s01", "--signs", "PIAN"]])
def test_rejects_names_and_unknown_signs(recorder, tmp_path, argv):
    with pytest.raises(SystemExit):
        recorder[0].main(argv + ["--out", str(tmp_path)])
    assert not any(tmp_path.iterdir())
