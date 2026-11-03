"""signvaani.predict_video with a fake MediaPipe and a fake model (no camera, no trained model)."""
from argparse import Namespace

import numpy as np
import pytest

import signvaani.inference
from signvaani.include_dataset import videos
from signvaani.landmarks import L_SHOULDER, L_WRIST, LEFT_HAND, N_POINTS, R_SHOULDER, RIGHT_HAND, preprocess
from signvaani.predict_video import classify, find_videos, main, summary, true_sign
from signvaani.spotter import trim_to_sign


class FakeHolistic:
    """Nothing detected in any frame."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def process(self, rgb):
        return Namespace(pose_landmarks=None, face_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None)


class FakeModel:
    """Always says HAPPY 90%, SAD 6%, LOUD 4%, and keeps every input it was given."""

    labels = ["HAPPY", "LOUD", "SAD"]
    seq_len = 32

    def __init__(self, path=None, trim=False):
        self.trim, self.inputs = trim, []

    def __call__(self, x):
        self.inputs.append(x)
        return np.array([0.90, 0.04, 0.06])


def write_video(path, frames=8):
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def test_true_sign_comes_from_the_nearest_sign_folder(tmp_path):
    labels = ["HAPPY", "HOUSE", "ME"]
    assert true_sign(tmp_path / "Adjectives_4of8/Adjectives/3. happy/MVI_5017.MOV", labels) == "HAPPY"
    assert true_sign(tmp_path / "Places/19. House/Extra/MVI_3439.MOV", labels) == "HOUSE"
    assert true_sign(tmp_path / "ME/clip1.mp4", labels) == "ME"
    assert true_sign(tmp_path / "my videos/clip1.mp4", labels) is None


def test_find_videos_searches_folders_and_rejects_missing_paths(tmp_path):
    for name in ["a/1. happy/x.MOV", "a/2. sad/y.mp4", "a/notes.txt"]:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    found = find_videos([tmp_path / "a"])
    assert [p.name for p in found] == ["x.MOV", "y.mp4"]
    assert find_videos([tmp_path / "a/2. sad/y.mp4"]) == [tmp_path / "a/2. sad/y.mp4"]
    with pytest.raises(FileNotFoundError):
        find_videos([tmp_path / "missing.mp4"])


def moving_clip():
    """Still, then the left wrist moves for 25 frames, then still again: one sign in the middle."""
    xs = [0.5] * 10 + list(0.5 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, 25))) + [0.5] * 20
    clip = np.full((len(xs), N_POINTS, 3), 0.5, np.float32)
    clip[:, L_SHOULDER, :2], clip[:, R_SHOULDER, :2] = (0.6, 0.6), (0.4, 0.6)
    clip[:, RIGHT_HAND] = np.nan
    clip[:, L_WRIST, 0] = xs
    clip[:, LEFT_HAND, 0] = np.array(xs)[:, None]
    return clip


def test_classify_cuts_the_clip_only_when_the_model_was_trained_that_way():
    raw = moving_clip()
    assert len(trim_to_sign(raw)) < len(raw)  # the cut really changes the clip
    for trim, seq in [(True, trim_to_sign(raw)), (False, raw)]:
        model = FakeModel(trim=trim)
        classify(raw, model)
        np.testing.assert_allclose(model.inputs[0], preprocess(seq, 32))


def test_main_prints_each_sign_the_accuracy_and_a_csv(tmp_path, monkeypatch, capsys):
    root = tmp_path / "Adjectives_4of8" / "Adjectives"
    write_video(root / "3. happy" / "MVI_0001.avi")
    write_video(root / "4. sad" / "MVI_0002.avi")
    write_video(tmp_path / "other" / "clip.avi")  # no sign folder: shown, not scored
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"")
    model = FakeModel()
    monkeypatch.setattr(signvaani.inference, "SignClassifier", lambda path: model)

    out = tmp_path / "runs" / "videos.csv"
    main(["--model", str(model_path), "--videos", str(tmp_path / "Adjectives_4of8"), str(tmp_path / "other"),
          "--out", str(out)], make_holistic=FakeHolistic)

    text = capsys.readouterr().out
    assert "MVI_0001.avi  ->  HAPPY 90%   (then SAD 6%, LOUD 4%)   true HAPPY: RIGHT" in text
    assert "MVI_0002.avi  ->  HAPPY 90%" in text and "true SAD: WRONG" in text
    assert "top-1 1/2 (50%), top-3 2/2 (100%)" in text
    assert all(x.shape == (32, 378) for x in model.inputs) and len(model.inputs) == 3
    rows = out.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "video,true,predicted,confidence,top3,correct,frames,hands"
    assert len(rows) == 4 and rows[1].endswith(",HAPPY,HAPPY,0.900,HAPPY SAD LOUD,1,8,0.00")


def test_include_split_keeps_only_official_test_videos(tmp_path, monkeypatch, capsys):
    test = next(v for v in videos("include") if v.split == "test")
    train = next(v for v in videos("include") if v.split == "train")
    for v in (test, train):
        (tmp_path / v.rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / v.rel).write_bytes(b"not a real video")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"")
    monkeypatch.setattr(signvaani.inference, "SignClassifier", lambda path: FakeModel())

    main(["--model", str(model_path), "--videos", str(tmp_path), "--include-split", "test"],
         make_holistic=FakeHolistic)

    text = capsys.readouterr().out
    assert text.startswith("1 videos,")
    assert f"{test.rel.split('/')[-1]}: could not be read" in text
    assert "1 videos, 1 unreadable" in text


def test_missing_model_or_videos_stop_with_a_message(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--model", str(tmp_path / "none.onnx"), "--videos", str(tmp_path)])
    assert "train one first" in capsys.readouterr().err
    (tmp_path / "m.onnx").write_bytes(b"")
    with pytest.raises(SystemExit):
        main(["--model", str(tmp_path / "m.onnx"), "--videos", str(tmp_path / "nothing.mp4")])
    assert "no video or folder" in capsys.readouterr().err


def test_summary_without_sign_folders_gives_no_accuracy():
    recs = [{"video": "a.mp4", "true": None, "top3": [("HAPPY", 0.9)], "frames": 5, "hands": 1.0}]
    assert summary(recs) == ["1 videos, 0 unreadable",
                             "No video is in a folder named after a sign the model knows, so no accuracy is given."]
