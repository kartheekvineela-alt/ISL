import csv

import numpy as np
import pytest

from signvaani.landmarks import L_SHOULDER, L_WRIST, LEFT_HAND, N_POINTS, R_SHOULDER, R_WRIST, RIGHT_HAND
from signvaani.qc import main, quality_check


def take(seed, path="circle", t=40, hands_frames=None):
    """A clip whose wrists and hands follow `path` (a circle or a vertical line) with small noise."""
    rng = np.random.default_rng(seed)
    clip = np.full((t, N_POINTS, 3), 0.5, np.float32) + rng.normal(0, 0.002, (t, N_POINTS, 3)).astype(np.float32)
    clip[:, L_SHOULDER, :2], clip[:, R_SHOULDER, :2] = (0.6, 0.6), (0.4, 0.6)
    s = np.linspace(0, 2 * np.pi, t)
    xy = (np.stack([0.5 + 0.1 * np.cos(s), 0.4 + 0.1 * np.sin(s)], 1) if path == "circle"
          else np.stack([np.full(t, 0.5), 0.3 + 0.3 * s / s[-1]], 1))
    for wrist, hand in ((L_WRIST, LEFT_HAND), (R_WRIST, RIGHT_HAND)):
        clip[:, wrist, :2] = xy + rng.normal(0, 0.003, xy.shape)
        clip[:, hand, :2] = xy[:, None] + rng.normal(0, 0.003, (t, 21, 2))
    if hands_frames is not None:
        clip[hands_frames:, LEFT_HAND.start:] = np.nan
    return clip


def save(root, sign, name, clip):
    (root / sign).mkdir(parents=True, exist_ok=True)
    np.save(root / sign / name, clip)


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "keypoints"
    for i in range(8):
        save(root, "PAIN", f"s01_{i}.npy", take(i))
    save(root, "PAIN", "s01_8.npy", take(8, hands_frames=16))  # hands in 40% of frames
    save(root, "PAIN", "s01_9.npy", take(9, path="line"))  # a different movement: wrong sign?
    for i in range(3):
        save(root, "IDLE", f"s01_{i}.npy", take(20 + i, hands_frames=0))  # IDLE may hide the hands
    return root


def test_reject_review_and_short(data):
    clips, reject, review, short = quality_check(data, min_takes=8)
    assert len(clips) == 13
    assert [(c.path.name, why) for c, why in reject] == [("s01_8.npy", "hands in 40% of frames (minimum 60%)")]
    assert [c.path.name for c, _, _ in review] == ["s01_9.npy"]
    assert ("s01", "IDLE", 3) in short and ("s01", "HEAD", 0) in short  # unrecorded vocabulary signs count
    assert not any(sign == "PAIN" for _, sign, _ in short)  # 9 good PAIN takes after removing one
    assert quality_check(data, min_takes=10)[3][0] == ("s01", "IDLE", 3)
    assert ("s01", "PAIN", 9) in quality_check(data, min_takes=10)[3]


def test_report_then_apply_moves_and_logs(data, capsys):
    assert main(["--data", str(data), "--min-takes", "3"]) == 1  # 13 vocabulary signs not recorded
    out = capsys.readouterr().out
    assert "REJECT  PAIN/s01_8.npy" in out and "REVIEW  PAIN/s01_9.npy" in out and "--apply" in out
    assert "python -m signvaani.record --signer s01 --signs ME DOCTOR FEVER" in out
    assert (data / "PAIN" / "s01_8.npy").exists()  # report only

    main(["--data", str(data), "--apply"])
    rejected = data.parent / "rejected"
    assert not (data / "PAIN" / "s01_8.npy").exists() and (rejected / "PAIN" / "s01_8.npy").exists()
    assert "| PAIN/s01_8.npy | hands in 40% of frames (minimum 60%) |" in capsys.readouterr().out
    assert "rejected" not in {p.name for p in data.iterdir()}  # never becomes a class folder

    main(["--data", str(data), "--reject", str(data / "PAIN" / "s01_9.npy"), "--reason", "two signs"])
    assert "python -m signvaani.record --signer s01 --signs PAIN" in capsys.readouterr().out
    rows = list(csv.DictReader(open(rejected / "qc_log.csv", encoding="utf-8")))
    assert [(r["file"].split("/")[-1], r["reason"]) for r in rows] == [
        ("s01_8.npy", "hands in 40% of frames (minimum 60%)"), ("s01_9.npy", "two signs")]


def test_review_plot(data, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "review.png"
    main(["--data", str(data), "--plot", "--plot-out", str(out)])
    assert out.stat().st_size > 1000
