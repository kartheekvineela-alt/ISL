import json

import numpy as np
import pytest

from signvaani.check_data import check, clip_problems, main, plot_wrists
from signvaani.landmarks import LEFT_HAND, N_POINTS


def clip(t=40, hands=True):
    c = np.random.default_rng(t).uniform(0.2, 0.8, size=(t, N_POINTS, 3)).astype(np.float32)
    if not hands:
        c[:, LEFT_HAND.start:] = np.nan  # both hands missing
    return c


def save(root, sign, name, arr):
    (root / sign).mkdir(parents=True, exist_ok=True)
    np.save(root / sign / name, arr)


def test_clip_format():
    good = clip()
    good[5:9, LEFT_HAND] = np.nan  # hand not detected: the whole hand is NaN
    assert clip_problems(good) == []
    assert clip_problems(good.astype(np.float64))[0].startswith("expected float32")
    assert clip_problems(good[:, :21])[0].startswith("expected float32")
    zeros, partial = good.copy(), good.copy()
    zeros[:, LEFT_HAND] = 0
    partial[3, LEFT_HAND.start] = np.nan
    assert clip_problems(zeros) == ["left hand saved as zeros in 40 frames (not detected must be NaN)"]
    assert clip_problems(partial) == ["left hand partly NaN in 1 frames"]


def test_pilot_session_passes_then_problems_fail(tmp_path, capsys):
    data, vocab = tmp_path / "keypoints", tmp_path / "vocabulary.json"
    vocab.write_text(json.dumps({"signs": ["IDLE", "PAIN", "HEAD"]}))
    for sign in ["IDLE", "PAIN", "HEAD"]:
        for take in range(2):
            save(data, sign, f"s01_{take}.npy", clip(30 + take))
    argv = ["--data", str(data), "--vocab", str(vocab), "--takes", "2"]
    assert main(argv) == 0 and "PASS: 3 signs x 2 takes for s01" in capsys.readouterr().out

    save(data, "HEAD", "rahul_1.npy", clip())  # a name instead of an anonymous id
    save(data, "PIAN", "s01_0.npy", clip())  # typo folder would become a 4th class
    save(data, "PAIN", "s01_2.npy", clip().astype(np.float64))
    (data / "IDLE" / "s01_1.npy").unlink()
    assert main(argv) == 1
    out = capsys.readouterr().out
    for problem in ["HEAD/rahul_1.npy: name", "PIAN/: not in the vocabulary", "PAIN/s01_2.npy: expected float32",
                    "IDLE: 1/2 takes for s01"]:
        assert problem in out


def test_missing_signer_and_weak_hands(tmp_path):
    for take in range(2):
        save(tmp_path, "PAIN", f"s01_{take}.npy", clip(hands=False))
    stats, people, errors, warnings = check(tmp_path, ["PAIN"], signers=["s01", "s02"], takes=2)
    assert people == ["s01", "s02"] and errors == ["PAIN: 0/2 takes for s02"]
    assert warnings[0].startswith("PAIN: hands visible in 0% of frames (target 90%)")


def test_subset_of_signs_and_idle_without_hands(tmp_path):
    for take in range(2):
        save(tmp_path, "PAIN", f"s01_{take}.npy", clip())
        save(tmp_path, "IDLE", f"s01_{take}.npy", clip(hands=False))
    save(tmp_path, "HEAD", "s01_0.npy", clip())  # in the vocabulary, not part of this check
    stats, people, errors, warnings = check(tmp_path, ["IDLE", "PAIN", "HEAD"], takes=2, signs=["PAIN", "IDLE"])
    assert errors == warnings == [] and list(stats) == ["PAIN", "IDLE"]
    assert stats["IDLE"]["hands"] == [0.0, 0.0]  # hands shown as NaN: reported, never warned for IDLE


def test_wrist_plot(tmp_path):
    pytest.importorskip("matplotlib")
    for take in range(3):
        save(tmp_path, "PAIN", f"s01_{take}.npy", clip(20 + take))
    out = plot_wrists(tmp_path, ["PAIN"], out=tmp_path / "plots" / "wrists.png")
    assert out.stat().st_size > 1000
