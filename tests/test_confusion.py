import json

import numpy as np
import pytest

from signvaani.confusion import main, pair_ratios, reason_hint, sign_profiles, top_pairs
from signvaani.landmarks import L_SHOULDER, L_WRIST, LEFT_HAND, N_POINTS, R_SHOULDER, R_WRIST, RIGHT_HAND


def sign_clip(seed, shape="flat", path="circle", t=40):
    """Both hands follow `path`; `shape` sets the finger layout around the wrist."""
    rng = np.random.default_rng(seed)
    clip = np.full((t, N_POINTS, 3), 0.5, np.float32)
    clip[:, L_SHOULDER, :2], clip[:, R_SHOULDER, :2] = (0.6, 0.6), (0.4, 0.6)
    s = np.linspace(0, 2 * np.pi, t)
    xy = (np.stack([0.5 + 0.08 * np.cos(s), 0.4 + 0.08 * np.sin(s)], 1) if path == "circle"
          else np.stack([np.full(t, 0.7), 0.2 + 0.3 * s / s[-1]], 1))
    k = np.arange(21)[:, None]
    fingers = np.hstack([0.004 * k, 0 * k]) if shape == "flat" else np.hstack([0 * k, 0.004 * k])
    for wrist, hand in ((L_WRIST, LEFT_HAND), (R_WRIST, RIGHT_HAND)):
        clip[:, wrist, :2] = xy
        clip[:, hand, :2] = xy[:, None] + fingers
    return clip + rng.normal(0, 0.001, clip.shape).astype(np.float32)


@pytest.fixture
def run_file(tmp_path):
    data = tmp_path / "keypoints"
    kinds = {"WATER": ("flat", "circle"), "MEDICINE": ("fist", "circle"), "HELP": ("flat", "line")}
    for sign, (shape, path) in kinds.items():
        (data / sign).mkdir(parents=True)
        for signer in ("s01", "s02"):
            for take in range(2):
                np.save(data / sign / f"{signer}_{take}.npy", sign_clip(take, shape, path))

    def rec(sign, signer, take, pred):
        return {"clip": (data / sign / f"{signer}_{take}.npy").as_posix(), "signer": signer, "true": sign,
                "pred": pred, "conf": 0.6, "top3": [pred, sign, "HELP"]}

    folds = []
    for s, wrong in (("s01", 2), ("s02", 1)):
        preds = [rec("WATER", s, t, "MEDICINE" if t < wrong else "WATER") for t in range(2)]
        preds += [rec("MEDICINE", s, t, "MEDICINE") for t in range(2)]
        preds += [rec("HELP", s, 0, "WATER"), rec("HELP", s, 1, "HELP")] if s == "s01" else \
            [rec("HELP", s, t, "HELP") for t in range(2)]
        top1 = sum(r["true"] == r["pred"] for r in preds) / len(preds)
        folds.append({"name": s, "test_signers": [s], "val_acc": 1.0, "top1": top1, "macro_f1": top1,
                      "n": len(preds), "predictions": preds})
    run = {"mode": "loso", "model": "gru", "data": data.as_posix(), "labels": ["HELP", "MEDICINE", "WATER"],
           "parameters": 1234, "mean": float(np.mean([f["top1"] for f in folds])),
           "std": 0.1, "macro_f1_mean": 0.5, "folds": folds}
    path = tmp_path / "loso_gru_test.json"
    path.write_text(json.dumps(run))
    return path


def test_top_pairs_ranked_by_count(run_file):
    run = json.loads(run_file.read_text())
    records = [r for f in run["folds"] for r in f["predictions"]]
    assert [(a, b, n) for a, b, n, _ in top_pairs(records)] == [("WATER", "MEDICINE", 3), ("HELP", "WATER", 1)]
    assert [(a, b) for a, b, *_ in top_pairs(records, k=1)] == [("WATER", "MEDICINE")]


def test_reason_hint_names_what_separates_two_signs(run_file):
    profiles = sign_profiles(run_file.parent / "keypoints", {"WATER", "MEDICINE", "HELP"})
    r = pair_ratios("WATER", "MEDICINE", profiles)  # same path and place, different fingers
    assert r["location"] < 0.1 and r["movement"] < 0.1 and r["hand shape"] > 0.75
    assert reason_hint(r) == "same location and movement; they differ mainly in hand shape"
    r = pair_ratios("WATER", "HELP", profiles)  # same fingers, different path and place
    assert r["hand shape"] < 0.1 and reason_hint(r).startswith("same hand shape; they differ mainly in")
    assert reason_hint({"hand shape": 0.1, "location": 0.2, "movement": 0.3}).startswith("alike in hand shape")
    assert reason_hint({"hand shape": 1.1, "location": 0.9, "movement": 2.0}).startswith("no aspect is alike")


def test_report_and_plots(run_file, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "analysis"
    main([str(run_file), "--out", str(out)])
    text = (out / "report.md").read_text(encoding="utf-8")
    assert text.strip() in capsys.readouterr().out
    assert "| Top-1 | 50.0% | 83.3% | 66.7% ± 10.0% |" in text
    assert ("| WATER | MEDICINE | 3 | s01 (2), s02 (1) | same location and movement; they differ mainly in "
            "hand shape |") in text
    assert "s01_0.npy` (fold s01, confidence 60%, top-3 MEDICINE, WATER, HELP)" in text
    for png in ("confusion_s01.png", "confusion_s02.png", "confusion_all.png", "confused_pairs.png"):
        assert (out / png).stat().st_size > 1000


def test_old_run_without_predictions_is_refused(tmp_path):
    old = tmp_path / "loso_gru_1.json"
    old.write_text(json.dumps({"per_signer": {"s01": 0.5}, "mean": 0.5, "std": 0.0}))
    with pytest.raises(SystemExit):
        main([str(old)])
