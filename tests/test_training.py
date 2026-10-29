import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_train_export_on_synthetic_data(tmp_path):
    data, ckpt = tmp_path / "syn", tmp_path / "m.pt"
    run = lambda *a: subprocess.run([sys.executable, "-m", *a], check=True, capture_output=True, text=True)
    run("signvaani.make_synthetic", "--out", str(data), "--signs", "3", "--signers", "3", "--takes", "4")
    out = run("signvaani.train", "--data", str(data), "--test-signers", "s02", "--epochs", "3",
              "--batch-size", "8", "--out", str(ckpt), "--runs", str(tmp_path / "runs")).stdout
    assert "TEST accuracy" in out and ckpt.exists()
    assert "class weights: SIGN0 1.00, SIGN1 1.00, SIGN2 1.00" in out  # equal classes: equal weights
    assert torch.load(ckpt, map_location="cpu")["trim"] is True
    out = run("signvaani.evaluate", "--ckpt", str(ckpt), "--data", str(data), "--signers", "s02",
              "--out", str(tmp_path / "eval")).stdout
    assert "clips 12 | top-1" in out

    import json
    record, = (tmp_path / "runs").glob("signer_gru_*.json")
    run_ = json.loads(record.read_text())
    fold, = run_["folds"]
    assert run_["mode"] == "signer" and fold["name"] == "s02" and fold["n"] == 3 * 4
    assert {p["signer"] for p in fold["predictions"]} == {"s02"} and run_["parameters"] > 0
    assert fold["top1"] == sum(p["true"] == p["pred"] for p in fold["predictions"]) / fold["n"]
    run("signvaani.confusion", str(record), "--out", str(tmp_path / "confusion"))
    assert "## Table R6" in (tmp_path / "confusion" / "report.md").read_text(encoding="utf-8")

    from signvaani.inference import SignClassifier
    import numpy as np
    clf = SignClassifier(ckpt)
    probs = clf(np.zeros((clf.seq_len, 378), np.float32))
    assert probs.shape == (3,) and abs(probs.sum() - 1) < 1e-5

    pytest.importorskip("onnxruntime")
    onnx_path = tmp_path / "m.onnx"
    run("signvaani.export_onnx", "--ckpt", str(ckpt), "--out", str(onnx_path))
    onnx_probs = SignClassifier(onnx_path)(np.zeros((clf.seq_len, 378), np.float32))
    np.testing.assert_allclose(probs, onnx_probs, atol=1e-4)


def test_loso_refuses_include_split_folders(tmp_path):
    import numpy as np
    for split in ("train", "val", "test"):
        (tmp_path / "HELLO").mkdir(exist_ok=True)
        np.save(tmp_path / "HELLO" / f"{split}_MVI_0001.npy", np.zeros((10, 63, 3), np.float32))
    res = subprocess.run([sys.executable, "-m", "signvaani.train", "--data", str(tmp_path), "--loso"],
                         capture_output=True, text=True)
    assert res.returncode == 2 and "--test-signers test --val-signers val" in res.stderr


def test_class_weights_favour_small_classes():
    from pathlib import Path
    from signvaani.data import Clip
    from signvaani.train import class_weights
    clips = [Clip(Path(f"a{i}.npy"), "LOUD", "train") for i in range(15)]
    clips += [Clip(Path(f"b{i}.npy"), "SAD", "train") for i in range(5)]
    w = class_weights(clips, ["LOUD", "SAD", "UGLY"]).tolist()
    assert abs(w[1] / w[0] - 3) < 1e-6 and w[2] == 0  # 5 clips weigh 3x as much as 15; no clips, no weight
    assert class_weights(clips[:5] + clips[15:], ["LOUD", "SAD"]).tolist() == [1.0, 1.0]


def test_balance_and_trim_can_be_switched_off(tmp_path):
    data, ckpt = tmp_path / "syn", tmp_path / "m.pt"
    run = lambda *a: subprocess.run([sys.executable, "-m", *a], check=True, capture_output=True, text=True)
    run("signvaani.make_synthetic", "--out", str(data), "--signs", "2", "--signers", "2", "--takes", "3")
    out = run("signvaani.train", "--data", str(data), "--test-signers", "s01", "--epochs", "1", "--batch-size", "4",
              "--no-balance", "--no-trim", "--out", str(ckpt), "--runs", str(tmp_path / "runs")).stdout
    assert "class weights" not in out and torch.load(ckpt, map_location="cpu")["trim"] is False
