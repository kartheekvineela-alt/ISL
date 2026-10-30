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
    out = run("signvaani.export_onnx", "--ckpt", str(ckpt), "--out", str(onnx_path)).stdout
    assert "on 8 random windows (OK" in out and "CPU time per sign: PyTorch" in out
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


def test_ablation_switches_and_final_model(tmp_path):
    data = tmp_path / "syn"
    run = lambda *a: subprocess.run([sys.executable, "-m", *a], check=True, capture_output=True, text=True)
    run("signvaani.make_synthetic", "--out", str(data), "--signs", "2", "--signers", "3", "--takes", "3")
    hands = tmp_path / "hands.pt"
    run("signvaani.train", "--data", str(data), "--test-signers", "s02", "--epochs", "1", "--batch-size", "4",
        "--parts", "hands", "--no-velocity", "--no-augment", "--out", str(hands), "--runs", str(tmp_path / "runs"))
    ck = torch.load(hands, map_location="cpu")
    assert ck["parts"] == ["hands"] and ck["velocity"] is False

    from signvaani.inference import SignClassifier
    import numpy as np
    clf = SignClassifier(hands)
    assert clf.parts == ["hands"] and clf.velocity is False
    x = np.random.default_rng(0).normal(size=(clf.seq_len, 378)).astype(np.float32)
    from signvaani.landmarks import mask_features
    assert np.allclose(clf(x), clf(mask_features(x, ["hands"], False)))  # hidden groups do not matter

    final = tmp_path / "final.pt"
    out = run("signvaani.train", "--data", str(data), "--final", "--val-signers", "s02", "--epochs", "1",
              "--batch-size", "4", "--out", str(final), "--runs", str(tmp_path / "final_runs")).stdout
    assert "train 12 / val 6 / test 0 clips" in out and final.exists()
    assert not (tmp_path / "final_runs").exists()  # no test set, so no run record
    res = subprocess.run([sys.executable, "-m", "signvaani.train", "--data", str(data), "--final", "--loso"],
                         capture_output=True, text=True)
    assert res.returncode == 2 and "no test set" in res.stderr


def test_experiments_compare_and_ablate(tmp_path):
    import numpy as np
    from signvaani.experiments import main
    for sign in ("HAPPY", "SAD"):
        (tmp_path / "inc" / sign).mkdir(parents=True)
    run = lambda *a: subprocess.run([sys.executable, "-m", *a], check=True, capture_output=True, text=True)
    run("signvaani.make_synthetic", "--out", str(tmp_path / "syn"), "--signs", "2", "--signers", "2", "--takes", "3")
    common = ["--epochs", "1", "--batch-size", "4", "--runs", str(tmp_path / "runs")]

    main(["compare", "--data", str(tmp_path / "syn"), "--out", str(tmp_path / "r2.md"), *common])
    r2 = (tmp_path / "r2.md").read_text(encoding="utf-8")
    assert "leave-one-signer-out, 2 folds" in r2
    assert all(f"| {m} |" in r2 for m in ("BiGRU + attention", "Transformer", "1D-CNN"))

    main(["ablate", "--data", str(tmp_path / "syn"), "--out", str(tmp_path / "r4.md"), *common])
    r4 = (tmp_path / "r4.md").read_text(encoding="utf-8")
    assert r4.count("\n| ") == 6  # header + 5 configurations
    assert '| Hands only |' in r4 and 'points vs "Hands + pose + face (full model)"' in r4
    assert len(list((tmp_path / "runs").glob("loso_gru_hands*.json"))) == 5

    # INCLUDE folders: the official split replaces leave-one-signer-out
    rng = np.random.default_rng(0)
    for sign in ("HAPPY", "SAD"):
        for split, n in (("train", 4), ("val", 1), ("test", 2)):
            for i in range(n):
                np.save(tmp_path / "inc" / sign / f"{split}_MVI_{i}.npy",
                        rng.uniform(0.3, 0.7, (30, 63, 3)).astype(np.float32))
    main(["compare", "--models", "gru", "--data", str(tmp_path / "inc"), "--out", str(tmp_path / "inc.md"), *common])
    assert "INCLUDE official split" in (tmp_path / "inc.md").read_text(encoding="utf-8")
