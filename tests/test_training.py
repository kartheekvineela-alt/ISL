import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_train_export_on_synthetic_data(tmp_path):
    data, ckpt = tmp_path / "syn", tmp_path / "m.pt"
    run = lambda *a: subprocess.run([sys.executable, "-m", *a], check=True, capture_output=True, text=True)
    run("signvaani.make_synthetic", "--out", str(data), "--signs", "3", "--signers", "3", "--takes", "4")
    out = run("signvaani.train", "--data", str(data), "--test-signers", "s02", "--epochs", "3",
              "--batch-size", "8", "--out", str(ckpt)).stdout
    assert "TEST accuracy" in out and ckpt.exists()

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
