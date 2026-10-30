"""Export a trained checkpoint to ONNX for fast CPU inference, and check the export (Day 4).

    python -m signvaani.export_onnx --ckpt models/best.pt --out models/sign_model.onnx

After exporting it runs the same random windows through PyTorch and ONNX Runtime and fails (exit code 1)
if their outputs differ by more than 1e-4. It also prints the CPU time per sign for both, the
"CPU ms / sign" column of Table R2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .models import build_model


def cpu_ms(fn, x, repeats=50):
    """Median wall time of fn(x) in milliseconds, after a few warm-up calls."""
    for _ in range(5):
        fn(x)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="models/best.pt")
    p.add_argument("--out", default="models/sign_model.onnx")
    args = p.parse_args(argv)

    ck = torch.load(args.ckpt, map_location="cpu")
    model = build_model(ck["model"], ck["feature_dim"], len(ck["labels"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    dummy = torch.zeros(1, ck["seq_len"], ck["feature_dim"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, out, input_names=["features"], output_names=["logits"],
                      dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps({"labels": ck["labels"], "seq_len": ck["seq_len"], "feature_dim": ck["feature_dim"],
                                **{k: ck[k] for k in ("trim", "parts", "velocity") if k in ck}}, indent=2))
    print(f"saved {out} and {meta}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(8, ck["seq_len"], ck["feature_dim"], generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = model(x).numpy()
    got = sess.run(None, {"features": x.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    print(f"ONNX check: largest difference from PyTorch {diff:.1e} on 8 random windows "
          f"({'OK' if diff <= 1e-4 else 'MISMATCH'}, limit 1e-4)")
    one = x[:1]
    with torch.no_grad():
        torch_ms = cpu_ms(model, one)
    onnx_ms = cpu_ms(lambda t: sess.run(None, {"features": t.numpy()}), one)
    print(f"CPU time per sign: PyTorch {torch_ms:.2f} ms, ONNX Runtime {onnx_ms:.2f} ms")
    return 0 if diff <= 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
