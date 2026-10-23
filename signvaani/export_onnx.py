"""Export a trained checkpoint to ONNX for fast CPU inference.

    python -m signvaani.export_onnx --ckpt models/best.pt --out models/sign_model.onnx
"""
import argparse
import json
from pathlib import Path

import torch

from .models import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/best.pt")
    p.add_argument("--out", default="models/sign_model.onnx")
    args = p.parse_args()

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
    meta.write_text(json.dumps({"labels": ck["labels"], "seq_len": ck["seq_len"],
                                "feature_dim": ck["feature_dim"]}, indent=2))
    print(f"saved {out} and {meta}")


if __name__ == "__main__":
    main()
