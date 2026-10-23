"""Load a trained model (.onnx preferred, .pt fallback) and classify feature windows."""
import json
from pathlib import Path

import numpy as np


class SignClassifier:
    def __init__(self, path):
        path = Path(path)
        if path.suffix == ".onnx":
            import onnxruntime as ort
            meta = json.loads(path.with_suffix(".json").read_text())
            self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._torch = None
        else:
            import torch
            from .models import build_model
            meta = torch.load(path, map_location="cpu")
            self._torch = build_model(meta["model"], meta["feature_dim"], len(meta["labels"]))
            self._torch.load_state_dict(meta["state_dict"])
            self._torch.eval()
        self.labels = meta["labels"]
        self.seq_len = meta["seq_len"]

    def __call__(self, features):
        """features: (seq_len, FEATURE_DIM) -> probability per label."""
        x = np.asarray(features, np.float32)[None]
        if self._torch is None:
            logits = self._sess.run(None, {"features": x})[0][0]
        else:
            import torch
            with torch.no_grad():
                logits = self._torch(torch.from_numpy(x))[0].numpy()
        e = np.exp(logits - logits.max())
        return e / e.sum()
