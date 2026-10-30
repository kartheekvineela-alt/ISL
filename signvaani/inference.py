"""Load a trained model (.onnx preferred, .pt fallback) and classify feature windows."""
import json
from pathlib import Path

import numpy as np

from .landmarks import ALL_GROUPS, mask_features


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
        # an ablation model only saw some landmark groups: hide the others the same way
        self.parts, self.velocity = meta.get("parts", ALL_GROUPS), meta.get("velocity", True)

    def __call__(self, features):
        """features: (seq_len, FEATURE_DIM) -> probability per label."""
        x = mask_features(np.asarray(features, np.float32), self.parts, self.velocity)[None]
        if self._torch is None:
            logits = self._sess.run(None, {"features": x})[0][0]
        else:
            import torch
            with torch.no_grad():
                logits = self._torch(torch.from_numpy(x))[0].numpy()
        e = np.exp(logits - logits.max())
        return e / e.sum()
