"""Evaluate a checkpoint: accuracy, top-3, macro-F1, confusion matrix, confused pairs.

    python -m signvaani.evaluate --ckpt models/best.pt --data data/keypoints --signers s05
    python -m signvaani.evaluate --ckpt models/best.pt --data data/unseen --signers s06 --out docs/results/s06

Day 5's unseen-signer result: s06 is recorded into its own folder (data/unseen), so the final model never
trains on it. The checkpoint lists the signers it was trained on, and a WARNING is printed when an
evaluated signer is one of them: then the number is not an unseen-signer result. --out gets metrics.json,
summary.md (the table for the report) and the confusion matrix.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import index_clips
from .landmarks import ALL_GROUPS, mask_features, preprocess
from .models import build_model
from .spotter import trim_to_sign


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="models/best.pt")
    p.add_argument("--data", default="data/keypoints")
    p.add_argument("--signers", nargs="*", default=[], help="evaluate only these signers (default: all)")
    p.add_argument("--out", default="runs/eval")
    args = p.parse_args(argv)

    ck = torch.load(args.ckpt, map_location="cpu")
    labels = ck["labels"]
    model = build_model(ck["model"], ck["feature_dim"], len(labels))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    clips = [c for c in index_clips(args.data) if not args.signers or c.signer in args.signers]
    clips = [c for c in clips if c.sign in labels]
    if not clips:
        p.error(f"no clips of the model's signs for signers {args.signers or 'any'} under {args.data}")
    people = sorted({c.signer for c in clips})
    trained_on = ck.get("train_signers")
    seen = sorted(set(people) & set(trained_on or []))
    if seen:
        print(f"WARNING: {', '.join(seen)} {'was' if len(seen) == 1 else 'were'} in this model's training data: "
              "this is NOT an unseen-signer result")
    elif trained_on is None:
        print("note: this checkpoint does not list its training signers (trained before Day 5); make sure "
              f"{', '.join(people)} did not train it")
    cut = trim_to_sign if ck.get("trim") else (lambda raw: raw)  # evaluate clips the way the model was trained
    parts, velocity = ck.get("parts", ALL_GROUPS), ck.get("velocity", True)
    x = torch.from_numpy(np.stack([mask_features(preprocess(cut(np.load(c.path)), ck["seq_len"]), parts, velocity)
                                   for c in clips]))
    y = np.array([labels.index(c.sign) for c in clips])
    with torch.no_grad():
        probs = torch.softmax(model(x), 1).numpy()
    pred = probs.argmax(1)
    top3 = np.mean([y[i] in np.argsort(-probs[i])[:3] for i in range(len(y))])

    acc = float((pred == y).mean())
    f1 = float(f1_score(y, pred, average="macro"))
    print(f"clips {len(y)} | top-1 {acc:.4f} | top-3 {top3:.4f} | macro-F1 {f1:.4f}")
    print(classification_report(y, pred, labels=range(len(labels)), target_names=labels, zero_division=0))

    cm = confusion_matrix(y, pred, labels=range(len(labels)))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels))
                    if off[i, j] > 0), reverse=True)[:10]
    print("Most confused (true -> predicted):")
    for n, a, b in pairs:
        print(f"  {a:>18} -> {b:<18} {n}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unseen = None if trained_on is None else not seen
    (out / "metrics.json").write_text(json.dumps(
        {"signers": people, "unseen": unseen, "top1": acc, "top3": float(top3), "macro_f1": f1, "n": int(len(y)),
         "confused_pairs": [[a, b, int(n)] for n, a, b in pairs]}, indent=2))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    lines = [f"# Evaluation of `{Path(args.ckpt).name}` on {', '.join(people)}", "",
             "| Signers | Unseen by the model | Clips | Top-1 | Top-3 | Macro-F1 |", "|---|---|---|---|---|---|",
             f"| {', '.join(people)} | { {True: 'yes', False: 'NO'}.get(unseen, 'not recorded') } | {len(y)} | "
             f"{acc:.1%} | {top3:.1%} | {f1:.3f} |", "", "| Sign | Clips | Recall |", "|---|---|---|"]
    lines += [f"| {l} | {int(cm[i].sum())} | {recall[i]:.0%} |" for i, l in enumerate(labels) if cm[i].sum()]
    lines += ["", "Most confused (true -> predicted): " + (", ".join(f"{a} -> {b} ({n})" for n, a, b in pairs) or "none")]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {out / 'metrics.json'} and {out / 'summary.md'}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.3),) * 2)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix (top-1 {acc:.1%})")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=150)
        print(f"saved {out / 'confusion_matrix.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
