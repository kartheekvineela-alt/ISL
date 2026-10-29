"""Confusion analysis of a training run (Day 3, 14:00-16:00; blueprint Tables R3 and R6).

    python -m signvaani.confusion runs/loso_gru_20261028-113000.json
    python -m signvaani.confusion runs/signer_gru_20261028-090000.json --top 10

Reads the per-clip predictions that signvaani.train saved and writes to --out (default
runs/confusion/<run name>/):
  confusion_<fold>.png  one confusion matrix per fold (per test signer for LOSO), plus confusion_all.png
  confused_pairs.png    normalised wrist paths of each top pair: the clips that were wrong next to clips
                        of the sign they were taken for
  report.md             Table R3 (top-1 per test signer), Table R6 (most confused pairs with a reason
                        hint) and the clips to inspect
The reason hint compares the two signs' average hand shape, location and movement over every clip of
both signs in the run's data folder, each measured against a typical pair of signs (1.0). It is a
starting point: watch the clips and the paths before a reason goes into the report.
"""
import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .data import index_clips
from .landmarks import L_WRIST, LEFT_HAND, N_POINTS, R_WRIST, RIGHT_HAND, preprocess

ASPECTS = ("hand shape", "location", "movement")


def confusion_matrix(records, labels):
    cm = np.zeros((len(labels), len(labels)), int)
    for r in records:
        cm[labels.index(r["true"]), labels.index(r["pred"])] += 1
    return cm


def top_pairs(records, k=5):
    """[(true, predicted, count, [records])] for the k most frequent mistakes."""
    wrong = defaultdict(list)
    for r in records:
        if r["true"] != r["pred"]:
            wrong[r["true"], r["pred"]].append(r)
    ranked = sorted(wrong.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:k]
    return [(a, b, len(rs), rs) for (a, b), rs in ranked]


def descriptors(raw, seq_len=32):
    """Hand shape, location and movement of one raw clip, in shoulder-normalised units."""
    pos = preprocess(raw, seq_len).reshape(seq_len, N_POINTS, 6)[..., :2]
    wrists = pos[:, [L_WRIST, R_WRIST]]
    shapes = []
    for hand in (LEFT_HAND, RIGHT_HAND):
        pts = pos[:, hand]  # (T, 21, 2); an absent hand is all zeros
        seen = np.abs(pts).sum(axis=(1, 2)) > 0
        if not seen.any():
            shapes.append(np.zeros((21, 2), np.float32))
            continue
        rel = pts[seen] - pts[seen, :1]  # relative to the hand's own wrist point
        size = np.linalg.norm(rel[:, 9], axis=-1)[:, None, None]  # wrist -> middle-finger knuckle
        shapes.append((rel / np.maximum(size, 1e-6)).mean(0))
    return {"hand shape": np.concatenate([s.ravel() for s in shapes]),
            "location": wrists.mean(0).ravel(),
            "movement": (wrists - wrists.mean(0)).ravel()}


def sign_profiles(data_root, signs):
    """Average descriptors of every sign over all its clips under data_root."""
    acc = defaultdict(list)
    for c in index_clips(data_root):
        if c.sign in signs:
            acc[c.sign].append(descriptors(np.load(c.path)))
    return {s: {a: np.mean([d[a] for d in ds], 0) for a in ASPECTS} for s, ds in acc.items()}


def pair_ratios(a, b, profiles):
    """Distance between two signs per aspect, divided by the median distance over all sign pairs."""
    typical = {}
    for aspect in ASPECTS:
        d = [np.linalg.norm(profiles[x][aspect] - profiles[y][aspect]) for x, y in combinations(profiles, 2)]
        typical[aspect] = max(float(np.median(d)), 1e-9)
    return {aspect: float(np.linalg.norm(profiles[a][aspect] - profiles[b][aspect]) / typical[aspect])
            for aspect in ASPECTS}


def reason_hint(ratios, alike=0.75):
    """'same location and movement; differ mainly in hand shape' from the per-aspect ratios."""
    same = [a for a in ASPECTS if ratios[a] < alike]
    if len(same) == len(ASPECTS):
        return ("alike in hand shape, location and movement: check that every signer uses the same variant, "
                "or replace one of the two signs (F2)")
    main = max(ASPECTS, key=ratios.get)
    if same:
        return f"same {' and '.join(same)}; they differ mainly in {main}"
    return (f"no aspect is alike on average (closest: {min(ASPECTS, key=ratios.get)}): look for signer-specific "
            "execution in the wrong clips")


def plot_matrix(cm, labels, title, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max(6, len(labels) * 0.32)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    if len(labels) <= 20:
        for i, j in zip(*np.nonzero(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_pairs(pairs, out, examples=5):
    """Per pair: wrong clips (red) and correctly recognised clips of the other sign (blue)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(len(pairs), 5)
    rows = -(-len(pairs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(4 * cols, 9), 4 * rows + 0.4), squeeze=False)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    for ax, (a, b, n, wrong, right) in zip(axes.flat, pairs):
        for records, color in ((right[:examples], "tab:blue"), (wrong, "tab:red")):
            for r in records:
                pos = preprocess(np.load(r["clip"]), 32).reshape(32, N_POINTS, 6)
                for wrist, style in ((R_WRIST, "-"), (L_WRIST, "--")):
                    ax.plot(pos[:, wrist, 0], pos[:, wrist, 1], style, color=color, lw=1)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(title=f"{a} -> {b} ({n}x)", xlabel="x / shoulder width", ylabel="y / shoulder width")
    fig.legend(handles=[plt.Line2D([], [], color="tab:red", label="true sign, predicted wrong"),
                        plt.Line2D([], [], color="tab:blue", label="the predicted sign, recognised right"),
                        plt.Line2D([], [], color="k", ls="--", label="left wrist (solid: right)")],
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.4 / (4 * rows + 0.4) + 0.03, 1, 1))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def report(run, run_name, pairs, ratios):
    folds, labels = run["folds"], run["labels"]
    title = {"loso": "leave-one-signer-out", "random": "random split (signer-dependent)",
             "signer": "held-out test signers"}.get(run["mode"], run["mode"])
    lines = [f"# Confusion analysis: {run_name}", "",
             f"Model `{run['model']}` ({run['parameters']:,} parameters), {title}, data `{run['data']}`, "
             f"{len(labels)} classes.", "",
             "## Table R3: top-1 per test fold", "",
             "| Test fold | " + " | ".join(f["name"] for f in folds) + " | Mean |",
             "|---|" + "---|" * (len(folds) + 1),
             "| Top-1 | " + " | ".join(f"{f['top1']:.1%}" for f in folds) + f" | {run['mean']:.1%} ± {run['std']:.1%} |",
             "| Macro-F1 | " + " | ".join(f"{f['macro_f1']:.3f}" for f in folds) + f" | {run['macro_f1_mean']:.3f} |",
             "| Clips | " + " | ".join(str(f["n"]) for f in folds) + f" | {sum(f['n'] for f in folds)} |", "",
             "## Table R6: most confused sign pairs", "",
             "| True sign | Predicted as | Count | Folds | Reason hint (confirm by eye) | Shape / location / movement | Fix tried |",
             "|---|---|---|---|---|---|---|"]
    for a, b, n, wrong, _ in pairs:
        where = ", ".join(f"{k} ({v})" for k, v in sorted(Counter(r["fold"] for r in wrong).items()))
        r = ratios.get((a, b))
        hint = reason_hint(r) if r else "no clips to compare"
        nums = " / ".join(f"{r[x]:.2f}" for x in ASPECTS) if r else "-"
        lines.append(f"| {a} | {b} | {n} | {where} | {hint} | {nums} | |")
    lines += ["", "Shape / location / movement: distance between the two signs' averages divided by the median "
              "distance over all sign pairs, so 1.00 is a typical pair and below 0.75 counts as alike.", "",
              "## Clips to inspect", ""]
    for a, b, n, wrong, _ in pairs:
        lines.append(f"**{a} -> {b}**")
        lines += [f"- `{r['clip']}` (fold {r['fold']}, confidence {r['conf']:.0%}, top-3 {', '.join(r['top3'])})"
                  for r in wrong]
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="run record written by signvaani.train, runs/*.json")
    p.add_argument("--top", type=int, default=5, help="how many confused pairs to list")
    p.add_argument("--data", help="clip folder for the reason hints (default: the run's data folder)")
    p.add_argument("--out", help="default: runs/confusion/<run name>")
    args = p.parse_args(argv)

    run = json.loads(Path(args.run).read_text())
    if "folds" not in run:
        p.error(f"{args.run} has no per-clip predictions; train again with this version of signvaani.train")
    name = Path(args.run).stem
    out = Path(args.out or Path("runs") / "confusion" / name)
    out.mkdir(parents=True, exist_ok=True)
    labels = run["labels"]
    records = [dict(r, fold=f["name"]) for f in run["folds"] for r in f["predictions"]]

    plotting = True
    try:
        for f in run["folds"]:
            plot_matrix(confusion_matrix(f["predictions"], labels), labels,
                        f"Test fold {f['name']}: top-1 {f['top1']:.1%}", out / f"confusion_{f['name']}.png")
        plot_matrix(confusion_matrix(records, labels), labels,
                    f"All folds: top-1 {run['mean']:.1%} (mean)", out / "confusion_all.png")
    except ImportError:
        plotting = False

    pairs = [(a, b, n, wrong, [r for r in records if r["true"] == b and r["pred"] == b])
             for a, b, n, wrong in top_pairs(records, args.top)]
    ratios = {}
    data = Path(args.data or run["data"])
    if pairs and data.is_dir():
        profiles = sign_profiles(data, set(labels))
        ratios = {(a, b): pair_ratios(a, b, profiles) for a, b, *_ in pairs if a in profiles and b in profiles}
    elif pairs:
        print(f"{data} not found: no reason hints (give --data)")
    if pairs and plotting and all(Path(r["clip"]).is_file() for *_, wrong, right in pairs for r in wrong + right):
        plot_pairs(pairs, out / "confused_pairs.png")

    text = report(run, name, pairs, ratios)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"saved {out / 'report.md'}" + (f" and the plots in {out}" if plotting else ""))


if __name__ == "__main__":
    main()
