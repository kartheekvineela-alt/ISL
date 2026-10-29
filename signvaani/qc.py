"""Quality control after a recording day (Day 2, 14:00-16:00; fallback F3).

    python -m signvaani.qc                      # report only: what to remove, review and re-record
    python -m signvaani.qc --plot               # + wrist paths of the takes to review (runs/qc/review.png)
    python -m signvaani.qc --apply              # move the REJECT clips to data/rejected/
    python -m signvaani.qc --reject data/keypoints/PAIN/s03_4.npy --reason "wrong sign"

Every clip is checked:
  REJECT  a sign take (not IDLE) with a hand in fewer than --min-hands (60%) of its frames.
  REVIEW  a take whose hand movement is far from the same signer's other takes of that sign: maybe a
          wrong sign, two signs or a false start. Look at it with --plot; if it is wrong, remove it
          with --reject.
  SHORT   a signer with fewer than --min-takes (8) good takes of a sign. Re-record with the printed
          command; the recorder records only the missing takes.
Removed clips are moved, never deleted, to data/rejected/<SIGN>/ and logged in data/rejected/qc_log.csv,
so a mistake is undone by moving the file back. Exit code 1 while any SHORT line is left.
"""
import argparse
import csv
import datetime
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import index_clips, vocabulary
from .landmarks import L_WRIST, N_FACE, N_POINTS, N_POSE, R_WRIST, hands_share, preprocess

# Arms and hands carry the sign: elbows (positions 5 and 6 of the selected points), wrists, both hands.
MOVING_POINTS = [5, 6, L_WRIST, R_WRIST, *range(N_POSE + N_FACE, N_POINTS)]


def movement(raw, seq_len=32):
    """Normalised (x, y) path of the arms and hands, (seq_len, len(MOVING_POINTS), 2)."""
    feats = preprocess(raw, seq_len).reshape(seq_len, N_POINTS, 6)
    return feats[:, MOVING_POINTS, :2]


def review_scores(paths):
    """Median distance of each take to the other takes of its group (same sign, same signer)."""
    moves = [movement(np.load(p)) for p in paths]
    d = np.array([[np.linalg.norm(a - b, axis=-1).mean() for b in moves] for a in moves])
    return np.array([np.median(np.delete(row, i)) for i, row in enumerate(d)])


def outliers(scores, ratio=1.25):
    """Takes far from their group: above Q3 + 1.5 IQR and at least `ratio` x the group median."""
    q1, med, q3 = np.percentile(scores, [25, 50, 75])
    return (scores > q3 + 1.5 * (q3 - q1)) & (scores > ratio * med)


def quality_check(root, signers=(), min_hands=0.6, min_takes=8):
    """-> (clips, reject [(clip, reason)], review [(clip, score / median, group)], short [(signer, sign, n)])."""
    clips = [c for c in index_clips(root) if not signers or c.signer in signers]
    reject, review = [], []
    for c in clips:
        share = hands_share(np.load(c.path))
        if c.sign != "IDLE" and share < min_hands:
            reject.append((c, f"hands in {share:.0%} of frames (minimum {min_hands:.0%})"))
    rejected = {c for c, _ in reject}

    groups = defaultdict(list)
    for c in clips:
        if c.sign != "IDLE" and c not in rejected:  # IDLE takes differ from each other on purpose
            groups[c.sign, c.signer].append(c)
    for group in groups.values():
        if len(group) < 4:
            continue
        scores = review_scores([c.path for c in group])
        for c, s, flag in zip(group, scores, outliers(scores)):
            if flag:
                review.append((c, s / np.median(scores), group))

    signs = sorted({c.sign for c in clips})
    vocab = vocabulary()
    if set(signs) <= set(vocab):
        signs = vocab  # own recordings: a sign nobody recorded yet is short too
    good = defaultdict(int)
    for c in clips:
        if c not in rejected:
            good[c.signer, c.sign] += 1
    people = sorted(set(signers) | {c.signer for c in clips})
    short = [(p, s, good[p, s]) for p in people for s in signs if good[p, s] < min_takes]
    return clips, reject, review, short


def move_to_rejected(path, rejected_dir, reason):
    """Move one clip to rejected_dir/<SIGN>/ and log it; returns the new path."""
    path = Path(path)
    dest = Path(rejected_dir) / path.parent.name / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while dest.exists():
        dest = dest.with_name(f"{path.stem}.{n}{path.suffix}")
        n += 1
    shutil.move(str(path), dest)
    log = Path(rejected_dir) / "qc_log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "file", "reason", "moved_to"])
        w.writerow([datetime.date.today().isoformat(), path.as_posix(), reason, dest.as_posix()])
    print(f"| {datetime.date.today().isoformat()} | {path.parent.name}/{path.name} | {reason} |")
    return dest


def plot_review(review, out="runs/qc/review.png"):
    """One panel per flagged take: its wrist paths in red over the signer's other takes in grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(len(review), 5)
    rows = -(-len(review) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(4 * cols, 9), 4 * rows + 0.4), squeeze=False)
    for ax in axes.flat[len(review):]:
        ax.axis("off")
    for ax, (clip, ratio, group) in zip(axes.flat, review):
        for c in group:
            raw = np.load(c.path)
            color, lw = ("red", 2) if c == clip else ("0.7", 1)
            for wrist, style in ((R_WRIST, "-"), (L_WRIST, "--")):
                ax.plot(raw[:, wrist, 0], raw[:, wrist, 1], style, color=color, lw=lw)
        ax.set(title=f"{clip.sign}/{clip.path.name} ({ratio:.1f}x)", xlim=(0, 1), ylim=(1, 0), aspect="equal")
    fig.legend(handles=[plt.Line2D([], [], color="red", label="take to review"),
                        plt.Line2D([], [], color="0.7", label="same signer's other takes"),
                        plt.Line2D([], [], color="k", ls="--", label="left wrist (solid: right)")],
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.4 / (4 * rows + 0.4) + 0.03, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/keypoints")
    p.add_argument("--signers", nargs="*", default=[], help="only these signers (default: all)")
    p.add_argument("--min-hands", type=float, default=0.6, help="reject sign takes below this hand share")
    p.add_argument("--min-takes", type=int, default=8, help="good takes every signer needs per sign")
    p.add_argument("--rejected", help="where removed clips go (default: data/rejected next to --data)")
    p.add_argument("--apply", action="store_true", help="move the REJECT clips out of the dataset")
    p.add_argument("--reject", nargs="+", metavar="CLIP", help="move these clips out (after reviewing them)")
    p.add_argument("--reason", default="wrong sign", help="logged with --reject")
    p.add_argument("--plot", action="store_true", help="save the wrist paths of the REVIEW takes")
    p.add_argument("--plot-out", default="runs/qc/review.png")
    args = p.parse_args(argv)
    rejected_dir = Path(args.rejected or Path(args.data).parent / "rejected")

    if args.reject:
        missing = [c for c in args.reject if not Path(c).is_file()]
        if missing:
            p.error(f"not found: {missing}")
        print("Rejected takes log rows (paste into docs/recording_protocol.md):")
        for c in args.reject:
            move_to_rejected(c, rejected_dir, args.reason)
        people = sorted({Path(c).stem.split("_")[0] for c in args.reject})
        signs = sorted({Path(c).parent.name for c in args.reject})
        print(f"Re-record: python -m signvaani.record --signer {people[0] if len(people) == 1 else '<sNN>'} "
              f"--signs {' '.join(signs)}")
        return 0

    clips, reject, review, short = quality_check(args.data, args.signers, args.min_hands, args.min_takes)
    for c, why in reject:
        print(f"REJECT  {c.sign}/{c.path.name}  {why}")
    for c, ratio, _ in review:
        print(f"REVIEW  {c.sign}/{c.path.name}  {ratio:.1f}x further from {c.signer}'s other {c.sign} takes "
              "than they are from each other")
    if args.plot and review:
        print(f"wrist paths of the takes to review: {plot_review(review, args.plot_out)}")

    if reject and args.apply:
        print(f"Moving {len(reject)} clips to {rejected_dir}. Rejected takes log rows "
              "(paste into docs/recording_protocol.md):")
        for c, why in reject:
            move_to_rejected(c.path, rejected_dir, why)
    elif reject:
        print(f"{len(reject)} REJECT clips stay in place: run again with --apply to move them to {rejected_dir}")

    for signer, sign, n in short:
        print(f"SHORT   {signer} {sign}: {n} good takes (need {args.min_takes})")
    by_signer = defaultdict(list)
    for signer, sign, _ in short:
        by_signer[signer].append(sign)
    if by_signer:
        print("Re-record (only the missing takes are recorded):")
        for signer, signs in by_signer.items():
            print(f"  python -m signvaani.record --signer {signer} --signs {' '.join(signs)}")

    print(f"{len(clips)} clips: {len(reject)} reject, {len(review)} to review, {len(clips) - len(reject)} good; "
          f"{len(short)} signer x sign pairs below {args.min_takes} good takes")
    return 1 if short else 0


if __name__ == "__main__":
    sys.exit(main())
