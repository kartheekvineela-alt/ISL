"""Check recorded keypoint clips (Day 1 pilot check; run it again after every recording session).

    python -m signvaani.check_data --signers s01                          # pilot: 16 signs x 10 takes
    python -m signvaani.check_data --signers s01 --plot PAIN HEAD WATER   # + wrist paths of 3 signs
    python -m signvaani.check_data                                        # every signer
    python -m signvaani.check_data --data runs/recorder_test --signs PAIN IDLE --takes 2   # a test recording

FAIL (exit code 1) when the data would train wrongly or a session is incomplete:
  - a folder or file that breaks data/keypoints/<SIGN>/<sNN>_<take>.npy with SIGN in
    configs/vocabulary.json (every folder becomes a class; the file name gives the signer);
  - an array that is not float32 (T, 63, 3);
  - a body part saved as zeros or only partly NaN: a part that was not detected must be all NaN;
  - a sign with fewer than --takes clips for a signer.
WARN when a sign (IDLE excluded) has hands visible in fewer than --min-hands of its frames:
fix the lighting or camera position and re-record it. The table's hands column is the share of
frames with at least one hand; the other frames hold NaN hands.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import CLIP_NAME, VOCAB, vocabulary
from .landmarks import L_WRIST, LEFT_HAND, N_FACE, N_POINTS, N_POSE, R_WRIST, RIGHT_HAND, hands_share

PARTS = {"pose": slice(0, N_POSE), "face": slice(N_POSE, N_POSE + N_FACE),
         "left hand": LEFT_HAND, "right hand": RIGHT_HAND}


def clip_problems(clip):
    """What is wrong with one raw clip; [] for float32 (T, 63, 3) with undetected parts all NaN."""
    if clip.dtype != np.float32 or clip.ndim != 3 or len(clip) == 0 or clip.shape[1:] != (N_POINTS, 3):
        return [f"expected float32 (T, {N_POINTS}, 3), got {clip.dtype} {clip.shape}"]
    problems = ["infinite values"] if np.isinf(clip).any() else []
    for name, part in PARTS.items():
        nan = np.isnan(clip[:, part])
        partial = nan.any(axis=(1, 2)) & ~nan.all(axis=(1, 2))
        zeros = (clip[:, part] == 0).all(axis=(1, 2))
        if partial.any():
            problems.append(f"{name} partly NaN in {partial.sum()} frames")
        if zeros.any():
            problems.append(f"{name} saved as zeros in {zeros.sum()} frames (not detected must be NaN)")
    return problems


def check(root, vocab, signers=(), takes=10, min_hands=0.9, signs=None):
    """Scan root (only `signs` if given) -> (stats per sign, signers, errors, warnings)."""
    root, errors, warnings = Path(root), [], []
    stats = {s: {"clips": defaultdict(int), "frames": [], "hands": []} for s in signs or vocab}
    if not root.is_dir():
        return stats, sorted(signers), [f"{root} does not exist"], warnings
    for sign_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if sign_dir.name not in vocab:
            errors.append(f"{sign_dir.name}/: not in the vocabulary (every folder becomes a class)")
            continue
        if sign_dir.name not in stats:
            continue
        for f in sorted(p for p in sign_dir.iterdir() if not p.name.startswith(".")):
            name = f"{sign_dir.name}/{f.name}"
            if f.suffix != ".npy":
                warnings.append(f"{name}: not a .npy clip, ignored")
                continue
            match = CLIP_NAME.fullmatch(f.stem)
            if not match:
                errors.append(f"{name}: name must be <signer>_<take>.npy with signer s01, s02, ...")
                continue
            if signers and match.group(1) not in signers:
                continue
            try:
                clip = np.load(f)
            except Exception as e:
                errors.append(f"{name}: unreadable ({e})")
                continue
            problems = clip_problems(clip)
            errors += [f"{name}: {p}" for p in problems]
            if problems:
                continue
            st = stats[sign_dir.name]
            st["clips"][match.group(1)] += 1
            st["frames"].append(len(clip))
            st["hands"].append(hands_share(clip))

    people = sorted(set(signers) | {s for st in stats.values() for s in st["clips"]})
    if not people:
        errors.append(f"no clips found under {root}")
    for sign, st in stats.items():
        errors += [f"{sign}: {st['clips'][p]}/{takes} takes for {p}" for p in people if st["clips"][p] < takes]
        if sign != "IDLE" and st["hands"] and np.mean(st["hands"]) < min_hands:
            warnings.append(f"{sign}: hands visible in {np.mean(st['hands']):.0%} of frames (target "
                            f"{min_hands:.0%}) - fix lighting / camera position and re-record")
    return stats, people, errors, warnings


def print_table(stats, people):
    print(f"{'sign':<10}" + "".join(f"{p:>6}" for p in people) + f"{'frames':>8}{'hands':>7}")
    for sign, st in stats.items():
        frames = f"{np.mean(st['frames']):.0f}" if st["frames"] else "-"
        hands = f"{np.mean(st['hands']):.0%}" if st["hands"] else "-"
        print(f"{sign:<10}" + "".join(f"{st['clips'][p]:>6}" for p in people) + f"{frames:>8}{hands:>7}")
    print(f"{'total':<10}" + "".join(f"{sum(st['clips'][p] for st in stats.values()):>6}" for p in people))


def plot_wrists(root, signs, signers=(), out="runs/pilot/wrists.png"):
    """Wrist paths of every take of each sign: takes of one sign should look alike."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(signs), figsize=(4 * len(signs), 4.4), squeeze=False)
    for ax, sign in zip(axes[0], signs):
        files = [f for f in sorted((Path(root) / sign).glob("*.npy"))
                 if (m := CLIP_NAME.fullmatch(f.stem)) and (not signers or m.group(1) in signers)]
        for f, color in zip(files, plt.cm.viridis(np.linspace(0, 1, max(len(files), 1)))):
            clip = np.load(f)
            for wrist, style in ((R_WRIST, "-"), (L_WRIST, "--")):
                ax.plot(clip[:, wrist, 0], clip[:, wrist, 1], style, color=color, lw=1)
                ax.plot(clip[0, wrist, 0], clip[0, wrist, 1], "o", color=color, ms=3)
        ax.set(title=f"{sign} ({len(files)} takes)", xlim=(0, 1), ylim=(1, 0), xlabel="x", ylabel="y")
    fig.legend(handles=[plt.Line2D([], [], color="k", ls="-", label="right wrist"),
                        plt.Line2D([], [], color="k", ls="--", label="left wrist"),
                        plt.Line2D([], [], color="k", ls="", marker="o", ms=3, label="first frame"),
                        plt.Line2D([], [], color=plt.cm.viridis(0.5), label="one colour per take")],
               loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/keypoints")
    p.add_argument("--signers", nargs="*", default=[], help="only these signers, e.g. s01 (default: all)")
    p.add_argument("--signs", nargs="*", default=[], help="only these signs (default: the whole vocabulary)")
    p.add_argument("--takes", type=int, default=10, help="takes every sign needs per signer")
    p.add_argument("--min-hands", type=float, default=0.9, help="share of frames with a hand visible")
    p.add_argument("--plot", nargs="*", metavar="SIGN", help="save the wrist paths of these signs")
    p.add_argument("--plot-out", default="runs/pilot/wrists.png")
    p.add_argument("--vocab", default=str(VOCAB))
    args = p.parse_args(argv)

    vocab = vocabulary(args.vocab)
    unknown = set(args.signs + (args.plot or [])) - set(vocab)
    if unknown:
        p.error(f"not in the vocabulary: {sorted(unknown)}")
    signs = args.signs or vocab
    stats, people, errors, warnings = check(args.data, vocab, args.signers, args.takes, args.min_hands, signs)
    print_table(stats, people)
    if args.plot:
        print(f"wrist paths saved to {plot_wrists(args.data, args.plot, args.signers, args.plot_out)}")
    for line in [f"[WARN] {w}" for w in warnings] + [f"[FAIL] {e}" for e in errors][:40]:
        print(line)
    if errors:
        print(f"FAIL: {len(errors)} problem(s)" + (" (first 40 shown)" if len(errors) > 40 else ""))
        return 1
    print(f"PASS: {len(signs)} signs x {args.takes} takes for {', '.join(people)}, "
          "all float32 (T, 63, 3) with NaN for undetected parts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
