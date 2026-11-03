"""Recognise the sign in video files with a trained model.

    python -m signvaani.predict_video --model models/sign_model.onnx --videos my_sign.mp4
    python -m signvaani.predict_video --model models/adjectives.pt --videos "C:/Users/dell/Downloads/Adjectives_4of8"
    python -m signvaani.predict_video --model models/adjectives.pt --videos <folder> --include-split test --out runs/videos.csv

Each video is one isolated sign, handled like a training clip: MediaPipe Holistic gives the 63 landmarks
of every frame (mirrored and padded to 4:3, as for INCLUDE and the webcam), the part the live spotter
would cut out is kept when the model was trained that way, and the model gives its top 3 signs. A
folder is searched for videos (.mp4, .mov, .avi, .mkv) in all its sub-folders.

When a video sits in a folder named after a sign the model knows (INCLUDE layout such as
'3. happy/MVI_5017.MOV', or your own HAPPY/clip1.mp4), that sign is the true answer and the summary
gives the accuracy over those videos. Videos the model was trained on score higher than new ones:
--include-split test keeps only INCLUDE's official test videos, which training never saw.

For a video with several signs one after another, play it through the live app instead:
    python -m signvaani.app --model models/sign_model.onnx --video my_sentence.mp4
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from .include_dataset import VIDEO_EXT, is_wanted, sign_label, video_keypoints, video_keys, videos
from .landmarks import hands_share, preprocess
from .spotter import trim_to_sign


def find_videos(paths):
    """Video files among `paths`; folders are searched in all their sub-folders."""
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXT)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"no video or folder at {p}")
    return found


def true_sign(path, labels):
    """The known sign named by the nearest folder above the video (e.g. '3. happy' -> HAPPY), or None."""
    for folder in Path(path).parent.parts[::-1]:
        label = sign_label(folder)
        if label in labels:
            return label
    return None


def classify(raw, clf):
    """Raw clip (T, 63, 3) -> probability per label, cut and preprocessed the way the model was trained."""
    seq = trim_to_sign(raw) if getattr(clf, "trim", False) else raw
    return clf(preprocess(seq, clf.seq_len))


def mediapipe_holistic(complexity=1):
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(model_complexity=complexity)


def predict(paths, clf, make_holistic=mediapipe_holistic, flip=True, aspect=4 / 3):
    """One record per video: true sign (or None), top-3 guesses with probabilities, frames, hand share."""
    records = []
    for path in paths:
        with make_holistic() as holistic:  # fresh tracker per video, as in include_dataset.convert
            raw = video_keypoints(path, holistic, flip, aspect)
        rec = {"video": str(path), "true": true_sign(path, clf.labels), "frames": 0 if raw is None else len(raw)}
        if raw is None:
            rec.update(top3=[], hands=0.0)
        else:
            probs = classify(raw, clf)
            rec.update(top3=[(clf.labels[k], float(probs[k])) for k in np.argsort(-probs)[:3]],
                       hands=hands_share(raw))
        records.append(rec)
    return records


def line(i, n, rec):
    """One printed line per video."""
    name = Path(rec["video"]).name
    if not rec["top3"]:
        return f"[{i}/{n}] {name}: could not be read"
    (sign, p), *rest = rec["top3"]
    text = f"[{i}/{n}] {name}  ->  {sign} {p:.0%}"
    if rest:
        text += "   (then " + ", ".join(f"{s} {q:.0%}" for s, q in rest) + ")"
    if rec["true"]:
        text += f"   true {rec['true']}: " + ("RIGHT" if sign == rec["true"] else "WRONG")
    if rec["hands"] < 0.5:
        text += f"   [hands seen in only {rec['hands']:.0%} of frames]"
    return text


def summary(records):
    """Accuracy lines over the videos whose folder names a known sign."""
    known = [r for r in records if r["true"] and r["top3"]]
    out = [f"{len(records)} videos, {sum(not r['top3'] for r in records)} unreadable"]
    if known:
        top1 = sum(r["top3"][0][0] == r["true"] for r in known)
        top3 = sum(r["true"] in [s for s, _ in r["top3"]] for r in known)
        out.append(f"Accuracy over {len(known)} videos in sign folders: top-1 {top1}/{len(known)} "
                   f"({top1 / len(known):.0%}), top-3 {top3}/{len(known)} ({top3 / len(known):.0%})")
    else:
        out.append("No video is in a folder named after a sign the model knows, so no accuracy is given.")
    return out


def main(argv=None, make_holistic=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/sign_model.onnx", help=".onnx (with its .json) or .pt checkpoint")
    p.add_argument("--videos", nargs="+", required=True, help="video files and/or folders")
    p.add_argument("--include-split", choices=["train", "val", "test"],
                   help="keep only the INCLUDE videos of this official split (test = never used in training)")
    p.add_argument("--out", help="CSV file with one row per video")
    p.add_argument("--no-flip", action="store_true", help="do not mirror (the webcam, recorder and INCLUDE are mirrored)")
    p.add_argument("--aspect", type=float, default=4 / 3, help="pad frames to this width/height (0 = off)")
    p.add_argument("--complexity", type=int, choices=[0, 1, 2], default=1, help="MediaPipe Holistic model")
    args = p.parse_args(argv)
    if not Path(args.model).is_file():
        p.error(f"no model at {args.model}: train one first (python -m signvaani.train ...)")
    try:
        paths = find_videos(args.videos)
    except FileNotFoundError as e:
        p.error(str(e))
    if args.include_split:
        keys = video_keys([v for v in videos("include") if v.split == args.include_split])
        paths = [v for v in paths if is_wanted(v.parts, keys)]
    if not paths:
        p.error("no videos found" + (f" from the INCLUDE {args.include_split} split" if args.include_split else ""))

    from .inference import SignClassifier
    clf = SignClassifier(args.model)
    print(f"{len(paths)} videos, model knows {len(clf.labels)} signs: {' '.join(clf.labels)}")
    holistic = make_holistic or (lambda: mediapipe_holistic(args.complexity))
    records = []
    for i, path in enumerate(paths, 1):
        rec = predict([path], clf, holistic, not args.no_flip, args.aspect)[0]
        print(line(i, len(paths), rec), flush=True)
        records.append(rec)
    for text in summary(records):
        print(text)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "true", "predicted", "confidence", "top3", "correct", "frames", "hands"])
            for r in records:
                pred, conf = r["top3"][0] if r["top3"] else ("", 0.0)
                w.writerow([r["video"], r["true"] or "", pred, f"{conf:.3f}",
                            " ".join(s for s, _ in r["top3"]), "" if not r["true"] else int(pred == r["true"]),
                            r["frames"], f"{r['hands']:.2f}"])
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
