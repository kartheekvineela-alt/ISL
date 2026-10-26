"""INCLUDE (AI4Bharat) isolated Indian Sign Language dataset -> SignVaani keypoint clips.

    python -m signvaani.include_dataset download --subset include50
    python -m signvaani.include_dataset convert --subset include50
    python -m signvaani.train --data data/include_keypoints/include50 --test-signers test --val-signers val

Subsets: include (all 263 signs), include50 (the official 50-sign subset) and hospital (the INCLUDE signs
that are also in configs/vocabulary.json, saved under the vocabulary label). The official split lists are
in data/include/splits. INCLUDE publishes no per-video signer IDs, so the split name takes the signer's
place in the file name and --test-signers test --val-signers val reproduces the official split:

    data/include_keypoints/<subset>/<SIGN>/<split>_<video>.npy     e.g. .../DOCTOR/test_MVI_5303.npy

Frames are mirrored and padded to 4:3 before MediaPipe Holistic, so the clips look like what record.py
and app.py get from a mirrored 640x480 webcam.
"""
import argparse
import functools
import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from .data import vocabulary
from .landmarks import frame_keypoints, hands_share

SPLITS = Path(__file__).resolve().parents[1] / "data" / "include" / "splits"
ZENODO_RECORD = "https://zenodo.org/api/records/4010759"
SUBSETS = ("include", "include50", "hospital")
VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv"}

# INCLUDE labels that differ from the configs/vocabulary.json label of the same sign.
RENAME = {"I": "ME"}


def hospital_label(sign, vocab):
    """Vocabulary label of an INCLUDE sign, or None when the vocabulary does not have it."""
    label = RENAME.get(sign, sign)
    return label if label in vocab else None


@dataclass(frozen=True)
class Video:
    rel: str  # path inside the INCLUDE release, e.g. "Jobs/87. Doctor/MVI_5303.MOV"
    category: str
    sign: str
    split: str

    @property
    def clip_name(self):
        return f"{self.split}_{PurePosixPath(self.rel).stem}.npy"


def sign_label(folder):
    """INCLUDE class folder -> SignVaani label, e.g. '46. you (plural)' -> 'YOU-PLURAL'."""
    name = re.sub(r"^\s*(\d+|Ex)\.\s*", "", folder)
    return re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")


def videos(subset="include", splits_dir=SPLITS):
    """Every video of a subset with its official split. The class is the second path part, so
    clips in 'Extra' sub-folders count towards their parent sign."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}, choose from {SUBSETS}")
    source = "include50" if subset == "include50" else "include"
    vocab = set(vocabulary()) if subset == "hospital" else None
    out = []
    for split in ("train", "val", "test"):
        for line in (Path(splits_dir) / f"{source}_{split}.txt").read_text().splitlines():
            if not line.strip():
                continue
            parts = PurePosixPath(line.strip()).parts
            sign = sign_label(parts[1])
            if vocab is not None:
                sign = hospital_label(sign, vocab)
                if sign is None:
                    continue
            out.append(Video(line.strip(), parts[0], sign, split))
    return out


# ---------------------------------------------------------------- download

def _get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def zenodo_files(record=ZENODO_RECORD):
    """[(name, size, md5 or None, url)] for every file of a Zenodo record."""
    meta = _get_json(record)
    files = meta.get("files")
    if not isinstance(files, list):  # newer API lists the files separately
        files = _get_json(record.rstrip("/") + "/files")["entries"]
    out = []
    for f in files:
        checksum = f.get("checksum") or ""
        md5 = checksum[4:] if checksum.startswith("md5:") else None
        out.append((f["key"], f.get("size"), md5, f["links"].get("content") or f["links"]["self"]))
    return out


def download_file(url, dest, size=None, md5=None):
    """Stream url to dest, checking the md5; an existing file of the right size is kept."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.md5()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            digest.update(chunk)
    if md5 and digest.hexdigest() != md5:
        tmp.unlink()
        raise IOError(f"checksum mismatch for {dest.name}, run the command again")
    tmp.replace(dest)


def _key(name):
    return re.sub(r"[^a-z]", "", name.lower())


def video_keys(wanted):
    """(sign label of the class folder, file name) of each wanted video."""
    return {(sign_label(PurePosixPath(v.rel).parts[1]), PurePosixPath(v.rel).name) for v in wanted}


def is_wanted(parts, keys):
    """True when the file at path parts (folders..., name) is a wanted video, however deep it is nested."""
    return any((sign_label(d), parts[-1]) in keys for d in parts[:-1])


def download(args):
    categories = sorted({v.category for v in videos(args.subset)})
    files = [f for f in zenodo_files(args.record) if f[0].lower().endswith(".zip")]
    if not args.all:
        missing = [c for c in categories if not any(_key(f[0]).startswith(_key(c)) for f in files)]
        if missing:
            raise SystemExit(f"no archive matches {missing}; archives in the record: {[f[0] for f in files]}. "
                             "Use --all to download everything.")
        files = [f for f in files if any(_key(f[0]).startswith(_key(c)) for c in categories)]
    # Only the subset's videos are unpacked: include50 needs every archive but keeps 958 of 4,292 videos.
    keys = video_keys(videos(args.subset))
    scope = "all" if args.all_videos or args.subset == "include" else args.subset
    print(f"{len(files)} archives, {sum(f[1] or 0 for f in files) / 1e9:.1f} GB to download:")
    for name, size, _, _ in files:
        print(f"  {name}  {(size or 0) / 1e9:.2f} GB")
    print("every video is unpacked" if scope == "all" else
          f"only the {len(keys)} {args.subset} videos are unpacked, one archive at a time")
    if args.dry_run:
        return
    raw, zips = Path(args.raw), Path(args.raw) / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    for name, size, md5, url in files:
        # name.extracted is the marker of earlier versions, which unpacked everything
        if any((zips / m).exists() for m in (f"{name}.{scope}.extracted", f"{name}.all.extracted",
                                            f"{name}.extracted")):
            continue
        print(f"downloading {name}")
        download_file(url, zips / name, size, md5)
        with zipfile.ZipFile(zips / name) as z:
            members = [m for m in z.namelist() if not m.endswith("/")
                       and (scope == "all" or is_wanted(PurePosixPath(m).parts, keys))]
            z.extractall(raw, members)
        print(f"  unpacked {len(members)} files")
        (zips / f"{name}.{scope}.extracted").touch()
        if not args.keep_zips:
            (zips / name).unlink()
    print(f"videos are under {raw}; next: python -m signvaani.include_dataset convert --subset {args.subset}")


# ---------------------------------------------------------------- convert

def pad_to_aspect(frame, aspect):
    """Pad with black borders to width / height = aspect (padding never cuts off a hand)."""
    import cv2

    h, w = frame.shape[:2]
    if aspect <= 0 or abs(w / h - aspect) < 1e-3:
        return frame
    if w / h > aspect:
        extra = round(w / aspect) - h
        return cv2.copyMakeBorder(frame, extra // 2, extra - extra // 2, 0, 0, cv2.BORDER_CONSTANT)
    extra = round(h * aspect) - w
    return cv2.copyMakeBorder(frame, 0, 0, extra // 2, extra - extra // 2, cv2.BORDER_CONSTANT)


def video_keypoints(path, holistic, flip=True, aspect=4 / 3):
    """Run Holistic over a video -> (T, 63, 3) float32 with NaN for missed points, or None if unreadable."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        frame = pad_to_aspect(frame, aspect)
        frames.append(frame_keypoints(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    return np.stack(frames).astype(np.float32) if frames else None


def find_videos(raw, wanted):
    """Locate each wanted Video under raw, however the archives were unpacked."""
    by_name = defaultdict(list)
    for p in Path(raw).rglob("*"):
        if p.suffix.lower() in VIDEO_EXT:
            by_name[p.name].append(p)
    found = {}
    for v in wanted:
        rel = PurePosixPath(v.rel)
        if (Path(raw) / rel).is_file():
            found[v] = Path(raw) / rel
            continue
        sign = sign_label(rel.parts[1])
        match = [p for p in by_name.get(rel.name, []) if any(sign_label(d) == sign for d in p.parts[:-1])]
        if match:
            found[v] = match[0]
    return found


def mediapipe_holistic():
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(model_complexity=1)


def convert_one(job, make_holistic=mediapipe_holistic):
    """(src, dest, flip, aspect) -> (dest, frames, share of frames with a hand); frames is None on failure."""
    src, dest, flip, aspect = job
    with make_holistic() as holistic:  # fresh tracker per video
        seq = video_keypoints(src, holistic, flip, aspect)
    if seq is None:
        return dest, None, 0.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, seq)
    return dest, len(seq), hands_share(seq)


def convert(args, make_holistic=mediapipe_holistic):
    wanted = videos(args.subset, args.splits)
    out = Path(args.out or f"data/include_keypoints/{args.subset}")
    found = find_videos(args.raw, wanted)
    missing = [v.rel for v in wanted if v not in found]
    jobs = [(found[v], out / v.sign / v.clip_name, not args.no_flip, args.aspect) for v in wanted
            if v in found and (args.overwrite or not (out / v.sign / v.clip_name).exists())]
    print(f"{len(wanted)} videos in {args.subset}: {len(found)} found, {len(jobs)} to convert")

    work = functools.partial(convert_one, make_holistic=make_holistic)
    failed, weak = [], []
    with ProcessPoolExecutor(args.workers) if args.workers > 1 else _Inline() as pool:
        for i, (dest, n, hands) in enumerate(pool.map(work, jobs), 1):
            if n is None:
                failed.append(str(dest))
            elif hands < 0.5:
                weak.append(f"{dest}  hands in {hands:.0%} of frames")
            print(f"[{i}/{len(jobs)}] {dest}  {n or 0} frames, hands {hands:.0%}")

    counts = Counter(p.parent.name for p in out.glob("*/*.npy"))
    print(f"{sum(counts.values())} clips in {out}, {len(counts)} signs")
    log = out / "conversion_log.txt"
    out.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join([f"# missing videos ({len(missing)})", *missing,
                              f"# unreadable videos ({len(failed)})", *failed,
                              f"# weak hand tracking, check by eye ({len(weak)})", *weak]) + "\n")
    if missing or failed or weak:
        print(f"{len(missing)} missing, {len(failed)} unreadable, {len(weak)} weak clips: see {log}")


class _Inline:
    """Stand-in for a process pool when --workers 1."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, items):
        return map(fn, items)


# ---------------------------------------------------------------- classes

def classes_csv(splits_dir=SPLITS):
    """One CSV row per INCLUDE sign: category, folder, label, clips per split, subsets."""
    rows = {}
    include50 = {v.sign for v in videos("include50", splits_dir)}
    vocab = set(vocabulary())
    for v in videos("include", splits_dir):
        folder = PurePosixPath(v.rel).parts[1]
        row = rows.setdefault(v.sign, {"category": v.category, "folder": folder, "train": 0, "val": 0, "test": 0})
        row[v.split] += 1
    lines = ["category,folder,sign,train,val,test,include50,hospital_sign"]
    for sign, r in sorted(rows.items(), key=lambda kv: (kv[1]["category"], kv[0])):
        lines.append(f'{r["category"]},"{r["folder"]}",{sign},{r["train"]},{r["val"]},{r["test"]},'
                     f'{"yes" if sign in include50 else ""},{hospital_label(sign, vocab) or ""}')
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download and unzip the INCLUDE videos from Zenodo")
    d.add_argument("--subset", choices=SUBSETS, default="include50")
    d.add_argument("--raw", default="data/raw_videos/include")
    d.add_argument("--record", default=ZENODO_RECORD)
    d.add_argument("--all", action="store_true", help="download every archive in the record")
    d.add_argument("--all-videos", action="store_true", help="unpack every video, not only the subset's")
    d.add_argument("--keep-zips", action="store_true")
    d.add_argument("--dry-run", action="store_true", help="only list the archives and their size")

    c = sub.add_parser("convert", help="videos -> data/include_keypoints/<subset>/<SIGN>/<split>_<video>.npy")
    c.add_argument("--subset", choices=SUBSETS, default="include50")
    c.add_argument("--raw", default="data/raw_videos/include")
    c.add_argument("--out", help="default: data/include_keypoints/<subset>")
    c.add_argument("--splits", default=str(SPLITS))
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--aspect", type=float, default=4 / 3, help="pad frames to this width/height (0 = off)")
    c.add_argument("--no-flip", action="store_true", help="do not mirror (record.py and app.py mirror)")
    c.add_argument("--overwrite", action="store_true")

    sub.add_parser("classes", help="print data/include/classes.csv")
    args = p.parse_args(argv)
    if args.cmd == "download":
        download(args)
    elif args.cmd == "convert":
        convert(args)
    else:
        print(classes_csv(), end="")


if __name__ == "__main__":
    main()
