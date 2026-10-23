"""Guided recorder for building your own ISL keypoint dataset (blueprint Appendix A).

    python -m signvaani.record --signer s01                         # every sign, 10 takes each
    python -m signvaani.record --signer s02 --signs PAIN WATER IDLE --takes 3

For every sign it shows the name, counts down, records ~2.5 s, checks quality
and saves data/keypoints/<SIGN>/<signer>_<take>.npy (keypoints only, no video).
--takes is how many takes each sign should have for this signer: takes saved
earlier are never overwritten, only the missing ones are recorded. So an
interrupted session resumes, and clips deleted in quality control are re-recorded.
Keys: q = quit, k = skip current sign, r = redo (delete) the take just saved.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .data import SIGNER_ID, recorded_takes, vocabulary
from .landmarks import frame_keypoints, hands_visible

# IDLE takes cycle through the natural non-sign movements of Appendix A.
IDLE_PROMPTS = ["rest your hands", "touch your face", "adjust your hair", "look at your phone", "small shrug"]


def put(img, text, y, scale=0.8, color=(255, 255, 255)):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--signer", required=True, help="anonymous signer id: s01, s02, ... (never a name)")
    p.add_argument("--signs", nargs="*", help="subset of signs (default: configs/vocabulary.json)")
    p.add_argument("--takes", type=int, default=10, help="takes each sign should have for this signer")
    p.add_argument("--seconds", type=float, default=2.5)
    p.add_argument("--out", default="data/keypoints")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args(argv)

    if not SIGNER_ID.fullmatch(args.signer):
        p.error(f"--signer must be an anonymous id like s01, not {args.signer!r}")
    vocab = vocabulary()
    signs = args.signs or vocab
    unknown = [s for s in signs if s not in vocab]
    if unknown:
        p.error(f"not in configs/vocabulary.json: {unknown}")
    missing = {s: max(0, args.takes - len(recorded_takes(Path(args.out) / s, args.signer))) for s in signs}
    print(f"{args.signer}: {sum(missing.values())} takes to record "
          f"({len(signs)} signs x {args.takes}, {len(signs) * args.takes - sum(missing.values())} already saved)")
    if not any(missing.values()):
        return

    cap = cv2.VideoCapture(args.camera)
    draw, holistic_mod = mp.solutions.drawing_utils, mp.solutions.holistic

    try:
        with holistic_mod.Holistic(model_complexity=1, min_detection_confidence=0.5) as holistic:
            def read():
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("camera not available")
                frame = cv2.flip(frame, 1)
                res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for lm, conn in [(res.pose_landmarks, holistic_mod.POSE_CONNECTIONS),
                                 (res.left_hand_landmarks, holistic_mod.HAND_CONNECTIONS),
                                 (res.right_hand_landmarks, holistic_mod.HAND_CONNECTIONS)]:
                    if lm:
                        draw.draw_landmarks(frame, lm, conn)
                return frame, res

            for sign in signs:
                out_dir = Path(args.out) / sign
                out_dir.mkdir(parents=True, exist_ok=True)
                saved = recorded_takes(out_dir, args.signer)
                take, next_id = len(saved), (saved[-1] + 1 if saved else 0)
                while take < args.takes:
                    hint = f"Do not sign - {IDLE_PROMPTS[take % len(IDLE_PROMPTS)]}" if sign == "IDLE" else ""
                    # Countdown
                    t0 = time.time()
                    while time.time() - t0 < 3:
                        frame, _ = read()
                        put(frame, f"Sign: {sign}   take {take + 1}/{args.takes}", 40, 1.0, (0, 255, 255))
                        put(frame, f"Starting in {3 - int(time.time() - t0)}...", 80)
                        put(frame, hint, 120)
                        cv2.imshow("SignVaani recorder", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            return
                        if key == ord("k"):
                            take = args.takes
                            break
                    if take >= args.takes:
                        break
                    # Record
                    seq, t0 = [], time.time()
                    while time.time() - t0 < args.seconds:
                        frame, res = read()
                        seq.append(frame_keypoints(res))
                        put(frame, f"RECORDING {sign}", 40, 1.0, (0, 0, 255))
                        put(frame, hint, 80)
                        cv2.imshow("SignVaani recorder", frame)
                        cv2.waitKey(1)
                    seq = np.stack(seq)
                    # Quality checks
                    vis = np.array([any(hands_visible(f)) for f in seq])
                    body_ok = ~np.isnan(seq[:, 3:5, 0]).any(axis=1)
                    problem = None
                    if body_ok.mean() < 0.9:
                        problem = "Shoulders not visible - step back"
                    elif sign != "IDLE" and vis.mean() < 0.6:
                        problem = f"Hands detected in only {vis.mean():.0%} of frames - improve light / framing"
                    if problem:
                        frame, _ = read()
                        put(frame, problem, 40, 0.7, (0, 0, 255))
                        put(frame, "Retrying...", 80)
                        cv2.imshow("SignVaani recorder", frame)
                        cv2.waitKey(1500)
                        continue
                    path = out_dir / f"{args.signer}_{next_id}.npy"
                    np.save(path, seq)
                    print(f"saved {path}  ({len(seq)} frames, hands {vis.mean():.0%})")
                    # One second to throw the take away and record it again.
                    t0 = time.time()
                    while time.time() - t0 < 1:
                        frame, _ = read()
                        put(frame, f"Saved take {take + 1}/{args.takes}   r = redo", 40, 0.8, (0, 255, 0))
                        cv2.imshow("SignVaani recorder", frame)
                        if cv2.waitKey(1) & 0xFF == ord("r"):
                            path.unlink()
                            print(f"deleted {path} (redo)")
                            break
                    else:
                        take, next_id = take + 1, next_id + 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
