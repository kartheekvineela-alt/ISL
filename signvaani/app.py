"""Live SignVaani app: webcam -> signs -> sentence -> speech.

    python -m signvaani.app --model models/sign_model.onnx --lang en
    python -m signvaani.app --model models/best.pt --lang hi --auto-speak
    python -m signvaani.app --preview          # no model yet: camera, skeleton and sign detection only

Keys: SPACE = speak sentence, c = clear, u = undo last sign, l = toggle language,
      s = toggle skeleton, f = full screen on/off, q = quit.
The window can also be resized or maximised; only the display grows, the camera stays at 640x480.
Top left: the model's top-3 guesses for the last movement, then whether that sign was added to the
sentence or not (and why: below --confidence, or IDLE). The signs line and the sentence at the bottom
keep the newest words on screen; c clears them, SPACE speaks and clears.
A sentence is also spoken automatically after --pause seconds without signing
when --auto-speak is on.

Tuning the sign spotter (Day 3): watch the motion value at the top. It should stay below --off at
rest and rise above --on during every sign. Signs missed: lower --on. False signs at rest: raise
--on (F4). Two signs merged: lower --quiet-frames or pause longer between signs (F6).
    python -m signvaani.app --model models/best.pt --on 0.03 --off 0.012 --quiet-frames 6

--preview needs no trained model. It shows the skeleton, the motion value and every sign the spotter
detects (as SIGN, with a count), so the camera, MediaPipe and the spotter thresholds can be checked
before Day 3. It does not recognise which sign it is and does not speak.
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .inference import SignClassifier
from .landmarks import frame_keypoints
from .sentence import SentenceBuilder
from .speech import speak


def put(img, text, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def text_width(text, scale):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]


def fit(text, width, scale):
    """The end of `text` that fits in `width` pixels, with '...' in place of the words dropped from the front,
    so the newest signs and words stay on screen."""
    words, dropped = text.split(" "), 0
    shown = lambda: " ".join(["..."] * bool(dropped) + words[dropped:])
    while len(words) - dropped > 1 and text_width(shown(), scale) > width:
        dropped += 1
    return shown()


class MotionOnly:
    """Stand-in classifier for --preview: every window the spotter closes counts as one SIGN."""
    labels = ["SIGN"]
    seq_len = 32

    def __call__(self, features):
        return np.ones(1, np.float32)


def main(argv=None):
    from .spotter import SignSpotter

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/sign_model.onnx")
    p.add_argument("--lang", default="en", choices=["en", "hi"])
    p.add_argument("--confidence", type=float, default=0.8)
    p.add_argument("--on", type=float, default=0.02, help="motion that opens a sign window")
    p.add_argument("--off", type=float, default=0.01, help="motion below which frames count as quiet")
    p.add_argument("--quiet-frames", type=int, default=8, help="quiet frames that close the window")
    p.add_argument("--auto-speak", action="store_true")
    p.add_argument("--pause", type=float, default=2.5)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fullscreen", action="store_true", help="start in full screen (f toggles it)")
    p.add_argument("--preview", action="store_true",
                   help="no model needed: camera, skeleton and sign detection only, nothing is recognised")
    args = p.parse_args(argv)
    if not args.preview and not Path(args.model).is_file():
        p.error(f"no model at {args.model}: train one first (docs/day3.md), or check the camera, skeleton and "
                "sign detection without a model: python -m signvaani.app --preview")

    clf = MotionOnly() if args.preview else SignClassifier(args.model)
    spotter = SignSpotter(clf, on_threshold=args.on, off_threshold=args.off, quiet_frames=args.quiet_frames,
                          confidence=args.confidence)
    builder = SentenceBuilder()
    glosses, last_sign_t, lang, show_skel = [], time.time(), args.lang, True
    holistic_mod, draw = mp.solutions.holistic, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)
    fps_t, fps = time.time(), 0.0
    window, full = "SignVaani", args.fullscreen
    shown_probs, verdict = None, None  # the last classified window, and whether its sign was added
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizable, keeps 4:3
    if full:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with holistic_mod.Holistic(model_complexity=1) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kp = frame_keypoints(res)

            hit = spotter.update(kp)
            if hit:
                glosses.append(hit[0])
                last_sign_t = time.time()
                print(f"sign: {hit[0]} ({hit[1]:.2f})")
            if spotter.last_probs is not shown_probs:  # a window was just classified: say what happened to it
                shown_probs = spotter.last_probs
                if hit:
                    verdict = (f"added: {hit[0]}", (0, 255, 0))
                elif clf.labels[int(np.argmax(shown_probs))] == spotter.idle_label:
                    verdict = ("not added: IDLE (no sign)", (0, 165, 255))
                else:
                    verdict = (f"not added: below {args.confidence:.0%}", (0, 165, 255))

            if args.preview:
                sentence = f"{len(glosses)} signs detected (preview, no model)"
            else:
                sentence, _ = builder.build(glosses, "en")
            if (args.auto_speak and not args.preview and glosses and not spotter.active
                    and time.time() - last_sign_t > args.pause):
                speak(builder.build(glosses, lang)[0], lang)
                glosses = []

            if show_skel:
                for lm, conn in [(res.pose_landmarks, holistic_mod.POSE_CONNECTIONS),
                                 (res.left_hand_landmarks, holistic_mod.HAND_CONNECTIONS),
                                 (res.right_hand_landmarks, holistic_mod.HAND_CONNECTIONS)]:
                    if lm:
                        draw.draw_landmarks(frame, lm, conn)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - fps_t, 1e-6)
            fps_t = now
            h, w = frame.shape[:2]
            status = ("SIGNING", (0, 0, 255)) if spotter.active else ("ready", (0, 255, 0))
            mode = "PREVIEW" if args.preview else f"lang {lang}"
            put(frame, f"{status[0]}   motion {spotter.energy:.3f} (on {args.on}, off {args.off})   "
                       f"{fps:.0f} FPS   {mode}", (15, 30), 0.6, status[1])
            if spotter.last_probs is not None and not args.preview:
                top = np.argsort(-spotter.last_probs)[:3]
                for i, k in enumerate(top):
                    put(frame, f"{clf.labels[k]:<14} {spotter.last_probs[k]:.0%}", (15, 60 + 25 * i), 0.55)
                put(frame, verdict[0], (15, 60 + 25 * len(top)), 0.55, verdict[1])
            put(frame, "Signs: " + fit(" ".join(glosses), w - 30 - text_width("Signs: ", 0.6), 0.6),
                (15, h - 50), 0.6, (0, 255, 255))
            put(frame, fit(sentence, w - 30, 0.8), (15, h - 18), 0.8)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if not args.preview:
                    speak(builder.build(glosses, lang)[0], lang)
                glosses = []
            elif key == ord("c"):
                glosses = []
            elif key == ord("u") and glosses:
                glosses.pop()
            elif key == ord("l"):
                lang = "hi" if lang == "en" else "en"
            elif key == ord("s"):
                show_skel = not show_skel
            elif key == ord("f"):
                full = not full
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
