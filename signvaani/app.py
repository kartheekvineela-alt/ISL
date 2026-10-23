"""Live SignVaani app: webcam -> signs -> sentence -> speech.

    python -m signvaani.app --model models/sign_model.onnx --lang en
    python -m signvaani.app --model models/best.pt --lang hi --auto-speak

Keys: SPACE = speak sentence, c = clear, u = undo last sign, l = toggle language,
      s = toggle skeleton, q = quit.
A sentence is also spoken automatically after --pause seconds without signing
when --auto-speak is on.
"""
import argparse
import time

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


def main():
    from .spotter import SignSpotter

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/sign_model.onnx")
    p.add_argument("--lang", default="en", choices=["en", "hi"])
    p.add_argument("--confidence", type=float, default=0.8)
    p.add_argument("--auto-speak", action="store_true")
    p.add_argument("--pause", type=float, default=2.5)
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    clf = SignClassifier(args.model)
    spotter = SignSpotter(clf, confidence=args.confidence)
    builder = SentenceBuilder()
    glosses, last_sign_t, lang, show_skel = [], time.time(), args.lang, True
    holistic_mod, draw = mp.solutions.holistic, mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)
    fps_t, fps = time.time(), 0.0

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

            sentence, _ = builder.build(glosses, "en")
            if args.auto_speak and glosses and not spotter.active and time.time() - last_sign_t > args.pause:
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
            put(frame, f"{status[0]}   motion {spotter.energy:.3f}   {fps:.0f} FPS   lang {lang}", (15, 30), 0.6, status[1])
            if spotter.last_probs is not None:
                top = np.argsort(-spotter.last_probs)[:3]
                for i, k in enumerate(top):
                    put(frame, f"{clf.labels[k]:<14} {spotter.last_probs[k]:.0%}", (15, 60 + 25 * i), 0.55)
            put(frame, "Signs: " + " ".join(glosses), (15, h - 50), 0.6, (0, 255, 255))
            put(frame, sentence, (15, h - 18), 0.8)
            cv2.imshow("SignVaani", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
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
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
