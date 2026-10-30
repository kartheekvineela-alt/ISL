"""Live SignVaani app: webcam -> signs -> sentence -> speech.

    python -m signvaani.app --model models/sign_model.onnx --lang en
    python -m signvaani.app --model models/best.pt --lang hi --auto-speak
    python -m signvaani.app --preview          # no model yet: camera, skeleton and sign detection only

Keys: SPACE = speak sentence, c = clear, u = undo last sign, l = toggle language,
      s = toggle skeleton, f = full screen on/off, q = quit.
The window can also be resized or maximised; only the display grows, the camera stays at 640x480.
Top left: the model's top-3 guesses for the last movement as confidence bars, then whether that sign was
added to the sentence or not (and why: below --confidence, or IDLE). The signs line and the sentence at the bottom
keep the newest words on screen; c clears them, SPACE speaks and clears.
A sentence is also spoken automatically after --pause seconds without signing
when --auto-speak is on.

Tuning the sign spotter (Day 3): watch the motion value at the top. It should stay below --off at
rest and rise above --on during every sign. Signs missed: lower --on. False signs at rest: raise
--on (F4). Two signs merged: lower --quiet-frames or pause longer between signs (F6).
    python -m signvaani.app --model models/best.pt --on 0.03 --off 0.012 --quiet-frames 6

Measuring speed (Day 4, TC-01 and TC-14): each accepted sign prints its latency, the time from the frame
where the hands stopped moving to the sign being accepted. On quit the app prints the average frame rate
and the median and worst latency; --log saves every sign to a CSV file as test evidence. The latency
starts when the motion value falls below --off, a frame or two after the hands really stop, and leaves out
the camera's own delay, so the filmed TC-14 measurement stays the official one.
    python -m signvaani.app --model models/sign_model.onnx --log runs/live_log.csv

--preview needs no trained model. It shows the skeleton, the motion value and every sign the spotter
detects (as SIGN, with a count), so the camera, MediaPipe and the spotter thresholds can be checked
before Day 3. It does not recognise which sign it is and does not speak.
"""
import argparse
import csv
import functools
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


def fit(text, width, scale, measure=None):
    """The end of `text` that fits in `width` pixels, with '...' in place of the words dropped from the front,
    so the newest signs and words stay on screen. measure(text) -> pixels defaults to OpenCV's font."""
    measure = measure or (lambda t: text_width(t, scale))
    words, dropped = text.split(" "), 0
    shown = lambda: " ".join(["..."] * bool(dropped) + words[dropped:])
    while len(words) - dropped > 1 and measure(shown()) > width:
        dropped += 1
    return shown()


# OpenCV's fonts cannot draw Devanagari: the Hindi caption is drawn with Pillow and the first font found.
HINDI_FONTS = ["C:/Windows/Fonts/Nirmala.ttf", "C:/Windows/Fonts/mangal.ttf",
               "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc", "/System/Library/Fonts/Kohinoor.ttc",
               "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
               "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
               "/usr/share/fonts/truetype/freefont/FreeSans.ttf"]


@functools.lru_cache(maxsize=1)
def hindi_font(size=26):
    """A Devanagari font with proper shaping (Pillow built with Raqm), or None: then the caption stays
    English while the speech is still Hindi."""
    try:
        from PIL import ImageFont, features
    except ImportError:
        return None
    if not features.check("raqm"):  # without it, vowel signs and conjuncts would be drawn wrongly
        return None
    path = next((f for f in HINDI_FONTS if Path(f).is_file()), None)
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM) if path else None


def put_unicode(img, text, org, font, color=(255, 255, 255)):
    """put() for text OpenCV cannot draw: Pillow draws it, outlined, straight onto the BGR frame."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text(org, text, font=font, fill=color, anchor="ls", stroke_width=3, stroke_fill=(0, 0, 0))
    img[:] = np.asarray(pil)


def summary(frames, seconds, latencies):
    """Lines printed on quit: average frame rate (TC-01) and sign latency (TC-14)."""
    lines = [f"Frame rate: {frames / max(seconds, 1e-6):.1f} FPS on average over {frames} frames "
             f"({seconds:.0f} s); TC-01 needs at least 15."]
    if latencies:
        lines.append(f"Latency, hands stop -> sign accepted: median {np.median(latencies):.0f} ms, "
                     f"worst {max(latencies):.0f} ms over {len(latencies)} signs; TC-14 needs at most 500 ms.")
    else:
        lines.append("Latency: no sign was accepted.")
    return lines


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
    p.add_argument("--log", help="CSV file that records every accepted sign with its latency")
    p.add_argument("--complexity", type=int, choices=[0, 1, 2], default=1,
                   help="MediaPipe Holistic model: 0 is fastest (fallback F5 when below 15 FPS), 2 most accurate")
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
    frames, started, latencies = 0, time.time(), []
    warned_hindi = False
    log = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log, "a", newline="", encoding="utf-8")
        log = csv.writer(log_file)
        if log_file.tell() == 0:
            log.writerow(["time", "sign", "confidence", "latency_ms", "fps"])
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizable, keeps 4:3
    if full:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with holistic_mod.Holistic(model_complexity=args.complexity) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            captured, frames = time.time(), frames + 1
            frame = cv2.flip(frame, 1)
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kp = frame_keypoints(res)

            hit = spotter.update(kp, captured)
            if hit:
                glosses.append(hit[0])
                last_sign_t = time.time()
                latency = None if spotter.quiet_since is None else 1000 * (last_sign_t - spotter.quiet_since)
                if latency is not None:
                    latencies.append(latency)
                print(f"sign: {hit[0]} ({hit[1]:.2f})" + ("" if latency is None else f"  latency {latency:.0f} ms"))
                if log:
                    log.writerow([time.strftime("%H:%M:%S"), hit[0], f"{hit[1]:.3f}",
                                  "" if latency is None else f"{latency:.0f}", f"{fps:.1f}"])
            if spotter.last_probs is not shown_probs:  # a window was just classified: say what happened to it
                shown_probs = spotter.last_probs
                if hit:
                    verdict = (f"added: {hit[0]}" + ("" if latency is None else f" ({latency:.0f} ms)"), (0, 255, 0))
                elif clf.labels[int(np.argmax(shown_probs))] == spotter.idle_label:
                    verdict = ("not added: IDLE (no sign)", (0, 165, 255))
                else:
                    verdict = (f"not added: below {args.confidence:.0%}", (0, 165, 255))

            caption_font = hindi_font() if lang == "hi" and not args.preview else None
            if lang == "hi" and not caption_font and not warned_hindi:
                print("Hindi caption needs Pillow with Raqm and a Devanagari font: the caption stays English, "
                      "the speech is Hindi")
                warned_hindi = True
            if args.preview:
                sentence = f"{len(glosses)} signs detected (preview, no model)"
            else:
                sentence, _ = builder.build(glosses, "hi" if caption_font else "en")
            if (args.auto_speak and not args.preview and glosses and not spotter.active
                    and time.time() - last_sign_t > args.pause):
                text = builder.build(glosses, lang)[0]
                print(f"spoken ({lang}): {text}")
                speak(text, lang)
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
                for i, k in enumerate(top):  # confidence bars: the best guess in green
                    y, prob = 60 + 25 * i, float(spotter.last_probs[k])
                    cv2.rectangle(frame, (15, y - 16), (175, y + 5), (40, 40, 40), -1)
                    cv2.rectangle(frame, (15, y - 16), (15 + int(160 * prob), y + 5),
                                  (0, 160, 0) if i == 0 else (110, 110, 110), -1)
                    put(frame, f"{clf.labels[k]} {prob:.0%}", (20, y), 0.55)
                put(frame, verdict[0], (15, 60 + 25 * len(top)), 0.55, verdict[1])
            put(frame, "Signs: " + fit(" ".join(glosses), w - 30 - text_width("Signs: ", 0.6), 0.6),
                (15, h - 50), 0.6, (0, 255, 255))
            if caption_font:
                put_unicode(frame, fit(sentence, w - 30, None, caption_font.getlength), (15, h - 18), caption_font)
            else:
                put(frame, fit(sentence, w - 30, 0.8), (15, h - 18), 0.8)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if not args.preview and glosses:
                    text = builder.build(glosses, lang)[0]
                    print(f"spoken ({lang}): {text}")
                    speak(text, lang)
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
    if log:
        log_file.close()
    for line in summary(frames, time.time() - started, latencies):
        print(line)


if __name__ == "__main__":
    main()
