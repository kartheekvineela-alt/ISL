"""Day 1 install check (1-week blueprint, Section 6): every team member runs it and must see OK.

    python -m signvaani.check_install               # Python, packages, MediaPipe models, speech engine
    python -m signvaani.check_install --camera 0    # + webcam frame rate with MediaPipe (target >= 15 FPS)

[FAIL] lines must be fixed. [WARN] lines differ from the blueprint's pinned setup but do not block.
If MediaPipe will not install, use fallback F1: Python 3.10, mediapipe==0.10.14 and "numpy<2".
"""
import argparse
import importlib
import os
import platform
import sys
import time
from importlib import metadata

# (module, pip install argument, version prefix that must match). Only mediapipe and numpy are
# checked (fallback F1): mediapipe installs its own opencv-contrib-python, so cv2 may be newer.
PACKAGES = [
    ("mediapipe", "mediapipe==0.10.14", "0.10.14"),
    ("cv2", "opencv-python==4.10.0.84", None),
    ("numpy", '"numpy<2"', "1."),
    ("torch", "torch --index-url https://download.pytorch.org/whl/cpu", None),
    ("onnx", "onnx", None),
    ("onnxruntime", "onnxruntime", None),
    ("sklearn", "scikit-learn", None),
    ("matplotlib", "matplotlib", None),
    ("pyttsx3", "pyttsx3", None),
    ("gtts", "gTTS", None),
    ("pygame", "pygame", None),
    ("pytest", "pytest", None),
]
DISTRIBUTION = {"cv2": "opencv-python", "sklearn": "scikit-learn", "gtts": "gTTS"}


def version_of(module, name):
    v = getattr(module, "__version__", None)
    if isinstance(v, str):
        return v
    try:
        return metadata.version(DISTRIBUTION.get(name, name))
    except metadata.PackageNotFoundError:
        return "?"


def check_python(version=sys.version_info):
    v = ".".join(map(str, version[:3]))
    if tuple(version[:2]) in [(3, 10), (3, 11)]:
        return "OK", f"Python {v}"
    return "WARN", f"Python {v}: the blueprint uses 3.10 or 3.11 (MediaPipe 0.10.14 wheels)"


def check_packages(import_module=importlib.import_module):
    """[(status, message)] per package, plus the modules that imported."""
    results, loaded = [], {}
    for name, pip_arg, pin in PACKAGES:
        try:
            module = import_module(name)
        except Exception as e:  # ImportError, or a binary built against another numpy
            detail = (str(e).strip().splitlines() or [""])[0]
            results.append(("FAIL", f"{name}: {type(e).__name__} {detail} -> pip install {pip_arg}"))
            continue
        loaded[name] = module
        v = version_of(module, name)
        if pin and not v.startswith(pin):
            results.append(("WARN", f"{name} {v}: the blueprint pins {pip_arg}"))
        else:
            results.append(("OK", f"{name} {v}"))
    return results, loaded


def check_holistic(mp, np):
    try:
        with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
            holistic.process(np.zeros((480, 640, 3), np.uint8))
    except Exception as e:
        return "FAIL", f"MediaPipe Holistic does not run ({type(e).__name__}: {e}) -> pip install mediapipe==0.10.14"
    return "OK", "MediaPipe Holistic loads its models and runs"


def check_speech(pyttsx3):
    try:
        pyttsx3.init().stop()
    except Exception as e:
        hint = " -> sudo apt install espeak-ng" if sys.platform.startswith("linux") else ""
        return "WARN", f"pyttsx3 speech engine does not start ({type(e).__name__}: {e}){hint}"
    return "OK", "pyttsx3 speech engine starts (offline English)"


def check_camera(cv2, mp, index, seconds):
    cap = cv2.VideoCapture(index)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return "FAIL", f"camera {index} gives no frames -> close other apps that use it, or try --camera 1"
    height, width = frame.shape[:2]
    frames = hands = 0
    with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
        t0 = time.time()
        while time.time() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            res = holistic.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
            frames += 1
            hands += bool(res.left_hand_landmarks or res.right_hand_landmarks)
        fps = frames / max(time.time() - t0, 1e-6)
    cap.release()
    msg = f"camera {index}: {width}x{height}, {fps:.1f} FPS with MediaPipe, hands in {hands}/{frames} frames"
    if fps >= 15:
        return "OK", msg
    return "WARN", msg + " - below 15 FPS: fallback F5 (model_complexity=0, 640x480, close other apps)"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, help="also test this webcam index, e.g. 0")
    p.add_argument("--seconds", type=float, default=5.0, help="length of the camera test")
    args = p.parse_args(argv)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

    results = []

    def report(status, msg):
        results.append(status)
        print(f"[{status:^4}] {msg}", flush=True)

    print(f"SignVaani install check on {platform.platform()}")
    report(*check_python())
    package_results, loaded = check_packages()
    for r in package_results:
        report(*r)
    if {"mediapipe", "numpy"} <= loaded.keys():
        report(*check_holistic(loaded["mediapipe"], loaded["numpy"]))
    if "pyttsx3" in loaded:
        report(*check_speech(loaded["pyttsx3"]))
    if args.camera is not None:
        if {"cv2", "mediapipe"} <= loaded.keys():
            print(f"Camera test ({args.seconds:.0f} s): sit 1-2 m in front of the camera and raise both hands.")
            report(*check_camera(loaded["cv2"], loaded["mediapipe"], args.camera, args.seconds))
        else:
            report("FAIL", "the camera test needs cv2 and mediapipe")

    failed = results.count("FAIL")
    if failed:
        print(f"NOT OK: fix the {failed} [FAIL] line(s). If MediaPipe is the problem, use fallback F1 "
              '(Python 3.10, mediapipe==0.10.14, "numpy<2").')
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
