"""Real-time sign spotting: decide when a sign starts/ends in a continuous stream.

A window opens when wrist motion energy rises above `on_threshold` and closes
after it stays below `off_threshold` for `quiet_frames` frames. The window is
classified; a sign is accepted only if it is confident and not the IDLE class.
"""
from collections import deque

import numpy as np

from .landmarks import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, hands_visible, preprocess


class SignSpotter:
    def __init__(self, classifier, on_threshold=0.02, off_threshold=0.01, quiet_frames=8,
                 min_frames=10, max_frames=90, pre_roll=5, confidence=0.8, cooldown=12,
                 idle_label="IDLE", smoothing=0.5):
        self.classifier = classifier
        self.on, self.off = on_threshold, off_threshold
        self.quiet_frames, self.min_frames, self.max_frames = quiet_frames, min_frames, max_frames
        self.confidence, self.cooldown, self.idle_label = confidence, cooldown, idle_label
        self.smoothing = smoothing
        self.history = deque(maxlen=pre_roll)
        self.buffer = []
        self.active = False
        self.energy = 0.0
        self.last_probs = None
        self._prev = None
        self._quiet = 0
        self._cool = 0

    def _motion(self, frame):
        left, right = hands_visible(frame)
        if not (left or right):
            self._prev = None
            return 0.0
        width = np.linalg.norm(frame[L_SHOULDER, :2] - frame[R_SHOULDER, :2])
        if not np.isfinite(width) or width < 1e-6:
            width = 1.0
        wrists = frame[[L_WRIST, R_WRIST], :2] / width
        if self._prev is None:
            self._prev = wrists
            return 0.0
        speed = np.linalg.norm(wrists - self._prev, axis=1)
        self._prev = wrists
        speed = speed[np.isfinite(speed)]
        return float(speed.max()) if speed.size else 0.0

    def update(self, frame):
        """Feed one (N_POINTS, 3) frame. Returns (label, prob) when a sign is accepted."""
        clip = self.segment(frame)
        if clip is None:
            return None
        probs = self.classifier(preprocess(clip, self.classifier.seq_len))
        self.last_probs = probs
        best = int(np.argmax(probs))
        label = self.classifier.labels[best]
        if label == self.idle_label or probs[best] < self.confidence:
            return None
        self._cool = self.cooldown
        return label, float(probs[best])

    def segment(self, frame):
        """Feed one frame. Returns the raw window (T, N_POINTS, 3) when a sign window closes, else None."""
        m = self._motion(frame)
        self.energy = self.smoothing * self.energy + (1 - self.smoothing) * m
        if self._cool > 0:
            self._cool -= 1

        if not self.active:
            self.history.append(frame)
            if self.energy > self.on and self._cool == 0:
                self.active, self.buffer, self._quiet = True, list(self.history), 0
            return None

        self.buffer.append(frame)
        self._quiet = self._quiet + 1 if self.energy < self.off else 0
        if self._quiet < self.quiet_frames and len(self.buffer) < self.max_frames:
            return None

        self.active = False
        clip = np.stack(self.buffer[:len(self.buffer) - self._quiet // 2])
        self.buffer, self.history = [], deque(maxlen=self.history.maxlen)
        return clip if len(clip) >= self.min_frames else None


def sign_windows(seq, **params):
    """The raw windows the live spotter cuts from a recorded clip, plus one still open when the clip ends."""
    spotter = SignSpotter(None, **params)
    windows = [w for w in (spotter.segment(f) for f in seq) if w is not None]
    if spotter.active and len(spotter.buffer) >= spotter.min_frames:
        windows.append(np.stack(spotter.buffer))
    return windows


def trim_to_sign(seq, **params):
    """The part of a recorded clip that the live app would classify: the spotter's longest window, or the
    whole clip when the spotter finds none (e.g. an IDLE take with the hands at rest). Training on this
    instead of the whole take keeps training clips like the live ones, which start just before the hands
    move and end soon after they stop."""
    windows = sign_windows(seq, **params)
    return max(windows, key=len) if windows else np.asarray(seq)
