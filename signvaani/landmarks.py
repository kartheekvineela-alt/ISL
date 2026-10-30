"""Landmark selection, gap filling, resampling and normalisation.

A raw clip is stored as an array of shape (T, N_POINTS, 3) holding MediaPipe
(x, y, z) coordinates, with NaN wherever a body part was not detected.
`preprocess` turns it into the fixed-size feature matrix the models consume.
"""
import numpy as np

# MediaPipe pose indices: nose, left eye, right eye, shoulders, elbows, wrists.
POSE_IDX = [0, 2, 5, 11, 12, 13, 14, 15, 16]
# MediaPipe face-mesh indices: lip corners/centres, eyebrows, eye corners.
# These carry ISL non-manual markers (questions, negation, mouth shapes).
FACE_IDX = [61, 291, 0, 17, 13, 14, 70, 105, 300, 334, 33, 263]
N_HAND = 21

N_POSE, N_FACE = len(POSE_IDX), len(FACE_IDX)
N_POINTS = N_POSE + N_FACE + 2 * N_HAND  # 63
FEATURE_DIM = N_POINTS * 3 * 2  # positions + velocities = 378

# Positions inside the selected point array.
L_SHOULDER, R_SHOULDER = 3, 4
L_WRIST, R_WRIST = 7, 8
LEFT_HAND = slice(N_POSE + N_FACE, N_POSE + N_FACE + N_HAND)
RIGHT_HAND = slice(N_POSE + N_FACE + N_HAND, N_POINTS)


def _points(landmark_list, idx, n):
    if landmark_list is None:
        return np.full((len(idx) if idx is not None else n, 3), np.nan, np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in landmark_list.landmark], np.float32)
    return pts[idx] if idx is not None else pts


def frame_keypoints(results):
    """Convert one MediaPipe Holistic result into a (N_POINTS, 3) array."""
    return np.concatenate([
        _points(results.pose_landmarks, POSE_IDX, None),
        _points(results.face_landmarks, FACE_IDX, None),
        _points(results.left_hand_landmarks, None, N_HAND),
        _points(results.right_hand_landmarks, None, N_HAND),
    ])


def hands_visible(frame):
    """(left_visible, right_visible) for one (N_POINTS, 3) frame."""
    return (not np.isnan(frame[LEFT_HAND]).any(), not np.isnan(frame[RIGHT_HAND]).any())


def hands_share(seq):
    """Share of the frames of a raw clip in which at least one hand was detected."""
    return float(np.mean([any(hands_visible(f)) for f in seq])) if len(seq) else 0.0


def interpolate_missing(seq, max_gap=6):
    """Linearly fill short detection gaps; longer gaps become 0 (treated as 'absent')."""
    seq = np.array(seq, np.float32, copy=True)
    t_len = len(seq)
    t = np.arange(t_len)
    for k in range(seq.shape[1]):
        valid = ~np.isnan(seq[:, k, 0])
        if not valid.any():
            seq[:, k] = 0.0
            continue
        if valid.all():
            continue
        for c in range(3):
            seq[:, k, c] = np.interp(t, t[valid], seq[valid, k, c])
        # Zero out gaps that were too long to trust the interpolation.
        start = None
        for i in range(t_len + 1):
            missing = i < t_len and not valid[i]
            if missing and start is None:
                start = i
            elif not missing and start is not None:
                if i - start > max_gap:
                    seq[start:i, k] = 0.0
                start = None
    return seq


def resample(seq, length=32):
    """Linearly resample a sequence along time to a fixed number of frames."""
    seq = np.asarray(seq, np.float32)
    if len(seq) == 1:
        return np.repeat(seq, length, axis=0)
    pos = np.linspace(0, len(seq) - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (pos - lo).reshape(-1, *([1] * (seq.ndim - 1)))
    return seq[lo] * (1 - w) + seq[hi] * w


def normalize(seq):
    """Centre on the shoulders, scale by shoulder width, append velocities.

    seq: (T, N_POINTS, 3) with no NaN (absent points are exactly 0).
    returns: (T, FEATURE_DIM) float32.
    """
    seq = np.asarray(seq, np.float32)
    absent = np.all(seq == 0, axis=-1, keepdims=True)
    centre = (seq[:, L_SHOULDER] + seq[:, R_SHOULDER]) / 2
    width = np.linalg.norm(seq[:, L_SHOULDER, :2] - seq[:, R_SHOULDER, :2], axis=-1).mean()
    if not np.isfinite(width) or width < 1e-6:
        width = 1.0
    out = (seq - centre[:, None]) / width
    out = np.where(absent, 0.0, out)
    vel = np.diff(out, axis=0, prepend=out[:1])
    vel = np.where(absent, 0.0, vel)
    return np.concatenate([out, vel], axis=-1).reshape(len(seq), -1).astype(np.float32)


def preprocess(raw_seq, length=32):
    """Raw clip (T, N_POINTS, 3) with NaN -> model input (length, FEATURE_DIM)."""
    return normalize(resample(interpolate_missing(raw_seq), length))


# Landmark groups for the ablation study (blueprint Table R4).
GROUPS = {"pose": slice(0, N_POSE), "face": slice(N_POSE, N_POSE + N_FACE), "hands": slice(N_POSE + N_FACE, N_POINTS)}
ALL_GROUPS = ("hands", "pose", "face")


def mask_features(x, parts=ALL_GROUPS, velocity=True):
    """Zero the features of the landmark groups not in `parts`, and all velocities when velocity=False.
    The size stays (T, FEATURE_DIM), so every model and the live app work unchanged."""
    if set(parts) >= set(ALL_GROUPS) and velocity:
        return x
    x = np.array(x, np.float32, copy=True).reshape(len(x), N_POINTS, 6)  # per point: x, y, z, vx, vy, vz
    for name, points in GROUPS.items():
        if name not in parts:
            x[:, points] = 0.0
    if not velocity:
        x[:, :, 3:] = 0.0
    return x.reshape(len(x), FEATURE_DIM)
