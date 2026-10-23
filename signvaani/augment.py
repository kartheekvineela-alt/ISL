"""Keypoint-level data augmentation applied to raw clips before preprocessing."""
import numpy as np

from .landmarks import L_SHOULDER, R_SHOULDER, LEFT_HAND, RIGHT_HAND, resample


def augment(seq, rng):
    """Randomly perturb a raw clip of shape (T, N_POINTS, 3) (NaN = missing)."""
    seq = np.array(seq, np.float32, copy=True)

    # Temporal: random start/end crop and speed change.
    t_len = len(seq)
    if t_len > 8:
        start = rng.integers(0, max(1, t_len // 10) + 1)
        end = t_len - rng.integers(0, max(1, t_len // 10) + 1)
        seq = seq[start:end]
    new_len = max(4, int(round(len(seq) * rng.uniform(0.8, 1.2))))
    seq = _resample_keep_nan(seq, new_len)

    # Spatial: rotation in the image plane, scale and shift around the shoulders.
    centre = np.nanmean((seq[:, L_SHOULDER, :2] + seq[:, R_SHOULDER, :2]) / 2, axis=0)
    if not np.all(np.isfinite(centre)):
        centre = np.array([0.5, 0.5], np.float32)
    angle = np.deg2rad(rng.uniform(-15, 15))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    scale = rng.uniform(0.9, 1.1)
    shift = rng.uniform(-0.05, 0.05, size=2)
    xy = (seq[..., :2] - centre) @ rot.T * scale + centre + shift
    seq[..., :2] = xy
    seq[..., 2] *= scale

    # Coordinate noise.
    seq += rng.normal(0, 0.004, size=seq.shape).astype(np.float32)

    # Hand drop-out: simulate tracking loss for a few frames.
    if rng.random() < 0.3:
        hand = LEFT_HAND if rng.random() < 0.5 else RIGHT_HAND
        n = len(seq)
        length = rng.integers(1, max(2, n // 5))
        s = rng.integers(0, max(1, n - length))
        seq[s:s + length, hand] = np.nan
    return seq


def _resample_keep_nan(seq, length):
    """Resample while keeping NaN where the nearest source frame was missing."""
    out = resample(np.nan_to_num(seq, nan=0.0), length)
    src = np.rint(np.linspace(0, len(seq) - 1, length)).astype(int)
    out[np.isnan(seq[src])] = np.nan
    return out
