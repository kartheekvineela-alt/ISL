import numpy as np

from signvaani.augment import augment
from signvaani.landmarks import (FEATURE_DIM, L_SHOULDER, LEFT_HAND, N_POINTS, R_SHOULDER,
                                 interpolate_missing, normalize, preprocess, resample)


def clip(t_len=40, seed=0):
    rng = np.random.default_rng(seed)
    seq = rng.uniform(0.2, 0.8, size=(t_len, N_POINTS, 3)).astype(np.float32)
    seq[:, L_SHOULDER, :2], seq[:, R_SHOULDER, :2] = (0.6, 0.6), (0.4, 0.6)
    return seq


def test_preprocess_shape_and_no_nan():
    raw = clip()
    raw[5:9, LEFT_HAND] = np.nan
    x = preprocess(raw, 32)
    assert x.shape == (32, FEATURE_DIM)
    assert np.isfinite(x).all()


def test_short_gap_interpolated_long_gap_zeroed():
    raw = clip(30)
    raw[10:12, LEFT_HAND] = np.nan          # short gap -> interpolated
    raw[15:28, LEFT_HAND.start] = np.nan    # long gap on one point -> zero
    out = interpolate_missing(raw, max_gap=6)
    assert np.isfinite(out).all()
    assert np.all(out[10:12, LEFT_HAND] != 0)
    assert np.all(out[15:28, LEFT_HAND.start] == 0)


def test_normalize_invariant_to_position_and_scale():
    raw = clip()
    moved = raw.copy()
    moved[..., :2] = raw[..., :2] * 1.7 + 0.2
    moved[..., 2] = raw[..., 2] * 1.7
    np.testing.assert_allclose(normalize(raw), normalize(moved), atol=1e-4)


def test_absent_points_stay_zero():
    raw = clip()
    raw[:, LEFT_HAND] = 0.0
    x = normalize(raw).reshape(len(raw), N_POINTS, 6)
    assert np.all(x[:, LEFT_HAND] == 0)


def test_resample_endpoints():
    seq = np.arange(10, dtype=np.float32)[:, None]
    out = resample(seq, 4)
    assert out[0, 0] == 0 and out[-1, 0] == 9


def test_augment_keeps_shape_and_preprocesses():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = augment(clip(), rng)
        assert a.shape[1:] == (N_POINTS, 3)
        assert np.isfinite(preprocess(a)).all()
