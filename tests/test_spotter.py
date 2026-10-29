import numpy as np

from signvaani.landmarks import L_SHOULDER, L_WRIST, LEFT_HAND, N_POINTS, R_SHOULDER, RIGHT_HAND
from signvaani.spotter import SignSpotter, trim_to_sign


class FakeClassifier:
    labels = ["IDLE", "HELP"]
    seq_len = 32

    def __init__(self, probs):
        self.probs = np.array(probs)
        self.calls = 0

    def __call__(self, x):
        assert x.shape == (32, 378)
        self.calls += 1
        return self.probs


def frame(wrist_x):
    f = np.full((N_POINTS, 3), 0.5, np.float32)
    f[L_SHOULDER, :2], f[R_SHOULDER, :2] = (0.6, 0.6), (0.4, 0.6)
    f[RIGHT_HAND] = np.nan
    f[L_WRIST, 0] = wrist_x
    f[LEFT_HAND, 0] = wrist_x
    return f


def stream(spotter, n_still=10, n_move=25, n_after=20):
    xs = [0.5] * n_still + list(0.5 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n_move))) + [0.5] * n_after
    return [r for r in (spotter.update(frame(x)) for x in xs) if r]


def test_detects_one_sign_from_motion():
    clf = FakeClassifier([0.05, 0.95])
    hits = stream(SignSpotter(clf))
    assert [h[0] for h in hits] == ["HELP"] and clf.calls == 1


def test_idle_and_low_confidence_rejected():
    assert stream(SignSpotter(FakeClassifier([0.9, 0.1]))) == []
    assert stream(SignSpotter(FakeClassifier([0.4, 0.6]))) == []


def test_no_motion_no_classification():
    clf = FakeClassifier([0.05, 0.95])
    sp = SignSpotter(clf)
    for _ in range(100):
        assert sp.update(frame(0.5)) is None
    assert clf.calls == 0


def test_trim_to_sign_keeps_the_window_the_live_spotter_classifies():
    xs = [0.5] * 15 + list(0.5 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, 25))) + [0.5] * 30
    take = np.stack([frame(x) for x in xs])
    cut = trim_to_sign(take)
    moving = lambda clip: int(np.sum(clip[:, L_WRIST, 0] != 0.5))
    first, last = np.flatnonzero(take[:, L_WRIST, 0] != 0.5)[[0, -1]]
    assert moving(cut) == moving(take)  # the whole movement is kept
    assert len(cut) <= (last - first + 1) + 5 + 10  # plus the pre-roll and a short quiet tail
    assert len(cut) < len(take) - 20  # most of the rest before and after is gone

    still = np.stack([frame(0.5)] * 40)
    assert np.array_equal(trim_to_sign(still), still, equal_nan=True)  # nothing detected: whole take kept

    unfinished = np.stack([frame(x) for x in [0.5] * 10 + list(0.5 + 0.1 * np.sin(np.linspace(0, 6 * np.pi, 30)))])
    assert moving(trim_to_sign(unfinished)) == moving(unfinished)  # still moving at the end of the take
