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


def test_quiet_since_marks_when_the_hands_stopped():
    sp = SignSpotter(FakeClassifier([0.05, 0.95]))
    xs = [0.5] * 10 + list(0.5 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, 25))) + [0.5] * 20
    hits = [(t, r) for t, r in enumerate(sp.update(frame(x), t=t) for t, x in enumerate(xs)) if r]
    (t_hit, _), = hits
    assert sp.quiet_since == t_hit - sp.quiet_frames + 1  # the first of the quiet frames that closed the window


def test_no_latency_when_the_window_closes_while_still_moving():
    clf = FakeClassifier([0.05, 0.95])
    sp = SignSpotter(clf, max_frames=20)
    xs = [0.5] * 10 + list(0.5 + 0.1 * np.sin(np.linspace(0, 12 * np.pi, 60)))
    hits = [r for r in (sp.update(frame(x), t=t) for t, x in enumerate(xs)) if r]
    assert hits and sp.quiet_since is None  # cut at max_frames, not by the hands stopping
