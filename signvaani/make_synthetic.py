"""Generate a small FAKE dataset to smoke-test the training pipeline.

    python -m signvaani.make_synthetic --out data/synthetic

Clips are random wrist trajectories, not real signs. Never report results on them.
"""
import argparse
from pathlib import Path

import numpy as np

from .landmarks import L_SHOULDER, L_WRIST, LEFT_HAND, N_POINTS, R_SHOULDER, R_WRIST, RIGHT_HAND


def fake_clip(rng, base, pattern, t_len):
    seq = np.repeat(base[None], t_len, axis=0)
    t = np.linspace(0, 2 * np.pi, t_len)
    dx, dy = pattern
    for wrist, hand, side in [(L_WRIST, LEFT_HAND, 1), (R_WRIST, RIGHT_HAND, -1)]:
        traj = np.stack([0.5 + side * 0.1 + 0.08 * np.sin(dx * t), 0.5 + 0.08 * np.cos(dy * t)], 1)
        seq[:, wrist, :2] = traj
        seq[:, hand, :2] = traj[:, None] + rng.normal(0, 0.01, size=(21, 2))
    return seq + rng.normal(0, 0.003, size=seq.shape).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--signs", type=int, default=5)
    p.add_argument("--signers", type=int, default=4)
    p.add_argument("--takes", type=int, default=6)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    # One fixed body layout per fake signer, like a real person standing still.
    bases = []
    for _ in range(args.signers):
        base = rng.uniform(0.3, 0.7, size=(N_POINTS, 3)).astype(np.float32)
        base[L_SHOULDER, :2], base[R_SHOULDER, :2] = (0.6, 0.6), (0.4, 0.6)
        bases.append(base)
    for k in range(args.signs):
        pattern = (k + 1, (k % 3) + 1)
        d = Path(args.out) / f"SIGN{k}"
        d.mkdir(parents=True, exist_ok=True)
        for s, base in enumerate(bases):
            for take in range(args.takes):
                np.save(d / f"s{s:02d}_{take}.npy", fake_clip(rng, base, pattern, int(rng.integers(50, 90))))
    print(f"wrote synthetic data to {args.out}")


if __name__ == "__main__":
    main()
