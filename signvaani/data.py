"""Dataset indexing, signer-based splits and the PyTorch Dataset.

Expected layout (one raw clip per file, shape (T, 63, 3), NaN = not detected):

    data/keypoints/<SIGN>/<signer>_<take>.npy      e.g. data/keypoints/PAIN/s03_2.npy
"""
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .augment import augment
from .landmarks import preprocess


@dataclass(frozen=True)
class Clip:
    path: Path
    sign: str
    signer: str


def index_clips(root):
    clips = []
    for sign_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(sign_dir.glob("*.npy")):
            clips.append(Clip(f, sign_dir.name, f.stem.split("_")[0]))
    if not clips:
        raise FileNotFoundError(f"No .npy clips found under {root}")
    return clips


def signers(clips):
    return sorted({c.signer for c in clips})


def split_by_signer(clips, test_signers, val_signers=None, val_fraction=0.15, seed=0):
    """Signer-independent test split. Validation uses held-out signers if given,
    otherwise a random fraction of the remaining clips."""
    test_signers, val_signers = set(test_signers), set(val_signers or [])
    test = [c for c in clips if c.signer in test_signers]
    rest = [c for c in clips if c.signer not in test_signers]
    if val_signers:
        val = [c for c in rest if c.signer in val_signers]
        train = [c for c in rest if c.signer not in val_signers]
    else:
        rest = rest[:]
        random.Random(seed).shuffle(rest)
        n_val = max(1, int(len(rest) * val_fraction))
        val, train = rest[:n_val], rest[n_val:]
    return train, val, test


def random_split(clips, val_fraction=0.15, test_fraction=0.15, seed=0):
    """Signer-dependent split (same people in train and test). Report it only
    alongside the signer-independent result."""
    clips = clips[:]
    random.Random(seed).shuffle(clips)
    n_test = int(len(clips) * test_fraction)
    n_val = int(len(clips) * val_fraction)
    return clips[n_test + n_val:], clips[n_test:n_test + n_val], clips[:n_test]


try:
    import torch
    from torch.utils.data import Dataset

    class SignDataset(Dataset):
        def __init__(self, clips, labels, seq_len=32, train=False, seed=0):
            self.clips = clips
            self.label_to_id = {l: i for i, l in enumerate(labels)}
            self.seq_len = seq_len
            self.train = train
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.clips)

        def __getitem__(self, i):
            clip = self.clips[i]
            raw = np.load(clip.path)
            if self.train:
                raw = augment(raw, self.rng)
            x = preprocess(raw, self.seq_len)
            return torch.from_numpy(x), self.label_to_id[clip.sign]
except ImportError:  # torch is only needed for training
    pass
