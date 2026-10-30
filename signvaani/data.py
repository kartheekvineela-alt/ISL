"""Dataset indexing, signer-based splits and the PyTorch Dataset.

Expected layout (one raw clip per file, shape (T, 63, 3), NaN = not detected):

    data/keypoints/<SIGN>/<signer>_<take>.npy      e.g. data/keypoints/PAIN/s03_2.npy
"""
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .augment import augment as augment_clip
from .landmarks import ALL_GROUPS, mask_features, preprocess
from .spotter import trim_to_sign

VOCAB = Path(__file__).resolve().parents[1] / "configs" / "vocabulary.json"
# Signers are anonymous IDs (s01, s02, ...), never names.
SIGNER_ID = re.compile(r"s\d+")
CLIP_NAME = re.compile(rf"({SIGNER_ID.pattern})_(\d+)")


def vocabulary(path=VOCAB):
    """The frozen sign list of configs/vocabulary.json, in recording order."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["signs"]


def recorded_takes(sign_dir, signer):
    """Take numbers already saved for one signer in one sign folder."""
    names = (CLIP_NAME.fullmatch(p.stem) for p in Path(sign_dir).glob(f"{signer}_*.npy"))
    return sorted(int(m.group(2)) for m in names if m and m.group(1) == signer)


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
        def __init__(self, clips, labels, seq_len=32, train=False, seed=0, trim=False, augment=True,
                     parts=ALL_GROUPS, velocity=True):
            """trim: use only the part of each clip the live spotter would cut out (spotter.trim_to_sign).
            augment, parts, velocity: the ablation switches (augmentation, landmark groups, velocities)."""
            self.clips = clips
            self.label_to_id = {l: i for i, l in enumerate(labels)}
            self.seq_len = seq_len
            self.train = train
            self.trim = trim
            self.augment, self.parts, self.velocity = augment, tuple(parts), velocity
            self.rng = np.random.default_rng(seed)
            self._trimmed = {}  # the cut is the same every epoch, so it is done once per clip

        def __len__(self):
            return len(self.clips)

        def __getitem__(self, i):
            clip = self.clips[i]
            if not self.trim:
                raw = np.load(clip.path)
            elif i in self._trimmed:
                raw = self._trimmed[i]
            else:
                raw = self._trimmed[i] = trim_to_sign(np.load(clip.path))
            if self.train and self.augment:
                raw = augment_clip(raw, self.rng)
            x = mask_features(preprocess(raw, self.seq_len), self.parts, self.velocity)
            return torch.from_numpy(x), self.label_to_id[clip.sign]
except ImportError:  # torch is only needed for training
    pass
