import numpy as np

from signvaani.dataset_stats import clip_stats, main, summary
from signvaani.landmarks import LEFT_HAND, N_POINTS


def save(root, sign, name, frames, hands_frames):
    clip = np.full((frames, N_POINTS, 3), 0.5, np.float32)
    clip[hands_frames:, LEFT_HAND.start:] = np.nan  # both hands lost after hands_frames
    (root / sign).mkdir(parents=True, exist_ok=True)
    np.save(root / sign / name, clip)


def test_table_r1_and_per_sign_rows(tmp_path, capsys):
    for take in range(3):
        save(tmp_path, "PAIN", f"s01_{take}.npy", 40, 40)
        save(tmp_path, "IDLE", f"s01_{take}.npy", 60, 0)  # resting hands out of view
    for take in range(2):
        save(tmp_path, "PAIN", f"s02_{take}.npy", 80, 40)
        save(tmp_path, "IDLE", f"s02_{take}.npy", 60, 0)

    stats = clip_stats(tmp_path)
    r = summary(stats)
    assert (r["signers"], r["classes"], r["takes_min"], r["takes_max"], r["clips"]) == (2, 2, 2, 3, 10)
    assert r["mean_frames"] == (3 * 40 + 5 * 60 + 2 * 80) / 10
    assert r["hands"] == (3 * 40 + 2 * 40) / (3 * 40 + 2 * 80)  # frame-weighted, IDLE left out
    assert [s.signer for s in clip_stats(tmp_path, ["s02"])] == ["s02"] * 4

    out = tmp_path / "results" / "stats.md"
    main(["--data", str(tmp_path), "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert text == capsys.readouterr().out.replace(f"saved {out}\n", "")
    assert "| 2 | 2 | 2-3 | 10 | 58 frames | 71% |" in text
    assert "| IDLE | 3 | 2 | 5 | 60 | 0% |" in text
    assert text.index("| IDLE |") < text.index("| PAIN |")  # vocabulary order
    assert "| PAIN | 3 | 2 | 5 | 56 | 80% |" in text
    assert "| **Total** | 6 | 4 | 10 | | |" in text and "| s02 | 4 | 70 | 50% |" in text


def test_include_layout_uses_split_as_signer(tmp_path):
    save(tmp_path, "HELLO", "train_MVI_0001.npy", 50, 50)
    save(tmp_path, "HELLO", "test_MVI_0002.npy", 50, 50)
    save(tmp_path, "BIG-LARGE", "train_MVI_0003.npy", 50, 50)
    r = summary(clip_stats(tmp_path))
    assert (r["signers"], r["classes"], r["takes_min"], r["clips"]) == (2, 2, 0, 3)
