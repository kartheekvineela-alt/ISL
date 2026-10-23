import functools
import hashlib
import http.server
import json
import threading
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from signvaani.data import index_clips
from signvaani.include_dataset import HOSPITAL_MAP, convert, main, sign_label, videos

VOCAB = Path(__file__).resolve().parents[1] / "configs" / "vocabulary.json"


def test_sign_labels():
    assert sign_label("87. Doctor") == "DOCTOR"
    assert sign_label("46. you (plural)") == "YOU-PLURAL"
    assert sign_label("Second (Number)") == "SECOND-NUMBER"
    assert sign_label("Ex. Monsoon") == "MONSOON"
    assert sign_label("42. T-Shirt") == "T-SHIRT"


def test_official_splits():
    full, small = videos("include"), videos("include50")
    assert len(full) == 4292 and len({v.sign for v in full}) == 263
    assert len(small) == 958 and len({v.sign for v in small}) == 50
    assert {v.sign for v in small} <= {v.sign for v in full}
    for subset in (full, small):
        assert len({v.rel for v in subset}) == len(subset)  # no video in two splits
        assert len({(v.sign, v.clip_name) for v in subset}) == len(subset)  # no file name clash
    extra = next(v for v in full if v.rel == "Places/19. House/Extra/MVI_3439.MOV")
    assert extra.sign == "HOUSE"


def test_hospital_subset_uses_vocabulary_labels():
    vocab = set(json.loads(VOCAB.read_text())["signs"])
    assert set(HOSPITAL_MAP) <= {v.sign for v in videos("include")}
    assert {v.sign for v in videos("hospital")} == set(HOSPITAL_MAP.values()) <= vocab


class FakeHolistic:
    shapes = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def process(self, rgb):
        FakeHolistic.shapes.append(rgb.shape)
        return Namespace(pose_landmarks=None, face_landmarks=None,
                         left_hand_landmarks=None, right_hand_landmarks=None)


def test_convert_writes_signvaani_layout(tmp_path):
    cv2 = pytest.importorskip("cv2")
    splits, raw, out = tmp_path / "splits", tmp_path / "raw", tmp_path / "out"
    splits.mkdir()
    rels = {"train": "Jobs/87. Doctor/MVI_0001.avi", "val": "Pronouns/40. I/MVI_0002.avi",
            "test": "Jobs/87. Doctor/Extra/MVI_0003.avi"}
    for split, rel in rels.items():
        (splits / f"include_{split}.txt").write_text(rel + "\nAnimals/1. Dog/MVI_0009.avi\n")
        path = raw / "unzipped" / rel  # one folder deeper than the split lists say
        path.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 36))
        for _ in range(6):
            writer.write(np.zeros((36, 64, 3), np.uint8))
        writer.release()

    FakeHolistic.shapes.clear()
    convert(Namespace(subset="hospital", splits=str(splits), raw=str(raw), out=str(out), workers=1,
                      aspect=4 / 3, no_flip=False, overwrite=False), make_holistic=FakeHolistic)

    files = sorted(p.relative_to(out).as_posix() for p in out.glob("*/*.npy"))
    assert files == ["DOCTOR/test_MVI_0003.npy", "DOCTOR/train_MVI_0001.npy", "ME/val_MVI_0002.npy"]
    clip = np.load(out / "ME" / "val_MVI_0002.npy")
    assert clip.shape == (6, 63, 3) and clip.dtype == np.float32 and np.isnan(clip).all()
    assert set(FakeHolistic.shapes) == {(48, 64, 3)}  # 16:9 padded to 4:3
    assert {c.signer for c in index_clips(out)} == {"train", "val", "test"}
    assert "weak hand tracking, check by eye (3)" in (out / "conversion_log.txt").read_text()


def test_download_picks_archives_checks_md5_and_unzips(tmp_path):
    srv, raw = tmp_path / "srv", tmp_path / "raw"
    srv.mkdir()
    for category in ["Jobs", "Pronouns", "People", "Society", "Days_and_Time", "Animals"]:
        with zipfile.ZipFile(srv / f"{category}_1of1.zip", "w") as z:
            z.writestr(f"{category}/1. Sign/MVI_0001.MOV", category.encode())
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(srv)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    (srv / "record.json").write_text(json.dumps({"files": [
        {"key": z.name, "size": z.stat().st_size, "checksum": "md5:" + hashlib.md5(z.read_bytes()).hexdigest(),
         "links": {"self": f"{base}/{z.name}"}} for z in srv.glob("*.zip")]}))
    try:
        main(["download", "--subset", "hospital", "--record", f"{base}/record.json", "--raw", str(raw)])
    finally:
        server.shutdown()
    assert (raw / "Jobs" / "1. Sign" / "MVI_0001.MOV").read_bytes() == b"Jobs"
    assert not (raw / "Animals").exists()  # no hospital sign in that category
    assert not list((raw / "zips").glob("*.zip"))  # archives removed after unpacking
