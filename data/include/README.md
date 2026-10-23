# INCLUDE: the public ISL dataset selected for SignVaani

[INCLUDE](https://zenodo.org/record/4010759) (AI4Bharat, ACM Multimedia 2020) is an **isolated Indian Sign
Language** dataset: 263 word signs in 15 categories, 4,292 videos, recorded in Chennai.
Each video holds one sign, which matches SignVaani's one-clip / one-label classifier. That is why it is the
first dataset for this repository (see `docs/SignVaani_ISL_Dataset_Guide.pdf`, sections 2 and 3).

| Option | Type | Fits SignVaani now? | Use |
|---|---|---|---|
| **INCLUDE** | isolated signs, 263 classes | **yes** | pipeline check, baseline, pre-training |
| iSign / CISLR | continuous translation, word-level corpus | no (sentence-level) | later continuous-translation work |
| ISL-CSLTR | sentence-level | no | exploratory only |
| Own hospital recordings | isolated signs, `configs/vocabulary.json` | yes | **required** for the final, unseen-signer result |

## What is in this folder

| Path | Content |
|---|---|
| `splits/include_{train,val,test}.txt` | Official INCLUDE split: 3,127 / 348 / 817 videos |
| `splits/include50_{train,val,test}.txt` | Official INCLUDE-50 split: 689 / 77 / 192 videos (a different split from the full one) |
| `splits/LICENSE` | MIT licence of [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE), where the split lists come from |
| `classes.csv` | All 263 signs: category, folder, SignVaani label, clips per split, INCLUDE-50 membership, hospital mapping |

The videos and keypoints are **not** committed. The full release is too large for git, and the dataset's
licence is on the Zenodo record, so check it before redistributing anything. Downloads go to
`data/raw_videos/include/` and converted clips to `data/include_keypoints/`. Both are git-ignored.

## Subsets

| Subset | Signs | Videos | Use |
|---|---|---|---|
| `include50` | 50 | 958 | Quick baseline on the official benchmark subset |
| `include` | 263 | 4,292 | Full benchmark and pre-training |
| `hospital` | 7 | 115 | INCLUDE clips of signs already in `configs/vocabulary.json`, saved under the vocabulary label: I→`ME`, `YOU`, `DOCTOR`, `FAMILY`, `MEDICINE`, `TODAY`, `YESTERDAY` |

Other INCLUDE signs that fit the hospital domain but are not in the vocabulary yet: `PATIENT`, `HOSPITAL`,
`SICK`, `HEALTHY`, `HOT`, `COLD`, `HELLO`, `THANK-YOU`, `BATHROOM`, `BED`, `MOTHER`, `FATHER`.

## Get and convert the data

```bash
python -m signvaani.include_dataset download --subset include50 --dry-run   # list archives and size
python -m signvaani.include_dataset download --subset include50             # Zenodo -> data/raw_videos/include
python -m signvaani.include_dataset convert  --subset include50 --workers 4
```

If the script cannot reach Zenodo, download the zip files from the record page in a browser, unzip them
anywhere under `data/raw_videos/include/`, and run `convert`. It finds each video by its sign folder and
file name, however the archives were nested. INCLUDE-50 videos are spread across all 15 category archives,
so `include50` needs the whole download. `hospital` only needs 5 categories.

`convert` runs MediaPipe Holistic on every frame. It saves `(T, 63, 3)` float32 arrays with NaN for missed
points, the same format that `signvaani.record` writes:

```
data/include_keypoints/<subset>/<SIGN>/<split>_<video>.npy      e.g. include50/HELLO/train_MVI_0089.npy
```

Frames are mirrored and padded to 4:3 first, so they match the mirrored 640×480 webcam frames of
`record.py` and `app.py` (`--no-flip`, `--aspect 0` turn this off). Missing videos, unreadable videos and
clips with hands in fewer than half the frames are listed in `conversion_log.txt` next to the clips.
Check the weak ones by eye before training.

## Train and evaluate

INCLUDE publishes no per-video signer IDs, so the split name stands where the signer ID usually goes.
This reproduces the official split:

```bash
python -m signvaani.train --data data/include_keypoints/include50 --test-signers test --val-signers val \
    --out models/include50.pt
python -m signvaani.evaluate --ckpt models/include50.pt --data data/include_keypoints/include50 --signers test
```

To use the `hospital` clips as extra training data, copy `data/include_keypoints/hospital/*` into
`data/keypoints/` and hold out your own signers with `--test-signers` / `--val-signers`. Do not use `--loso`
on INCLUDE folders, because `train`, `val` and `test` would each become a fake "signer" fold.

## Limitations

- The official split is **not signer-independent**: the same people can appear in train and test. Report
  it as "INCLUDE official split", never as the unseen-signer result.
- All recordings come from Chennai, so other regional ISL variants may differ.
- Only 7 of the 41 vocabulary signs exist in INCLUDE, and there is no `IDLE` class. The final hospital
  model still needs your own consented recordings (guide, section 2.4).

## Citation

```bibtex
@inproceedings{10.1145/3394171.3413528,
  author    = {Sridhar, Advaith and Ganesan, Rohith Gandhi and Kumar, Pratyush and Khapra, Mitesh},
  title     = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
  year      = {2020},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3394171.3413528},
  booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
  series    = {MM '20}
}
```
