# SignVaani ISL Keypoint Dataset — Datasheet

Fill this in on Day 2 (see the 1-week blueprint, Section 7, and `docs/day2.md`). Two datasets are used:

- **Part A, own hospital recordings**: the data behind the final, unseen-signer result.
- **Part B, INCLUDE**: public ISL videos, used to check the pipeline on real signing and as a baseline.

Every number comes from a command, never from an estimate. Run the commands after quality control and
copy the values from their output.

## Part A: own hospital recordings (`data/keypoints/`)

```bash
python -m signvaani.qc --apply                                          # quality control first
python -m signvaani.dataset_stats --out docs/results/dataset_stats.md   # Table R1 and per-sign counts
```

| Field | Value |
|---|---|
| Domain | Hospital communication |
| Classes | 15 signs + IDLE (see `configs/vocabulary.json`) |
| Signers | [XX] (IDs s01–s0X; no names stored) |
| Takes per class per signer | [XX] after QC (Table R1, "Takes / class / signer") |
| Total clips | [XX] (Table R1) |
| Mean clip length | [XX] frames (Table R1) |
| Frames with hands | [XX]% of frames, IDLE excluded (Table R1) |
| Removed in QC | [XX] clips: [XX] hands below 60%, [XX] wrong sign (`data/rejected/qc_log.csv`, `docs/recording_protocol.md`) |
| Format | `data/keypoints/<SIGN>/<signer>_<take>.npy`, float32 array (T, 63, 3), NaN = not detected |
| Points | 9 pose + 12 face + 21 left hand + 21 right hand (MediaPipe Holistic) |
| Frame rate | ~30 FPS, ~2.5 s per take |
| Sign source | ISLRTC ISL dictionary; verified by [name / organisation] (see `docs/sign_verification.md`) |
| Recording conditions | See the table below |
| Consent | Written consent from every signer, stored offline by the team (register: `docs/recording_protocol.md`) |
| Video stored? | No — only landmark coordinates |
| Licence | [e.g. CC BY 4.0, only if every signer agreed to public release; otherwise "not released"] |
| Known limitations | [regional variants, small number of signers, one camera, ...] |

Recording conditions, one row per signer (no names). Blueprint Day 2 asks for varied clothing, background,
lighting and distance (1-2 m) between signers.

| Signer | Date | Room / background | Lighting | Camera, distance | Clothing | Dominant hand |
|---|---|---|---|---|---|---|
| s01 | | | | | | |
| s02 | | | | | | |
| s03 | | | | | | |
| s04 | | | | | | |
| s05 | | | | | | |

## Part B: INCLUDE (`data/include_keypoints/`)

```bash
python -m signvaani.dataset_stats --data data/include_keypoints/include50 --out docs/results/include50_stats.md
```

| Field | Value |
|---|---|
| Source | INCLUDE (AI4Bharat, ACM Multimedia 2020), [Zenodo record 4010759](https://zenodo.org/record/4010759); details in `data/include/README.md` |
| Classes | 263 word signs in 15 categories; the INCLUDE-50 subset has 50 |
| Videos | 4,292 in the official split lists (3,127 train / 348 val / 817 test); INCLUDE-50: 958 (689 / 77 / 192) |
| Signers | No per-video signer IDs are published; the official split is **not** signer-independent |
| Region | Recorded in Chennai |
| Overlap with the frozen vocabulary | 3 signs: ME (INCLUDE "I"), DOCTOR, MEDICINE, 49 videos (`--subset hospital`); no IDLE class |
| Conversion | `signvaani.include_dataset convert`: frames mirrored and padded to 4:3, MediaPipe Holistic, same (T, 63, 3) NaN format as Part A |
| Clips after conversion | [XX] of 958 (`conversion_log.txt` lists missing, unreadable and weak clips) |
| Mean clip length | [XX] frames (Table R1 of `include50_stats.md`) |
| Frames with hands | [XX]% (Table R1 of `include50_stats.md`) |
| Licence | As stated on the Zenodo record. Videos and keypoints are not redistributed in this repository |
| Use in SignVaani | Day 3 pipeline check on real signing and public baseline (official split). Never reported as the unseen-signer result |
