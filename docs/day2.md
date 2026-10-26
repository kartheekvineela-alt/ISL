# Day 2: full dataset and quality control

Blueprint Section 7, Day 2. By tonight signers s02-s05 are recorded, every bad clip is out, the
dataset stats table and the datasheet are committed, and the sentence tests pass for S1-S10. One laptop
also starts fetching the real INCLUDE videos, so Day 3 can check the pipeline on real signing.

## Ready in the repository

| Blueprint task | What is in the repo |
|---|---|
| Record signers s02-s05 | `signvaani.record` (Day 1) and `docs/recording_protocol.md`. Run `check_data` after every signer. |
| Dataset-stats script | `python -m signvaani.dataset_stats`: clips per sign and signer, mean clip length, share of frames with a hand. It prints Table R1 and the per-sign table as Markdown; `--out` saves them for the report. |
| Sentence rules and unit tests for S1-S10 | `tests/test_sentence.py` reads `configs/sentences.json` and checks every demo sentence in English and Hindi, also when the spotter repeats a sign or lets IDLE through. The current `rules.json` passes all of them. |
| Quality control | `python -m signvaani.qc`: REJECT for sign takes with a hand in under 60% of frames, REVIEW for takes whose movement is far from the same signer's other takes (possible wrong sign), SHORT for a signer with under 8 good takes of a sign, with the record command that fixes it. `--apply` moves rejected clips to `data/rejected/` and logs them; nothing is deleted. |
| Use-case and sequence diagrams | `docs/diagrams/usecase.drawio` and `sequence.drawio` with SVG previews, now Figures 3 and 4 of `docs/SRS.md`. |
| DATASHEET.md | `data/DATASHEET.md`: Part A for your recordings (every number from `dataset_stats`), Part B for INCLUDE. |
| Real-data pipeline check (prepares Day 3) | `signvaani.include_dataset` now unpacks only the chosen subset's videos, so INCLUDE-50 keeps 958 of the 4,292 videos on disk. |

## Step by step

### 09:00-13:00, A and D: record s02-s05

Consent form first, then one signer at a time (about 40 minutes each with breaks):

```bash
python -m signvaani.record --signer s02
python -m signvaani.check_data --signers s02          # PASS: 16 signs x 10 takes for s02
```

Vary the conditions between signers and write them in the conditions table of `data/DATASHEET.md`:
clothing colour, background, room, lighting, distance (1-2 m). A left-handed signer helps. Never put a
name in a file name, commit or chat.

### 09:00-11:00, B: dataset stats

```bash
python -m signvaani.dataset_stats                     # all signers recorded so far
```

Run it again after quality control (below) for the numbers that go into the report.

### 09:00, B or C: start the INCLUDE download on one laptop

The download needs access to `zenodo.org`; `--dry-run` shows how much it fetches. Start it in the morning,
let it run alongside the recording, and convert in the evening or overnight:

```bash
python -m signvaani.include_dataset download --subset include50 --dry-run   # archives and their size
python -m signvaani.include_dataset download --subset include50             # one archive at a time
python -m signvaani.include_dataset convert --subset include50 --workers 4  # MediaPipe on every frame
python -m signvaani.dataset_stats --data data/include_keypoints/include50 --out docs/results/include50_stats.md
```

INCLUDE-50 videos are spread over every category archive, so every archive is downloaded. Each archive
is unpacked for the subset's videos only and then deleted, so the disk holds one archive plus the 958
INCLUDE-50 videos. If the laptop cannot reach Zenodo, download the zip files in a browser, unzip them
anywhere under `data/raw_videos/include/`, and run `convert` (see `data/include/README.md`).

### 11:00-13:00, C: sentence rules and tests

```bash
python -m pytest tests/test_sentence.py -v           # S1 ... S10b, English and Hindi, plus live noise
```

All demo sentences already pass. If a native speaker corrects a Hindi sentence in
`docs/sign_verification.md`, change it in both `configs/sentences.json` and the matching pattern in
`signvaani/rules.json`, then run the tests again.

### 14:00-16:00, A and B: quality control

```bash
python -m signvaani.qc --plot              # report: REJECT, REVIEW and SHORT lines; plots in runs/qc/
python -m signvaani.qc --apply             # move the REJECT clips to data/rejected/
```

1. **REJECT** (hands in under 60% of frames): `--apply` moves them out. Paste the printed log rows into
   the rejected-takes log of `docs/recording_protocol.md`.
2. **REVIEW** (movement unlike the signer's other takes): open `runs/qc/review.png`. The red path is the
   flagged take, the grey ones are the same signer's other takes. If it is a wrong sign, two signs or a
   false start, remove it:
   `python -m signvaani.qc --reject data/keypoints/HEAD/s03_7.npy --reason "wrong sign"`.
   If it is only a little sloppy, keep it: variation helps the model.
3. **SHORT** (under 8 good takes): run the printed command, e.g.
   `python -m signvaani.record --signer s03 --signs PAIN WATER`. It records only the missing takes.
4. Run `python -m signvaani.qc` again until it prints no REJECT and no SHORT lines (exit code 0).

### 14:00-16:00, D: diagrams and report

The use-case and sequence diagrams are in `docs/diagrams/`. Open the `.drawio` files in draw.io, adjust
if needed, and export PNGs for the report. Start report Chapter 1 (introduction) and Chapter 2
(literature survey, from `docs/ISL_Literature_Survey_with_Links.pdf`), in your own words.

### 16:00-18:00, A: datasheet

```bash
python -m signvaani.dataset_stats --out docs/results/dataset_stats.md
```

Copy the Table R1 values into Part A of `data/DATASHEET.md`, fill in the conditions table, sign source,
consent and licence rows. The licence row comes from the consent register: public release only if every
signer ticked Yes.

### 18:00, all: commit

```bash
python -m pytest -q
git add docs data/DATASHEET.md configs signvaani/rules.json
git commit -m "Day 2: s02-s05 recorded, QC done, dataset stats and datasheet"
git push
```

Never commit `data/keypoints/`, `data/rejected/`, the INCLUDE videos or keypoints, or the signed consent
forms.

## Exit criteria

- [ ] About 800 clips: 5 signers x 16 classes x at least 8 good takes. `python -m signvaani.qc` exits
      with code 0 (no SHORT line), and Table R1 shows the totals.
- [ ] `docs/results/dataset_stats.md` and `data/DATASHEET.md` committed.
- [x] Sentence unit tests pass for S1-S10 (`tests/test_sentence.py`).

If a criterion fails, apply the fallback before Day 3 (blueprint Section 10):

- **F3, too few clean clips:** record the missing takes with the command QC prints; if a signer is not
  available, add a sixth signer for the weak classes.
- **F1, MediaPipe problems on the recording laptop:** record on a teammate's laptop; the folder layout
  is the same everywhere, so clips can be copied together into `data/keypoints/`.
