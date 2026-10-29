# Day 3: baseline model and honest evaluation

Blueprint Section 7, Day 3. By tonight the BiGRU baseline has a leave-one-signer-out (LOSO) result with
a number per signer, the most confused sign pairs are listed with a reason for each, and any data
problem behind them is fixed.

The blueprint starts the day with a smoke test on synthetic data. This plan uses real signing instead:
the INCLUDE-50 videos converted on Day 2. A pipeline bug shows up there just as clearly, and the result is
a real, citable baseline. `make_synthetic` stays only for the automated tests.

## Ready in the repository

| Blueprint task | What is in the repo |
|---|---|
| Smoke-test the pipeline | INCLUDE-50, official split: `train --test-signers test --val-signers val` on `data/include_keypoints/include50`. |
| Train BiGRU: random split, then LOSO; save `runs/*.json` | Every `signvaani.train` run now writes `runs/<mode>_<model>_<time>.json` with top-1 and macro-F1 per fold, the parameter count and the prediction for every test clip. LOSO prints the per-signer table and says when the mean is below 60%. `--loso` refuses INCLUDE folders, whose "signers" are split names. |
| Live app with the first checkpoint, tune the spotter | `signvaani.app` takes `--on`, `--off` and `--quiet-frames` and shows them next to the live motion value. |
| Confusion matrix per LOSO fold, top-5 confused pairs, inspect the clips | `python -m signvaani.confusion runs/<run>.json`: one matrix per fold and one for all folds, Table R3, Table R6 with a reason hint per pair, the list of wrong clips, and their wrist paths next to the sign they were taken for. |
| Fix data causes | `signvaani.qc --reject` moves a wrong clip out and logs it; `signvaani.record` re-records only the missing takes. |

## Step by step

### 09:00-10:00, B: pipeline check on real ISL (INCLUDE-50)

Needs the converted clips from Day 2 (`data/include_keypoints/include50/`).

```bash
python -m signvaani.train --data data/include_keypoints/include50 --test-signers test --val-signers val \
    --out models/include50.pt
python -m signvaani.confusion runs/signer_gru_<time>.json --out runs/confusion/include50
```

- Chance level is 2% (50 signs). A top-1 near chance, or training accuracy that never rises, means a bug
  in the pipeline or the conversion: stop and fix it before training on your own data.
- Report this number only as "INCLUDE-50, official split". The same people can appear in its train and
  test parts, so it is not an unseen-signer result. For comparison, quote the baselines from the INCLUDE
  paper (citation in `data/include/README.md`) with their exact setting.
- Copy `runs/confusion/include50/report.md` to `docs/results/include50_baseline.md` for the report.

### 10:00-13:00, B: own data, random split then LOSO

```bash
python -m signvaani.train --data data/keypoints --random-split --out models/random.pt   # signer-dependent
python -m signvaani.train --data data/keypoints --loso                                 # 5 folds, the main result
```

- The LOSO run prints top-1 and macro-F1 for every test signer, then the mean ± standard deviation, and
  saves `runs/loso_gru_<time>.json`. Write the per-signer numbers in the daily log.
- Expect LOSO to be lower than the random split. The gap is a finding: different hand sizes, speeds
  and signing styles. Explain it in the report rather than hide it.

### 10:00-13:00, C: live app with the first checkpoint, tune the spotter

```bash
python -m signvaani.app --model models/random.pt
```

Watch the motion value at the top of the window:

- At rest it should stay below `off` (0.01). During every sign it should rise above `on` (0.02).
- Signs missed: lower `--on`, e.g. `--on 0.015`. False signs while sitting still (F4): raise `--on`.
- Two signs merged into one (F6): lower `--quiet-frames`, or drop the hands briefly between signs.

```bash
python -m signvaani.app --model models/random.pt --on 0.03 --off 0.012 --quiet-frames 6
```

Write the values that work, and the room they were tuned in, in the daily log. Day 4 uses them.

### 14:00-16:00, B and A: confusion analysis

```bash
python -m signvaani.confusion runs/loso_gru_<time>.json
```

It writes `runs/confusion/loso_gru_<time>/`:

- `confusion_<signer>.png` for each LOSO fold and `confusion_all.png` for all folds together.
- `report.md` with Table R3 (top-1 per test signer) and Table R6 (the 5 most confused pairs, with the
  folds they happened in and a reason hint).
- `confused_pairs.png`: for each pair, the wrongly recognised clips (red) next to clips of the sign they
  were taken for (blue), as wrist paths in shoulder-width units.

For each of the 5 pairs:

1. Read the reason hint. It compares the two signs' average hand shape, location and movement over all
   their clips: "same location and movement; they differ mainly in hand shape" means only the fingers
   tell the two signs apart, so a few badly tracked finger frames are enough to swap them.
2. Check it: look at `confused_pairs.png`, have one of you perform both signs side by side, and compare
   with the "easy to confuse" table in `docs/sign_verification.md`.
3. Is the cause the data? A red path that looks like neither sign is a wrong or sloppy take. A pair that
   fails for one test signer only points to that signer's execution.
4. Write the reason (hand shape / location / movement) in the Reason column of Table R6.

Copy `report.md` to `docs/results/loso_baseline.md`, with the reasons filled in, and
`confusion_all.png` next to it.

### 16:00-18:00, A: fix data causes

- **Wrong label** (a clip of sign X saved under Y), **wrong or sloppy sign**: move it out and log it,
  `python -m signvaani.qc --reject data/keypoints/HEAD/s03_7.npy --reason "wrong sign, looks like PAIN"`,
  then re-record it: `python -m signvaani.record --signer s03 --signs HEAD`.
- **Inconsistent execution** (one signer does a different variant): re-record that signer's takes of
  the sign with the variant in `docs/sign_verification.md`.
- Run `python -m signvaani.qc` (no SHORT line), then LOSO again, and note before and after in the log.

### 14:00-18:00, D: report Chapters 3 and 4

Chapter 3 (requirements) comes from `docs/SRS.md`. Chapter 4 (design) uses the architecture, DFD 0,
use-case and sequence diagrams in `docs/diagrams/`, the module table and model architecture of blueprint
Sections 5.2-5.4, and the parameter count from any run record (`"parameters"`).

### 18:00, all: commit

```bash
python -m pytest -q
git add docs
git commit -m "Day 3: INCLUDE-50 check, BiGRU baseline, LOSO and confusion analysis"
git push
```

`runs/` and `models/*.pt` are git-ignored; the tables to keep go in `docs/results/`.

## Exit criteria

- [ ] Baseline LOSO mean accuracy recorded (any value) with per-signer numbers (Table R3).
- [ ] Confused pairs identified with a reason for each (Table R6).
- [ ] If the LOSO mean is below 60%, fallback F2 / F3 applied before Day 4.

Fallbacks (blueprint Section 10):

- **F2, LOSO below 60% with confusion concentrated on a few pairs:** re-record the confused signs
  consistently; if a pair stays confused, replace one of its signs with a more distinct one. That
  changes the frozen vocabulary, so it needs the whole team and a line in the daily log.
- **F3, too few clean clips:** record 5 more takes for the weak classes, or add a sixth signer.
