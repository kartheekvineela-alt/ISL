# Day 4: improve, compare, ablate, export

Blueprint Section 7, Day 4. By tonight the three models are compared (Table R2), the best one is
ablated (Table R4), the final model is exported to ONNX and checked, the app runs from it, and the frame
rate and latency are measured on the weakest laptop.

All of this needs the team's recordings (`data/keypoints`, s01-s05, after Day 2 quality control). The
same commands also run on converted INCLUDE folders, where the official split replaces
leave-one-signer-out: useful to try the tools before your own data is complete.

## Ready in the repository

| Blueprint task | What is in the repo |
|---|---|
| Compare GRU vs Transformer vs 1D-CNN with LOSO, same data and seed | `python -m signvaani.experiments compare`: random split and LOSO for each model, plus parameters and CPU ms per sign. Writes Table R2 as Markdown and a run record per training. |
| Ablation: hands only / + pose / + face / no augmentation / no velocity | `python -m signvaani.experiments ablate --model <best>`: five LOSO runs, Table R4 with the change of each row. The switches are also on `train`: `--parts`, `--no-augment`, `--no-velocity`. |
| Final model on all signers except a validation signer; export ONNX; ONNX output = PyTorch output | `train --final --val-signers s05`, then `export_onnx`, which now checks ONNX against PyTorch (fails if they differ by more than 1e-4) and prints the CPU time per sign for both. |
| App: status, top-3 bars, undo / clear / language keys, auto-speak | Top-3 confidence bars, "added / not added" line, keys `u` `c` `l` `s` `f`, `--auto-speak --pause 2.5`. |
| Measure FPS and latency (TC-01, TC-14) on the weakest laptop | On quit the app prints the average frame rate and the median and worst latency; each sign prints its latency; `--log` saves them as CSV. `--complexity 0` is fallback F5. |
| Test-case document (Section 8) | `docs/test_cases.md`: TC-01 to TC-18 with how to run, expected result, linked unit tests, and columns for the Day 6 results. |

## Step by step

### The evening before, B: start the long runs

Each training takes a few minutes on a laptop CPU; compare runs 3 x (signers + 1) trainings and ablate
5 x signers. Start compare in the evening (plug the laptop in):

```bash
python -m signvaani.experiments compare --data data/keypoints --out docs/results/model_comparison.md
```

For a quicker first look, add `--epochs 40` (report only the full run).

### 09:00-12:00, B: model comparison (Table R2)

Read `docs/results/model_comparison.md`. Its last line names the best signer-independent model. Compare
it with your Day 3 LOSO result: the BiGRU row should be close, since it is the same model and seed.

### 12:00-14:00, B: ablation (Table R4)

```bash
python -m signvaani.experiments ablate --model gru --data data/keypoints --out docs/results/ablation.md
```

Use the best model from the comparison instead of `gru` if another one won. Expect hands only to be the
weakest and the full model the strongest; if face or augmentation makes no difference, say so in the
report. A result that is not the expected one is still a result.

### 14:00-15:00, B: final model and ONNX

```bash
python -m signvaani.train --data data/keypoints --final --val-signers s05 --out models/best.pt
python -m signvaani.export_onnx --ckpt models/best.pt --out models/sign_model.onnx
```

Add `--model transformer` or `--model cnn` to the first line if that model won. The export must print
`OK` on the ONNX check line; write the two CPU times in the daily log. Then:

```bash
python -m signvaani.app --model models/sign_model.onnx
git add models/sign_model.onnx models/sign_model.json
```

`models/*.pt` stays out of git; the ONNX model (about 3 MB) and its `.json` are committed.

### 09:00-15:00, C: live app

Check the overlay against TC-13 and the keys against TC-16 (`docs/test_cases.md`). Try auto-speak:

```bash
python -m signvaani.app --model models/sign_model.onnx --auto-speak --pause 2.5
```

Use the spotter values tuned on Day 3 (`--on`, `--off`, `--quiet-frames`) if they differ from the
defaults.

### 15:00-18:00, C and B: frame rate and latency on the weakest laptop

```bash
python -m signvaani.app --model models/sign_model.onnx --log runs/live_log.csv
```

Sign about 20 signs with the hands dropped between them, then press `q`. Write down the two lines it prints:

- `Frame rate: ... FPS on average` for TC-01. Below 15: apply F5, run again with `--complexity 0`
  (also record with `--complexity 0` from now on), close other apps, measure again.
- `Latency, hands stop -> sign accepted: median ... ms, worst ... ms` for TC-14. Film the screen at
  60 FPS for the official TC-14 number on Day 6; the app's number is a little lower because it leaves out
  the camera's own delay.

### 09:00-18:00, D: test cases and report

Review `docs/test_cases.md` with C, then draft report Chapter 5 (implementation): the recorder and
quality control, landmark extraction and normalisation, the model and training (class balance,
live-window trim), the spotter, the sentence rules and speech, the UI.

### 18:00, all: commit

```bash
python -m pytest -q
git add docs models/sign_model.onnx models/sign_model.json
git commit -m "Day 4: model comparison, ablation, final ONNX model, FPS and latency"
git push
```

## Exit criteria

- [ ] Model comparison (R2) and ablation (R4) tables filled from `docs/results/`.
- [ ] `models/sign_model.onnx` and `.json` committed; the ONNX check printed OK; the app runs from ONNX.
- [ ] At least 15 FPS on the weakest laptop (TC-01), otherwise fallback F5 applied.

Fallback F5 (blueprint Section 10): `--complexity 0` for the app and the recorder, the ONNX model, close
other apps.
