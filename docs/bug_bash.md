# Bug bash: Day 5, 14:00-17:00

Every member uses the app for 20 minutes and writes down **every** failure, however small. 17:00-18:00
C fixes the blockers and majors; anything not fixed goes into the report's limitations. GitHub Issues
works too: then put the issue number in the ID column.

```bash
python -m signvaani.app --model models/sign_model.onnx --log runs/bugbash_<member>.csv
```

## What to try (20 minutes each)

| Who | Focus |
|---|---|
| A | All 15 signs, 3 times each, in the order of `configs/vocabulary.json`; a sign not in the vocabulary |
| B | S1-S10 in English, then press `l` and S1-S10 in Hindi; SPACE after each; `--auto-speak` |
| C | Fast and slow signing; two signs without a pause; very long pauses; `u`, `c`, `l`, `s`, `f`, `q` |
| D | Sitting still for 3 minutes; touching the face, adjusting hair, using a phone; another person walking behind; dim and bright light |

## Severity

- **Blocker:** crash, freeze, or a demo sentence that cannot be produced. Fix before the code freeze.
- **Major:** a wrong sign or sentence that happens more than once, a key that does not work, no speech.
- **Minor:** cosmetic, rare, or has a workaround.

## Issue list

| ID | Found by | What happened | Steps to repeat | Expected | Severity | Status | Fixed in (commit) |
|---|---|---|---|---|---|---|---|
| B01 | | | | | | | |
| B02 | | | | | | | |
| B03 | | | | | | | |
| B04 | | | | | | | |
| B05 | | | | | | | |

When a failure is a wrong sign, also write the top-3 bars and the "added / not added" line you saw, and
keep the `runs/bugbash_<member>.csv` log: it tells B whether to re-record, retrain or tune the spotter.
