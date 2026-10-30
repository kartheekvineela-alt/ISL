# Day 5: integration, sentence and speech

Blueprint Section 7, Day 5. By tonight S1-S10 come out right and are spoken in English and Hindi, the
model has been tested offline on a sixth person it never saw (s06), the bug bash is done, and the code is
frozen with the tag `v1.0`.

## Ready in the repository

| Blueprint task | What is in the repo |
|---|---|
| Integrate SentenceBuilder and speech; S1-S10 end to end in English and Hindi | The app builds the sentence from the accepted signs and speaks it on SPACE or after `--auto-speak --pause`. Every spoken sentence is printed (`spoken (hi): ...`). In Hindi mode the caption is drawn in Hindi when Pillow has Raqm and a Devanagari font is found (Windows: Nirmala UI or Mangal), otherwise it stays English and the app says so. `tests/test_app.py` signs every S1-S10 sentence through the app and checks the exact English and Hindi sentence that is spoken. |
| Check the speech by ear | `python -m signvaani.speech --demo --lang en` and `--lang hi` speak S1-S10 one after the other. |
| Hindi without internet at the review (fallback F7) | `python -m signvaani.speech --cache` saves the Hindi audio of S1-S10; after that Hindi plays offline. |
| Record s06 for testing only | `signvaani.record --signer s06 --out data/unseen`: a separate, git-ignored folder, so the final model cannot train on s06. |
| Offline test-signer result | `signvaani.evaluate --data data/unseen --signers s06 --out docs/results/s06` writes `summary.md` (top-1, top-3, macro-F1, recall per sign). Checkpoints now list their training signers, and evaluate prints a WARNING if s06 was one of them. |
| Bug bash | `docs/bug_bash.md`: who tests what, severity levels and the issue list. |
| Speech robustness | Sentences are spoken one at a time, so two quick SPACE presses no longer start two speech engines at once. |

## Step by step

### 09:00-12:00, C: sentences and speech

```bash
python -m pytest tests/test_sentence.py tests/test_app.py -q
python -m signvaani.speech --demo --lang en
python -m signvaani.speech --cache
python -m signvaani.speech --demo --lang hi
```

The tests check the words; the demo checks that you can hear them. `--cache` needs internet once. Then
in the live app, sign S1-S10 in English, press `l`, and sign them again in Hindi:

```bash
python -m signvaani.app --model models/sign_model.onnx --log runs/day5_sentences.csv
```

Each SPACE prints `spoken (en): ...` or `spoken (hi): ...`: copy these lines into TC-07 and TC-08 of
`docs/test_cases.md`. Have a native speaker listen to the Hindi (`docs/sign_verification.md`).

### 09:00-12:00, A: record s06, for testing only

Consent form first. s06 is someone who did **not** record on Days 1-2:

```bash
python -m signvaani.record --signer s06 --takes 3 --out data/unseen
python -m signvaani.check_data --data data/unseen --signers s06 --takes 3
```

Never copy these clips into `data/keypoints`.

### 12:00-14:00, B: offline test on s06

```bash
python -m signvaani.evaluate --ckpt models/best.pt --data data/unseen --signers s06 --out docs/results/s06
```

There must be no WARNING line. Write top-1 and macro-F1 in the daily log; `docs/results/s06/summary.md`
and its confusion matrix go into the report. If the final model was trained before Day 5, the
checkpoint cannot list its signers and evaluate says so: check by hand that s06 is not in
`data/keypoints`.

### 14:00-17:00, all: bug bash

Follow `docs/bug_bash.md`: 20 minutes each, every failure in the issue list.

### 17:00-18:00, C: fix the top bugs, freeze the code

Fix the blockers and majors (with B), run the tests, then freeze the code. After the tag, only bug
fixes go in.

```bash
python -m pytest -q
git add docs
git commit -m "Day 5: sentences and speech checked, s06 result, bug bash"
git push
git tag -a v1.0 -m "Code freeze, Day 5"
git push origin v1.0
```

## Exit criteria

- [ ] S1-S10 produce the correct English and Hindi sentence and are spoken aloud (TC-07, TC-08, TC-15).
- [ ] Offline test-signer result recorded (`docs/results/s06/summary.md`, no WARNING).
- [ ] Code freeze tag `v1.0` on GitHub.

Fallback F7 (blueprint Section 10): no internet in the hall means no new Hindi audio, so run
`python -m signvaani.speech --cache` on the presentation laptop the day before. English always works
offline.
