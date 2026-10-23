# Day 1: setup, vocabulary, recorder, pilot data

Blueprint Section 7, Day 1. By tonight every laptop works, the scope is frozen, and signer s01's pilot
data is recorded and checked.

## Ready in the repository

| Blueprint task | What is in the repo |
|---|---|
| Environment setup, push skeleton | Package skeleton on GitHub. `python -m signvaani.check_install` checks Python, the pinned packages, that MediaPipe Holistic loads, and the speech engine. With `--camera 0` it also measures the webcam frame rate with MediaPipe. |
| Freeze vocabulary | `configs/vocabulary.json`: the 15 signs + IDLE of Section 2.1. It replaces the 41-sign list of the 8-week plan. |
| Freeze sentence list | `configs/sentences.json`: S1-S10 with the exact English and Hindi output. `tests/test_frozen_scope.py` fails if the vocabulary or the sentences drift. The current `signvaani/rules.json` already produces all ten; the Day 2 unit tests will confirm it. |
| Run the recorder, verify NaN for missing hands, check file layout | `signvaani.record` keeps earlier takes: re-running resumes or tops up and never overwrites. It accepts only anonymous IDs (s01, ...) and vocabulary signs, gives 1 s to redo a take, and prompts natural movements during IDLE takes. `signvaani.check_data` checks folder and file names, float32 (T, 63, 3) arrays, NaN (never zeros) for undetected parts, takes per sign, and hand visibility. |
| Write SRS, draw architecture and DFD 0 | `docs/SRS.md`; `docs/diagrams/architecture.drawio` and `dfd0.drawio` (open in draw.io / diagrams.net) with the SVG previews used in the SRS. |
| Verify signs with an ISL user | `docs/sign_verification.md`: sign table, Hindi sentence check, message template. |
| Pilot recording | `docs/recording_protocol.md`: session checklist, consent form, consent register, rejected-takes log. |
| Inspect pilot data | `python -m signvaani.check_data --signers s01 --plot PAIN HEAD WATER` |
| Daily log | `docs/daily_log.md`, with the Day 1 plan filled in. |

## Step by step

### 09:00-11:00, all: environment setup

```bash
git clone https://github.com/kartheekvineela-alt/ISL.git && cd ISL
python -m venv .venv
.venv\Scripts\activate                 # Windows
source .venv/bin/activate              # Linux / macOS
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m signvaani.check_install --camera 0     # must end with OK
python -m pytest -q
```

Fix every `[FAIL]` line; the line says how. Each member writes "install check OK" in the daily log.

### 09:00-11:00, A and D: learn the signs

The vocabulary is frozen. Learn each sign from the ISLRTC dictionary videos, paste the link into
`docs/sign_verification.md`, and practise.

### 11:00-13:00, B and C: test the recorder

```bash
python -m signvaani.record --signer s01 --signs PAIN IDLE --takes 2 --out runs/recorder_test
python -m signvaani.check_data --data runs/recorder_test --signs PAIN IDLE --takes 2
```

- PASS means the file names, folder layout and array format are right.
- Do one IDLE take with both hands out of the frame. The IDLE `hands` column then drops below 100%, and
  because the format check still passes, the missing hands were saved as NaN.
- Hold your hands out of the frame during a PAIN take: the recorder must reject it and retry (TC-04).
- `runs/` is git-ignored; delete `runs/recorder_test` when done.

### 11:00-13:00, D: SRS and diagrams

Review `docs/SRS.md` with the team and fill in the [XX] cells of Section 3.6 for your laptops. Open
`docs/diagrams/*.drawio` in draw.io, adjust if needed, and export PNGs for the report
(File > Export as > PNG).

### 14:00-16:00, A: confirm the signs

Send the message in `docs/sign_verification.md`. Record who confirmed each sign and any variant notes.

### 14:00-17:00, all: pilot recording of s01 (about 30 minutes)

Consent form first (`docs/recording_protocol.md`). Then:

```bash
python -m signvaani.record --signer s01
```

If takes keep getting rejected, bring the light to the front of the signer or move the camera back.

### 17:00-18:00, B: inspect the pilot data

```bash
python -m signvaani.check_data --signers s01 --plot PAIN HEAD WATER
```

- It must print `PASS: 16 signs x 10 takes for s01` (exit criterion 2).
- The `hands` column should be at least 90% for every sign. WARN lines name the signs to re-record after
  fixing the lighting.
- Open `runs/pilot/wrists.png`: the takes of one sign should overlap. A take whose path goes elsewhere
  is a wrong or sloppy sign. Delete it, log it in `docs/recording_protocol.md`, and run
  `python -m signvaani.record --signer s01 --signs PAIN` to record it again.

### 18:00, all: commit

```bash
python -m pytest -q
git add docs configs
git commit -m "Day 1: environment, frozen scope, pilot data checked"
git push
```

Never commit `data/keypoints/` or the signed consent forms.

## Exit criteria

- [ ] Every member's machine prints OK in the install check.
- [ ] `python -m signvaani.check_data --signers s01` prints PASS: 16 folders x 10 clips for s01, and the
      recorder rejected the bad takes.
- [x] Vocabulary and sentence list frozen and committed (`configs/vocabulary.json`,
      `configs/sentences.json`).

If a criterion fails, apply the fallback before Day 2 (blueprint Section 10):

- **F1, MediaPipe install fails:** Python 3.10 exactly, `mediapipe==0.10.14`, `"numpy<2"`; or record on
  a teammate's laptop.
- **F5, below 15 FPS in `check_install --camera 0`:** set `model_complexity=0` where `record.py` and
  `app.py` create `Holistic`, use 640x480, close other apps.
- **F3, too few clean clips:** run `python -m signvaani.record --signer s01 --signs <SIGN>` again. It
  records only the missing takes.
