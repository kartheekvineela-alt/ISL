# ISL — SignVaani: Real-Time Indian Sign Language to Speech

B.Tech CSE III-year mini-project. SignVaani recognises Indian Sign Language (ISL) signs from a normal webcam,
turns them into a grammatical sentence and speaks it in English or Hindi, on a laptop CPU.

Scope (frozen on Day 1 of the 1-week blueprint): 15 hospital signs + `IDLE` in `configs/vocabulary.json`
(ME, DOCTOR, PAIN, FEVER, HEAD, STOMACH, WATER, MEDICINE, TOILET, HELP, NEED, CALL, WHERE, YES, NO) and the demo
sentences S1-S10 in `configs/sentences.json`. Runbooks: [Day 1](docs/day1.md), [Day 2](docs/day2.md),
[Day 3](docs/day3.md).

```
Webcam -> MediaPipe Holistic (hands, pose, face) -> 63 landmarks -> normalise + velocity (32 x 378)
       -> motion-based sign spotting + IDLE class -> BiGRU classifier (ONNX)
       -> ISL-order sentence rules (en / hi) -> text-to-speech
```

## Documents (`docs/`)

| File | What it is |
|---|---|
| `SignVaani_1_Week_Execution_Blueprint.pdf` | Day-by-day 7-day plan, SRS, DFD/UML, test cases, report & viva kit |
| `ISL_Sign_to_Speech_Project_Blueprint.pdf` | Full project blueprint (8-week version, tech stack, features) |
| `SignVaani_Research_Paper_IEEE_Draft.pdf` | IEEE-format paper template (replace every `[XX]` with measured results) |
| `ISL_Literature_Survey_with_Links.pdf` | 24 related papers with links and the research gap |
| `SignVaani_ISL_Dataset_Guide.pdf` | Dataset choice, recording protocol, datasheet and privacy checklist |
| `day1.md`, `day2.md`, `day3.md` | Runbooks for Days 1-3: tasks, commands and exit criteria |
| `SRS.md`, `diagrams/` | Software requirements with traceability; architecture, DFD 0, use-case and sequence diagrams (`.drawio` + `.svg`) |
| `results/` | Measured tables for the report (dataset stats, baselines, confused pairs), added as each day produces them |
| `recording_protocol.md` | Recording session checklist, consent form and consent register |
| `sign_verification.md` | Sign references, ISL-user check and Hindi sentence check |
| `daily_log.md` | Daily stand-up log for Days 1-7 |

## Dataset

- **Public baseline: [INCLUDE](data/include/README.md)** (AI4Bharat). It has isolated ISL signs, 263 classes and
  4,292 videos. The official split lists and class table are in `data/include/`. The videos are downloaded
  and converted by `signvaani.include_dataset`.
- **Final result: your own hospital recordings** (`signvaani.record`, `configs/vocabulary.json`), tested on
  signers who are not in the training set. Fill in `data/DATASHEET.md`.

```bash
python -m signvaani.include_dataset download --subset include50     # unpacks only the 958 INCLUDE-50 videos
python -m signvaani.include_dataset convert --subset include50 --workers 4
python -m signvaani.train --data data/include_keypoints/include50 --test-signers test --val-signers val \
    --out models/include50.pt
```

INCLUDE-50 is the pipeline check on real signing (Day 3) before and alongside your own data.

## Setup

Python **3.10 or 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m signvaani.check_install    # must end with OK; add --camera 0 to measure webcam FPS
python -m pytest -q                  # all tests should pass
```

## Usage

```bash
# 1. Record your dataset (keypoints only, no video is saved; re-running tops up, never overwrites)
python -m signvaani.record --signer s01                # 16 signs x 10 takes
python -m signvaani.check_data --signers s01           # layout, format, takes, hand tracking
python -m signvaani.qc --apply                         # remove weak takes, flag wrong signs, list re-recordings
python -m signvaani.dataset_stats --out docs/results/dataset_stats.md   # Table R1

# 2. Train and evaluate (signer-independent is the number to report); every run saves runs/*.json
python -m signvaani.train --data data/keypoints --loso
python -m signvaani.train --data data/keypoints --random-split
python -m signvaani.train --data data/keypoints --test-signers s05 --val-signers s04
python -m signvaani.confusion runs/loso_gru_<time>.json     # confusion matrices, Tables R3 and R6
python -m signvaani.evaluate --ckpt models/best.pt --signers s05

# 3. Export and run live
python -m signvaani.export_onnx --ckpt models/best.pt
python -m signvaani.app --model models/sign_model.onnx --lang en
```

Live app keys: `SPACE` speak, `c` clear, `u` undo, `l` switch English/Hindi, `s` skeleton on/off, `q` quit.

Spotter tuning (Day 3): `--on`, `--off` and `--quiet-frames` set the motion thresholds shown at the top of
the window.

To check the pipeline before your own recordings are complete, train on the real INCLUDE-50 clips (see
Dataset). `signvaani.make_synthetic` only feeds the automated tests; never train or report on it.

## Code layout

| Module | Purpose |
|---|---|
| `signvaani/landmarks.py` | Landmark selection, gap filling, resampling, normalisation |
| `signvaani/augment.py` | Keypoint augmentation (speed, rotation, scale, noise, hand drop-out) |
| `signvaani/data.py` | Dataset index, signer-independent / random splits, PyTorch dataset |
| `signvaani/models.py` | BiGRU + attention (main), Transformer and 1D-CNN (comparison) |
| `signvaani/train.py`, `evaluate.py`, `export_onnx.py`, `inference.py` | Training, metrics, ONNX export, inference |
| `signvaani/spotter.py` | Real-time sign start/end detection |
| `signvaani/sentence.py`, `rules.json` | ISL gloss order -> English / Hindi sentences |
| `signvaani/speech.py` | Text-to-speech (pyttsx3 offline English, gTTS Hindi) |
| `signvaani/record.py`, `app.py` | Guided recorder and live app |
| `signvaani/check_install.py`, `check_data.py` | Install check; recorded-data check and wrist-path plots |
| `signvaani/qc.py`, `dataset_stats.py` | Quality control (weak takes, possible wrong signs, re-record list); dataset statistics (Table R1) |
| `signvaani/confusion.py` | Confusion matrices per fold, most confused pairs with a reason hint (Tables R3, R6) |
| `signvaani/include_dataset.py` | INCLUDE download and video-to-keypoint conversion |
| `configs/vocabulary.json`, `sentences.json` | Frozen sign list (15 signs + IDLE) and demo sentences S1-S10 |

## Results

Fill in from your own runs (see blueprint Section 9).

| Setting | Top-1 accuracy |
|---|---|
| Signer-dependent (random split) | [XX] |
| Signer-independent (leave-one-signer-out) | [XX] ± [XX] |

## Limitations

Isolated signs from one domain only; small number of signers; no full ISL grammar; Hindi speech needs internet
unless cached. Signs and Hindi sentences must be verified by an ISL user and a native speaker. This tool supports,
and does not replace, human interpreters.
