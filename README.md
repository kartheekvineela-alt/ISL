# ISL — SignVaani: Real-Time Indian Sign Language to Speech

B.Tech CSE III-year mini-project. SignVaani recognises Indian Sign Language (ISL) signs from a normal webcam,
turns them into a grammatical sentence and speaks it in English or Hindi, on a laptop CPU.

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

## Setup

Python **3.10 or 3.11**.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q                  # all tests should pass
```

## Usage

```bash
# 1. Record your dataset (keypoints only, no video is saved)
python -m signvaani.record --signer s01 --takes 10

# 2. Train and evaluate (signer-independent is the number to report)
python -m signvaani.train --data data/keypoints --loso
python -m signvaani.train --data data/keypoints --random-split
python -m signvaani.train --data data/keypoints --test-signers s05 --val-signers s04
python -m signvaani.evaluate --ckpt models/best.pt --signers s05

# 3. Export and run live
python -m signvaani.export_onnx --ckpt models/best.pt
python -m signvaani.app --model models/sign_model.onnx --lang en
```

Live app keys: `SPACE` speak, `c` clear, `u` undo, `l` switch English/Hindi, `s` skeleton on/off, `q` quit.

To check the pipeline before you have real data:
`python -m signvaani.make_synthetic --out data/synthetic` then train on `data/synthetic`
(fake data, never report these numbers).

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
| `configs/vocabulary.json` | Sign list |

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
