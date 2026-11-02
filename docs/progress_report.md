# SignVaani: Project Progress Report

**Real-Time Indian Sign Language to Speech**, B.Tech CSE mini-project, progress up to 2 November 2026

Team: Name 1 (USN 1), Name 2 (USN 2), Name 3 (USN 3), Name 4 (USN 4) · Guide: [Guide's name], [Designation] ·
Department of Computer Science & Engineering, B.M.S. College of Engineering

## 1. Project at a glance

| Item | Details |
|---|---|
| Problem | Deaf patients who use Indian Sign Language (ISL) cannot easily talk to hospital staff who do not know ISL, and certified interpreters are scarce |
| Solution | Laptop webcam -> MediaPipe body, hand and face landmarks -> sign recognition -> grammatical sentence -> English or Hindi speech, on a CPU, without a GPU |
| Vocabulary (frozen) | 15 hospital signs + IDLE: ME, DOCTOR, PAIN, FEVER, HEAD, STOMACH, WATER, MEDICINE, TOILET, HELP, NEED, CALL, WHERE, YES, NO |
| Demo sentences (frozen) | S1-S10, e.g. ME STOMACH PAIN -> "My stomach hurts." / "मेरे पेट में दर्द है।" |
| Main model | BiGRU with attention pooling (about 0.69 M parameters), compared with a Transformer and a 1D-CNN |
| Evaluation plan | Signer-independent: leave-one-signer-out over 5 signers, plus a 6th person never seen in training |
| Technology | Python 3.10/3.11, MediaPipe 0.10.14, OpenCV, PyTorch (CPU), ONNX Runtime, scikit-learn, pyttsx3, gTTS |
| Work so far | 23 October to 2 November 2026, 15 commits on GitHub |

**Summary of status**

- **Software: implemented.** Every module needed for recording, quality control, training, evaluation,
  model export, the live app, sentence generation and speech is written, with 67 automated unit tests.
- **Design and documents: complete.** SRS, four design diagrams, test plan with 18 test cases, literature
  survey of 24 papers, IEEE paper draft, recording protocol and consent form, PPT content.
- **Pending:** recording the team's own dataset, training runs and their measured results, the live test
  cases, the final report and the demo video.

## 2. Objectives and their status

| # | Objective | Implemented | Measured | Evidence / next step |
|---|---|---|---|---|
| 1 | Recognise 15 signs + IDLE from a webcam in real time (at least 15 FPS) on a CPU | Done | To do | `app.py`, `models.py`, ONNX export; FPS and accuracy measured after training (TC-01, TC-05) |
| 2 | Detect the start and end of each sign automatically, ignore non-sign movement | Done | To do | `spotter.py` with motion gate + IDLE class; to be measured live (TC-09, TC-10) |
| 3 | Turn ISL word order into grammatical English and Hindi sentences and speak them | Done | Verified | All 11 demo sentences correct in both languages (Section 6) |
| 4 | Evaluate honestly on unseen signers; never store video | Done | To do | `train.py --loso`, `evaluate.py`; recorder saves only landmarks (TC-18) |

## 3. Work timeline

From the commit history on GitHub.

| Date | Milestone | What was delivered |
|---|---|---|
| 23&nbsp;Oct | Project set-up | Python package skeleton; project blueprint (1-week and 8-week plans), literature survey with 24 papers, IEEE paper draft, dataset guide |
| 23&nbsp;Oct | Public dataset | Tool to download the INCLUDE ISL dataset (AI4Bharat, 263 signs, 4,292 videos) and convert the videos to keypoints; official split lists |
| 23&nbsp;Oct | Day 1: scope and set-up | Vocabulary and sentences frozen (and locked by tests); install check; recorded-data check; guided recorder that resumes and never overwrites; SRS; architecture and DFD level 0 diagrams; recording protocol, consent form, sign verification sheet |
| 26&nbsp;Oct | Days 2-3: data and baseline tools | Dataset statistics (Table R1); quality control (reject, review, re-record list); unit tests for S1-S10; confusion-matrix analysis (Tables R3, R6); use-case and sequence diagrams; datasheet; INCLUDE-50 real-data check |
| 29&nbsp;Oct | App and training improvements | Camera preview mode before a model exists; resizable and full-screen window; class-balanced loss; training clips cut to the live spotter window; on-screen "added / not added" message |
| 30&nbsp;Oct | Day 4: models and speed | Model comparison (BiGRU, Transformer, 1D-CNN); ablation study; final model export to ONNX with a check that ONNX equals PyTorch; FPS and latency measurement; test case document TC-01 to TC-18 |
| 30&nbsp;Oct | Day 5: sentences and speech | Sentence + speech end to end in English and Hindi; Hindi audio cache for offline use; unseen-signer (s06) evaluation; bug-bash plan; speech made one-at-a-time |
| 2&nbsp;Nov | Review preparation | Slide-by-slide PPT content (with PDF), diagram images, this progress report |

## 4. What has been built

**Figure:** `docs/diagrams/architecture.png`. The top row runs on every frame; the bottom row runs once
per accepted sign or sentence.

### Modules (21 Python modules, about 2,900 lines of code)

| Area | Modules | Lines |
|---|---|---|
| Data collection and checks | `record.py`, `check_install.py`, `check_data.py`, `qc.py`, `dataset_stats.py` | 787 |
| Public dataset (INCLUDE) | `include_dataset.py` | 345 |
| Features and augmentation | `landmarks.py`, `augment.py`, `data.py` | 301 |
| Models and training | `models.py`, `train.py`, `experiments.py` | 463 |
| Evaluation | `evaluate.py`, `confusion.py` | 337 |
| Deployment | `export_onnx.py`, `inference.py` | 113 |
| Real-time app | `app.py`, `spotter.py` | 382 |
| Language and speech | `sentence.py` + `rules.json`, `speech.py` | 156 |
| Test data generator | `make_synthetic.py` (for automated tests only) | 51 |

### Documents and design

| Document | Content | Status |
|---|---|---|
| `docs/SRS.md` | 12 functional and 7 non-functional requirements, each traced to a module and a test case; measurable targets; feasibility | Done |
| `docs/diagrams/` | Architecture, DFD level 0, use-case and sequence diagrams (draw.io, SVG and PNG) | Done |
| `docs/test_cases.md` | 18 system test cases with how to run, expected result and linked unit test | Done |
| `docs/recording_protocol.md` | Recording checklist, consent form, consent register | Done |
| `docs/sign_verification.md` | Sign references (ISLRTC dictionary), ISL-user and Hindi checks | Done |
| `data/DATASHEET.md` | Datasheet for our recordings and for INCLUDE | In progress |
| `docs/presentation_content.md` | Content for every slide of the review PPT | Done |
| Project report | Chapters 1-6 | To do |

### Testing

- **67 automated unit tests** in 14 test files (1,246 lines) cover landmark processing, sign spotting,
  training and splits, sentences, speech, recorder, quality control, the app and the INCLUDE tool.
  Run `python -m pytest -q` on the demo laptop before the review and quote the pass count.
- **18 system test cases** (TC-01 to TC-18) are written and will be run with the final model.

## 5. Method implemented

- **Landmarks:** MediaPipe Holistic gives body, hand and face points; we keep 63 per frame (9 upper-body,
  12 face, 21 per hand). Missing points are stored as NaN; gaps of up to 6 frames are filled.
- **Features:** each sign is resampled to 32 frames, centred on the shoulders and divided by shoulder
  width; velocity is added, giving a **32 x 378** input.
- **Augmentation:** speed 0.8-1.2x, rotation ±15°, scale 0.9-1.1, shift, noise, and hiding a hand for a few frames.
- **Model:** Linear 378 -> 256 + LayerNorm -> 2-layer bidirectional GRU (hidden 128) -> attention
  pooling -> 16 classes. Transformer and 1D-CNN versions use the same input for comparison.
- **Training:** AdamW (learning rate 0.001), cosine decay, label smoothing 0.1, class-balanced loss, up to
  120 epochs with early stopping (patience 25), batch 64.
- **Evaluation:** random split (same people) and leave-one-signer-out (new people), top-1 accuracy,
  macro-F1, confusion matrices, and a final test on a 6th signer.
- **Live use:** a sign starts when the wrists move and ends after 8 still frames; it is accepted only if
  confidence is at least 0.8 and it is not IDLE. Accepted signs go to the sentence rules and are spoken
  (pyttsx3 for English offline, gTTS with a cache for Hindi).

## 6. Results so far

| Item | Result | Status |
|---|---|---|
| Sentence generation, English (S1-S10, 11 sentences) | 11 / 11 exactly as expected | Verified |
| Sentence generation, Hindi (S1-S10, 11 sentences) | 11 / 11 exactly as expected | Verified |
| Classifier size (BiGRU + attention) | 694,801 parameters (about 0.69 M), small enough for a CPU | Verified |
| Frozen scope | Vocabulary and sentences locked by `tests/test_frozen_scope.py` | Done |
| Accuracy, random split and leave-one-signer-out | [XX] % and [XX] ± [XX] % | To do |
| Frame rate, latency, false triggers while idle | [XX] FPS, [XX] ms, [XX] per minute | To do |
| Unseen signer s06 (60 attempts) | [XX] / 60 | To do |

Accuracy and speed numbers are filled in only from the team's own runs after training.

## 7. Work remaining

| Task | Status | Command / note |
|---|---|---|
| Record 5 signers x 16 classes x 10 takes (800 clips) and run quality control | To do | `python -m signvaani.record --signer s01`, then `python -m signvaani.qc --apply` |
| Extra experiment on public data: INCLUDE "Adjectives" part 4 of 8 | In progress | Downloaded and unzipped; next: `include_dataset convert`, then `train` with the official split |
| Pipeline check on INCLUDE-50 | To do | `python -m signvaani.include_dataset download --subset include50` |
| Training: random split, LOSO, model comparison, ablation | To do | `train --loso`, `experiments compare`, `experiments ablate` |
| Final model, ONNX export, live app tuning | To do | `train --final`, `export_onnx`, `app --model models/sign_model.onnx` |
| Record s06 (test only) and evaluate | To do | `evaluate --data data/unseen --signers s06` |
| Run system test cases TC-01 to TC-18 | To do | On the weakest laptop, in the review room |
| Report, demo video, final PPT | In progress | PPT content ready; fill in measured numbers |

## 8. Challenges and how we handled them

| Challenge | How we handled it |
|---|---|
| MediaPipe installs only on certain Python versions | Pinned Python 3.10/3.11, MediaPipe 0.10.14 and NumPy below 2; an install-check script tests every laptop |
| The system could add words while the user is just resting | An IDLE class (resting hands, touching the face, adjusting hair), a motion gate, a 0.8 confidence threshold and a cool-down |
| Training clips looked different from live windows | Training clips are cut to the same window the live spotter classifies |
| Signs with fewer clips were predicted less often | Class-balanced loss: rarer signs get a larger weight |
| Accuracy looks too high when the same person is in training and test | Leave-one-signer-out evaluation plus a sixth, unseen person |
| Hindi speech needs internet, which the review hall may not have | Hindi audio for S1-S10 is cached in advance; English works fully offline |
| Two quick SPACE presses started two speech engines | Sentences are now spoken one at a time |

## 9. Likely review questions

| Question | Short answer |
|---|---|
| Why landmarks instead of raw video? | Landmarks are small (378 numbers per frame), ignore the background and clothing, run fast on a CPU, and protect privacy because no video is stored |
| Why a BiGRU? | Our dataset is small; a 0.69 M-parameter recurrent model learns well from little data and runs in real time. We still compare it with a Transformer and a 1D-CNN |
| What is leave-one-signer-out? | Train on 4 people, test on the 5th, repeat for each person. It shows how the system works for a new patient |
| How does it know where a sign starts and ends? | Wrist speed above a threshold opens a window; 8 quiet frames close it; the IDLE class and the 0.8 confidence threshold reject non-signs |
| How is ISL grammar handled? | ISL order such as ME STOMACH PAIN is matched to patterns like [PRONOUN, BODY, PAIN] and turned into "My stomach hurts." and the Hindi sentence |
| What if there is no internet? | Recognition and English speech are fully offline; Hindi audio is cached beforehand |
| What are the limitations? | Isolated signs from one domain, a small number of signers, no full ISL grammar or fingerspelling. The tool supports interpreters; it does not replace them |
