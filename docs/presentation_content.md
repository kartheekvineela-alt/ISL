# SignVaani: PPT content, slide by slide

Content for the BMSCE mini-project PPT template (`MIP_PPT_template`), in the template's slide order.
Each slide has what goes **on the slide**, the **figure** to insert, and **speaker notes** (what to say).
Slides marked *optional* are extra slides you can add where shown.

Before the review:

- Replace every `[XX]` with a number you measured yourself. The command that produces each number is
  given on the Results slides. Never copy example numbers.
- The diagrams are in `docs/diagrams/`: use the `.png` files (Google Slides, any PowerPoint) or the
  `.svg` files (PowerPoint 2016 / 365 keeps them sharp).
- Reference numbers `[1]` to `[21]` match the References slides.

---

## Slide 1: Title

**On the slide**

- **Project title:** SignVaani: Real-Time Indian Sign Language to Speech
- Subtitle (smaller, optional): *A landmark-based, CPU-only system that turns ISL signs into English and Hindi speech*
- Name 1, USN 1
- Name 2, USN 2
- (Name 3, USN 3 and Name 4, USN 4 if you are a team of four)
- Guide: [Guide's name]
- Designation: [Assistant Professor / Associate Professor]
- Department of Computer Science & Engineering
- B.M.S. College of Engineering

**Speaker notes**

"Good morning. Our mini-project is SignVaani, which means 'voice of signs'. It recognises Indian Sign
Language from an ordinary laptop webcam and speaks the meaning aloud in English or Hindi."

---

## Slide 2: Outline

1. Introduction
2. Problem Statement and Objectives
3. Literature Survey
4. High Level Design
5. Detailed Design
6. Software and Hardware Requirements
7. Implementation
8. Results
9. Applications
10. Conclusion and Future Scope
11. References

---

## Slide 3: Introduction

**On the slide**

- Indian Sign Language (ISL) is the main language of India's deaf and hard-of-hearing community. It is a
  complete language with its own vocabulary and grammar.
- ISL is different from American Sign Language (ASL):
  - many signs use both hands;
  - facial expressions and head movements carry meaning (non-manual markers), e.g. raised eyebrows for a question;
  - sentences mostly follow subject-object-verb order, e.g. **ME STOMACH PAIN** = "My stomach hurts".
- The 2011 Census counted over **5 million** people in India with a hearing disability, but the country
  has only a few hundred certified ISL interpreters.
- So deaf people face daily communication barriers in hospitals, banks, railway stations and government
  offices, where staff do not know ISL.
- Most sign language recognition research and tools are built for ASL. ISL data and ready-to-use tools
  are scarce [18], [19].
- **SignVaani** recognises ISL signs from a normal laptop webcam, turns them into a grammatical sentence
  and speaks it in English or Hindi, in real time, on a laptop CPU, without a GPU or special sensor.
- Chosen domain: **hospital communication**, 15 signs + IDLE:
  ME, DOCTOR, PAIN, FEVER, HEAD, STOMACH, WATER, MEDICINE, TOILET, HELP, NEED, CALL, WHERE, YES, NO.

**Figure:** a simple flow across the bottom of the slide:
`ISL sign -> Webcam -> Body/hand landmarks -> Sign recognition -> Sentence -> Speech`

**Key terms (small box, optional)**

| Term | Meaning |
|---|---|
| Gloss | Written label of one sign, e.g. `PAIN` |
| Landmark | One (x, y, z) body point detected by MediaPipe |
| IDLE | The "no sign" class: resting hands, touching the face, adjusting hair |
| Signer-independent | The person tested never appears in the training data |

**Speaker notes**

"ISL is not signed English. It has its own grammar, uses both hands and the face, and puts the verb at
the end. Hospital staff rarely know it, and there are very few interpreters. Most research tools are for
American Sign Language and need GPUs or depth cameras. We wanted something that runs on the laptop a
hospital already has."

---

## Slide 4: Problem Statement

**On the slide**

**Problem statement**

> Deaf patients who use Indian Sign Language cannot easily communicate with hospital staff who do not
> know ISL, and certified interpreters are scarce. Existing sign language recognition systems mostly
> target ASL, need GPUs, depth cameras or gloves, stop at text output, and are usually evaluated on the
> same people they were trained on. There is a need for a low-cost system that recognises ISL signs from
> an ordinary webcam in real time, converts them into grammatical sentences, speaks them in English and
> Hindi, and works reliably for people it has never seen.

**Objectives**

1. Recognise 15 hospital ISL signs and an IDLE class from a laptop webcam in real time (at least 15 FPS) on a CPU.
2. Detect the start and end of each sign automatically, and ignore natural non-sign movements.
3. Convert the recognised signs (in ISL word order) into grammatical English and Hindi sentences and speak them.
4. Evaluate honestly with a signer-independent (leave-one-signer-out) protocol, and keep only landmarks, never video, for privacy.

**Scope**

| In scope | Out of scope |
|---|---|
| Isolated recognition of 16 classes from a webcam | Continuous, open-vocabulary ISL translation |
| Automatic sign start/end detection | Fingerspelling A-Z |
| Rule-based ISL order to English / Hindi sentences | Neural translation, LLM sentence polishing |
| Speech: English offline (pyttsx3), Hindi via gTTS with offline cache | Fully offline Hindi TTS (future scope) |
| Signer-independent evaluation | Large user study or clinical validation |
| Own dataset: 5 signers x 16 classes x 10 takes = 800 clips | Mobile app |

**Speaker notes**

"We deliberately fixed a small vocabulary so that every sign works on a new person. The golden rule we
followed: 15 signs that work on someone the model has never seen are worth more than 50 signs that only
work for the person who recorded them."

---

## Slide 5: Literature Survey (1/4): ISL datasets and benchmarks

**On the slide** (table)

| No. | Paper (authors, venue, year) | Methodology / contribution | Limitations / gap |
|---|---|---|---|
| [1] | INCLUDE: A Large Scale Dataset for ISL Recognition (Sridhar et al., ACM Multimedia 2020) | Word-level ISL dataset: 4,287 videos, 263 signs in 15 categories, signed by deaf students. Best reported accuracy 94.5% on INCLUDE-50 and 85.6% on the full set | Lab dataset and recognition only; no live sentence building or speech output |
| [2] | CISLR: Corpus for Indian Sign Language Recognition (Joshi et al., EMNLP 2022) | Word-level ISL corpus with about 4,700 words; one-shot recognition baseline using features learned from ASL | One-shot recognition from video; no real-time sign spotting or speech pipeline |
| [3] | ISLTranslate: Dataset for Translating ISL (Joshi et al., Findings of ACL 2023) | About 31k ISL-English sentence pairs from educational videos; pose-based transformer translation baseline | Continuous translation accuracy is still low; heavy models; no speech output |
| [4] | iSign: A Benchmark for ISL Processing (Joshi et al., Findings of ACL 2024) | Large ISL-English benchmark (118k+ video-sentence pairs) with tasks such as SignVideo2Text, SignPose2Text and Text2Pose | A benchmark, not a deployable assistant; no domain focus or offline speech |

**Speaker notes**

"ISL research has grown mainly through datasets. INCLUDE is the standard word-level dataset and we use
its INCLUDE-50 subset to test our pipeline on real signing. But these works stop at recognition or
translation scores; none gives a live, speaking assistant."

---

## Slide 6: Literature Survey (2/4): ISL and multi-signer recognition

**On the slide** (table)

| No. | Paper (authors, venue, year) | Methodology / contribution | Limitations / gap |
|---|---|---|---|
| [5] | OpenHands: Pose-based Pretrained Models across Languages (Selvaraj et al., ACL 2022) | Uses pose keypoints instead of raw video; releases pose-based models for 6 sign languages including ISL, and a large ISL pretraining set | Research toolkit, not a real-time end-user app; no Indian-language speech |
| [6] | ISL-CSLTR: Dataset for Continuous Sign Language Translation and Recognition (Elakkiya and Natarajan, 2021) | Sentence-level ISL corpus: 700 videos of 100 sentences by 7 signers, plus word-level images | Small vocabulary and few signers; dataset only |
| [7] | Nearest Neighbour Classification of ISL Gestures using Kinect (Ansari and Harit, Sadhana 2016) | 140 ISL signs from 18 subjects captured with a Kinect depth camera; k-NN classifier; about 90.7% on the fingerspelling subset | Needs a special depth camera; static gestures; no real-time sentences |
| [8] | Continuous SLR Handling Multiple Signers (Koller, Forster and Ney, CVIU 2015) | Statistical continuous recognition on RWTH-PHOENIX-Weather and SIGNUM; shows how hard multi-signer recognition is | German Sign Language; pre-deep-learning pipeline |

**Speaker notes**

"OpenHands showed that pose keypoints work well for sign recognition, which is the approach we follow.
Older ISL work used Kinect depth cameras, which a hospital does not have. Koller's work shows that
accuracy drops when the signer changes, which is why we evaluate on unseen signers."

---

## Slide 7: Literature Survey (3/4): large-scale recognition and translation

**On the slide** (table)

| No. | Paper (authors, venue, year) | Methodology / contribution | Limitations / gap |
|---|---|---|---|
| [9] | WLASL: Word-level Deep SLR from Video (Li et al., WACV 2020) | 2,000+ ASL words from 100+ signers; compares appearance-based (I3D) and pose-based (Pose-TGCN) models | ASL only; models too heavy for real-time use on a laptop CPU |
| [10] | MS-ASL: Large-Scale Data Set and Benchmark for ASL (Vaezi Joze and Koller, BMVC 2019) | 25,000+ videos of 1,000 ASL signs from 200+ signers; signer-independent splits; I3D baseline | ASL only; RGB video models need GPUs |
| [11] | Neural Sign Language Translation (Camgoz et al., CVPR 2018) | Treats sign language translation as neural machine translation; releases RWTH-PHOENIX-Weather 2014T | German Sign Language; offline research setting; no speech |
| [12] | Sign Language Transformers (Camgoz et al., CVPR 2020) | Transformer with CTC loss learns recognition and translation together; 21.80 BLEU-4 on PHOENIX14T | Large model; not ISL; no real-time deployment |

**Speaker notes**

"The large ASL and German datasets drove the deep learning methods, but they use RGB video models that
need GPUs. MS-ASL is important for us because it uses signer-independent splits, the same honest
evaluation we use."

---

## Slide 8: Literature Survey (4/4): pose and keypoint-based methods

**On the slide** (table)

| No. | Paper (authors, venue, year) | Methodology / contribution | Limitations / gap |
|---|---|---|---|
| [13] | SPOTER: Sign Pose-based Transformer (Bohacek and Hruz, WACV Workshops 2022) | Lightweight transformer on 2D body landmarks with normalisation and rotation augmentation; 63.18% on WLASL100 | ASL / LSA64 only; no ISL, sentence or speech module |
| [14] | SAM-SLR: Skeleton Aware Multi-modal SLR (Jiang et al., CVPR Workshops 2021) | Skeleton graph network (SL-GCN) plus RGB and depth streams; won the 2021 signer-independent isolated SLR challenge (98.42% RGB track) | Heavy multi-modal ensemble; Turkish SL; not real-time on CPU |
| [15] | SignBERT: Hand-Model-Aware Pre-training (Hu et al., ICCV 2021) | Self-supervised pre-training on hand-pose tokens, then fine-tuning; state of the art on MS-ASL and WLASL | Hands only (no face or body); heavy pre-training; not ISL |
| [16] | ST-GCN: Spatial Temporal Graph Convolutional Networks (Yan, Xiong and Lin, AAAI 2018) | Graph convolution over skeleton joints in space and time; basis of many skeleton-based SLR models | Built for action recognition, not sign language; a possible future model for us |

**Tools used** (one line under the table): MediaPipe [17] and MediaPipe Hands [18] give real-time pose,
hand and face landmarks on a CPU.

**Speaker notes**

"Keypoint methods are compact, ignore the background and protect privacy, because no video is needed.
SPOTER inspired our shoulder-based normalisation and rotation augmentation. SignBERT uses hands only;
we add body and face points because ISL meaning depends on where the hand is and on facial expression."

---

## Slide 9: Comparison of the existing methods

**On the slide** (table)

| Method | Sign language | Input | Reported result | Main limitation |
|---|---|---|---|---|
| Ansari and Harit [7] | ISL | Kinect depth camera | about 90.7% (fingerspelling subset) | Special sensor; static gestures |
| INCLUDE [1] | ISL | RGB video | 94.5% (INCLUDE-50), 85.6% (263 signs) | No live sentences or speech |
| OpenHands [5] | ISL + 5 others | Pose keypoints | Pretrained pose models | Toolkit, not a real-time app |
| SPOTER [13] | ASL, LSA64 | 2D body landmarks | 63.18% (WLASL100) | No ISL; no sentence or speech |
| SAM-SLR [14] | Turkish SL | Skeleton + RGB + depth | 98.42% (RGB track) | Heavy ensemble; not real-time on CPU |
| Sign Language Transformers [12] | German SL | RGB video | 21.80 BLEU-4 (PHOENIX14T) | Large model; no real-time use |
| **SignVaani (ours)** | **ISL (15 hospital signs + IDLE)** | **Laptop webcam, 63 hand, body and face landmarks** | **See Results** | **Small vocabulary of isolated signs** |

**Research gap** (box under the table)

None of the reviewed works combines all of the following, and SignVaani is designed to:

1. real-time ISL recognition on a laptop CPU;
2. hand + body + face landmarks;
3. automatic sign spotting with an IDLE class;
4. ISL-order to grammatical sentence generation;
5. speech output in an Indian language;
6. offline, privacy-preserving operation (no video stored);
7. signer-independent evaluation.

**Speaker notes**

"The numbers in this table come from different datasets, so they are not directly comparable. The point
is the last column: each work solves one part of the problem. Our contribution is putting the parts
together into one working, honest system on cheap hardware."

---

## Slide 10: High Level Design

**Figure:** `docs/diagrams/architecture.png` (system architecture). Optional on the same or the next
slide: `docs/diagrams/dfd0.png` (context diagram, DFD level 0).

**On the slide** (one line per block, beside or under the figure)

Top row: runs on every frame (30 FPS)

- **Webcam:** OpenCV captures frames at 30 FPS.
- **MediaPipe Holistic:** finds pose, both-hand and face landmarks in each frame.
- **Preprocess:** keeps 63 points, fills short gaps, normalises to the shoulders, adds velocity.
- **Sign spotter:** hand-motion gate decides when a sign starts and ends; IDLE class rejects non-signs.
- **Classifier:** BiGRU with attention, exported to ONNX for fast CPU inference.

Bottom row: runs once per accepted sign or sentence

- **Gloss buffer:** list of accepted signs (undo / clear by keyboard).
- **Sentence builder:** ISL gloss order to a grammatical sentence (rules in `rules.json`).
- **Translation:** English and Hindi templates.
- **TTS:** pyttsx3 (English, offline), gTTS (Hindi, with an offline cache).
- **Speaker:** audio out; the caption stays on screen.

**Design decisions** (small box, optional)

- Landmarks instead of raw video: small input, background-independent, privacy-preserving.
- Explicit IDLE class + motion gate: no false words while the user is resting.
- Rule-based sentences: predictable, cannot invent content that was not signed.
- ONNX Runtime: fast inference on any laptop CPU.

**Speaker notes**

"The top row runs 30 times a second. The bottom row runs only when a sign is accepted or the user asks
for speech. Only landmark coordinates are kept, never video, which makes the system private and light."

### Optional extra slide after slide 10: High Level Design (contd.): Use cases

**Figure:** `docs/diagrams/usecase.png`.

**On the slide**

| Actor | What they do |
|---|---|
| Deaf user (signer), e.g. a patient | Signs in front of the webcam; sees the recognised signs and the sentence; undoes a wrong sign |
| Hearing person (listener), e.g. hospital staff | Hears the sentence and reads the caption; presses SPACE to speak or `c` to clear |
| Developer (project team) | Records signers, checks the data, trains, evaluates and exports models |

---

## Slide 11: Detailed Design

**On the slide**

**1. Feature vector (per frame)**

- 63 landmarks per frame: 9 upper-body pose points (nose, eyes, shoulders, elbows, wrists) + 12 face
  points (lips, eyebrows, eye corners) + 21 left-hand + 21 right-hand points.
- 63 x (x, y, z) = 189 position values + 189 velocity values = **378 features per frame**.
- Each sign is resampled to 32 frames, so the model input is **32 x 378**.
- A body part that was not detected is stored as NaN (not zero); gaps of up to 6 frames are interpolated.

**2. Normalisation** (removes the effect of where the person stands and their body size)

- p' = (p - shoulder centre) / shoulder width
- v(t) = p'(t) - p'(t-1)

**3. Classifier: BiGRU with attention pooling**

| Layer | Output shape |
|---|---|
| Input | 32 x 378 |
| Linear 378 -> 256 + LayerNorm + GELU + Dropout 0.2 | 32 x 256 |
| 2-layer bidirectional GRU, hidden 128, dropout 0.3 | 32 x 256 |
| Attention pooling: a(t) = softmax(w . h(t)), z = sum of a(t) h(t) | 256 |
| Dropout + Linear 256 -> 16 | 16 class scores (15 signs + IDLE) |

- About **0.69 M parameters** (694,801), small enough for real-time CPU inference.
- Comparison models with the same input: Transformer encoder (4 layers, 4 heads, width 192) and a
  temporal 1D-CNN (3 convolution layers, 192 channels).

**4. Sign spotter (real-time segmentation)**

- Motion energy = largest wrist movement between frames, divided by shoulder width.
- Window opens when energy > 0.02; closes after energy < 0.01 for 8 frames (min 10, max 90 frames).
- A sign is accepted only if confidence >= 0.8 and the class is not IDLE; 12-frame cool-down stops duplicates.

**Figure:** `docs/diagrams/sequence.png` (one sign, then speak). If the slide gets crowded, put the
sequence diagram on its own slide, "Detailed Design (contd.)".

**Speaker notes**

"Each frame becomes a 378-number vector. Dividing by shoulder width means a tall person and a short
person, near or far from the camera, give the same numbers. The GRU reads the 32 frames in both
directions, and attention lets the model focus on the frames where the hand shape is clearest."

---

## Slide 12: Hardware and Software Requirements

**On the slide**

**Hardware**

| Item | Minimum | Used by us |
|---|---|---|
| Processor | Intel Core i5 8th gen / AMD Ryzen 5 (no GPU needed) | [XX] |
| RAM | 8 GB | [XX] |
| Camera | 720p front-facing webcam | [XX] |
| Audio | Speakers or headphones | [XX] |

**Software**

| Item | Version / tool |
|---|---|
| Operating system | Windows 10/11, Ubuntu 22.04 or macOS 12+ |
| Language | Python 3.10 or 3.11 |
| Landmark detection | MediaPipe Holistic 0.10.14 |
| Video and UI | OpenCV 4.10 |
| Deep learning | PyTorch (CPU build) for training |
| Inference | ONNX, ONNX Runtime (CPU) |
| Data and evaluation | NumPy, scikit-learn, Matplotlib |
| Speech | pyttsx3 (offline English), gTTS + pygame (Hindi) |
| Testing and tools | pytest, Git and GitHub, draw.io |

Training: a laptop CPU is enough for 800 clips (a few minutes per model); Google Colab is optional.

**Speaker notes**

"Everything is open source and runs on a normal student laptop. Internet is needed only to generate
Hindi speech the first time; after caching, the whole system works offline."

---

## Slide 13: Functionalities of the modules

**On the slide** (table)

| Module (file) | Input -> output | What it does |
|---|---|---|
| `record.py` | Webcam -> keypoint clips (.npy) | Guided recorder: shows the sign, counts down, records about 2.5 s, checks quality, retries; saves keypoints only, never video |
| `check_data.py`, `qc.py` | Recorded clips -> report | Checks layout and hand tracking; removes weak takes, flags possible wrong signs, lists re-recordings |
| `landmarks.py` | MediaPipe result -> (63, 3) per frame; clip -> 32 x 378 | Selects 63 points; fills gaps; resamples to 32 frames; shoulder normalisation; velocity |
| `augment.py` | Clip -> varied clip | Speed change, rotation, scaling, shift, noise and hand drop-out, so the model sees more variety |
| `models.py`, `train.py` | Dataset -> trained model | BiGRU (main), Transformer and 1D-CNN; signer-independent and random splits; class-balanced loss |
| `evaluate.py`, `confusion.py` | Model + data -> metrics | Top-1, top-3, macro-F1, per-sign recall, confusion matrices and most-confused pairs |
| `export_onnx.py`, `inference.py` | PyTorch model -> ONNX | Export and check that ONNX output equals PyTorch output; fast CPU inference |
| `spotter.py` | Frame stream -> (sign, confidence) | Detects sign start and end from wrist motion; rejects IDLE and low confidence |
| `sentence.py`, `rules.json` | Gloss list -> English / Hindi sentence | Matches ISL patterns such as [PRONOUN, BODY, PAIN]; word-by-word fallback |
| `speech.py` | Text + language -> audio | pyttsx3 for English, gTTS for Hindi with a cache, speaks one sentence at a time |
| `app.py` | Webcam -> live UI and speech | Skeleton, status, FPS, top-3 confidence bars, signs, sentence; keys SPACE, u, c, l, s, f, q |

**Speaker notes**

"The code is a modular Python package. Each module has one job and its own unit tests, and the
vocabulary and sentence rules live in JSON files, so adding a sign does not need code changes."

---

## Slide 14: Implementation

Split into two slides if it does not fit.

**On the slide**

**A. Data collection and quality control**

- Guided recorder: 16 classes x 10 takes per signer, about 2.5 s per take, at 30 FPS.
- IDLE takes cover natural non-sign movements: resting hands, touching the face, adjusting hair,
  looking at a phone, shrugging.
- Only landmark coordinates are saved (`data/keypoints/<SIGN>/<signer>_<take>.npy`); signers are
  anonymous IDs (s01, s02, ...); written consent from every signer.
- Quality control rejects takes where a hand is tracked in fewer than 60% of frames, flags takes whose
  movement differs from the same signer's other takes (possibly a wrong sign), and lists every sign with
  fewer than 8 good takes for re-recording. Rejected clips are moved and logged, never deleted.
- Public INCLUDE-50 dataset (958 videos, 50 signs) converted to the same keypoint format, used to check
  the pipeline on real signing.

**B. Training**

- Augmentation: speed 0.8-1.2x, rotation ±15°, scale 0.9-1.1, shift, coordinate noise, hand drop-out.
- Optimiser AdamW (learning rate 0.001, weight decay 0.01) with cosine learning-rate decay.
- Cross-entropy with label smoothing 0.1; signs with fewer clips get a larger loss weight.
- Up to 120 epochs, batch size 64, early stopping after 25 epochs without improvement.
- Training clips are cut to the same window the live spotter would see, so training matches live use.

**C. Evaluation protocol**

- Signer-dependent: random split of clips (same people in training and test).
- Signer-independent: leave-one-signer-out (LOSO); train on 4 signers, test on the 5th, repeat 5 times.
- Final check on a 6th person (s06) recorded only for testing.

**D. Deployment and live app**

- Final model exported to ONNX; the export fails if ONNX and PyTorch outputs differ by more than 0.0001.
- Live app shows the skeleton, READY / SIGNING status, FPS, top-3 confidence bars, the signs and the sentence.
- Keys: SPACE speak, `u` undo, `c` clear, `l` English/Hindi, `s` skeleton, `f` full screen, `q` quit;
  optional auto-speak after a 2.5 s pause.

**E. Sentence generation example**

| Signed (ISL order) | Pattern matched | English | Hindi |
|---|---|---|---|
| ME STOMACH PAIN | [PRONOUN, BODY, PAIN] | My stomach hurts. | मेरे पेट में दर्द है। |
| ME NEED WATER | [PRONOUN, NEED, THING] | I need water. | मुझे पानी चाहिए। |
| DOCTOR CALL | [PERSON, CALL] | Please call the doctor. | कृपया डॉक्टर को बुलाइए। |

**Figure (optional):** a screenshot of the live app with the skeleton and a sentence on screen (only
with the consent of the person shown).

**Speaker notes**

"We built our own dataset because no public ISL dataset has these hospital signs recorded with an IDLE
class. Leave-one-signer-out is the key part of our evaluation: the test person is never in the training
data, which is exactly the situation in a real hospital."

---

## Slide 15: Results

Split into two slides: 15a (what is already verified) and 15b (measured accuracy and speed).

### 15a. Sentence generation (verified)

**On the slide**

| ID | Signed (ISL order) | English output | Hindi output |
|---|---|---|---|
| S1 | ME HEAD PAIN | My head hurts. | मेरे सिर में दर्द है। |
| S2 | ME STOMACH PAIN | My stomach hurts. | मेरे पेट में दर्द है। |
| S3 | ME FEVER | I have a fever. | मुझे बुखार है। |
| S4 | ME NEED WATER | I need water. | मुझे पानी चाहिए। |
| S5 | ME NEED MEDICINE | I need medicine. | मुझे दवा चाहिए। |
| S6 | ME NEED TOILET | I need to use the toilet. | मुझे शौचालय जाना है। |
| S7 | TOILET WHERE | Where is the toilet? | शौचालय कहाँ है? |
| S8 | DOCTOR CALL | Please call the doctor. | कृपया डॉक्टर को बुलाइए। |
| S9 | HELP | Please help me. | कृपया मेरी मदद कीजिए। |
| S10 | YES / NO | Yes. / No. | हाँ। / नहीं। |

- The sentence engine produces all 11 expected sentences exactly: **11/11 in English and 11/11 in Hindi**
  (automated tests `tests/test_sentence.py` and `tests/test_app.py`).
- Classifier size: **about 0.69 M parameters**.
- 67 automated unit tests cover landmarks, spotting, training, sentences, speech, recorder and app.
  Run `python -m pytest -q` on your laptop and quote the pass count.

### 15b. Accuracy and real-time performance (fill in from your runs)

**On the slide**

| Metric | Target | Measured |
|---|---|---|
| Top-1 accuracy, signer-dependent (random split) | >= 95% | [XX]% |
| Top-1 accuracy, signer-independent (LOSO mean ± std) | >= 80% | [XX] ± [XX]% |
| Unseen person s06, live (TC-12) | >= 42 / 60 | [XX] / 60 |
| False triggers while idle (TC-10) | <= 1 per minute | [XX] |
| Frame rate on the weakest laptop (TC-01) | >= 15 FPS | [XX] FPS |
| Hands stop -> text on screen (TC-14) | <= 500 ms | [XX] ms |
| Sentence correctness S1-S10 (TC-07, TC-08) | 10 / 10 | [XX] / 10 live; 11 / 11 automated |

Model comparison (Table R2), if you have time for a third results slide:

| Model | Random split top-1 | LOSO top-1 | LOSO macro-F1 | Parameters | CPU ms / sign |
|---|---|---|---|---|---|
| BiGRU + attention | [XX] | [XX] ± [XX] | [XX] | 0.69 M | [XX] |
| Transformer | [XX] | [XX] ± [XX] | [XX] | [XX] | [XX] |
| 1D-CNN | [XX] | [XX] ± [XX] | [XX] | [XX] | [XX] |

Also show the confusion matrix image from `signvaani.confusion` and name the two most confused sign
pairs with the reason (same hand shape, different location or movement).

Where each number comes from:

| Number | Command |
|---|---|
| Random split | `python -m signvaani.train --data data/keypoints --random-split` |
| LOSO | `python -m signvaani.train --data data/keypoints --loso` |
| Model comparison (R2) | `python -m signvaani.experiments compare --out docs/results/model_comparison.md` |
| Ablation (R4) | `python -m signvaani.experiments ablate --model gru --out docs/results/ablation.md` |
| Confusion matrix (R3, R6) | `python -m signvaani.confusion runs/loso_gru_<time>.json` |
| Unseen signer s06 | `python -m signvaani.evaluate --ckpt models/best.pt --data data/unseen --signers s06 --out docs/results/s06` |
| FPS and latency | `python -m signvaani.app --model models/sign_model.onnx --log runs/live_log.csv`, then read the lines printed on `q` |

**Speaker notes**

"Accuracy on the random split is higher than on unseen signers. That gap is a finding, not a failure:
different people have different hand sizes, speeds and styles. Leave-one-signer-out is the number that
matters for a real hospital, because every patient is a new person."

---

## Slide 16: Applications

**On the slide** (six short blocks, each with an icon)

1. **Hospitals and clinics:** patients tell doctors and nurses about pain, fever, medicine, water or
   the toilet without waiting for an interpreter.
2. **Public service counters:** banks, railway stations, post offices and government offices, where
   staff do not know ISL.
3. **Emergencies and help desks:** quick requests such as HELP or CALL DOCTOR, spoken aloud at once.
4. **Education:** deaf students and ISL learners practise signs and get instant feedback; hearing
   classmates and teachers can follow.
5. **Low-cost and offline settings:** runs on an ordinary laptop CPU with a webcam, with no internet
   for English, so it suits rural health centres.
6. **Research:** a privacy-preserving keypoint dataset and a signer-independent evaluation protocol
   that others can reuse and extend to new domains.

**Speaker notes**

"The same pipeline works for any domain: only the vocabulary file and the sentence rules change. The
tool supports human interpreters; it does not replace them in medical, legal or emergency decisions."

---

## Optional extra slide after slide 16: Conclusion and Future Scope

**On the slide**

**Conclusion**

- Built SignVaani, a complete webcam-to-speech pipeline for ISL that runs in real time on a laptop CPU.
- A compact representation of 63 hand, body and face landmarks with shoulder normalisation and velocity.
- Automatic sign spotting with an IDLE class, so the user signs naturally without pressing a button per sign.
- An ISL-order sentence engine produces correct English and Hindi sentences for all demo sentences.
- Honest, signer-independent evaluation, and a privacy-first design: no video is ever stored.

**Limitations**

- Isolated signs from one domain (15 signs + IDLE); a small number of signers.
- No full ISL grammar (spatial referencing, classifiers) and no fingerspelling.
- Hindi speech needs internet unless cached in advance.

**Future scope**

- Larger vocabulary and continuous sentence-level ISL translation, trained on datasets such as iSign [4].
- More Indian languages (Telugu, Kannada, Tamil) with IndicTrans2 [20] and offline Indic-TTS [21].
- Graph-based models such as ST-GCN [16] for better accuracy on similar signs.
- Android / mobile app and a reverse speech-to-sign avatar for two-way conversation.
- User-specific fine-tuning on a few samples from each new user.

---

## Slide 17: References

Follow the template's format. Split over two or three slides if needed.

[1] Sridhar, A., Ganesan, R.G., Kumar, P. and Khapra, M.M., "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition", Proceedings of the 28th ACM International Conference on Multimedia, 2020, DOI 10.1145/3394171.3413528.

[2] Joshi, A., Bhat, A., S, P., Gole, P., Gupta, S., Agarwal, S. and Modi, A., "CISLR: Corpus for Indian Sign Language Recognition", Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022, Page no.10357-10366.

[3] Joshi, A., Agrawal, S. and Modi, A., "ISLTranslate: Dataset for Translating Indian Sign Language", Findings of the Association for Computational Linguistics: ACL 2023, Page no.10466-10475.

[4] Joshi, A., Mohanty, R., Kanakanti, M., Mangla, A., Choudhary, S., Barbate, M. and Modi, A., "iSign: A Benchmark for Indian Sign Language Processing", Findings of the Association for Computational Linguistics: ACL 2024, Page no.10827-10844.

[5] Selvaraj, P., NC, G., Kumar, P. and Khapra, M.M., "OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained Models across Languages", Proceedings of the 60th Annual Meeting of the ACL, 2022, Page no.2114-2133.

[6] Elakkiya, R. and Natarajan, B., "ISL-CSLTR: Indian Sign Language Dataset for Continuous Sign Language Translation and Recognition", Mendeley Data, Version 1, 2021, DOI 10.17632/kcmpdxky7p.1.

[7] Ansari, Z.A. and Harit, G., "Nearest Neighbour Classification of Indian Sign Language Gestures using Kinect Camera", Sadhana, Volume 41, Issue 2, 2016, Page no.161-182.

[8] Koller, O., Forster, J. and Ney, H., "Continuous Sign Language Recognition: Towards Large Vocabulary Statistical Recognition Systems Handling Multiple Signers", Computer Vision and Image Understanding, Volume 141, 2015, Page no.108-125.

[9] Li, D., Rodriguez, C., Yu, X. and Li, H., "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison", Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV), 2020.

[10] Vaezi Joze, H.R. and Koller, O., "MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language", Proceedings of the British Machine Vision Conference (BMVC), 2019.

[11] Camgoz, N.C., Hadfield, S., Koller, O., Ney, H. and Bowden, R., "Neural Sign Language Translation", Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[12] Camgoz, N.C., Koller, O., Hadfield, S. and Bowden, R., "Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[13] Bohacek, M. and Hruz, M., "Sign Pose-based Transformer for Word-level Sign Language Recognition", Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops, 2022, Page no.182-191.

[14] Jiang, S., Sun, B., Wang, L., Bai, Y., Li, K. and Fu, Y., "Skeleton Aware Multi-modal Sign Language Recognition", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2021.

[15] Hu, H., Zhao, W., Zhou, W., Wang, Y. and Li, H., "SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition", Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[16] Yan, S., Xiong, Y. and Lin, D., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

[17] Lugaresi, C. et al., "MediaPipe: A Framework for Building Perception Pipelines", arXiv:1906.08172, 2019.

[18] Rastgoo, R., Kiani, K. and Escalera, S., "Sign Language Recognition: A Deep Survey", Expert Systems with Applications, Volume 164, 2021.

[19] Koller, O., "Quantitative Survey of the State of the Art in Sign Language Recognition", arXiv:2008.09918, 2020.

[20] Gala, J. et al., "IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages", Transactions on Machine Learning Research, 2023.

[21] Kumar, G.K., Praveen, S.V., Kumar, P., Khapra, M.M. and Nandakumar, K., "Towards Building Text-to-Speech Systems for the Next Billion Users", Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023.

Check the volume and page numbers on the DOI, ACL Anthology or CVF page before submission; where none
is given above, the publisher page has them.

---

## Slide 18: Thank You

**On the slide**

- THANK YOU
- (Optional, smaller) Questions?

**Speaker notes**

Be ready to show the live demo (3 minutes): sit still for 15 s to show no false signs, sign
ME STOMACH PAIN and press SPACE, press `l` and sign ME NEED WATER in Hindi, then have a teammate who
was not in the training data sign DOCTOR CALL. If the camera or room fails, play the recorded demo video.
