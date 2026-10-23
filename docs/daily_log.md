# Daily stand-up log

Blueprint Appendix B. Fill it in at each morning stand-up and commit it with the day's work. Examiners
often ask "what went wrong and how did you fix it?", and the Blockers column answers that.

## Day 1: setup, vocabulary, recorder, pilot data. Date: ________

| Member | Done yesterday | Plan today | Blockers | Exit criteria met? |
|---|---|---|---|---|
| A | - | Freeze vocabulary; learn the 15 signs from ISLRTC; confirm them with an ISL user; pilot recording s01 | | |
| B | - | Install check; test the recorder (NaN for missing hands, file layout); pilot recording; inspect pilot data (wrist paths, hands >= 90%) | | |
| C | - | Install check; test the recorder; pilot recording | | |
| D | - | Install check; SRS, architecture and DFD 0 diagrams; pilot recording | | |

Exit criteria: every machine prints OK in the install check; `data/keypoints` has 16 folders x 10 clips
for s01; vocabulary and sentence list frozen and committed.

## Day 2: full dataset and quality control. Date: ________

| Member | Done yesterday | Plan today | Blockers | Exit criteria met? |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |

Exit criteria: about 800 clips (5 signers x 16 classes x at least 8 good takes); dataset stats table and
datasheet committed; sentence unit tests pass for S1-S10.

## Day 3: baseline model and honest evaluation. Date: ________

| Member | Done yesterday | Plan today | Blockers | Exit criteria met? |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |

Exit criteria: baseline LOSO mean accuracy recorded with per-signer numbers; confused pairs identified
with a reason for each; if LOSO is below 60%, apply fallback F2/F3 before Day 4.

## Day 4: improve, compare, ablate, export. Date: ________

| Member | Done yesterday | Plan today | Blockers | Exit criteria met? |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |

Exit criteria: model comparison and ablation tables filled; `models/sign_model.onnx` and `.json`
committed (or stored if large) and the app runs from ONNX; at least 15 FPS on the weakest laptop,
otherwise fallback F5.

## Day 5: integration, sentence and speech. Date: ________

| Member | Done yesterday | Plan today | Blockers | Exit criteria met? |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |

Exit criteria: S1-S10 produce the correct English and Hindi sentence and are spoken aloud; offline
test-signer result recorded; code freeze tag v1.0 on GitHub.

## Day 6: testing with unseen users, results, demo video. Date: ________

| Member | Done yesterday | Plan today | Blockers | Exit criteria met? |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |

Exit criteria: all test cases executed with evidence; results tables contain only measured numbers; demo
video saved in two places.

## Day 7: report, presentation, viva rehearsal, submission. Date: ________

| Member | Done yesterday | Plan today | Blockers | Exit criteria met? |
|---|---|---|---|---|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |

Exit criteria: report, PPT, code, datasheet and demo video submitted; every member can explain every
module and number; live demo tested on the exact laptop, camera and room.
