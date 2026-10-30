# System test cases

Blueprint Section 8.2. Written on Day 4 (D), executed on Day 6 (A, D) with evidence. Run every case on
the **weakest laptop** of the team, in the room of the review, with the final model:

```bash
python -m signvaani.app --model models/sign_model.onnx --log runs/live_log.csv
```

Fill in **Actual** with what was measured or seen, **P/F** with pass or fail, and **Evidence** with the
screenshot or file name (keep screenshots in `docs/evidence/`, never with a face that has not consented).
**Automated** names the unit test that covers the logic behind the case; it does not replace the live run.

| ID | Req. | Test | How to run | Expected | Automated | Actual | P/F | Evidence |
|---|---|---|---|---|---|---|---|---|
| TC-01 | FR-1, NFR-1 | Frame rate | Run the app for 60 s while signing, press `q`, read the "Frame rate" line | At least 15 FPS | - | | | |
| TC-02 | FR-2 | Skeleton | Stand 1.5 m from the camera and raise both hands | Skeleton on the body and on both hands | - | | | |
| TC-03 | FR-3 | Short tracking loss | Hide one hand for about 0.1 s in the middle of a sign | The sign is still recognised | `test_short_gap_interpolated_long_gap_zeroed` | | | |
| TC-04 | FR-11 | Recorder rejects bad takes | `python -m signvaani.record --signer s01 --signs PAIN --takes 1 --out runs/tc04`, keep both hands out of view | The take is rejected and retried | `test_record` | | | |
| TC-05 | FR-5 | Each sign | Perform each of the 15 signs once | Correct label with its confidence bar | - | | | |
| TC-06 | FR-12 | LOSO training | `python -m signvaani.train --data data/keypoints --loso` | Per-signer accuracy and mean ± std printed and saved in `runs/loso_*.json` | `test_training` | | | |
| TC-07 | FR-8 | Sentences, English | Sign S1-S10 (`configs/sentences.json`), press SPACE after each | The exact English sentences | `test_demo_sentences_s1_to_s10` | | | |
| TC-08 | FR-8 | Sentences, Hindi | Press `l`, sign S1-S10 again | The exact Hindi sentences | `test_demo_sentences_s1_to_s10` | | | |
| TC-09 | FR-4 | Sign spotting | Sign 3 signs with short pauses (hands down between signs) | Exactly 3 signs on the signs line | `test_detects_one_sign_from_motion` | | | |
| TC-10 | FR-6 | No false signs | Sit naturally without signing for 3 minutes; count the signs added | At most 3 | `test_no_motion_no_classification` | | | |
| TC-11 | FR-6 | Unknown sign | Perform a sign that is not in the vocabulary | "not added: below 80%" or "IDLE" | `test_idle_and_low_confidence_rejected` | | | |
| TC-12 | NFR-5 | Unseen person | Signer s06 (not in training) makes 60 sign attempts, 4 per sign | At least 42 of 60 correct (70%) | - | | | |
| TC-13 | FR-7 | Overlay | Look at the window while signing | Status, motion, FPS, top-3 bars, added / not added, signs and sentence visible | `test_app` | | | |
| TC-14 | NFR-1 | Latency | Film the screen and hands at 60 FPS; count frames from hands down to text on screen. Also read the app's latency line on quit | At most 500 ms | `test_quiet_since_marks_when_the_hands_stopped` | | | |
| TC-15 | FR-9 | Speech | Sign S4, press SPACE; then run with `--auto-speak` and wait 2.5 s after S4 | The sentence is spoken in the chosen language | - | | | |
| TC-16 | FR-10 | Keys | Press `u`, `c`, `l` after some signs | Undo the last sign, clear, switch language | `test_app` | | | |
| TC-17 | NFR-2 | Offline | Disconnect the internet, run the app in English | Recognition and English speech work | - | | | |
| TC-18 | NFR-3 | Privacy | After a recording and a live session, search the project for `.mp4`, `.avi`, `.mov`, `.jpg`, `.png` written by the app or recorder | No video or image files from the camera | - | | | |

## Notes on measuring

- **TC-01:** the number to report is the "Frame rate" line printed when you press `q`, the average
  over the whole run, not the changing value on screen. Below 15 FPS, apply fallback F5: run with
  `--complexity 0` and close other apps, then measure again.
- **TC-14:** each accepted sign prints `latency ... ms`, the time from the frame where the hands stopped
  moving to the sign being accepted, and `--log` saves it per sign. It starts when the motion value falls
  below `--off`, a frame or two after the hands really stop, and leaves out the camera's own delay, so it
  is a little lower than what the camera sees. The filmed measurement is the official one; report both.
- **TC-12:** s06 must not be in `data/keypoints` when the final model is trained. Count a sign as correct
  only if it appears on the signs line, not if it is only in the top-3 bars.
- **TC-10:** count only signs added to the signs line. "not added" lines are correct rejections.

## Summary for the report (Table R5)

| FPS (weakest laptop) | Latency, app (median / worst) | Latency, filmed (TC-14) | False triggers / min (TC-10) | Unseen person (TC-12) | Sentences (TC-07) |
|---|---|---|---|---|---|
| [XX] | [XX] / [XX] ms | [XX] ms | [XX] | [XX] / 60 | [XX] / 10 |
