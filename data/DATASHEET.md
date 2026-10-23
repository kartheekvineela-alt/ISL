# SignVaani ISL Keypoint Dataset — Datasheet

Fill this in on Day 2 (see the 1-week blueprint, Section 7).

| Field | Value |
|---|---|
| Domain | Hospital communication |
| Classes | 15 signs + IDLE (see `configs/vocabulary.json`) |
| Signers | [XX] (IDs s01–s0X; no names stored) |
| Takes per class per signer | [XX] |
| Total clips | [XX] |
| Format | `data/keypoints/<SIGN>/<signer>_<take>.npy`, float32 array (T, 63, 3), NaN = not detected |
| Points | 9 pose + 12 face + 21 left hand + 21 right hand (MediaPipe Holistic) |
| Frame rate | ~30 FPS, ~2.5 s per take |
| Sign source | ISLRTC ISL dictionary; verified by [name / organisation] |
| Recording conditions | [rooms, lighting, backgrounds, camera model, distance] |
| Consent | Written consent from every signer, stored offline by the team |
| Video stored? | No — only landmark coordinates |
| Licence | [e.g. CC BY 4.0, only if every signer agreed to public release] |
| Known limitations | [regional variants, small number of signers, ...] |
