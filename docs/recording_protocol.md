# Recording protocol and consent

Blueprint Appendix A. Use it for the Day 1 pilot (s01) and the Day 2 sessions (s02-s05).
**Never record anyone without a signed consent form.**

## Before the session

- [ ] The participant has read and signed the consent form below. Give them the next free ID (s01, s02,
      ...) and write it on the form. Names never go into file names, folders, commits or chats.
- [ ] The signs follow the variants in `docs/sign_verification.md`. Let the signer practise each one.
- [ ] Camera at chest height, 1-2 m away, upper body and both hands fully in frame, light from the
      front, plain background if possible.
- [ ] `python -m signvaani.check_install --camera 0` prints OK (at least 15 FPS).

## Recording

```bash
python -m signvaani.record --signer s01        # 16 signs x 10 takes, about 30 minutes
```

- Each take is a 3 s countdown, then 2.5 s of capture. Start from rest, perform one sign naturally,
  return to rest.
- **IDLE** takes: do not sign. The recorder prompts natural movements: rest your hands, touch your face,
  adjust your hair, look at your phone, small shrug.
- The recorder rejects a take if the shoulders are out of view in more than 10% of the frames, or (for
  every sign except IDLE) the hands are visible in under 60% of the frames, and then retries.
- Keys: `r` throws away the take just saved (within 1 s) and records it again; `k` skips a sign; `q`
  quits.
- Running the same command again resumes. Saved takes are kept and only missing ones are recorded.
- Between signers, vary clothing colour, background, room, time of day and distance. Include
  left-handed signers if available.

## After the session

```bash
python -m signvaani.check_data --signers s01
python -m signvaani.check_data --signers s01 --plot PAIN HEAD WATER    # Day 1: wrist paths in runs/pilot/
```

- **[FAIL] lines** must be fixed: a wrong folder or file name, a broken file, or missing takes.
- **[WARN] hands** means the hands were often not tracked. Fix the lighting or camera position and
  re-record that sign.
- Delete takes with a wrong sign, two signs, or a blocked camera, and log them below. Then run the
  recorder again for that sign (`--signs PAIN`); it tops the sign back up to 10 takes. A sign with fewer
  than 8 good takes must be re-recorded (fallback F3).
- `data/keypoints/` is git-ignored. Keypoints stay off GitHub unless every signer agreed to public
  release.

### Rejected takes log

| Date | File | Reason |
|---|---|---|
| | | |

## Consent form

Print one copy per participant. Signed forms carry names: keep them offline with the team and never
commit, photograph or upload them.

> **SignVaani: participant consent**
>
> Project: SignVaani, real-time Indian Sign Language to speech (B.Tech CSE mini-project)
>
> Institution: ____________________   Guide: ____________________
>
> I, ____________________, agree to take part in the SignVaani mini-project at ____________________.
>
> I understand that:
>
> - the recording tool stores only body-landmark coordinates (points on my hands, arms, shoulders and
>   face), and no video or photographs;
> - my data will be identified only by a code (s__), never by my name;
> - movement data is not perfectly anonymous, so the team keeps it private unless I agree below;
> - my data will be used for academic purposes: training and testing the model, the project report and
>   the presentation;
> - taking part is voluntary, and I may withdraw at any time before submission by contacting the team,
>   and my recordings will then be deleted.
>
> I agree that my data may be released publicly in anonymised keypoint form:  [ ] Yes   [ ] No
>
> Signature: ____________________   Date: __________   Signer ID: s___
>
> Team contact: ____________________

## Consent register (no names)

Feeds the consent and licence rows of `data/DATASHEET.md`. Public release is possible only if every
signer ticked Yes.

| Signer ID | Consent date | Public release (Yes / No) | Signed form kept by |
|---|---|---|---|
| s01 | | | |
| s02 | | | |
| s03 | | | |
| s04 | | | |
| s05 | | | |
| s06 (test only) | | | |
