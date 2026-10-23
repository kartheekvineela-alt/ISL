# Sign verification sheet

Owner: A (Data lead). Day 1 plan:

- **09:00-11:00 (A, D):** learn every sign from the ISLRTC ISL dictionary videos
  ([islrtc.nic.in](https://islrtc.nic.in), also on ISLRTC's official YouTube channel) and practise it.
- **14:00-16:00 (A):** ask an ISL user or a teacher of the deaf to confirm each sign. A phone or video call
  is fine.

Rules:

- Choose one variant per sign: the one the ISL user confirms. Every signer performs that variant.
- After watching the reference, describe the sign in your own words: one or two hands, handshape, where
  it is made, and the movement. The report appendix uses these descriptions, and the report must note
  the exact variant used.
- Do not add or rename signs. The vocabulary is frozen in `configs/vocabulary.json`.

## Signs

| # | Sign | Group | ISLRTC reference (link) | Hands (1 / 2) | Description: handshape, location, movement | Confirmed by (name or role, with permission) | Date | OK? |
|---|---|---|---|---|---|---|---|---|
| 1 | ME | People | | | | | | |
| 2 | DOCTOR | People | | | | | | |
| 3 | PAIN | Symptoms | | | | | | |
| 4 | FEVER | Symptoms | | | | | | |
| 5 | HEAD | Body parts | | | | | | |
| 6 | STOMACH | Body parts | | | | | | |
| 7 | WATER | Needs | | | | | | |
| 8 | MEDICINE | Needs | | | | | | |
| 9 | TOILET | Needs | | | | | | |
| 10 | HELP | Needs | | | | | | |
| 11 | NEED | Needs | | | | | | |
| 12 | CALL | Actions / questions | | | | | | |
| 13 | WHERE | Actions / questions | | | | | | |
| 14 | YES | Answers | | | | | | |
| 15 | NO | Answers | | | | | | |
| 16 | IDLE | Non-sign | not a sign | - | Hands resting; natural movements: touching the face, adjusting hair, looking at a phone, small shrugs | - | - | - |

Signs that the ISL user finds easy to confuse with each other (note them now; Day 3 checks the confusion
matrix against this list):

| Sign pair | Why they look alike |
|---|---|
| | |

## Hindi sentences

The demo sentences are frozen in `configs/sentences.json`. A native Hindi speaker checks each one before
Day 5.

| # | Signs (ISL order) | English | Hindi | Checked by | OK? |
|---|---|---|---|---|---|
| S1 | ME HEAD PAIN | My head hurts. | मेरे सिर में दर्द है। | | |
| S2 | ME STOMACH PAIN | My stomach hurts. | मेरे पेट में दर्द है। | | |
| S3 | ME FEVER | I have a fever. | मुझे बुखार है। | | |
| S4 | ME NEED WATER | I need water. | मुझे पानी चाहिए। | | |
| S5 | ME NEED MEDICINE | I need medicine. | मुझे दवा चाहिए। | | |
| S6 | ME NEED TOILET | I need to use the toilet. | मुझे शौचालय जाना है। | | |
| S7 | TOILET WHERE | Where is the toilet? | शौचालय कहाँ है? | | |
| S8 | DOCTOR CALL | Please call the doctor. | कृपया डॉक्टर को बुलाइए। | | |
| S9 | HELP | Please help me. | कृपया मेरी मदद कीजिए। | | |
| S10 | YES / NO | Yes. / No. | हाँ। / नहीं। | | |

## Message to the ISL user (template)

> Hello, we are B.Tech CSE students at [college]. For our mini-project, SignVaani, we are building a
> tool that recognises 15 Indian Sign Language signs used in a hospital (ME, DOCTOR, PAIN, FEVER, HEAD,
> STOMACH, WATER, MEDICINE, TOILET, HELP, NEED, CALL, WHERE, YES, NO) and speaks them aloud. Could you
> spare about 20 minutes on a video call to check that we perform these signs correctly? We will show
> each sign once and will not record the call. Thank you!
