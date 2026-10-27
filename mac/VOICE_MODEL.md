# 2026-10-05 RVC voice model

Model weights and the retrieval index are published as GitHub Release assets:

https://github.com/olivaciiiiigar-stack/AudioSep/releases/tag/voice-zyc-20261005

Download the desired `.pth` checkpoint and `zyc_20261005_A.index` together.
The release includes checkpoints at 75, 150 and 200 epochs. The same index
applies to all three checkpoints; it is built from the training features,
not from a particular epoch's generator weights. `SHA256SUMS.txt` and
`model_manifest.json` describe and verify the files.

## Training recipe

- RVC v2 / HiFi-GAN; official `f0G40k.pth` and `f0D40k.pth` initialization.
- Sample rate: 40000 Hz; content encoder: `chinese-hubert-base` (768 dimensions).
- Pitch extraction: RMVPE. Single speaker, ID 0.
- Batch size: 8; learning rate: 1e-4; epoch decay: 0.999875.
- 666 prepared clips from the 2026-10-05 recordings, plus 2 preset mute entries.
- Whole recording `恋人.wav` excluded from training and index construction.
- Preprocessing: whole-file normalization, automatic slicing, 3-second chunks
  with 0.3-second overlap; no training denoising or dereverberation.
- Final checkpoint: 200 epochs / 17000 steps, completed 2026-10-06.

## Inference

Use `chinese-hubert-base` and RMVPE, with retrieval strength 0.3 as the
listened-to baseline. Disable additional reverb. Match the pitch setting to
the intended song key; the `我不难过` experiment uses -4 semitones and shifts
the accompaniment and backing vocals by the same amount at constant tempo.

Place the selected weights and index in an Applio experiment directory, or
pass their explicit paths to `mac/cover.py:rvc_infer`. For model discovery
under `models/`, use `models/<name>/<name>.pth` and `<name>.index`.

## Experiment scripts

`train_cover_1005.py`, `preview_1005_75.py`, `preview_1005_index05.py` and
`cover_wobunanguo_150.py` reproduce the local experiments. These scripts
contain the original machine's absolute Applio/source paths. Update those
paths for another machine; they are not portable general-purpose CLIs.
`audit_training_1005.py` reads and audits the prepared dataset and old outputs.

The index=0.5 experiment's Python source includes a later requested mix change
(instrumental and backing gain 0.675, about 2 dB below the original 0.85).
The source was edited while its first process was running, so that existing
audio output still uses the original gains. Rerun the updated script to remix
the existing converted vocals with the lower accompaniment level.

Separation still needs listening comparisons: the current three-stage chain
can reduce quiet consonants and tails. These weights do not by themselves
restore information removed during separation. No claim that 200 epochs
sounds better than the retained 75/150-epoch checkpoints is made.

Original recordings, copyrighted songs, generated covers, optimizer states,
base pretrained models and downloaded Seed-VC models are not included.
