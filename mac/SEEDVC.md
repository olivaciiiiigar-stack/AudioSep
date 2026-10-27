# Seed-VC singing conversion experiment

Independent checkout: `tools/seed-vc`, upstream
https://github.com/Plachtaa/seed-vc at
`51383efd921027683c89e5348211d93ff12ac2a8` (archived upstream).
Python 3.12.13; installed versions recorded in `seedvc-requirements.lock.txt`.
This environment is separate from AudioSep and Applio.

To recreate from the project root:

```sh
git clone https://github.com/Plachtaa/seed-vc.git tools/seed-vc
git -C tools/seed-vc checkout 51383efd921027683c89e5348211d93ff12ac2a8
uv venv tools/seed-vc/.venv --python 3.12
uv pip sync --python tools/seed-vc/.venv/bin/python mac/seedvc-requirements.lock.txt
git -C tools/seed-vc apply ../../mac/seedvc-mps.patch
```

Mac compatibility patch: `seedvc-mps.patch` converts RMVPE NumPy pitch arrays
to float32 before transferring to MPS, which does not support float64.
The patch is already applied to the local checkout.
It also runs BigVGAN on CPU when diffusion uses MPS: torch 2.5.1 rejects its
long grouped transposed convolution on MPS even with fallback enabled.

Use prepared dry vocals and a clean 10–20 second target recording:

```sh
uv run mac/seedvc.py \
  --source 'output/seedvc_谁稀罕/stems/2_纯主唱_干声_Dry_Vocals.flac' \
  --reference 'output/seedvc_谁稀罕/我的参考干声.wav' \
  --output 'output/seedvc_谁稀罕/converted' \
  --steps 30 --pitch 0
```

The CLI selects the 44.1 kHz F0-conditioned singing model, keeps source pitch,
and disables automatic pitch adjustment. FP16 diffusion is disabled for this
initial Mac test. Upstream selects MPS when available, otherwise CPU.
First run downloads checkpoints into the checkout's `checkpoints/` directory.
Voice recordings are processed locally.

Test preparation: original song seconds 60–90, three-stage high-quality
separation; reference seconds 80–95 of the new recording exported from Audacity.
The reference is an independently sung excerpt, not necessarily aligned to the
original. No reference denoising or dereverberation was applied.

Assess dry conversion first: intelligibility, sustained notes, artifacts and
target voice resemblance. Waveform validation alone does not establish quality.
Original RVC checkpoints are incompatible with Seed-VC.

## First local run, 2026-10-05

Succeeded on Apple Silicon using MPS diffusion and CPU BigVGAN. Conversion
time (excluding model loading/downloads): approximately 287 seconds for
30 seconds, RTF 9.57. Output 29.9886 seconds, mono 44.1 kHz float WAV;
peak -0.85 dBFS, finite samples, no samples at or above 0.999 amplitude.
This establishes execution and basic file validity, not perceptual quality.

Listening copies in `output/seedvc_谁稀罕/`: original and converted dry vocals
matched by whole-clip RMS using constant gain only (not exact perceived
loudness); converted vocals mixed with original accompaniment and backing
vocals, with a global gain reduction to leave 1 dB peak headroom. No additional
reverb, denoising, EQ or limiting. Parameters and gains in `test_manifest.json`.

Upstream logs skipped checkpoint keys due to shape mismatch during loading;
this is unmodified loader behavior and should be reviewed before treating
the integration as production-ready. No claim of superiority over RVC has
been established by this test.
