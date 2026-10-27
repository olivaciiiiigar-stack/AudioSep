"""Run the independent Seed-VC 44.1 kHz singing model on prepared dry vocals."""

import argparse
from pathlib import Path

from procutil import run_streaming

ROOT = Path(__file__).resolve().parent.parent


def convert(source, reference, output, steps=30, pitch=0):
    upstream = ROOT / "tools" / "seed-vc"
    python = upstream / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"Seed-VC environment missing: {python}")
    source, reference = Path(source).resolve(), Path(reference).resolve()
    for path in (source, reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected = output / f"vc_{source.name.split('.')[0]}_{reference.name.split('.')[0]}_1.0_{steps}_0.7.wav"
    before = expected.stat().st_mtime_ns if expected.exists() else None
    command = [str(python), "inference.py", "--source", str(source),
               "--target", str(reference), "--output", str(output),
               "--diffusion-steps", str(steps), "--f0-condition", "True",
               "--auto-f0-adjust", "False", "--semi-tone-shift", str(pitch),
               "--fp16", "False"]
    run_streaming(command, cwd=str(upstream), env_extra={
        "HF_HOME": "./checkpoints/hf_home", "HF_HUB_DISABLE_XET": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    })
    if not expected.exists() or expected.stat().st_size <= 44 or expected.stat().st_mtime_ns == before:
        raise RuntimeError("Seed-VC did not produce a new audio file")
    return str(expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Separated dry singing vocals")
    parser.add_argument("--reference", required=True, help="Clean target voice, ideally 10–20 seconds")
    parser.add_argument("--output", default="output/seedvc")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--pitch", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    print(convert(args.source, args.reference, args.output, args.steps, args.pitch))
