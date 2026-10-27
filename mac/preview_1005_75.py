"""Render saved epoch 75 without interfering with the ongoing experiment."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import cover

ROOT = Path(__file__).resolve().parents[1]
APPLIO = Path('/Users/zhangyichi/Workspace/Applio')
NAME = 'zyc_20261005_A'
OUT = ROOT / 'output/那天下雨了_1005_A_75e试听'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'epoch': 75, 'index_rate': .3, 'pitch': 0, 'reverb': False,
             'stems_source': str(ROOT / 'output/那天下雨了'),
             'note': 'Use previously separated stems for fast model comparison; 200e pipeline will re-separate.'}
    log_file = (OUT / 'preview.log').open('a', buffering=1)
    def log(message):
        line = time.strftime('%H:%M:%S ') + message
        print(line, flush=True)
        log_file.write(line+'\n')
    def phase(p):
        state['phase'] = p
        (OUT / 'preview.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
        log(p)
    try:
        exp = APPLIO / 'logs' / NAME
        weights = list(exp.glob(f'{NAME}_75e_*s.pth'))
        assert len(weights) == 1
        model_dir = OUT / NAME
        model_dir.mkdir(exist_ok=True)
        pth = model_dir / weights[0].name
        if not pth.exists():
            shutil.copy2(weights[0], pth)
        feature_link = model_dir / 'extracted'
        if not feature_link.exists():
            feature_link.symlink_to(exp / 'extracted', target_is_directory=True)
        phase('index')
        env = dict(os.environ, OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        proc = subprocess.Popen([str(APPLIO / '.venv/bin/python'),
            str(APPLIO / 'rvc/train/process/extract_index.py'), str(model_dir), 'Auto'],
            cwd=APPLIO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log(line.rstrip())
        assert proc.wait() == 0, 'Index generation failed'
        index = model_dir / f'{NAME}.index'
        assert index.is_file()
        phase('infer')
        dry = OUT / '翻唱干声_A_75e.flac'
        stems = ROOT / 'output/那天下雨了'
        cover.rvc_infer(str(stems / '2_纯主唱_干声_Dry_Vocals.flac'), str(dry),
                        str(pth), str(index), reverb=False, applio_dir=str(APPLIO), log=log)
        phase('mix')
        temp = OUT / 'mix_float.wav'
        subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-v', 'error',
            '-i', str(stems / '1_纯伴奏_Instrumental.flac'), '-i', str(dry),
            '-i', str(stems / '3_纯和声_Backing_Vocals.flac'),
            '-filter_complex', '[0:a]aresample=44100,volume=0.85[a];[1:a]aresample=44100,volume=1.3[b];[2:a]aresample=44100,volume=0.85[c];[a][b][c]amix=inputs=3:duration=longest:normalize=0',
            '-c:a', 'pcm_f32le', str(temp)], check=True)
        samples, rate = sf.read(temp, dtype='float32', always_2d=True)
        assert np.isfinite(samples).all()
        peak = float(np.max(np.abs(samples)))
        gain = min(1., 10**(-1/20) / max(peak, 1e-9))
        final = OUT / '那天下雨了_翻唱_A_75e.flac'
        sf.write(final, samples*gain, rate, subtype='PCM_24')
        temp.unlink()
        state.update(final=str(final), weights=str(pth), index=str(index),
                     global_gain=gain, duration_seconds=len(samples)/rate,
                     peak_dbfs=float(20*np.log10(max(peak*gain,1e-9))))
        # A short player-friendly excerpt, while keeping the complete lossless result.
        excerpt = OUT / '试听片段_30秒.mp3'
        subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-v', 'error', '-ss', '45',
                        '-i', str(final), '-t', '30', '-c:a', 'libmp3lame',
                        '-b:a', '192k', str(excerpt)], check=True)
        phase('complete')
    except BaseException as e:
        state['error'] = repr(e)
        phase('failed')
        raise


if __name__ == '__main__':
    main()
