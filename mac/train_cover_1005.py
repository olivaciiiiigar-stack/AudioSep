"""2026-10-05 A experiment: isolated dataset, 200 epochs, then cover."""
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
APPLIO = Path('/Users/zhangyichi/Workspace/Applio')
NAME = 'zyc_20261005_A'
OUT = ROOT / 'output' / '那天下雨了_1005_A_200e'


def main():
    import trainer
    import cover
    from main import separate_track
    import soundfile as sf
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'model': NAME, 'epochs_requested': 200, 'sample_rate': 40000,
             'embedder': 'chinese-hubert-base', 'f0': 'rmvpe',
             'holdout': '恋人.wav', 'index_rate': .3, 'pitch': 0,
             'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    log_file = (OUT / 'pipeline.log').open('a', buffering=1)
    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        log_file.write(line + '\n')
    def phase(value):
        state['phase'] = value
        (OUT / 'run.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
        log('阶段: ' + value)
    try:
        phase('prepare')
        exp = APPLIO / 'logs' / NAME
        source = ROOT / 'data/preprocessed/voice_20261005_40k'
        if not exp.exists():
            exp.mkdir()
            for folder in ('sliced_audios', 'sliced_audios_16k'):
                (exp / folder).mkdir()
                for p in sorted((source / folder).glob('*.wav')):
                    if p.name.split('_')[1] != '9':
                        shutil.copy2(p, exp / folder / p.name)
            shutil.copy2(source / 'model_info.json', exp / 'model_info.json')
        clips = list((exp / 'sliced_audios').glob('*.wav'))
        assert len(clips) == 666, f'unexpected dataset count: {len(clips)}'
        assert not any(p.name.split('_')[1] == '9' for p in clips)
        info = json.loads((exp / 'model_info.json').read_text())
        import subprocess
        holdout_seconds = float(subprocess.check_output([
            '/opt/homebrew/bin/ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            str(ROOT / 'data/original_voice/voice_20261005/恋人.wav')]))
        info['total_seconds'] = 2039.3066 - holdout_seconds
        info['total_dataset_duration'] = time.strftime('%H:%M:%S', time.gmtime(info['total_seconds']))
        (exp / 'model_info.json').write_text(json.dumps(info, indent=2))
        state['training_clips'] = len(clips)
        trainer.ensure_prerequisites(str(APPLIO), log=log)
        if not (exp / 'filelist.txt').exists():
            phase('extract')
            trainer.extract(NAME, 40000, str(APPLIO), cpu_cores=4, log=log)
        lines = (exp / 'filelist.txt').read_text().splitlines()
        real_rows = [row for row in lines if f'logs/{NAME}/' in row]
        assert len(real_rows) == len(clips), f'incomplete extraction: {len(real_rows)}'
        for row in lines:
            wav, feat, coarse, continuous, sid = row.split('|')
            paths = [APPLIO / p for p in (wav, feat, coarse, continuous)]
            assert all(p.is_file() for p in paths), f'Missing training pair: {row}'
            features, bins, pitch = [np.load(p, allow_pickle=False) for p in paths[1:]]
            assert features.ndim == 2 and features.shape[1] == 768, str(paths[1])
            assert all(np.isfinite(a).all() for a in (features, bins, pitch)), row
            assert len(bins) == len(pitch) and abs(2*len(features)-len(pitch)) <= 4, row
            assert sf.info(paths[0]).samplerate == 40000, wav
        for kind in ('G', 'D'):
            assert (APPLIO / f'rvc/models/pretraineds/hifi-gan/f0{kind}40k.pth').is_file()
        config = json.loads((exp / 'config.json').read_text())
        assert config['train']['learning_rate'] == .0001, config['train']
        state['learning_rate'] = config['train']['learning_rate']
        phase('train')
        trainer.train(NAME, 200, 40000, str(APPLIO), batch_size=8,
                      save_every_epoch=25, log=log)
        weights = list(exp.glob(f'{NAME}_200e_*s.pth'))
        assert weights, 'Training ended without a 200-epoch model; cover cancelled'
        pth = max(weights, key=lambda p: p.stat().st_mtime)
        index = exp / f'{NAME}.index'
        assert index.exists(), 'Missing retrieval index'
        model_out = OUT / 'model'
        model_out.mkdir(exist_ok=True)
        for p in (pth, index, exp / 'config.json'):
            shutil.copy2(p, model_out / p.name)
        state['weights'] = str(pth)
        phase('separate')
        stems = OUT / 'stems'
        separate_track(str(ROOT / 'data/audio/周杰伦-那天下雨了.flac'), str(stems), log=log, fast=False)
        phase('infer')
        dry = OUT / '翻唱干声_200e.flac'
        cover.rvc_infer(str(stems / '2_纯主唱_干声_Dry_Vocals.flac'), str(dry),
                        str(pth), str(index), reverb=False, applio_dir=str(APPLIO), log=log)
        phase('mix')
        # Float intermediate preserves peaks; apply only global attenuation if required.
        intermediate = OUT / 'mix_float.wav'
        subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-v', 'error',
                        '-i', str(stems / '1_纯伴奏_Instrumental.flac'),
                        '-i', str(dry), '-i', str(stems / '3_纯和声_Backing_Vocals.flac'),
                        '-filter_complex', '[0:a]aresample=44100,volume=0.85[a];[1:a]aresample=44100,volume=1.3[b];[2:a]aresample=44100,volume=0.85[c];[a][b][c]amix=inputs=3:duration=longest:normalize=0',
                        '-c:a', 'pcm_f32le', str(intermediate)], check=True)
        audio, rate = sf.read(intermediate, dtype='float32', always_2d=True)
        assert np.isfinite(audio).all()
        peak = float(np.abs(audio).max())
        gain = min(1., 10**(-1/20) / max(peak, 1e-9))
        final = OUT / '那天下雨了_翻唱_A_200e.flac'
        sf.write(final, audio * gain, rate, subtype='PCM_24')
        intermediate.unlink()
        state.update(final=str(final), duration_seconds=len(audio)/rate,
                     global_gain=gain, final_peak_dbfs=float(20*np.log10(max(peak*gain,1e-9))))
        phase('complete')
    except KeyboardInterrupt:
        phase('paused')
        raise
    except BaseException:
        state['error'] = traceback.format_exc()
        phase('failed')
        raise


if __name__ == '__main__':
    main()
