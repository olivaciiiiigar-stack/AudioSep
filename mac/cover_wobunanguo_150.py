"""150-epoch cover, all stems transposed -4 semitones at constant tempo."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np
import soundfile as sf
import cover
from main import separate_track

ROOT = Path(__file__).resolve().parents[1]
APPLIO = Path('/Users/zhangyichi/Workspace/Applio')
NAME = 'zyc_20261005_A'
OUT = ROOT / 'output/我不难过_1005_A_150e_降4调'
SOURCE = Path('/Users/zhangyichi/Music/QQ音乐/孙燕姿-我不难过.flac')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'epoch':150, 'pitch_semitones':-4, 'index_rate':.3,
             'reverb':False, 'source':str(SOURCE), 'accompaniment_transposed':True}
    log_file = (OUT/'pipeline.log').open('a', buffering=1)
    def log(message):
        line = time.strftime('%H:%M:%S ') + message
        print(line, flush=True)
        log_file.write(line+'\n')
    def phase(value):
        state['phase'] = value
        (OUT/'run.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
        log(value)
    try:
        model = OUT/'model'
        model.mkdir(exist_ok=True)
        weights = list((APPLIO/'logs'/NAME).glob(f'{NAME}_150e_*s.pth'))
        assert len(weights)==1
        pth=model/weights[0].name
        shutil.copy2(weights[0],pth)
        old_index=ROOT/'output/那天下雨了_1005_A_75e试听'/NAME/f'{NAME}.index'
        assert old_index.is_file()
        index=model/f'{NAME}.index'
        if not index.exists():
            shutil.copy2(old_index,index)
        stems=OUT/'stems'
        expected=['1_纯伴奏_Instrumental.flac','2_纯主唱_干声_Dry_Vocals.flac','3_纯和声_Backing_Vocals.flac']
        if not all((stems/f).is_file() for f in expected):
            phase('separate')
            groups=set()
            for line in subprocess.check_output(['ps','-axo','pid,ppid,pgid,command'],text=True).splitlines():
                fields=line.split(None,3)
                if len(fields)==4 and fields[3].startswith(str(APPLIO/'.venv/bin/python')+' core.py train --model-name '+NAME+' '):
                    groups.add(int(fields[2]))
            try:
                for pgid in groups:
                    os.killpg(pgid,signal.SIGSTOP)
                    log(f'暂时暂停训练进程组 {pgid}')
                separate_track(str(SOURCE),str(stems),log=log,fast=False)
                assert all((stems/f).is_file() for f in expected)
            finally:
                for pgid in groups:
                    try:
                        os.killpg(pgid,signal.SIGCONT)
                        log(f'已恢复训练进程组 {pgid}')
                    except ProcessLookupError:
                        pass
        phase('infer')
        dry=OUT/'翻唱干声_A_150e_降4调.flac'
        if not dry.exists():
            cover.rvc_infer(str(stems/expected[1]),str(dry),str(pth),str(index),
                            pitch=-4,reverb=False,applio_dir=str(APPLIO),log=log)
        phase('transpose_accompaniment')
        import librosa
        shifted=[]
        for filename, dest in [(expected[0],'伴奏_降4调.wav'),(expected[2],'和声_降4调.wav')]:
            out=OUT/dest
            if not out.exists():
                data,rate=sf.read(stems/filename,dtype='float32',always_2d=True)
                transposed=librosa.effects.pitch_shift(data.T,sr=rate,n_steps=-4).T
                assert np.isfinite(transposed).all() and len(transposed)==len(data)
                sf.write(out,transposed,rate,subtype='FLOAT')
            shifted.append(out)
            log('已降调: '+dest)
        phase('mix')
        temp=OUT/'mix_float.wav'
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error',
            '-i',str(shifted[0]),'-i',str(dry),'-i',str(shifted[1]),
            '-filter_complex','[0:a]aresample=44100,volume=0.85[a];[1:a]aresample=44100,volume=1.3[b];[2:a]aresample=44100,volume=0.85[c];[a][b][c]amix=inputs=3:duration=longest:normalize=0',
            '-c:a','pcm_f32le',str(temp)],check=True)
        samples,rate=sf.read(temp,dtype='float32',always_2d=True)
        assert np.isfinite(samples).all()
        peak=float(np.abs(samples).max())
        gain=min(1.,10**(-1/20)/max(peak,1e-9))
        final=OUT/'我不难过_翻唱_A_150e_降4调.flac'
        sf.write(final,samples*gain,rate,subtype='PCM_24')
        temp.unlink()
        excerpt=OUT/'试听片段_30秒.mp3'
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error','-ss','75','-i',str(final),
                        '-t','30','-c:a','libmp3lame','-b:a','192k',str(excerpt)],check=True)
        state.update(final=str(final),duration_seconds=len(samples)/rate,global_gain=gain,
                     peak_dbfs=float(20*np.log10(max(peak*gain,1e-9))),weights=str(pth))
        phase('complete')
    except BaseException as e:
        state['error']=repr(e)
        phase('failed')
        raise


if __name__=='__main__':
    main()
