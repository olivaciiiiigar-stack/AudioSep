"""Render 200e index=0.5 against the same separated source as index=0.3."""
from pathlib import Path
import json
import subprocess
import time
import numpy as np
import soundfile as sf
import cover

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/那天下雨了_1005_A_200e'
OUT = ROOT / 'output/那天下雨了_1005_A_200e_index05'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logfile = (OUT/'pipeline.log').open('a', buffering=1)
    state = {'epoch':200,'index_rate':0.5,'pitch':0,'reverb':False,
             'vocal_gain':1.3,'instrumental_gain':0.675,'backing_gain':0.675,
             'stems_source':str(BASE/'stems')}
    def log(msg):
        line=time.strftime('%H:%M:%S ')+msg
        print(line,flush=True)
        logfile.write(line+'\n')
    def phase(value):
        state['phase']=value
        (OUT/'run.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
        log(value)
    try:
        model=BASE/'model'
        weights=list(model.glob('*200e*.pth'))
        indexes=list(model.glob('*.index'))
        assert len(weights)==len(indexes)==1
        dry=OUT/'翻唱干声_200e_index05.flac'
        stems=BASE/'stems'
        phase('infer')
        if not dry.exists():
            cover.rvc_infer(str(stems/'2_纯主唱_干声_Dry_Vocals.flac'),str(dry),
                            str(weights[0]),str(indexes[0]),index_rate=.5,reverb=False,
                            applio_dir='/Users/zhangyichi/Workspace/Applio',log=log)
        phase('mix')
        temp=OUT/'mix_float.wav'
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error',
            '-i',str(stems/'1_纯伴奏_Instrumental.flac'),'-i',str(dry),
            '-i',str(stems/'3_纯和声_Backing_Vocals.flac'),
            '-filter_complex','[0:a]aresample=44100,volume=0.675[a];[1:a]aresample=44100,volume=1.3[b];[2:a]aresample=44100,volume=0.675[c];[a][b][c]amix=inputs=3:duration=longest:normalize=0',
            '-c:a','pcm_f32le',str(temp)],check=True)
        audio,rate=sf.read(temp,dtype='float32',always_2d=True)
        assert np.isfinite(audio).all()
        peak=float(np.abs(audio).max())
        gain=min(1.,10**(-1/20)/max(peak,1e-9))
        final=OUT/'那天下雨了_翻唱_A_200e_index05.flac'
        sf.write(final,audio*gain,rate,subtype='PCM_24')
        temp.unlink()
        # Same passage, matched RMS and sufficient headroom for a fair quick comparison.
        ref=OUT/'comparison_index03_float.wav'
        subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error',
            '-ss','45','-t','30','-i',str(stems/'1_纯伴奏_Instrumental.flac'),
            '-ss','45','-t','30','-i',str(BASE/'翻唱干声_200e.flac'),
            '-ss','45','-t','30','-i',str(stems/'3_纯和声_Backing_Vocals.flac'),
            '-filter_complex','[0:a]aresample=44100,volume=0.675[a];[1:a]aresample=44100,volume=1.3[b];[2:a]aresample=44100,volume=0.675[c];[a][b][c]amix=inputs=3:duration=longest:normalize=0',
            '-c:a','pcm_f32le',str(ref)],check=True)
        a,sr=sf.read(ref,dtype='float32',always_2d=True)
        b,_=sf.read(final,dtype='float32',always_2d=True)
        b=b[45*sr:75*sr]
        ref.unlink()
        rms_a=float(np.sqrt(np.mean(a.astype('float64')**2)))
        rms_b=float(np.sqrt(np.mean(b.astype('float64')**2)))
        b*=rms_a/max(rms_b,1e-9)
        shared=min(1.,10**(-1/20)/max(float(np.abs(a).max()),float(np.abs(b).max()),1e-9))
        for label,clip in [('03',a),('05',b)]:
            wav=OUT/f'对比片段_index{label}.wav'
            sf.write(wav,clip*shared,sr,subtype='PCM_24')
            subprocess.run(['/opt/homebrew/bin/ffmpeg','-y','-v','error','-i',str(wav),
                '-c:a','libmp3lame','-b:a','192k',str(OUT/f'对比片段_index{label}.mp3')],check=True)
        state.update(final=str(final),duration_seconds=len(audio)/rate,global_gain=gain,
                     peak_dbfs=float(20*np.log10(max(peak*gain,1e-9))),
                     comparison_seconds=[45,75],comparison_rms_matched=True)
        phase('complete')
    except BaseException as e:
        state['error']=repr(e)
        phase('failed')
        raise


if __name__=='__main__':
    main()
