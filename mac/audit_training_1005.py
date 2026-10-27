"""Read-only audit of the prepared A dataset and older cover files."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
APPLIO = Path('/Users/zhangyichi/Workspace/Applio')
EXP = APPLIO / 'logs/zyc_20261005_A'
report = {'dataset': {}, 'old_models': [], 'old_outputs': []}
errors, pitch_values, mismatch, short, silent = [], [], [], [], []
rows = (EXP / 'filelist.txt').read_text().splitlines()
for row in rows:
    wav, feat, coarse, continuous, sid = row.split('|')
    paths = [APPLIO / p for p in (wav, feat, coarse, continuous)]
    if not all(p.exists() for p in paths):
        errors.append({'wav': wav, 'error': 'missing pair'})
        continue
    info = sf.info(paths[0])
    f, c, p = [np.load(q, allow_pickle=False) for q in paths[1:]]
    if not all(np.isfinite(v).all() for v in (f,c,p)):
        errors.append({'wav':wav,'error':'nonfinite'})
    if info.samplerate != 40000 or f.ndim != 2 or f.shape[1] != 768:
        errors.append({'wav':wav,'error':'shape/rate'})
    if len(p) != len(c) or abs(2*len(f)-len(p)) > 4:
        mismatch.append({'wav':wav,'feature_frames_x2':2*len(f),'pitch_frames':len(p)})
    if info.duration <= .5:
        short.append({'wav':wav,'seconds':info.duration})
    if not np.any(p>0):
        silent.append(wav)
    pitch_values.extend(p[p>0].tolist())
report['dataset'] = {'filelist_rows':len(rows), 'audio_clips':len(list((EXP/'sliced_audios').glob('*.wav'))),
                     'errors':errors,'frame_mismatches':mismatch,'short_clips':short,
                     'all_unvoiced':silent,'voiced_pitch_percentiles_hz':np.percentile(pitch_values,[0,1,50,95,99,100]).tolist(),
                     'config':json.loads((EXP/'config.json').read_text())}
for p in (APPLIO/'logs/zyc_cn_mandarin_dry').glob('*200e*.pth'):
    ckpt=torch.load(p,map_location='cpu',weights_only=True)
    report['old_models'].append({'path':str(p),'metadata':{k:v for k,v in ckpt.items() if k!='weight' and isinstance(v,(str,int,float,bool,list,dict))}})
for p in (ROOT/'output/那天下雨了').glob('*.flac'):
    samples = total = 0
    peak = 0.
    with sf.SoundFile(p) as audio:
        duration=audio.frames/audio.samplerate
        subtype=audio.subtype
        for block in audio.blocks(blocksize=262144,dtype='float32'):
            peak=max(peak,float(np.max(np.abs(block))))
            samples+=int(np.count_nonzero(np.abs(block)>=.999))
            total+=block.size
    report['old_outputs'].append({'path':str(p),'duration':duration,'subtype':subtype,
                                  'peak_dbfs':float(20*np.log10(max(peak,1e-10))),
                                  'samples_at_or_above_0_999':samples,'sample_count':total})
dst=ROOT/'output/训练代码检查_1005.json'
dst.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
print(json.dumps(report,ensure_ascii=False,indent=2,default=str))
