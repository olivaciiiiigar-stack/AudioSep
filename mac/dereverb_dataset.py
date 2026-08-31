"""把训练素材整批过一遍 UVR-DeEcho-DeReverb，输出到 <目录名>_dry/。

浴室之类的硬质空间录音，房间反射会被模型当成音色的一部分学进去，
每次输出都自带回声。实测这一步对频谱几乎无损（各频段 ±0.5 dB 以内，
高频反而略升），比降噪安全得多——降噪会明显削高频。

用法：
    uv run mac/dereverb_dataset.py                      # 处理默认的 voice_zyc_cn_ma
    uv run mac/dereverb_dataset.py <目录>                # 处理指定目录
"""

import logging
import os
import sys
import time
import warnings

import settings
import trainer
from audio_separator.separator import Separator
from main import MODEL_DIR, QUALITY_PARAMS
from procutil import configure_console_encoding, ensure_tools_on_path, safe_print

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

DEFAULT = os.path.join("data", "original_voice", "voice_zyc_cn_ma")


def dereverb_folder(src, log=safe_print):
    ensure_tools_on_path()
    src = settings.project_path(src).rstrip("/")
    dst = src + "_dry"
    os.makedirs(dst, exist_ok=True)

    files = trainer.collect_audio_files(src)
    if not files:
        raise SystemExit(f"目录里没有音频: {src}")

    log(f"源目录 {src}")
    log(f"输出   {dst}")
    log(f"{len(files)} 个文件\n")

    sep = Separator(output_dir=dst, output_format="flac",
                    model_file_dir=MODEL_DIR, **QUALITY_PARAMS)
    sep.load_model(model_filename="UVR-DeEcho-DeReverb.pth")

    t0 = time.time()
    done = skipped = 0
    for i, f in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(dst, base + ".flac")
        if os.path.exists(out):
            skipped += 1
            continue
        log(f"[{i}/{len(files)}] {os.path.basename(f)}")
        for produced in sep.separate(f):
            p = os.path.join(dst, produced)
            # 只要干声那一路，被切掉的混响直接丢弃
            if "(No Reverb)" in produced:
                os.replace(p, out)
            else:
                os.remove(p)
        done += 1

    log(f"\n完成 {done} 个（跳过已存在的 {skipped} 个），耗时 {(time.time()-t0)/60:.1f} 分钟")
    n, secs = trainer.dataset_stats(trainer.collect_audio_files(dst))
    log(f"输出目录: {n} 个文件，{secs/60:.1f} 分钟")
    return dst


if __name__ == "__main__":
    configure_console_encoding()
    dereverb_folder(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
