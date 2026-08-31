"""看一眼数据集攒了多少：文件数、总时长、格式分布。

用法：uv run mac/dataset_info.py [目录]
默认看 data/original_voice/myvoice/
"""

import os
import sys

import settings
import trainer
from procutil import configure_console_encoding, safe_print


def main(folder=None):
    folder = folder or settings.project_path(
        os.path.join(settings.DEFAULTS["dataset_root"], "myvoice"))

    files = trainer.collect_audio_files(folder)
    if not files:
        safe_print(f"📂 {folder}\n   还没有音频文件")
        return

    n, secs = trainer.dataset_stats(files)
    mins = secs / 60

    safe_print(f"📂 {folder}")
    safe_print(f"   {n} 个文件，共 {mins:.1f} 分钟")

    # 格式分布，顺便标出哪些要转码
    exts = {}
    for f in files:
        exts[os.path.splitext(f)[1].lower()] = exts.get(os.path.splitext(f)[1].lower(), 0) + 1
    need = len(trainer.needs_conversion(files))
    safe_print("   格式: " + "、".join(f"{e} × {c}" for e, c in sorted(exts.items()))
               + (f"（其中 {need} 个训练前会自动转成 FLAC）" if need else ""))

    # 对着建议的档位给个进度条
    for label, target in [("最低可用", 10), ("推荐", 25), ("充裕", 45)]:
        pct = min(1.0, mins / target)
        bar = "█" * int(pct * 24) + "░" * (24 - int(pct * 24))
        safe_print(f"   {label:<5}({target:>2} 分钟) {bar} {pct*100:5.1f}%")


if __name__ == "__main__":
    configure_console_encoding()
    main(sys.argv[1] if len(sys.argv) > 1 else None)
