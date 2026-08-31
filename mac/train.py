"""训练音色模型：预处理 → 特征提取 → 训练 → 生成索引。（macOS 版）

改下面的配置区就能跑：

    uv run mac/train.py

三步之间的产物都会校验（Applio 失败时退出码仍是 0，见 trainer.py 的注释）。
预处理和特征提取是一次性的，重训时把 SKIP_PREPROCESS / SKIP_EXTRACT 打开可以跳过。
"""

import os
import sys
import time

import settings
import trainer
from procutil import configure_console_encoding, safe_print

# ================= 配置区域 =================
MODEL_NAME = "zyc_cn_mandarin"
DATASET_DIR = "data/original_voice/voice_zyc_cn_ma"   # 相对路径按项目根目录算

TOTAL_EPOCH = 200
SAMPLE_RATE = 40000        # 素材是 44.1kHz、内容到 20kHz，40k 最匹配
BATCH_SIZE = 8
SAVE_EVERY_EPOCH = 25      # 每 25 epoch 存一次档，中途可以拿出来试听

F0_METHOD = "rmvpe"
EMBEDDER_MODEL = "chinese-hubert-base"   # 唱中文；推理时必须用同一个
VOCODER = "HiFi-GAN"

# 重训时跳过前两步（产物已经在 Applio/logs/<模型名>/ 下）
SKIP_PREPROCESS = True    # 产物已在 Applio/logs/zyc_cn_mandarin/，续训跳过
SKIP_EXTRACT = True
# ===========================================


def main():
    cfg = settings.load()
    applio_dir = cfg["applio_dir"]
    dataset = settings.project_path(DATASET_DIR)

    files = trainer.collect_audio_files(dataset)
    if not files:
        raise SystemExit(f"数据集目录里没有音频: {dataset}")
    n, secs = trainer.dataset_stats(files)

    safe_print("=" * 60)
    safe_print(f"  模型      {MODEL_NAME}")
    safe_print(f"  数据集    {dataset}")
    safe_print(f"            {n} 个文件，{secs/60:.1f} 分钟")
    safe_print(f"  采样率    {SAMPLE_RATE}   批大小 {BATCH_SIZE}")
    safe_print(f"  总 epoch  {TOTAL_EPOCH}    每 {SAVE_EVERY_EPOCH} 存一次档")
    safe_print(f"  音高      {F0_METHOD}    特征 {EMBEDDER_MODEL}")
    safe_print(f"  Applio    {applio_dir}")
    safe_print("=" * 60)

    t0 = time.time()

    # m4a 之类 Applio 不认的格式在这里转成 FLAC，原目录不动
    ds = trainer.resolve_dataset(MODEL_NAME, dataset, None,
                                 settings.project_path(cfg["dataset_root"]))

    out = trainer.run_training(
        model_name=MODEL_NAME,
        dataset_path=ds,
        total_epoch=TOTAL_EPOCH,
        sample_rate=SAMPLE_RATE,
        applio_dir=applio_dir,
        batch_size=BATCH_SIZE,
        save_every_epoch=SAVE_EVERY_EPOCH,
        f0_method=F0_METHOD,
        embedder_model=EMBEDDER_MODEL,
        vocoder=VOCODER,
        skip_preprocess=SKIP_PREPROCESS,
        skip_extract=SKIP_EXTRACT,
    )

    safe_print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟")
    safe_print(f"产物目录 {out}")


if __name__ == "__main__":
    configure_console_encoding()
    main()
