"""整条 AI 翻唱链路：分离 → RVC 变声 → 混音。（macOS 版）

和 Windows 版的三处差别：

1. Applio 目录默认在 `~/workspace/Applio`，解释器是 `.venv/bin/python`（见 settings.py）。
2. ffmpeg。Windows 版 Applio 目录下自带 ffmpeg.exe，mac 版没有，
   只能靠 `brew install ffmpeg`，所以找不到时直接给这条命令。
3. 推理设备。Applio 的 rvc/configs/config.py 里写死了 `"cuda:0" if cuda else "cpu"`，
   **没有 MPS 分支**，所以 mac 上 RVC 推理是跑在 CPU 上的，会比 N 卡慢不少。
   这不是本项目能改的，只在开跑前提示一句，免得以为是卡住了。
"""

import glob
import os
import re

import settings
import trainer
from main import separate_track
from procutil import (APPLIO_INFER_ENV, configure_console_encoding, ensure_tools_on_path,
                      require_tool, run_streaming, safe_print)
from settings import applio_python

# ================= 配置区域 =================
APPLIO_DIR = settings.load()["applio_dir"]   # 默认 ~/workspace/Applio，可在 mac/config.json 里改
MODELS_DIR = "models"          # 用户提供的现成模型（回退用）
OUTPUT_DIR = "output"

# (歌曲文件, 输出子目录名, 用哪个声音模型)
JOBS = [
    ("data/audio/周杰伦 - 那天下雨了.flac", "那天下雨了", "zyccn"),
    ("data/audio/Imagine Dragons - Blank Space_Stand By Me (Live from Spotify London).flac", "Blank Space", "zycen"),
]

F0_METHOD = "rmvpe"
EMBEDDER_MODEL = "chinese-hubert-base"  # 训练特征提取时用的同一个，推理要保持一致
PITCH_SHIFT = 0          # 目标音色和源歌手音域不一致时手动调（半音）
INDEX_RATE = 0.3

# 转换后人声加一点混响，模拟录音棚的空间感（偏干声、少量湿声，不要洗澡效果）
REVERB_ROOM_SIZE = 0.15
REVERB_DAMPING = 0.5
REVERB_WET_GAIN = 0.2
REVERB_DRY_GAIN = 0.8
REVERB_WIDTH = 0.5

# 最终混音的相对音量：人声突出一点，伴奏和和声往后退一点
VOCAL_GAIN = 1.3
INSTRUMENTAL_GAIN = 0.85
BACKING_VOCAL_GAIN = 0.85
# ===========================================

F0_METHODS = ["rmvpe", "crepe", "crepe-tiny", "fcpe"]
EMBEDDER_MODELS = [
    "chinese-hubert-base",
    "contentvec",
    "spin",
    "spin-v2",
    "japanese-hubert-base",
    "korean-hubert-base",
]


def find_ffmpeg():
    """mac 上只认 PATH 和 Homebrew 的 bin，没有 Applio 自带的那份。"""
    ensure_tools_on_path()
    return require_tool("ffmpeg")


def list_models(applio_dir=APPLIO_DIR, models_dir=MODELS_DIR):
    """扫出所有能用的音色模型名，喂给界面的下拉框。"""
    names = set()

    logs_dir = os.path.join(os.path.expanduser(applio_dir), "logs")
    if os.path.isdir(logs_dir):
        for entry in os.listdir(logs_dir):
            exp_dir = os.path.join(logs_dir, entry)
            if os.path.isdir(exp_dir) and glob.glob(os.path.join(exp_dir, f"{entry}_*e_*s.pth")):
                names.add(entry)

    models_dir = settings.project_path(models_dir)
    if os.path.isdir(models_dir):
        for entry in os.listdir(models_dir):
            if os.path.exists(os.path.join(models_dir, entry, f"{entry}.pth")):
                names.add(entry)

    return sorted(names)


def find_trained_model(model_name, applio_dir=APPLIO_DIR, models_dir=MODELS_DIR):
    """
    优先用 Applio 训练产出的模型 (Applio/logs/<model_name>/)；
    找不到就回退到用户手动放在 models/<model_name>/ 下的现成模型。
    """
    exp_dir = os.path.join(os.path.expanduser(applio_dir), "logs", model_name)
    index_path = os.path.join(exp_dir, f"{model_name}.index")

    candidates = glob.glob(os.path.join(exp_dir, f"{model_name}_*e_*s.pth"))
    if candidates:
        def epoch_of(path):
            m = re.search(rf"{re.escape(model_name)}_(\d+)e_", os.path.basename(path))
            return int(m.group(1)) if m else -1
        pth_path = max(candidates, key=epoch_of)
        # index 不在也照样返回。Applio 只在**整个训练跑完**之后才生成 .index，
        # 而 --save-every-weights 会在每个存档点就导出可分发模型 ——
        # 想中途试听某个 checkpoint 时，必然是「有 pth、没 index」这个状态。
        # 以前这里 index 不在就往下掉到 models/ 兜底、最后报「找不到模型」，
        # 明明模型就在眼前。rvc_infer 对 index=None 本来就有处理（跳过特征检索）。
        return pth_path, (index_path if os.path.exists(index_path) else None)

    models_dir = settings.project_path(models_dir)
    fallback_pth = os.path.join(models_dir, model_name, f"{model_name}.pth")
    fallback_index = os.path.join(models_dir, model_name, f"{model_name}.index")
    if os.path.exists(fallback_pth):
        return fallback_pth, fallback_index if os.path.exists(fallback_index) else None

    raise FileNotFoundError(f"找不到模型 {model_name}：既没有训练产物也没有现成模型")


def rvc_infer(input_path, output_path, pth_path, index_path,
              pitch=PITCH_SHIFT,
              f0_method=F0_METHOD,
              embedder_model=EMBEDDER_MODEL,
              index_rate=INDEX_RATE,
              reverb=True,
              reverb_room_size=REVERB_ROOM_SIZE,
              reverb_damping=REVERB_DAMPING,
              reverb_wet_gain=REVERB_WET_GAIN,
              reverb_dry_gain=REVERB_DRY_GAIN,
              reverb_width=REVERB_WIDTH,
              applio_dir=APPLIO_DIR,
              log=safe_print,
              on_start=None):
    tone = "带混响" if reverb else "干声"
    log(f"  → RVC 推理({tone}, pitch={pitch}): {os.path.basename(pth_path)}")
    log("  ℹ️ Applio 的推理端没有 MPS 分支，mac 上这步跑 CPU，耐心等")

    applio_dir = os.path.expanduser(applio_dir)
    python_exe = applio_python(applio_dir)
    if not os.path.exists(python_exe):
        raise FileNotFoundError(
            f"找不到 Applio 的 Python 解释器: {python_exe}\n"
            f"确认 Applio 装在 {applio_dir}，且已经跑过 ./run-install.sh"
        )

    # 推理同样要用音高预测器（rmvpe.pt 之类），而那些只有开 Applio 的 GUI 才会自动下。
    # 少了它 Applio 不会明说，只会转出一段没有音高的噪音。详见 trainer.py 里的注释。
    trainer.ensure_prerequisites(applio_dir, f0_method=f0_method,
                                 embedder_model=embedder_model, pretrained=False, log=log)

    # Applio 的 --index-path 是 required 的。没有 index 文件时传空串，
    # 它内部对空串和不存在的路径都有兜底，会自动跳过特征检索。
    resolved_index = os.path.abspath(index_path) if index_path and os.path.exists(index_path) else ""
    if not resolved_index:
        log("  ⚠️ 该模型没有 .index 文件，本次推理不使用特征检索")
        index_rate = 0

    cmd = [
        python_exe, "core.py", "infer",
        "--input-path", os.path.abspath(input_path),
        "--output-path", os.path.abspath(output_path),
        "--pth-path", os.path.abspath(pth_path),
        "--index-path", resolved_index,
        "--f0-method", f0_method,
        "--embedder-model", embedder_model,
        "--pitch", str(pitch),
        "--index-rate", str(index_rate),
        "--export-format", "FLAC",
    ]
    if reverb:
        cmd += [
            "--post-process",
            "--reverb",
            "--reverb-room-size", str(reverb_room_size),
            "--reverb-damping", str(reverb_damping),
            "--reverb-wet-gain", str(reverb_wet_gain),
            "--reverb-dry-gain", str(reverb_dry_gain),
            "--reverb-width", str(reverb_width),
        ]

    run_streaming(cmd, cwd=applio_dir, log=log, on_start=on_start,
                  env_extra=APPLIO_INFER_ENV)

    # Applio 的 core.py 失败时退出码仍然是 0，只能靠产物在不在判断
    if not os.path.exists(output_path):
        raise RuntimeError(f"RVC 推理失败：没有生成 {output_path}（Applio 的报错见上面的日志）")


def mix_final(vocals_path, instrumental_path, backing_vocals_path, output_path,
              vocal_gain=VOCAL_GAIN,
              instrumental_gain=INSTRUMENTAL_GAIN,
              backing_vocal_gain=BACKING_VOCAL_GAIN,
              include_backing=True,
              log=safe_print,
              on_start=None):
    use_backing = bool(include_backing and backing_vocals_path and os.path.exists(backing_vocals_path))
    layout = "人声+伴奏+和声" if use_backing else "人声+伴奏"
    log(f"  → 混音({layout}): {os.path.basename(output_path)}")

    inputs = ["-i", instrumental_path, "-i", vocals_path]
    chains = [
        f"[0:a]aresample=44100,volume={instrumental_gain}[a0];",
        f"[1:a]aresample=44100,volume={vocal_gain}[a1];",
    ]
    labels = "[a0][a1]"
    count = 2

    if use_backing:
        inputs += ["-i", backing_vocals_path]
        chains.append(f"[2:a]aresample=44100,volume={backing_vocal_gain}[a2];")
        labels += "[a2]"
        count = 3

    filter_complex = "".join(chains) + f"{labels}amix=inputs={count}:duration=longest:normalize=0"

    cmd = [
        find_ffmpeg(), "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-ar", "44100",
        output_path,
        "-loglevel", "error",
    ]
    run_streaming(cmd, log=log, on_start=on_start)


def run_cover(song_file, song_name, model_name, pitch=PITCH_SHIFT,
              f0_method=F0_METHOD,
              embedder_model=EMBEDDER_MODEL,
              index_rate=INDEX_RATE,
              reverb=True,
              reverb_room_size=REVERB_ROOM_SIZE,
              reverb_damping=REVERB_DAMPING,
              reverb_wet_gain=REVERB_WET_GAIN,
              reverb_dry_gain=REVERB_DRY_GAIN,
              reverb_width=REVERB_WIDTH,
              vocal_gain=VOCAL_GAIN,
              instrumental_gain=INSTRUMENTAL_GAIN,
              backing_vocal_gain=BACKING_VOCAL_GAIN,
              include_backing=True,
              output_root=OUTPUT_DIR,
              applio_dir=APPLIO_DIR,
              models_dir=MODELS_DIR,
              pth_path=None,
              index_path=None,
              log=safe_print,
              should_stop=None,
              on_start=None):
    """
    整条翻唱链路：分离 -> RVC 转换 -> 混音。

    pth_path/index_path 显式给了就直接用（界面上「浏览」手选的外部模型），
    否则按 model_name 去 Applio logs / models 里找。
    """
    log(f"\n=== 🎤 {song_name}（{model_name}） ===")

    # 混音要用 ffmpeg，与其跑完分离再倒在最后一步，不如开跑前就查
    find_ffmpeg()

    song_output_dir = os.path.join(settings.project_path(output_root), song_name)

    log("[1/3] 人声分离...")
    stems = separate_track(settings.project_path(song_file), song_output_dir,
                           log=log, should_stop=should_stop)

    if pth_path is None:
        pth_path, index_path = find_trained_model(model_name, applio_dir, models_dir)

    log("[2/3] AI 翻唱转换...")
    converted_vocals_path = os.path.join(song_output_dir, f"6_AI翻唱人声_{model_name}.flac")
    rvc_infer(stems["dry_vocals"], converted_vocals_path, pth_path, index_path,
              pitch=pitch, f0_method=f0_method, embedder_model=embedder_model,
              index_rate=index_rate, reverb=reverb,
              reverb_room_size=reverb_room_size, reverb_damping=reverb_damping,
              reverb_wet_gain=reverb_wet_gain, reverb_dry_gain=reverb_dry_gain,
              reverb_width=reverb_width, applio_dir=applio_dir,
              log=log, on_start=on_start)

    log("[3/3] 人声+伴奏+和声混音...")
    final_path = os.path.join(song_output_dir, f"7_AI翻唱成品_{model_name}.flac")
    mix_final(converted_vocals_path, stems["instrumental"], stems["backing_vocals"], final_path,
              vocal_gain=vocal_gain, instrumental_gain=instrumental_gain,
              backing_vocal_gain=backing_vocal_gain, include_backing=include_backing,
              log=log, on_start=on_start)

    log(f"✅ {song_name} 完成 → {final_path}")
    return final_path


def main():
    for song_file, song_name, model_name in JOBS:
        run_cover(song_file, song_name, model_name)


if __name__ == "__main__":
    configure_console_encoding()
    main()
