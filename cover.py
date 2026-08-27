import glob
import os
import re
import shutil

from main import separate_track
from procutil import configure_console_encoding, run_streaming, safe_print
from settings import applio_python

# ================= 配置区域 =================
APPLIO_DIR = r"C:\workspace\Applio"
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


def find_ffmpeg(applio_dir=APPLIO_DIR):
    """优先用 PATH 里的 ffmpeg，没有就用 Applio 自带的那个。"""
    found = shutil.which("ffmpeg")
    if found:
        return found
    bundled = os.path.join(applio_dir, "ffmpeg.exe")
    if os.path.exists(bundled):
        return bundled
    raise FileNotFoundError("找不到 ffmpeg：PATH 里没有，Applio 目录下也没有 ffmpeg.exe")


def list_models(applio_dir=APPLIO_DIR, models_dir=MODELS_DIR):
    """扫出所有能用的音色模型名，喂给界面的下拉框。"""
    names = set()

    logs_dir = os.path.join(applio_dir, "logs")
    if os.path.isdir(logs_dir):
        for entry in os.listdir(logs_dir):
            exp_dir = os.path.join(logs_dir, entry)
            if os.path.isdir(exp_dir) and glob.glob(os.path.join(exp_dir, f"{entry}_*e_*s.pth")):
                names.add(entry)

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
    exp_dir = os.path.join(applio_dir, "logs", model_name)
    index_path = os.path.join(exp_dir, f"{model_name}.index")

    candidates = glob.glob(os.path.join(exp_dir, f"{model_name}_*e_*s.pth"))
    if candidates:
        def epoch_of(path):
            m = re.search(rf"{re.escape(model_name)}_(\d+)e_", os.path.basename(path))
            return int(m.group(1)) if m else -1
        pth_path = max(candidates, key=epoch_of)
        if os.path.exists(index_path):
            return pth_path, index_path

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

    # Applio 的 --index-path 是 required 的。没有 index 文件时传空串，
    # 它内部对空串和不存在的路径都有兜底，会自动跳过特征检索。
    resolved_index = os.path.abspath(index_path) if index_path and os.path.exists(index_path) else ""
    if not resolved_index:
        log("  ⚠️ 该模型没有 .index 文件，本次推理不使用特征检索")
        index_rate = 0

    cmd = [
        applio_python(applio_dir), "core.py", "infer",
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

    run_streaming(cmd, cwd=applio_dir, log=log, on_start=on_start)


def mix_final(vocals_path, instrumental_path, backing_vocals_path, output_path,
              vocal_gain=VOCAL_GAIN,
              instrumental_gain=INSTRUMENTAL_GAIN,
              backing_vocal_gain=BACKING_VOCAL_GAIN,
              include_backing=True,
              applio_dir=APPLIO_DIR,
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
        find_ffmpeg(applio_dir), "-y",
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
    song_output_dir = os.path.join(output_root, song_name)

    log("[1/3] 人声分离...")
    stems = separate_track(song_file, song_output_dir, log=log, should_stop=should_stop)

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
              applio_dir=applio_dir, log=log, on_start=on_start)

    log(f"✅ {song_name} 完成 → {final_path}")
    return final_path


def main():
    for song_file, song_name, model_name in JOBS:
        run_cover(song_file, song_name, model_name)


if __name__ == "__main__":
    configure_console_encoding()
    main()
