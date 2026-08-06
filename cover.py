import glob
import os
import re
import subprocess

from main import separate_track

# ================= 配置区域 =================
APPLIO_DIR = r"C:\workspace\Applio"
APPLIO_PYTHON = os.path.join(APPLIO_DIR, ".venv", "Scripts", "python.exe")
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
VOCAL_GAIN = 1.15
INSTRUMENTAL_GAIN = 0.85
BACKING_VOCAL_GAIN = 0.85
# ===========================================


def find_trained_model(model_name):
    """
    优先用 Applio 训练产出的模型 (Applio/logs/<model_name>/)；
    找不到就回退到用户手动放在 models/<model_name>/ 下的现成模型。
    """
    exp_dir = os.path.join(APPLIO_DIR, "logs", model_name)
    index_path = os.path.join(exp_dir, f"{model_name}.index")

    candidates = glob.glob(os.path.join(exp_dir, f"{model_name}_*e_*s.pth"))
    if candidates:
        def epoch_of(path):
            m = re.search(rf"{re.escape(model_name)}_(\d+)e_", os.path.basename(path))
            return int(m.group(1)) if m else -1
        pth_path = max(candidates, key=epoch_of)
        if os.path.exists(index_path):
            return pth_path, index_path

    fallback_pth = os.path.join(MODELS_DIR, model_name, f"{model_name}.pth")
    fallback_index = os.path.join(MODELS_DIR, model_name, f"{model_name}.index")
    if os.path.exists(fallback_pth):
        return fallback_pth, fallback_index if os.path.exists(fallback_index) else None

    raise FileNotFoundError(f"找不到模型 {model_name}：既没有训练产物也没有现成模型")


def rvc_infer(input_path, output_path, pth_path, index_path):
    print(f"  → RVC 推理(带混响): {os.path.basename(pth_path)}")
    cmd = [
        APPLIO_PYTHON, "core.py", "infer",
        "--input-path", os.path.abspath(input_path),
        "--output-path", os.path.abspath(output_path),
        "--pth-path", os.path.abspath(pth_path),
        "--f0-method", F0_METHOD,
        "--embedder-model", EMBEDDER_MODEL,
        "--pitch", str(PITCH_SHIFT),
        "--index-rate", str(INDEX_RATE),
        "--export-format", "FLAC",
        "--post-process",
        "--reverb",
        "--reverb-room-size", str(REVERB_ROOM_SIZE),
        "--reverb-damping", str(REVERB_DAMPING),
        "--reverb-wet-gain", str(REVERB_WET_GAIN),
        "--reverb-dry-gain", str(REVERB_DRY_GAIN),
        "--reverb-width", str(REVERB_WIDTH),
    ]
    if index_path:
        cmd += ["--index-path", os.path.abspath(index_path)]
    subprocess.run(cmd, cwd=APPLIO_DIR, check=True)


def mix_final(vocals_path, instrumental_path, backing_vocals_path, output_path):
    print(f"  → 混音(人声+伴奏+和声): {os.path.basename(output_path)}")
    cmd = [
        "ffmpeg", "-y",
        "-i", instrumental_path,
        "-i", vocals_path,
        "-i", backing_vocals_path,
        "-filter_complex",
        f"[0:a]aresample=44100,volume={INSTRUMENTAL_GAIN}[a0];"
        f"[1:a]aresample=44100,volume={VOCAL_GAIN}[a1];"
        f"[2:a]aresample=44100,volume={BACKING_VOCAL_GAIN}[a2];"
        "[a0][a1][a2]amix=inputs=3:duration=longest:normalize=0",
        "-ar", "44100",
        output_path,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)


def run_cover(song_file, song_name, model_name):
    print(f"\n=== 🎤 {song_name}（{model_name}） ===")
    song_output_dir = os.path.join(OUTPUT_DIR, song_name)

    print("[1/3] 人声分离...")
    stems = separate_track(song_file, song_output_dir)

    pth_path, index_path = find_trained_model(model_name)

    print("[2/3] AI 翻唱转换...")
    converted_vocals_path = os.path.join(song_output_dir, f"6_AI翻唱人声_{model_name}.flac")
    rvc_infer(stems["dry_vocals"], converted_vocals_path, pth_path, index_path)

    print("[3/3] 人声+伴奏+和声混音...")
    final_path = os.path.join(song_output_dir, f"7_AI翻唱成品_{model_name}.flac")
    mix_final(converted_vocals_path, stems["instrumental"], stems["backing_vocals"], final_path)

    print(f"✅ {song_name} 完成 → {final_path}")


def main():
    for song_file, song_name, model_name in JOBS:
        run_cover(song_file, song_name, model_name)


if __name__ == "__main__":
    main()
