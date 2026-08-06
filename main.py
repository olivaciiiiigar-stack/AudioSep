import os
import shutil
import logging
from audio_separator.separator import Separator

# ================= 配置区域 =================
INPUT_FILE = "data/audio/test.flac"   # 你的文件名
OUTPUT_DIR = "output"     # 输出文件夹
OUTPUT_FORMAT = "flac"    # 强烈建议用 FLAC，mp3 反复编码会加重电音感
# ===========================================

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rename_file(old_path, output_dir, new_name):
    """文件重命名工具"""
    if os.path.exists(old_path):
        new_path = os.path.join(output_dir, new_name)
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(old_path, new_path)
        print(f"✨ 重命名: {new_name}")
        return new_path
    return None

def separate_track(input_file, output_dir, output_format="flac"):
    """
    对一个音频文件跑完整的 3 阶段分离流程：
    伴奏/人声 -> 主唱/和声 -> 去混响。
    返回各产物的绝对/相对路径字典，供上层流程（比如 AI 翻唱）复用。
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件: {input_file}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("--- 🚀 启动 3阶段 极智分离引擎 (去电音+去混响) ---")

    # GPU 加速后耗时不再是瓶颈，把各架构的质量参数直接拉高
    sep = Separator(
        output_dir=output_dir,
        output_format=output_format,
        mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 4, "overlap": 8, "pitch_shift": 0},
        vr_params={"batch_size": 4, "window_size": 512, "aggression": 10, "enable_tta": True, "enable_post_process": False, "post_process_threshold": 0.2, "high_end_process": False},
    )

    # =======================================================
    # [Stage 1] 伴奏分离 (高精度去电音)
    # =======================================================
    print("\n[1/3] 正在分离伴奏 (高重叠率模式)...")
    sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    step1_files = sep.separate(input_file)

    instrumental_path = None
    full_vocals_path = None
    for f in step1_files:
        path = os.path.join(output_dir, f)
        if "(Instrumental)" in f:
            instrumental_path = rename_file(path, output_dir, "1_纯伴奏_Instrumental.flac")
        elif "(Vocals)" in f:
            full_vocals_path = rename_file(path, output_dir, "temp_full_vocals.flac")

    # =======================================================
    # [Stage 2] 主唱/和声分离
    # 模型：Mel-Band Roformer Karaoke
    # =======================================================
    backing_vocals_path = None
    lead_vocals_path = None
    if full_vocals_path:
        print("\n[2/3] 正在分离 主唱 与 和声...")
        sep.load_model(model_filename="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
        step2_files = sep.separate(full_vocals_path)

        for f in step2_files:
            path = os.path.join(output_dir, f)
            if "(Instrumental)" in f:
                # 在 Karaoke 模型里，Instrumental 其实就是和声
                backing_vocals_path = rename_file(path, output_dir, "3_纯和声_Backing_Vocals.flac")
            elif "(Vocals)" in f:
                # 这是一个中间产物：带混响的主唱
                lead_vocals_path = rename_file(path, output_dir, "temp_lead_vocals_reverb.flac")

    # =======================================================
    # [Stage 3] 去混响 (De-Reverb) - 解决"回音"问题
    # 模型：UVR-DeEcho-DeReverb (VR 架构)
    # =======================================================
    dry_vocals_path = None
    reverb_only_path = None
    if lead_vocals_path:
        print("\n[3/3] 正在对主唱进行【深度去混响】...")

        sep.load_model(model_filename="UVR-DeEcho-DeReverb.pth")
        step3_files = sep.separate(lead_vocals_path)

        for f in step3_files:
            path = os.path.join(output_dir, f)

            # UVR-DeEcho-DeReverb 模型输出标签是 (No Reverb) / (Reverb)，不是 (Vocals)/(Instrumental)
            if "(Reverb)" in f and "(No Reverb)" not in f:
                reverb_only_path = rename_file(path, output_dir, "5_被切除的混响_Reverb_Only.flac")
            elif "(No Reverb)" in f:
                dry_vocals_path = rename_file(path, output_dir, "2_纯主唱_干声_Dry_Vocals.flac")

        # 清理临时文件
        if full_vocals_path and os.path.exists(full_vocals_path):
            os.remove(full_vocals_path)
        if lead_vocals_path and os.path.exists(lead_vocals_path):
            os.remove(lead_vocals_path)

        print("\n🎉 === 分离完成 === 🎉")
    else:
        raise RuntimeError("流程中断：未生成主唱文件")

    return {
        "instrumental": instrumental_path,
        "dry_vocals": dry_vocals_path,
        "backing_vocals": backing_vocals_path,
        "reverb_only": reverb_only_path,
    }

def main():
    separate_track(INPUT_FILE, OUTPUT_DIR, OUTPUT_FORMAT)

if __name__ == "__main__":
    main()
