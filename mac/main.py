"""三阶段音频分离：伴奏/人声 → 主唱/和声 → 去混响。（macOS 版）

分离这一段本身是纯 Python + torch，没有调外部程序，所以和 Windows 版的逻辑
一模一样。mac 上要处理的只有两件事：

1. 设备。audio-separator 会自己认出 Apple Silicon 并把 torch 设备设成 MPS
   （onnx 那侧走 CoreML），这里不需要干预，只是把认到的设备打出来，
   免得跑得慢的时候搞不清是掉到 CPU 上了还是本来就这么慢。
2. 速度。原来那套参数（overlap=8、TTA 开）是按 CUDA 显卡调的，M 系列上会明显更慢。
   参数原样保留，另外给一个 FAST_MODE 开关，嫌慢时打开，画质换时间。
"""

import logging
import os
import shutil

from audio_separator.separator import Separator

import settings
from procutil import (Cancelled, configure_console_encoding, ensure_tools_on_path,
                      require_tool, safe_print)

# ================= 配置区域 =================
INPUT_FILE = "data/audio/test.flac"   # 你的文件名（相对路径按项目根目录算）
OUTPUT_DIR = "output"     # 输出文件夹
OUTPUT_FORMAT = "flac"    # 强烈建议用 FLAC，mp3 反复编码会加重电音感

# True = 降低重叠率、关掉 TTA，快好几倍，画质略降。M 系列上嫌慢就打开。
FAST_MODE = False

# 分离模型缓存目录。audio-separator 默认下到 /tmp/audio-separator-models，
# 而 macOS 会定期清理 /tmp —— 那三个模型加起来 1.3GB，被清掉就得重下。
# 放到系统正规的缓存目录里，重启也还在。
MODEL_DIR = os.path.expanduser("~/Library/Caches/audio-separator-models")
# ===========================================

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU 加速后耗时不再是瓶颈，把各架构的质量参数直接拉高（与 Windows 版一致）
QUALITY_PARAMS = {
    "mdxc_params": {"segment_size": 256, "override_model_segment_size": False,
                    "batch_size": 4, "overlap": 8, "pitch_shift": 0},
    "vr_params": {"batch_size": 4, "window_size": 512, "aggression": 10,
                  "enable_tta": True, "enable_post_process": False,
                  "post_process_threshold": 0.2, "high_end_process": False},
}

# 重叠率从 8 降到 2、TTA 关掉，是这套流程里最大的两个耗时来源
FAST_PARAMS = {
    "mdxc_params": {"segment_size": 256, "override_model_segment_size": False,
                    "batch_size": 2, "overlap": 2, "pitch_shift": 0},
    "vr_params": {"batch_size": 2, "window_size": 512, "aggression": 10,
                  "enable_tta": False, "enable_post_process": False,
                  "post_process_threshold": 0.2, "high_end_process": False},
}


def describe_device():
    """返回一句人话，说明这次分离会跑在什么设备上。"""
    import torch

    if torch.backends.mps.is_available():
        return "MPS（Apple Silicon GPU）"
    if torch.cuda.is_available():
        return f"CUDA（{torch.cuda.get_device_name(0)}）"
    return "CPU（没检测到 MPS，会非常慢）"


def rename_file(old_path, output_dir, new_name, log=safe_print):
    """文件重命名工具"""
    if os.path.exists(old_path):
        new_path = os.path.join(output_dir, new_name)
        if os.path.exists(new_path):
            os.remove(new_path)
        os.rename(old_path, new_path)
        log(f"✨ 重命名: {new_name}")
        return new_path
    return None


def separate_track(input_file, output_dir, output_format="flac",
                   clean_output=False, log=safe_print, should_stop=None,
                   fast=None):
    """
    对一个音频文件跑完整的 3 阶段分离流程：
    伴奏/人声 -> 主唱/和声 -> 去混响。
    返回各产物的绝对/相对路径字典，供上层流程（比如 AI 翻唱）复用。

    clean_output=True 会先把 output_dir 整个删掉重建。界面上用户可能随手选了
    桌面或某个已有文件夹，所以默认不清空——只有命令行里指定的专用目录才开。

    should_stop 是个无参回调，在每个阶段开始前检查，返回 True 就抛 Cancelled。
    分离是在进程内跑的，没法瞬间掐断，最细只能做到阶段边界。

    fast=None 时用模块顶上的 FAST_MODE。
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件: {input_file}")

    def check_stop():
        if should_stop and should_stop():
            raise Cancelled("已停止")

    if clean_output and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # audio-separator 在 Separator() 里就要 `ffmpeg -version`，而且是裸命令名走 PATH，
    # 所以纯分离也必须有 ffmpeg。先把 Homebrew 的 bin 补进 PATH，再自己查一遍，
    # 好在它抛出一句英文 FileNotFoundError 之前给出能照抄的安装命令。
    ensure_tools_on_path()
    require_tool("ffmpeg")

    log("--- 🚀 启动 3阶段 极智分离引擎 (去电音+去混响) ---")

    use_fast = FAST_MODE if fast is None else fast
    log(f"🖥  计算设备: {describe_device()}｜画质档位: {'快速' if use_fast else '高质量'}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    sep = Separator(
        output_dir=output_dir,
        output_format=output_format,
        model_file_dir=MODEL_DIR,
        **(FAST_PARAMS if use_fast else QUALITY_PARAMS),
    )

    # =======================================================
    # [Stage 1] 伴奏分离 (高精度去电音)
    # =======================================================
    check_stop()
    log("\n[1/3] 正在分离伴奏 (高重叠率模式)...")
    sep.load_model(model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    step1_files = sep.separate(input_file)

    instrumental_path = None
    full_vocals_path = None
    for f in step1_files:
        path = os.path.join(output_dir, f)
        if "(Instrumental)" in f:
            instrumental_path = rename_file(path, output_dir, "1_纯伴奏_Instrumental.flac", log)
        elif "(Vocals)" in f:
            full_vocals_path = rename_file(path, output_dir, "temp_full_vocals.flac", log)

    # =======================================================
    # [Stage 2] 主唱/和声分离
    # 模型：Mel-Band Roformer Karaoke
    # =======================================================
    backing_vocals_path = None
    lead_vocals_path = None
    if full_vocals_path:
        check_stop()
        log("\n[2/3] 正在分离 主唱 与 和声...")
        sep.load_model(model_filename="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
        step2_files = sep.separate(full_vocals_path)

        for f in step2_files:
            path = os.path.join(output_dir, f)
            if "(Instrumental)" in f:
                # 在 Karaoke 模型里，Instrumental 其实就是和声
                backing_vocals_path = rename_file(path, output_dir, "3_纯和声_Backing_Vocals.flac", log)
            elif "(Vocals)" in f:
                # 这是一个中间产物：带混响的主唱
                lead_vocals_path = rename_file(path, output_dir, "temp_lead_vocals_reverb.flac", log)

    # =======================================================
    # [Stage 3] 去混响 (De-Reverb) - 解决"回音"问题
    # 模型：MelBand Roformer De-Reverb Less Aggressive (MDXC 架构)
    #
    # 原来用的是 UVR-DeEcho-DeReverb（VR 架构，2023）。它削得太狠，会把歌手自己
    # 的乐句尾音当混响整段抹掉——实测在《心的距离》里打出 7 个 0.3~0.4 秒的洞，
    # 每个洞的位置都是「前 0.5s 大声、后 0.5s 安静」的衰减段。调 vr_params 的
    # aggression 没用（10 降到 0，洞的电平几乎不变），是模型本身的问题。
    # 换成这个之后：整体过度削减从 6.93 dB 降到 2.28 dB，洞从 22 帧降到 10 帧。
    # =======================================================
    dry_vocals_path = None
    reverb_only_path = None
    if lead_vocals_path:
        check_stop()
        log("\n[3/3] 正在对主唱进行【深度去混响】...")

        sep.load_model(
            model_filename="dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt")
        step3_files = sep.separate(lead_vocals_path)

        for f in step3_files:
            path = os.path.join(output_dir, f)

            # 去混响模型的输出标签是「有没有混响」，不是 (Vocals)/(Instrumental)。
            # 而且各架构的大小写和空格不一致：VR 那个是 (No Reverb)/(Reverb)，
            # Roformer 这个是 (noreverb)/(reverb)。统一压平了再比，换模型才不会
            # 悄无声息地匹配不上——那样 dry_vocals_path 会是 None，直接抛
            # 「流程中断」，比拿到半成品好，但没必要让它发生。
            tag = f.lower().replace(" ", "")
            if "(noreverb)" in tag:
                dry_vocals_path = rename_file(path, output_dir, "2_纯主唱_干声_Dry_Vocals.flac", log)
            elif "(reverb)" in tag:
                reverb_only_path = rename_file(path, output_dir, "5_被切除的混响_Reverb_Only.flac", log)

        # 清理临时文件
        if full_vocals_path and os.path.exists(full_vocals_path):
            os.remove(full_vocals_path)
        if lead_vocals_path and os.path.exists(lead_vocals_path):
            os.remove(lead_vocals_path)

        log("\n🎉 === 分离完成 === 🎉")
    else:
        raise RuntimeError("流程中断：未生成主唱文件")

    return {
        "instrumental": instrumental_path,
        "dry_vocals": dry_vocals_path,
        "backing_vocals": backing_vocals_path,
        "reverb_only": reverb_only_path,
    }


def main():
    # 命令行跑的是固定的专用目录，可以放心清空
    separate_track(settings.project_path(INPUT_FILE),
                   settings.project_path(OUTPUT_DIR),
                   OUTPUT_FORMAT, clean_output=True)


if __name__ == "__main__":
    configure_console_encoding()
    main()
