"""把 Applio 的 preprocess → extract → train → index 四步串成一条能在界面里跑的流水线。

Applio 自己的 core.py train 命令跑完会自动接 index 生成，所以这里只需要串三步。

两个必须绕开的坑：

1. Applio 的 preprocess 只认 .wav/.mp3/.flac/.ogg，扔 .m4a 进去它只会说
   "No audio files found" —— 而手机录音基本都是 m4a。所以这里先转码再喂。
2. core.py 每一步失败时只是 click.echo 一行英文，**退出码仍然是 0**。
   光靠 CalledProcessError 根本发现不了失败，整条流水线会一路绿灯跑完。
   所以每步之后都得自己去查产物在不在。
"""

import glob
import os
import shutil
import subprocess

from procutil import CREATE_NO_WINDOW, run_streaming, safe_print
from settings import applio_python

# 用户那边能选的格式
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".wma", ".opus")
# Applio 的 preprocess 实际认的格式，其余一律转码
APPLIO_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")

SAMPLE_RATES = ["32000", "40000", "48000"]
F0_METHODS = ["rmvpe", "crepe", "crepe-tiny", "fcpe"]
EMBEDDER_MODELS = [
    "chinese-hubert-base",
    "contentvec",
    "spin",
    "spin-v2",
    "japanese-hubert-base",
    "korean-hubert-base",
]
VOCODERS = ["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]
CUT_PREPROCESS = ["Automatic", "Simple", "Skip"]
INDEX_ALGORITHMS = ["Auto", "Faiss", "KMeans"]


class _Tail:
    """转发日志的同时留最后几行，报错时好把 Applio 的原话带出来。"""

    def __init__(self, log, keep=10):
        self.log = log
        self.keep = keep
        self.lines = []

    def __call__(self, message):
        self.lines.append(message)
        del self.lines[:-self.keep]
        self.log(message)

    def text(self):
        return "\n".join(self.lines)


def collect_audio_files(folder):
    """把一个文件夹里的音频文件列出来（只扫一层，Applio 的 preprocess 也不递归）。"""
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(AUDIO_EXTS)
    )


def needs_conversion(files):
    return [f for f in files if not f.lower().endswith(APPLIO_AUDIO_EXTS)]


def _find_tool(name, applio_dir):
    found = shutil.which(name)
    if found:
        return found
    bundled = os.path.join(applio_dir, f"{name}.exe")
    return bundled if os.path.exists(bundled) else None


def _probe_duration(path, ffprobe):
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, creationflags=CREATE_NO_WINDOW, timeout=20,
        )
        return float(out.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0.0


def dataset_stats(files, applio_dir):
    """返回 (文件数, 总秒数)。总时长探不到就返回 0，界面上只显示文件数。"""
    ffprobe = _find_tool("ffprobe", applio_dir)
    if not ffprobe:
        return len(files), 0.0
    return len(files), sum(_probe_duration(f, ffprobe) for f in files)


def _unique_path(dataset_dir, base, ext):
    """同名文件加序号后缀，避免不同来源目录下的「录音.m4a」互相覆盖。"""
    dst = os.path.join(dataset_dir, base + ext)
    n = 1
    while os.path.exists(dst):
        dst = os.path.join(dataset_dir, f"{base}_{n}{ext}")
        n += 1
    return dst


def build_dataset(model_name, files, dataset_root, applio_dir, log=safe_print):
    """把用户散着选的音频汇总到 dataset_root/<model_name>/，返回该目录。

    Applio 不认的格式（m4a / aac / wma / opus）在这里转成 flac，无损且体积小。
    """
    dataset_dir = os.path.join(dataset_root, model_name)
    os.makedirs(dataset_dir, exist_ok=True)

    to_convert = needs_conversion(files)
    ffmpeg = _find_tool("ffmpeg", applio_dir) if to_convert else None
    if to_convert and not ffmpeg:
        raise FileNotFoundError(
            f"有 {len(to_convert)} 个文件需要转码成 Applio 支持的格式，但找不到 ffmpeg。"
            "请安装 ffmpeg，或先自行把素材转成 wav/flac/mp3/ogg。"
        )

    copied = converted = 0
    for src in files:
        base, ext = os.path.splitext(os.path.basename(src))
        if ext.lower() in APPLIO_AUDIO_EXTS:
            dst = _unique_path(dataset_dir, base, ext)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
                copied += 1
        else:
            dst = _unique_path(dataset_dir, base, ".flac")
            result = subprocess.run(
                [ffmpeg, "-y", "-i", src, "-loglevel", "error", dst],
                capture_output=True, text=True, creationflags=CREATE_NO_WINDOW,
            )
            if result.returncode != 0 or not os.path.exists(dst):
                raise RuntimeError(f"转码失败: {src}\n{result.stderr.strip()}")
            converted += 1

    if converted:
        log(f"🔄 已把 {converted} 个 Applio 不支持的文件转成 FLAC")
    if copied:
        log(f"📥 已复制 {copied} 个文件")
    log(f"📁 数据集: {os.path.abspath(dataset_dir)}")
    return dataset_dir


def resolve_dataset(model_name, folder, picked_files, dataset_root, applio_dir, log=safe_print):
    """决定最终喂给 Applio 的数据集目录。

    用户选的文件夹如果本来就全是 Applio 认识的格式，就直接用，不做任何复制；
    只要掺了 m4a 之类的，或者是散装挑的文件，才汇总/转码到我们自己的目录。
    """
    if picked_files:
        return build_dataset(model_name, picked_files, dataset_root, applio_dir, log)

    files = collect_audio_files(folder)
    if not files:
        raise RuntimeError(f"数据集目录里没有音频文件: {folder}")

    unsupported = needs_conversion(files)
    if not unsupported:
        log(f"📁 数据集: {os.path.abspath(folder)}（{len(files)} 个音频）")
        return folder

    log(f"ℹ️ 目录里有 {len(unsupported)} 个 Applio 不支持的格式，转码后汇总到独立目录（原目录不动）")
    return build_dataset(model_name, files, dataset_root, applio_dir, log)


# ---------------- 每步之后的产物校验 ----------------

def _model_dir(applio_dir, model_name):
    return os.path.join(applio_dir, "logs", model_name)


def _nonempty_dir(path):
    return os.path.isdir(path) and bool(os.listdir(path))


def _verify(condition, step, tail, hint=""):
    """Applio 失败时退出码是 0，只能靠产物在不在来判断。"""
    if condition:
        return
    message = f"{step}失败。"
    if hint:
        message += hint
    detail = tail.text().strip()
    if detail:
        message += f"\nApplio 的输出：\n{detail}"
    raise RuntimeError(message)


# ---------------- 三个步骤 ----------------

def preprocess(model_name, dataset_path, sample_rate, applio_dir,
               cpu_cores=None,
               cut_preprocess="Automatic",
               process_effects=False,
               noise_reduction=False,
               noise_reduction_strength=0.7,
               chunk_len=3.0,
               overlap_len=0.3,
               normalization_mode="none",
               log=safe_print,
               on_start=None):
    cmd = [
        applio_python(applio_dir), "core.py", "preprocess",
        "--model-name", model_name,
        "--dataset-path", os.path.abspath(dataset_path),
        "--sample-rate", str(sample_rate),
        "--cut-preprocess", cut_preprocess,
        "--noise-reduction-strength", str(noise_reduction_strength),
        "--chunk-len", str(chunk_len),
        "--overlap-len", str(overlap_len),
        "--normalization-mode", normalization_mode,
    ]
    if cpu_cores:
        cmd += ["--cpu-cores", str(cpu_cores)]
    if process_effects:
        cmd.append("--process-effects")
    if noise_reduction:
        cmd.append("--noise-reduction")

    tail = _Tail(log)
    run_streaming(cmd, cwd=applio_dir, log=tail, on_start=on_start)
    _verify(_nonempty_dir(os.path.join(_model_dir(applio_dir, model_name), "sliced_audios_16k")),
            "预处理", tail, "没有切出任何音频片段，检查数据集里是不是没有有效人声。")


def extract(model_name, sample_rate, applio_dir,
            f0_method="rmvpe",
            embedder_model="chinese-hubert-base",
            cpu_cores=None,
            gpu="0",
            include_mutes=2,
            log=safe_print,
            on_start=None):
    cmd = [
        applio_python(applio_dir), "core.py", "extract",
        "--model-name", model_name,
        "--sample-rate", str(sample_rate),
        "--f0-method", f0_method,
        "--embedder-model", embedder_model,
        "--gpu", str(gpu),
        "--include-mutes", str(include_mutes),
    ]
    if cpu_cores:
        cmd += ["--cpu-cores", str(cpu_cores)]

    tail = _Tail(log)
    run_streaming(cmd, cwd=applio_dir, log=tail, on_start=on_start)
    model_dir = _model_dir(applio_dir, model_name)
    _verify(os.path.exists(os.path.join(model_dir, "config.json"))
            and os.path.exists(os.path.join(model_dir, "filelist.txt")),
            "特征提取", tail)


def train(model_name, total_epoch, sample_rate, applio_dir,
          batch_size=8,
          save_every_epoch=10,
          gpu="0",
          pretrained=True,
          vocoder="HiFi-GAN",
          cache_data_in_gpu=False,
          checkpointing=False,
          cleanup=False,
          save_only_latest=False,
          index_algorithm="Auto",
          custom_pretrained=False,
          g_pretrained_path=None,
          d_pretrained_path=None,
          log=safe_print,
          on_start=None):
    cmd = [
        applio_python(applio_dir), "core.py", "train",
        "--model-name", model_name,
        "--total-epoch", str(total_epoch),
        "--sample-rate", str(sample_rate),
        "--batch-size", str(batch_size),
        "--save-every-epoch", str(save_every_epoch),
        "--gpu", str(gpu),
        "--vocoder", vocoder,
        "--index-algorithm", index_algorithm,
        "--pretrained" if pretrained else "--no-pretrained",
    ]
    if cache_data_in_gpu:
        cmd.append("--cache-data-in-gpu")
    if checkpointing:
        cmd.append("--checkpointing")
    if cleanup:
        cmd.append("--cleanup")
    if save_only_latest:
        cmd.append("--save-only-latest")
    if custom_pretrained and g_pretrained_path and d_pretrained_path:
        cmd += [
            "--custom-pretrained",
            "--g-pretrained-path", os.path.abspath(g_pretrained_path),
            "--d-pretrained-path", os.path.abspath(d_pretrained_path),
        ]

    tail = _Tail(log)
    run_streaming(cmd, cwd=applio_dir, log=tail, on_start=on_start)
    model_dir = _model_dir(applio_dir, model_name)
    weights = glob.glob(os.path.join(model_dir, f"{model_name}_*e_*s.pth"))
    _verify(bool(weights), "训练", tail, "没有产出任何模型权重。")
    return weights


def run_training(model_name, dataset_path, total_epoch, sample_rate, applio_dir,
                 batch_size=8,
                 save_every_epoch=10,
                 f0_method="rmvpe",
                 embedder_model="chinese-hubert-base",
                 vocoder="HiFi-GAN",
                 gpu="0",
                 cpu_cores=None,
                 pretrained=True,
                 cut_preprocess="Automatic",
                 process_effects=False,
                 noise_reduction=False,
                 noise_reduction_strength=0.7,
                 include_mutes=2,
                 cache_data_in_gpu=False,
                 checkpointing=False,
                 cleanup=False,
                 save_only_latest=False,
                 index_algorithm="Auto",
                 skip_preprocess=False,
                 skip_extract=False,
                 log=safe_print,
                 should_stop=None,
                 on_start=None):
    """完整训练流程。三步之间检查一次停止标志。"""

    def check_stop():
        from procutil import Cancelled
        if should_stop and should_stop():
            raise Cancelled("已停止")

    python_exe = applio_python(applio_dir)
    if not os.path.exists(python_exe):
        raise FileNotFoundError(f"找不到 Applio 的 Python 解释器: {python_exe}")

    if not skip_preprocess:
        check_stop()
        log("\n[1/3] 预处理数据集 (切片 + 重采样)...")
        preprocess(model_name, dataset_path, sample_rate, applio_dir,
                   cpu_cores=cpu_cores, cut_preprocess=cut_preprocess,
                   process_effects=process_effects, noise_reduction=noise_reduction,
                   noise_reduction_strength=noise_reduction_strength,
                   log=log, on_start=on_start)
    else:
        log("\n[1/3] 跳过预处理（沿用上次结果）")

    if not skip_extract:
        check_stop()
        log("\n[2/3] 提取音高与特征...")
        extract(model_name, sample_rate, applio_dir,
                f0_method=f0_method, embedder_model=embedder_model,
                cpu_cores=cpu_cores, gpu=gpu, include_mutes=include_mutes,
                log=log, on_start=on_start)
    else:
        log("\n[2/3] 跳过特征提取（沿用上次结果）")

    check_stop()
    log(f"\n[3/3] 开始训练，共 {total_epoch} epoch（跑完会自动生成 .index）...")
    weights = train(model_name, total_epoch, sample_rate, applio_dir,
                    batch_size=batch_size, save_every_epoch=save_every_epoch, gpu=gpu,
                    pretrained=pretrained, vocoder=vocoder,
                    cache_data_in_gpu=cache_data_in_gpu, checkpointing=checkpointing,
                    cleanup=cleanup, save_only_latest=save_only_latest,
                    index_algorithm=index_algorithm, log=log, on_start=on_start)

    out_dir = _model_dir(applio_dir, model_name)
    log(f"\n🎉 === 训练完成 === 产物在 {out_dir}")
    for w in sorted(weights):
        log(f"   {os.path.basename(w)}")
    index_file = os.path.join(out_dir, f"{model_name}.index")
    if os.path.exists(index_file):
        log(f"   {model_name}.index")
    else:
        log("   ⚠️ 没生成 .index，翻唱时特征检索会被跳过")
    return out_dir
