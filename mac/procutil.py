"""跑子进程并把输出逐行喂给回调，同时支持中途整棵进程树杀掉。（macOS 版）

和 Windows 版的两处关键差别：

1. `creationflags` 在 POSIX 上传非 0 会直接抛 ValueError，所以这里整个参数不传。
2. Applio 的 core.py 自己还会再 spawn 一层（train.py / extract.py），
   Windows 靠 `taskkill /T` 连根拔起；mac 上没有这东西，
   只 `terminate()` 杀得掉 core.py，杀不掉它下面那层 train.py——
   训练会变成"界面显示已停止，后台还在烧 GPU"。
   所以这里让子进程自己开一个进程组（start_new_session），停止时 killpg 整组。
"""

import os
import shutil
import signal
import subprocess
import sys

# Homebrew 的 bin 不在 GUI（launchd）继承的 PATH 里。
# 从终端跑一般没事，双击/从访达启动时 shutil.which("ffmpeg") 会直接找不到。
_HOMEBREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")

# 调 Applio 的 core.py 时必须补的环境变量。
# Applio 自己的 run-applio.sh 在 mac 上会先 export 这两个再起 app.py，
# 而我们是绕过那个脚本直接调 core.py 的，不补就享受不到：
#   PYTORCH_ENABLE_MPS_FALLBACK=1   MPS 没实现的算子回退到 CPU，
#                                   否则训练跑到一半直接抛 "not implemented for MPS"
#   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  关掉 MPS 显存水位限制（跟上游保持一致）
APPLIO_MPS_ENV = {
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
}

# RVC 推理专用。除上面两个外还要把 OpenMP 摁成单线程：
#
# faiss-cpu 和 torch 在 macOS 上各自链接了一份 libomp.dylib
# （otool -L 能看到 faiss 用 @loader_path/.dylibs/libomp.dylib，
#  torch 用 @rpath/libomp.dylib），同一个进程里初始化两份 OpenMP 运行时
# 会报 "OMP: Error #15" 然后直接 SIGSEGV。
# 推理正好同时踩这两个：torch 跑模型 + faiss 做特征检索，
# 所以只要传了 --index-path 就必崩，不传就没事 —— 实测四种组合已确认与混响无关。
#
# 坑中坑：网上通行的 KMP_DUPLICATE_LIB_OK=TRUE 在这里**没用**，实测照样 rc=139。
# 它只是放行重复加载，然后在 faiss 的并行区里崩。真正管用的是 OMP_NUM_THREADS=1，
# 连跑 3 次全过。
#
# 只给推理用：训练/特征提取不涉及 faiss 与 torch 同进程，
# 摁成单线程只会白白拖慢本来就跑 CPU 的提取那步。
APPLIO_INFER_ENV = {**APPLIO_MPS_ENV, "OMP_NUM_THREADS": "1"}


def configure_console_encoding():
    """mac 上 stdout 本来就是 UTF-8，这里基本是空转。

    保留是因为重定向到文件、或者从 launchd 起（locale 是 C）时仍可能不是 UTF-8，
    而日志里全是 emoji 和中文。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def find_tool(name):
    """找 ffmpeg / ffprobe 这类命令行工具，找不到返回 None。

    先看 PATH，再补一遍 Homebrew 的两个 bin 目录（Intel 装在 /usr/local）。
    Windows 版 Applio 目录下自带 ffmpeg.exe，mac 版没有，所以这里不找 Applio 目录。
    """
    found = shutil.which(name)
    if found:
        return found

    for directory in _HOMEBREW_BIN_DIRS:
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def ensure_tools_on_path():
    """把 Homebrew 的 bin 补进本进程的 PATH。

    非做不可：audio-separator 在 `Separator.__init__` 里就会
    `subprocess.check_output(["ffmpeg", "-version"])` —— 裸命令名，只认 PATH。
    我们这边 find_tool 能在 /opt/homebrew/bin 里找到也没用，它自己找不到就直接抛
    FileNotFoundError，连分离都起不来。

    从终端跑通常 PATH 里本来就有；从 launchd / GUI 起的进程继承的是一份精简 PATH，
    里面没有 /opt/homebrew/bin。以后做 mac 版 app 时这一步同样要先调。

    子进程也顺带受益：run_streaming 是 copy 当前 os.environ 的。
    """
    parts = os.environ.get("PATH", "").split(os.pathsep)
    added = [d for d in _HOMEBREW_BIN_DIRS if os.path.isdir(d) and d not in parts]
    if added:
        os.environ["PATH"] = os.pathsep.join([*parts, *added])
    return added


def require_tool(name):
    """找不到就抛错，并给一句能直接照抄的安装命令。"""
    found = find_tool(name)
    if not found:
        raise FileNotFoundError(
            f"找不到 {name}。请先装 ffmpeg：\n    brew install ffmpeg"
        )
    return found


class Cancelled(RuntimeError):
    """用户中途点了停止。"""


def safe_print(message):
    """print 的安全版。

    控制台编码不支持某个字符时降级成占位符，而不是抛 UnicodeEncodeError ——
    绝不能因为日志里的一个 emoji 就把跑了几小时的训练打断。
    """
    try:
        print(message)
    except UnicodeEncodeError:
        enc = (sys.stdout.encoding if sys.stdout else None) or "utf-8"
        print(str(message).encode(enc, errors="replace").decode(enc, errors="replace"))


def kill_tree(proc):
    """连子孙进程一起杀。proc 为 None 或已退出时静默返回。

    run_streaming 用 start_new_session 把子进程放进了它自己的进程组（组号 == 它的 pid），
    所以 killpg 能一次带走 Applio 再 spawn 出来的那层。

    这里要先校验"它确实是组长"：万一传进来的 proc 不是 run_streaming 起的，
    它的进程组可能就是我们自己这一组，killpg 会把脚本/界面本身也一起干掉。
    """
    if proc is None or proc.poll() is not None:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None

    # 不是自成一组，就退回到只杀直接子进程——杀不干净也好过杀到自己头上
    if pgid is None or pgid != proc.pid or pgid == os.getpgid(0):
        proc.terminate()
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
        return

    # 给 3 秒收尾（torch 要放显存、写 checkpoint），赖着不走就 SIGKILL
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()


def run_streaming(cmd, cwd=None, log=safe_print, on_start=None, check=True, env_extra=None):
    """启动 cmd，把 stdout/stderr 合并后逐行回调给 log。

    tqdm 之类的进度条只吐 \r 不吐 \n，所以这里按字符读、\r 和 \n 都当断行，
    否则训练跑几小时界面上一行日志都不会动。

    on_start(proc) 用来把 Popen 对象交给上层，方便点停止时 kill_tree。

    env_extra 用来补子进程专属的环境变量（调 Applio 时传 APPLIO_MPS_ENV）。
    """
    ensure_tools_on_path()   # 子进程继承的是这份 PATH，Applio 内部也要调 ffmpeg
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update(env_extra)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        # 单独开一个进程组（顺带脱离终端），kill_tree 才能一次带走整棵树
        start_new_session=True,
    )
    if on_start:
        on_start(proc)

    buf = []
    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in ("\r", "\n"):
                line = "".join(buf).rstrip()
                buf.clear()
                if line:
                    log(line)
            else:
                buf.append(ch)
        tail = "".join(buf).rstrip()
        if tail:
            log(tail)
    finally:
        proc.stdout.close()
        code = proc.wait()

    if check and code != 0:
        raise subprocess.CalledProcessError(code, cmd)
    return code
