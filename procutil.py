"""跑子进程并把输出逐行喂给回调，同时支持中途整棵进程树杀掉。

Applio 的 core.py 自己还会再 spawn 一层（train.py / extract.py），
所以停止时必须 taskkill /T，只 terminate 直接子进程杀不干净。
"""

import os
import subprocess
import sys

# 打包成 windowed exe 后不弹黑框
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def configure_console_encoding():
    """Windows 控制台默认是 GBK，日志里的 emoji 会直接抛 UnicodeEncodeError 把程序崩掉。

    打包成 exe 后没有控制台，stdout 可能是 None，所以每一步都得容错。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


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
    """连子孙进程一起杀。proc 为 None 或已退出时静默返回。"""
    if proc is None or proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
        )
    else:
        proc.terminate()


def run_streaming(cmd, cwd=None, log=safe_print, on_start=None, check=True):
    """启动 cmd，把 stdout/stderr 合并后逐行回调给 log。

    tqdm 之类的进度条只吐 \r 不吐 \n，所以这里按字符读、\r 和 \n 都当断行，
    否则训练跑几小时界面上一行日志都不会动。

    on_start(proc) 用来把 Popen 对象交给上层，方便点停止时 kill_tree。
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

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
        creationflags=CREATE_NO_WINDOW,
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
