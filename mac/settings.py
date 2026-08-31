"""路径类设置，存在 mac/config.json 里。（macOS 版）

和 Windows 版的差别只有两处，但都是致命的：

1. Applio 虚拟环境的解释器在 `.venv/bin/python`，不是 `.venv\\Scripts\\python.exe`。
2. 默认安装目录不再是 `C:\\workspace\\Applio`。

配置文件单独放在 mac/ 下，不去碰根目录那份 Windows 的 config.json。
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

CONFIG_PATH = os.path.join(_HERE, "config.json")

DEFAULTS = {
    # Applio 安装目录，翻唱和训练都靠它。对应 Windows 上的 C:\workspace\Applio
    "applio_dir": os.path.expanduser("~/workspace/Applio"),
    # 现成模型的回退目录
    "models_dir": "models",
    # 训练素材汇总到这里
    "dataset_root": os.path.join("data", "original_voice"),
    # 各页上次用的输出目录
    "separate_output_dir": "output",
    "cover_output_dir": "output",
}

# Applio 的安装脚本建的就是 .venv，但手动装/换过环境的情况也兜一下
_VENV_DIRS = (".venv", "venv", "env")


def load():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass  # 配置坏了就用默认值，不要卡住启动
    return cfg


def save(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def applio_python(applio_dir):
    """Applio 虚拟环境里的 python 解释器。

    mac 上是 `<applio>/.venv/bin/python`。找不到时返回默认的那条路径，
    让上层的 "找不到解释器: xxx" 报错里能带上一个用户看得懂的地址。
    """
    applio_dir = os.path.expanduser(applio_dir)
    for venv in _VENV_DIRS:
        candidate = os.path.join(applio_dir, venv, "bin", "python")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(applio_dir, ".venv", "bin", "python")


def project_path(path):
    """把配置里的相对路径按项目根目录解析。

    脚本在 mac/ 下，但 data/ 和 output/ 都在项目根，
    不这么兜一下的话从别的 cwd 跑就会到处找不到文件。
    """
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)
