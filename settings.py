"""界面上那些「路径」类设置，存在 config.json 里，下次开界面自动带出来。"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    # Applio 安装目录，翻唱和训练都靠它
    "applio_dir": r"C:\workspace\Applio",
    # 现成模型的回退目录
    "models_dir": "models",
    # 训练素材汇总到这里
    "dataset_root": os.path.join("data", "original_voice"),
    # 各页上次用的输出目录
    "separate_output_dir": "output",
    "cover_output_dir": "output",
}


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
    return os.path.join(applio_dir, ".venv", "Scripts", "python.exe")
