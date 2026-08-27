"""AI 翻唱工坊 —— 分离 / 翻唱 / 训练 三合一界面。

长任务一律丢后台线程跑，子进程输出通过队列回主线程刷日志，界面不会卡死。
"""

import os
import queue
import threading
import traceback
from tkinter import filedialog

import customtkinter as ctk

import cover
import settings
import trainer
from main import separate_track
from procutil import Cancelled, configure_console_encoding, kill_tree

configure_console_encoding()
print("⏳ 正在加载界面，请稍候...")

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

UI_FONT = ("Microsoft YaHei UI", 13)
TITLE_FONT = ("Microsoft YaHei UI", 20, "bold")
SECTION_FONT = ("Microsoft YaHei UI", 14, "bold")
LOG_FONT = ("Consolas", 12)

AUDIO_FILETYPES = [
    ("音频文件", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.opus *.wma"),
    ("所有文件", "*.*"),
]


# ============================ 可复用小控件 ============================

class PathRow(ctk.CTkFrame):
    """一行「说明 + 可编辑路径框 + 浏览按钮」。mode 为 file / dir / files。"""

    def __init__(self, master, label, mode="file", filetypes=None,
                 initial="", label_width=110, button_text="浏览"):
        super().__init__(master, fg_color="transparent")
        self.mode = mode
        self.filetypes = filetypes or AUDIO_FILETYPES
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text=label, width=label_width, anchor="w", font=UI_FONT).grid(
            row=0, column=0, padx=(0, 8), sticky="w")
        self.entry = ctk.CTkEntry(self, font=UI_FONT)
        self.entry.grid(row=0, column=1, sticky="ew")
        if initial:
            self.entry.insert(0, initial)
        ctk.CTkButton(self, text=button_text, width=70, font=UI_FONT,
                      command=self._browse).grid(row=0, column=2, padx=(8, 0))

    def _browse(self):
        if self.mode == "dir":
            path = filedialog.askdirectory(title="选择文件夹")
        else:
            path = filedialog.askopenfilename(title="选择文件", filetypes=self.filetypes)
        if path:
            self.set(path)

    def get(self):
        return self.entry.get().strip().strip('"')

    def set(self, value):
        self.entry.delete(0, "end")
        self.entry.insert(0, value)


class Slider(ctk.CTkFrame):
    """一行「说明 + 滑块 + 当前值」。"""

    def __init__(self, master, label, from_, to, default,
                 steps=None, fmt="{:.2f}", label_width=110):
        super().__init__(master, fg_color="transparent")
        self.fmt = fmt
        self.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self, text=label, width=label_width, anchor="w", font=UI_FONT).grid(
            row=0, column=0, padx=(0, 8), sticky="w")
        self.slider = ctk.CTkSlider(self, from_=from_, to=to, number_of_steps=steps,
                                    command=self._on_change)
        self.slider.grid(row=0, column=1, sticky="ew")
        self.slider.set(default)
        self.value_label = ctk.CTkLabel(self, text=fmt.format(default), width=48, font=UI_FONT)
        self.value_label.grid(row=0, column=2, padx=(8, 0))

    def _on_change(self, value):
        self.value_label.configure(text=self.fmt.format(value))

    def get(self):
        return self.slider.get()

    def get_int(self):
        return int(round(self.slider.get()))


class LabeledEntry(ctk.CTkFrame):
    """一行「说明 + 小输入框」，用于 epoch、batch 这种要精确填的数字。"""

    def __init__(self, master, label, default="", width=90, label_width=110):
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, width=label_width, anchor="w", font=UI_FONT).grid(
            row=0, column=0, padx=(0, 8), sticky="w")
        self.entry = ctk.CTkEntry(self, width=width, font=UI_FONT)
        self.entry.grid(row=0, column=1, sticky="w")
        if default != "":
            self.entry.insert(0, str(default))

    def get(self):
        return self.entry.get().strip()

    def get_int(self, default):
        try:
            return int(self.get())
        except ValueError:
            return default

    def get_float(self, default):
        try:
            return float(self.get())
        except ValueError:
            return default


class LabeledMenu(ctk.CTkFrame):
    """一行「说明 + 下拉框」。"""

    def __init__(self, master, label, values, default=None, width=180, label_width=110):
        super().__init__(master, fg_color="transparent")
        values = list(values) or ["(空)"]
        ctk.CTkLabel(self, text=label, width=label_width, anchor="w", font=UI_FONT).grid(
            row=0, column=0, padx=(0, 8), sticky="w")
        self.var = ctk.StringVar(value=default if default in values else values[0])
        self.menu = ctk.CTkOptionMenu(self, values=values, variable=self.var,
                                      width=width, font=UI_FONT)
        self.menu.grid(row=0, column=1, sticky="w")

    def get(self):
        return self.var.get()

    def set_values(self, values, keep=True):
        values = list(values) or ["(空)"]
        current = self.var.get()
        self.menu.configure(values=values)
        if not (keep and current in values):
            self.var.set(values[0])


def section(master, text):
    return ctk.CTkLabel(master, text=text, font=SECTION_FONT, anchor="w")


# ============================ 主窗口 ============================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI 翻唱工坊 — 分离 / 翻唱 / 训练")
        self.geometry("980x900")
        self.minsize(880, 700)

        self.cfg = settings.load()
        self.msg_queue = queue.Queue()
        self.busy = False
        self.stop_requested = False
        self.active_proc = None
        self.run_buttons = []
        self.train_files = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)
        self.grid_rowconfigure(3, weight=2)

        self._build_topbar()
        self._build_tabs()
        self._build_statusbar()
        self._build_log()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(60, self._drain_queue)

        self.log("✨ 界面启动成功。")
        self.log(f"💡 Applio 目录: {self.cfg['applio_dir']}")
        if not os.path.exists(settings.applio_python(self.cfg["applio_dir"])):
            self.log("⚠️ 该目录下没找到 Applio 的 .venv/Scripts/python.exe，翻唱和训练会失败，请在上方改成正确路径。")

    # ---------------- 布局 ----------------

    def _build_topbar(self):
        bar = ctk.CTkFrame(self)
        bar.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        bar.grid_columnconfigure(0, weight=1)

        self.applio_row = PathRow(bar, "Applio 目录", mode="dir",
                                  initial=self.cfg["applio_dir"], label_width=90)
        self.applio_row.grid(row=0, column=0, padx=12, pady=10, sticky="ew")
        ctk.CTkButton(bar, text="保存路径", width=90, font=UI_FONT,
                      command=self.save_paths).grid(row=0, column=1, padx=(0, 12))

    def _build_tabs(self):
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")
        self.tab_sep = self.tabs.add("🎚 分离")
        self.tab_cover = self.tabs.add("🎤 翻唱")
        self.tab_train = self.tabs.add("🏋 训练")
        for tab in (self.tab_sep, self.tab_cover, self.tab_train):
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        self._build_separate_tab()
        self._build_cover_tab()
        self._build_train_tab()

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self)
        bar.grid(row=2, column=0, padx=12, pady=6, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(bar, text="就绪", anchor="w", font=UI_FONT)
        self.status_label.grid(row=0, column=0, padx=12, pady=8, sticky="ew")

        self.progress = ctk.CTkProgressBar(bar, width=180, mode="indeterminate")
        self.progress.grid(row=0, column=1, padx=(0, 12))
        self.progress.set(0)

        self.stop_button = ctk.CTkButton(bar, text="⏹ 停止", width=90, state="disabled",
                                         fg_color="#a63a3a", hover_color="#7d2c2c",
                                         font=UI_FONT, command=self.request_stop)
        self.stop_button.grid(row=0, column=2, padx=(0, 12))

    def _build_log(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=3, column=0, padx=12, pady=(6, 12), sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.grid(row=0, column=0, padx=12, pady=(8, 0), sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="运行日志", font=SECTION_FONT, anchor="w").grid(
            row=0, column=0, sticky="w")
        ctk.CTkButton(header, text="清空", width=60, font=UI_FONT,
                      command=self.clear_log).grid(row=0, column=1)

        self.log_box = ctk.CTkTextbox(frame, font=LOG_FONT, wrap="none")
        self.log_box.grid(row=1, column=0, padx=12, pady=(6, 12), sticky="nsew")
        self.log_box.configure(state="disabled")

    # ---------------- 分离页 ----------------

    def _build_separate_tab(self):
        body = ctk.CTkScrollableFrame(self.tab_sep, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        r = 0

        ctk.CTkLabel(body, text="三阶段分离：伴奏 / 主唱干声 / 和声 / 混响残留",
                     font=TITLE_FONT).grid(row=r, column=0, pady=(6, 14), sticky="w")
        r += 1

        self.sep_input = PathRow(body, "音频文件")
        self.sep_input.grid(row=r, column=0, pady=6, sticky="ew")
        r += 1

        self.sep_output = PathRow(body, "输出目录", mode="dir",
                                  initial=self.cfg["separate_output_dir"])
        self.sep_output.grid(row=r, column=0, pady=6, sticky="ew")
        r += 1

        self.sep_format = LabeledMenu(body, "输出格式", ["flac", "wav", "mp3"], "flac")
        self.sep_format.grid(row=r, column=0, pady=6, sticky="ew")
        r += 1

        self.sep_subdir = ctk.CTkCheckBox(body, text="输出到以歌名命名的子文件夹（推荐，避免多首歌互相覆盖）",
                                          font=UI_FONT)
        self.sep_subdir.select()
        self.sep_subdir.grid(row=r, column=0, pady=(6, 14), sticky="w")
        r += 1

        btn = ctk.CTkButton(body, text="🚀 开始分离", height=46,
                            font=("Microsoft YaHei UI", 16, "bold"),
                            command=self.start_separate)
        btn.grid(row=r, column=0, pady=8, sticky="ew")
        self.run_buttons.append(btn)

    # ---------------- 翻唱页 ----------------

    def _build_cover_tab(self):
        body = ctk.CTkScrollableFrame(self.tab_cover, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        r = 0

        ctk.CTkLabel(body, text="AI 翻唱：分离 → RVC 变声 → 混音",
                     font=TITLE_FONT).grid(row=r, column=0, pady=(6, 14), sticky="w")
        r += 1

        self.cov_input = PathRow(body, "音频文件")
        self.cov_input.grid(row=r, column=0, pady=6, sticky="ew")
        r += 1

        self.cov_output = PathRow(body, "输出目录", mode="dir",
                                  initial=self.cfg["cover_output_dir"])
        self.cov_output.grid(row=r, column=0, pady=6, sticky="ew")
        r += 1

        # --- 模型 ---
        model_row = ctk.CTkFrame(body, fg_color="transparent")
        model_row.grid(row=r, column=0, pady=6, sticky="ew")
        model_row.grid_columnconfigure(0, weight=1)
        self.cov_model = LabeledMenu(model_row, "音色模型", self._scan_models(), width=240)
        self.cov_model.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(model_row, text="刷新", width=60, font=UI_FONT,
                      command=self.refresh_models).grid(row=0, column=1, padx=8)
        r += 1

        ctk.CTkLabel(body, text="或手动指定模型文件（两个都留空则用上面的下拉）",
                     font=UI_FONT, text_color="gray").grid(row=r, column=0, pady=(8, 2), sticky="w")
        r += 1

        self.cov_pth = PathRow(body, ".pth 文件",
                               filetypes=[("RVC 模型", "*.pth"), ("所有文件", "*.*")])
        self.cov_pth.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.cov_index = PathRow(body, ".index 文件",
                                 filetypes=[("检索索引", "*.index"), ("所有文件", "*.*")])
        self.cov_index.grid(row=r, column=0, pady=(4, 14), sticky="ew")
        r += 1

        # --- 转换参数 ---
        section(body, "转换参数").grid(row=r, column=0, pady=(6, 4), sticky="w")
        r += 1

        self.cov_pitch = Slider(body, "升降调(半音)", -12, 12, 0, steps=24, fmt="{:+.0f}")
        self.cov_pitch.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.cov_index_rate = Slider(body, "特征检索强度", 0, 1, cover.INDEX_RATE, steps=100)
        self.cov_index_rate.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.cov_f0 = LabeledMenu(body, "音高算法", cover.F0_METHODS, cover.F0_METHOD)
        self.cov_f0.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.cov_embedder = LabeledMenu(body, "特征提取器", cover.EMBEDDER_MODELS,
                                        cover.EMBEDDER_MODEL, width=220)
        self.cov_embedder.grid(row=r, column=0, pady=(4, 14), sticky="ew")
        r += 1

        # --- 混响 ---
        self.cov_reverb_on = ctk.CTkCheckBox(body, text="给转换后的人声加混响", font=SECTION_FONT,
                                             command=self._toggle_reverb)
        self.cov_reverb_on.select()
        self.cov_reverb_on.grid(row=r, column=0, pady=(6, 6), sticky="w")
        r += 1

        self.reverb_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.reverb_frame.grid(row=r, column=0, pady=(0, 14), sticky="ew")
        self.reverb_frame.grid_columnconfigure(0, weight=1)
        r += 1

        self.cov_room = Slider(self.reverb_frame, "房间大小", 0, 1, cover.REVERB_ROOM_SIZE, steps=100)
        self.cov_room.grid(row=0, column=0, pady=4, sticky="ew")
        self.cov_damping = Slider(self.reverb_frame, "阻尼", 0, 1, cover.REVERB_DAMPING, steps=100)
        self.cov_damping.grid(row=1, column=0, pady=4, sticky="ew")
        self.cov_wet = Slider(self.reverb_frame, "湿声(混响)", 0, 1, cover.REVERB_WET_GAIN, steps=100)
        self.cov_wet.grid(row=2, column=0, pady=4, sticky="ew")
        self.cov_dry = Slider(self.reverb_frame, "干声(原声)", 0, 1, cover.REVERB_DRY_GAIN, steps=100)
        self.cov_dry.grid(row=3, column=0, pady=4, sticky="ew")
        self.cov_width = Slider(self.reverb_frame, "立体声宽度", 0, 1, cover.REVERB_WIDTH, steps=100)
        self.cov_width.grid(row=4, column=0, pady=4, sticky="ew")

        # --- 混音 ---
        section(body, "混音音量").grid(row=r, column=0, pady=(6, 4), sticky="w")
        r += 1

        self.cov_vocal_gain = Slider(body, "AI 人声", 0, 2, cover.VOCAL_GAIN, steps=200)
        self.cov_vocal_gain.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.cov_inst_gain = Slider(body, "伴奏", 0, 2, cover.INSTRUMENTAL_GAIN, steps=200)
        self.cov_inst_gain.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.cov_backing_gain = Slider(body, "和声", 0, 2, cover.BACKING_VOCAL_GAIN, steps=200)
        self.cov_backing_gain.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.cov_include_backing = ctk.CTkCheckBox(body, text="把原曲和声混进成品", font=UI_FONT)
        self.cov_include_backing.select()
        self.cov_include_backing.grid(row=r, column=0, pady=(6, 14), sticky="w")
        r += 1

        btn = ctk.CTkButton(body, text="🎤 开始翻唱", height=46,
                            font=("Microsoft YaHei UI", 16, "bold"),
                            command=self.start_cover)
        btn.grid(row=r, column=0, pady=8, sticky="ew")
        self.run_buttons.append(btn)

    def _toggle_reverb(self):
        state = "normal" if self.cov_reverb_on.get() else "disabled"
        for s in (self.cov_room, self.cov_damping, self.cov_wet, self.cov_dry, self.cov_width):
            s.slider.configure(state=state)

    def _scan_models(self):
        try:
            found = cover.list_models(self.applio_dir(), self.cfg["models_dir"])
        except OSError:
            found = []
        return found or ["(未找到模型)"]

    def refresh_models(self):
        models = self._scan_models()
        self.cov_model.set_values(models)
        self.log(f"🔄 扫描到 {len(models)} 个音色模型: {', '.join(models)}")

    # ---------------- 训练页 ----------------

    def _build_train_tab(self):
        body = ctk.CTkScrollableFrame(self.tab_train, fg_color="transparent")
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        r = 0

        ctk.CTkLabel(body, text="训练音色模型：预处理 → 特征提取 → 训练",
                     font=TITLE_FONT).grid(row=r, column=0, pady=(6, 14), sticky="w")
        r += 1

        self.tr_name = LabeledEntry(body, "模型名", "", width=220)
        self.tr_name.grid(row=r, column=0, pady=6, sticky="ew")
        r += 1

        # --- 数据集 ---
        section(body, "训练素材").grid(row=r, column=0, pady=(10, 4), sticky="w")
        r += 1

        self.tr_dataset = PathRow(body, "数据集文件夹", mode="dir")
        self.tr_dataset.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        file_bar = ctk.CTkFrame(body, fg_color="transparent")
        file_bar.grid(row=r, column=0, pady=4, sticky="ew")
        file_bar.grid_columnconfigure(2, weight=1)
        ctk.CTkButton(file_bar, text="➕ 添加文件(可多选)", width=150, font=UI_FONT,
                      command=self.add_train_files).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkButton(file_bar, text="清空", width=60, font=UI_FONT,
                      command=self.clear_train_files).grid(row=0, column=1, padx=(0, 8))
        self.tr_files_label = ctk.CTkLabel(file_bar, text="未选择散装文件", anchor="w",
                                           font=UI_FONT, text_color="gray")
        self.tr_files_label.grid(row=0, column=2, sticky="ew")
        r += 1

        ctk.CTkLabel(body,
                     text="两种方式二选一：选一个现成文件夹，或用「添加文件」挑散落各处的音频"
                          "（会汇总到 data/original_voice/<模型名>/）。"
                          "Applio 只认 wav/mp3/flac/ogg，m4a 等格式会自动转成 FLAC，原文件不动。",
                     font=UI_FONT, text_color="gray", wraplength=800, justify="left").grid(
            row=r, column=0, pady=(2, 14), sticky="w")
        r += 1

        # --- 主要参数 ---
        section(body, "主要参数").grid(row=r, column=0, pady=(6, 4), sticky="w")
        r += 1

        self.tr_epoch = LabeledEntry(body, "总 epoch", 200)
        self.tr_epoch.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.tr_batch = LabeledEntry(body, "批大小", 8)
        self.tr_batch.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.tr_sr = LabeledMenu(body, "采样率", trainer.SAMPLE_RATES, "40000")
        self.tr_sr.grid(row=r, column=0, pady=4, sticky="ew")
        r += 1

        self.tr_save_every = LabeledEntry(body, "保存间隔", 10)
        self.tr_save_every.grid(row=r, column=0, pady=(4, 12), sticky="ew")
        r += 1

        # --- 高级设置（折叠） ---
        self.adv_button = ctk.CTkButton(body, text="▶ 高级设置", anchor="w", height=34,
                                        fg_color="transparent", border_width=1,
                                        text_color=("black", "white"), font=UI_FONT,
                                        command=self.toggle_advanced)
        self.adv_button.grid(row=r, column=0, pady=(6, 4), sticky="ew")
        r += 1

        self.adv_frame = ctk.CTkFrame(body)
        self.adv_frame.grid(row=r, column=0, pady=(0, 12), sticky="ew")
        self.adv_frame.grid_columnconfigure(0, weight=1)
        self._build_advanced(self.adv_frame)
        self.adv_frame.grid_remove()
        self.adv_visible = False
        r += 1

        btn = ctk.CTkButton(body, text="🏋 开始训练", height=46,
                            font=("Microsoft YaHei UI", 16, "bold"),
                            command=self.start_train)
        btn.grid(row=r, column=0, pady=8, sticky="ew")
        self.run_buttons.append(btn)

    def _build_advanced(self, parent):
        pad = {"padx": 12, "pady": 4, "sticky": "ew"}
        i = 0

        self.tr_f0 = LabeledMenu(parent, "音高算法", trainer.F0_METHODS, "rmvpe")
        self.tr_f0.grid(row=i, column=0, **pad); i += 1

        self.tr_embedder = LabeledMenu(parent, "特征提取器", trainer.EMBEDDER_MODELS,
                                       "chinese-hubert-base", width=220)
        self.tr_embedder.grid(row=i, column=0, **pad); i += 1

        self.tr_vocoder = LabeledMenu(parent, "声码器", trainer.VOCODERS, "HiFi-GAN")
        self.tr_vocoder.grid(row=i, column=0, **pad); i += 1

        self.tr_cut = LabeledMenu(parent, "切片方式", trainer.CUT_PREPROCESS, "Automatic")
        self.tr_cut.grid(row=i, column=0, **pad); i += 1

        self.tr_index_algo = LabeledMenu(parent, "索引算法", trainer.INDEX_ALGORITHMS, "Auto")
        self.tr_index_algo.grid(row=i, column=0, **pad); i += 1

        self.tr_gpu = LabeledEntry(parent, "GPU 编号", "0", width=90)
        self.tr_gpu.grid(row=i, column=0, **pad); i += 1

        self.tr_cpu = LabeledEntry(parent, "CPU 核数", "", width=90)
        self.tr_cpu.grid(row=i, column=0, **pad); i += 1

        self.tr_mutes = LabeledEntry(parent, "静音样本数", 2, width=90)
        self.tr_mutes.grid(row=i, column=0, **pad); i += 1

        self.tr_nr_strength = LabeledEntry(parent, "降噪强度", 0.7, width=90)
        self.tr_nr_strength.grid(row=i, column=0, **pad); i += 1

        checks = ctk.CTkFrame(parent, fg_color="transparent")
        checks.grid(row=i, column=0, padx=12, pady=(8, 12), sticky="ew")

        def check(text, row, col, selected=False):
            cb = ctk.CTkCheckBox(checks, text=text, font=UI_FONT)
            cb.grid(row=row, column=col, padx=(0, 20), pady=4, sticky="w")
            if selected:
                cb.select()
            return cb

        self.tr_pretrained = check("使用预训练底模（强烈建议）", 0, 0, selected=True)
        self.tr_noise_reduction = check("预处理时降噪", 0, 1)
        self.tr_process_effects = check("预处理时应用滤波", 1, 0)
        self.tr_cache_gpu = check("数据缓存到显存", 1, 1)
        self.tr_checkpointing = check("梯度检查点（省显存、更慢）", 2, 0)
        self.tr_save_only_latest = check("只保留最新检查点", 2, 1)
        self.tr_cleanup = check("清理上次训练残留（从头开始）", 3, 0)
        self.tr_skip_preprocess = check("跳过预处理", 4, 0)
        self.tr_skip_extract = check("跳过特征提取", 4, 1)

    def toggle_advanced(self):
        if self.adv_visible:
            self.adv_frame.grid_remove()
            self.adv_button.configure(text="▶ 高级设置")
        else:
            self.adv_frame.grid()
            self.adv_button.configure(text="▼ 高级设置")
        self.adv_visible = not self.adv_visible

    def add_train_files(self):
        paths = filedialog.askopenfilenames(title="选择训练音频（可多选）",
                                            filetypes=AUDIO_FILETYPES)
        if not paths:
            return
        for p in paths:
            if p not in self.train_files:
                self.train_files.append(p)
        self.tr_files_label.configure(text=f"已选 {len(self.train_files)} 个文件，正在统计时长...")
        threading.Thread(target=self._update_dataset_stats, daemon=True).start()

    def clear_train_files(self):
        self.train_files = []
        self.tr_files_label.configure(text="未选择散装文件")

    def _update_dataset_stats(self):
        files = list(self.train_files)
        count, seconds = trainer.dataset_stats(files, self.applio_dir())
        if seconds > 0:
            text = f"已选 {count} 个文件，总时长 {int(seconds // 60)} 分 {int(seconds % 60)} 秒"
        else:
            text = f"已选 {count} 个文件"
        self.msg_queue.put(("train_files", text))

    # ---------------- 通用：日志 / 任务调度 ----------------

    def applio_dir(self):
        return self.applio_row.get() or self.cfg["applio_dir"]

    def save_paths(self):
        self.cfg["applio_dir"] = self.applio_row.get()
        self.cfg["separate_output_dir"] = self.sep_output.get()
        self.cfg["cover_output_dir"] = self.cov_output.get()
        settings.save(self.cfg)
        self.log(f"💾 已保存路径设置到 {settings.CONFIG_PATH}")
        self.refresh_models()

    def log(self, message):
        self.msg_queue.put(("log", str(message)))

    def clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", payload + "\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif kind == "status":
                    self.status_label.configure(text=payload)
                elif kind == "train_files":
                    self.tr_files_label.configure(text=payload)
                elif kind == "finish":
                    self._on_task_finished(payload)
        except queue.Empty:
            pass
        self.after(60, self._drain_queue)

    def set_busy(self, busy):
        self.busy = busy
        for btn in self.run_buttons:
            btn.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        if busy:
            self.progress.start()
        else:
            self.progress.stop()
            self.progress.set(0)

    def should_stop(self):
        return self.stop_requested

    def on_proc_start(self, proc):
        self.active_proc = proc
        # 停止是在子进程起来之前点的，补一刀
        if self.stop_requested:
            kill_tree(proc)

    def request_stop(self):
        if not self.busy:
            return
        self.stop_requested = True
        self.status_label.configure(text="正在停止...")
        self.log("⏹ 已请求停止（分离阶段会在当前阶段跑完后退出，子进程会被立即结束）")
        kill_tree(self.active_proc)

    def start_task(self, name, func):
        if self.busy:
            self.log("⚠️ 已有任务在跑，请先等它结束或点停止。")
            return
        self.stop_requested = False
        self.active_proc = None
        self.set_busy(True)
        self.status_label.configure(text=f"正在{name}...")
        threading.Thread(target=self._run_task, args=(name, func), daemon=True).start()

    def _run_task(self, name, func):
        try:
            func()
            self.msg_queue.put(("finish", f"✅ {name}完成"))
        except Cancelled:
            self.msg_queue.put(("finish", f"⏹ {name}已停止"))
        except Exception as exc:
            if self.stop_requested:
                # 停止时子进程被 kill，报错是预期的，不当成失败
                self.msg_queue.put(("finish", f"⏹ {name}已停止"))
            else:
                self.msg_queue.put(("log", traceback.format_exc()))
                self.msg_queue.put(("finish", f"❌ {name}失败: {exc}"))

    def _on_task_finished(self, message):
        self.set_busy(False)
        self.active_proc = None
        self.status_label.configure(text=message)
        self.log("\n" + message)

    def on_close(self):
        if self.busy:
            self.stop_requested = True
            kill_tree(self.active_proc)
        self.destroy()

    # ---------------- 三个任务的入口 ----------------

    def _resolve_output_dir(self, base_dir, song_path, use_subdir=True):
        base_dir = base_dir or "output"
        if not use_subdir:
            return base_dir
        stem = os.path.splitext(os.path.basename(song_path))[0]
        return os.path.join(base_dir, stem)

    def start_separate(self):
        song = self.sep_input.get()
        if not song:
            self.log("⚠️ 请先选择要分离的音频文件。")
            return
        if not os.path.exists(song):
            self.log(f"⚠️ 文件不存在: {song}")
            return

        out_dir = self._resolve_output_dir(self.sep_output.get(), song,
                                           bool(self.sep_subdir.get()))
        fmt = self.sep_format.get()

        def job():
            self.log(f"📂 输出目录: {os.path.abspath(out_dir)}")
            stems = separate_track(song, out_dir, fmt,
                                   log=self.log, should_stop=self.should_stop)
            for key, path in stems.items():
                if path:
                    self.log(f"   {key}: {path}")

        self.start_task("分离", job)

    def start_cover(self):
        song = self.cov_input.get()
        if not song:
            self.log("⚠️ 请先选择要翻唱的音频文件。")
            return
        if not os.path.exists(song):
            self.log(f"⚠️ 文件不存在: {song}")
            return

        pth = self.cov_pth.get()
        index = self.cov_index.get()
        model_name = self.cov_model.get()

        if pth:
            if not os.path.exists(pth):
                self.log(f"⚠️ .pth 文件不存在: {pth}")
                return
            model_name = os.path.splitext(os.path.basename(pth))[0]
        elif model_name in ("(未找到模型)", "(空)"):
            self.log("⚠️ 没有可用的音色模型：先训练一个，或用「浏览」手动指定 .pth。")
            return

        song_name = os.path.splitext(os.path.basename(song))[0]
        output_root = self.cov_output.get() or "output"
        reverb_on = bool(self.cov_reverb_on.get())

        params = dict(
            pitch=self.cov_pitch.get_int(),
            f0_method=self.cov_f0.get(),
            embedder_model=self.cov_embedder.get(),
            index_rate=round(self.cov_index_rate.get(), 3),
            reverb=reverb_on,
            reverb_room_size=round(self.cov_room.get(), 3),
            reverb_damping=round(self.cov_damping.get(), 3),
            reverb_wet_gain=round(self.cov_wet.get(), 3),
            reverb_dry_gain=round(self.cov_dry.get(), 3),
            reverb_width=round(self.cov_width.get(), 3),
            vocal_gain=round(self.cov_vocal_gain.get(), 3),
            instrumental_gain=round(self.cov_inst_gain.get(), 3),
            backing_vocal_gain=round(self.cov_backing_gain.get(), 3),
            include_backing=bool(self.cov_include_backing.get()),
            output_root=output_root,
            applio_dir=self.applio_dir(),
            models_dir=self.cfg["models_dir"],
            pth_path=pth or None,
            index_path=index or None,
        )

        def job():
            cover.run_cover(song, song_name, model_name,
                            log=self.log, should_stop=self.should_stop,
                            on_start=self.on_proc_start, **params)

        self.start_task("翻唱", job)

    def start_train(self):
        model_name = self.tr_name.get()
        if not model_name:
            self.log("⚠️ 请先填模型名（会作为 Applio logs 下的文件夹名）。")
            return

        folder = self.tr_dataset.get()
        picked = list(self.train_files)
        if not folder and not picked:
            self.log("⚠️ 请选一个数据集文件夹，或用「添加文件」挑几个音频。")
            return
        if folder and not os.path.isdir(folder):
            self.log(f"⚠️ 数据集文件夹不存在: {folder}")
            return

        applio_dir = self.applio_dir()
        dataset_root = self.cfg["dataset_root"]
        skip_pre = bool(self.tr_skip_preprocess.get())

        params = dict(
            total_epoch=self.tr_epoch.get_int(200),
            sample_rate=int(self.tr_sr.get()),
            batch_size=self.tr_batch.get_int(8),
            save_every_epoch=self.tr_save_every.get_int(10),
            f0_method=self.tr_f0.get(),
            embedder_model=self.tr_embedder.get(),
            vocoder=self.tr_vocoder.get(),
            cut_preprocess=self.tr_cut.get(),
            index_algorithm=self.tr_index_algo.get(),
            gpu=self.tr_gpu.get() or "0",
            cpu_cores=self.tr_cpu.get_int(0) or None,
            include_mutes=self.tr_mutes.get_int(2),
            noise_reduction_strength=self.tr_nr_strength.get_float(0.7),
            pretrained=bool(self.tr_pretrained.get()),
            noise_reduction=bool(self.tr_noise_reduction.get()),
            process_effects=bool(self.tr_process_effects.get()),
            cache_data_in_gpu=bool(self.tr_cache_gpu.get()),
            checkpointing=bool(self.tr_checkpointing.get()),
            save_only_latest=bool(self.tr_save_only_latest.get()),
            cleanup=bool(self.tr_cleanup.get()),
            skip_preprocess=skip_pre,
            skip_extract=bool(self.tr_skip_extract.get()),
        )

        def job():
            if skip_pre:
                # 沿用上次的切片结果，数据集路径只是占位，Applio 不会再读它
                dataset_path = folder or os.path.join(dataset_root, model_name)
                self.log("↩️ 跳过预处理，沿用 Applio logs 里上次的切片结果")
            else:
                dataset_path = trainer.resolve_dataset(model_name, folder, picked,
                                                       dataset_root, applio_dir,
                                                       log=self.log)

            trainer.run_training(model_name, dataset_path, applio_dir=applio_dir,
                                 log=self.log, should_stop=self.should_stop,
                                 on_start=self.on_proc_start, **params)
            self.msg_queue.put(("log", "🔄 记得回翻唱页点「刷新」，新模型才会出现在下拉框里。"))

        self.start_task("训练", job)


if __name__ == "__main__":
    App().mainloop()
