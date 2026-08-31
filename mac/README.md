# Mac 版脚本

根目录那套是 Windows 版，**一行没动**。这里是同一套流程的 macOS（Apple Silicon）版本，
两套并存，互不影响。暂时只有命令行脚本，没有 GUI。

## 装环境

```bash
uv sync
brew install ffmpeg
```

`uv sync` 会按 `pyproject.toml` 里的平台标记自动挑：mac 装 `audio-separator[cpu]`
和 PyPI 官方的 arm64 torch（自带 MPS），Windows 那边照旧是 `[gpu]` + cu128，行为不变。

**ffmpeg 是硬性依赖，纯分离也要。** audio-separator 在 `Separator.__init__` 里就会
`subprocess.check_output(["ffmpeg", "-version"])`，缺了直接抛 FileNotFoundError。

## 跑

```bash
uv run mac/main.py
```

```bash
uv run mac/cover.py
```

配置还是改各自文件顶上的「配置区域」。相对路径按**项目根目录**算，不是 `mac/`，
所以 `data/audio/xxx.flac`、`output/` 和 Windows 版指的是同一个地方。

## 和 Windows 版差在哪

| | Windows | Mac |
|---|---|---|
| Applio 解释器 | `.venv\Scripts\python.exe` | `.venv/bin/python` |
| Applio 默认目录 | `C:\workspace\Applio` | `~/workspace/Applio` |
| ffmpeg | Applio 目录自带 `ffmpeg.exe` | 只能 `brew install ffmpeg` |
| 停止正在跑的任务 | `taskkill /F /T` | 独立进程组 + `killpg` |
| 分离 | CUDA | MPS |
| 特征提取 | CUDA | **CPU**（Applio 没写 MPS 分支） |
| 训练 | CUDA | MPS |
| RVC 推理 | CUDA | **CPU**（Applio 没写 MPS 分支） |
| 配置文件 | `config.json` | `mac/config.json` |
| 分离模型缓存 | `/tmp/audio-separator-models` | `~/Library/Caches/audio-separator-models` |

## mac 上新踩的坑（已在代码里绕开）

- **`--gpu` 不能传 `0`。** Applio 的 `rvc/train/extract/extract.py` 里是
  `devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" ...]` ——
  除了字面量 `-`，任何值都会被拼成 `cuda:N`，mac 上直接崩。
  所以 `trainer.py` 里 `DEFAULT_GPU = "-"`。
  训练那步不受影响：`train.py` 自己会 `cuda → mps → cpu` 挑设备，非 CUDA 时不看这个值。

- **`creationflags` 在 POSIX 上传非 0 会抛 ValueError。** Windows 版靠它隐藏黑框，
  mac 版整个参数不传。

- **只 `terminate()` 杀不干净。** Applio 的 `core.py` 下面还会 spawn 一层 `train.py`，
  Windows 靠 `taskkill /T` 连根拔起。mac 上得让子进程 `start_new_session` 自成一个进程组，
  停止时 `killpg` 整组，否则会变成"显示已停止，后台还在烧 GPU"。

- **Homebrew 的 bin 不在 GUI 继承的 PATH 里。** 从终端跑没事，以后做 app 双击启动时
  `shutil.which("ffmpeg")` 会直接找不到，所以 `find_tool()` 额外兜了
  `/opt/homebrew/bin` 和 `/usr/local/bin`。

- **Applio 的 MPS 环境变量得自己补。** `run-applio.sh` 里会先
  `export PYTORCH_ENABLE_MPS_FALLBACK=1` 再启动，而我们是绕过它直接调 `core.py` 的。
  少了 `MPS_FALLBACK=1`，MPS 上没实现的算子不会回退到 CPU，而是直接抛异常把训练打断。
  所以 `procutil.APPLIO_MPS_ENV` 会在每次调 `core.py` 时注入这两个变量。

- **分离模型别留在 `/tmp`。** audio-separator 默认下到 `/tmp/audio-separator-models`，
  三个模型加起来 1.7GB，而 macOS 会定期清理 `/tmp`。`main.py` 里改成了
  `~/Library/Caches/audio-separator-models`。

- **Applio 的前置模型必须自己下。** 音高预测器（`rvc/models/predictors/rmvpe.pt`）和
  HiFi-GAN 预训练权重**只有开 Applio 的 GUI 时才会自动下载**，`core.py` 从来不管。
  缺了它的表现极其隐蔽：`extract.py` 的 `run_pitch_extraction` 用
  `ProcessPoolExecutor` + `concurrent.futures.wait()`，而 `wait()` **不看结果** ——
  worker 因 `FileNotFoundError` 整个崩掉，外面照样打印 "Pitch extraction completed"，
  `f0/` 却是空的，一路滑到训练那步才报 "Not enough data present in the training set"。
  所以 `trainer.ensure_prerequisites()` 会先查后补。
  （特征提取器 embedder 反而**不用**管，Applio 的 `load_embedding` 会按需去 HuggingFace 拉。）

- **`assets/config.json` 也得自己造。** 只有 Applio 的 `app.py`（GUI）启动时会从
  `config_template.json` 复制一份，`core.py` 不会。缺了它训练能完整跑完、日志还写
  "Training has been successfully completed"，但把 checkpoint 导出成可分发模型那步会
  `An error occurred extracting the model` 然后**继续往下走**，最后照样报
  "trained successfully"、退出码 0 —— `logs/<model>/` 下只剩 `G_*.pth` / `D_*.pth`，
  没有 `<model>_<epoch>e_<step>s.pth`。`ensure_applio_config()` 负责补。

- **RVC 推理挂 `.index` 会段错误（SIGSEGV）。** faiss-cpu 和 torch 在 macOS 上各自链接了
  一份 `libomp.dylib`，同进程初始化两份 OpenMP 运行时会报 `OMP: Error #15` 然后崩。
  推理正好同时用到两者（torch 跑模型 + faiss 做特征检索），所以传了 `--index-path` 必崩、
  不传就没事（和混响无关，四种组合验过）。
  **网上通行的 `KMP_DUPLICATE_LIB_OK=TRUE` 在这里没用**，实测照样 rc=139；
  管用的是 `OMP_NUM_THREADS=1`（见 `procutil.APPLIO_INFER_ENV`），连跑 3 次全过。
  只在推理时加，训练和提取不受影响。

- **推理时的 embedder 必须和训练时一致。** 不一致会在推理早期直接段错误，不会给你好脸色。

- **推理和提取跑 CPU 是正常的，不是卡死。** 见上表，Applio 那两处源码里就没有 MPS 分支。

- **audio-separator 是用裸命令名 `ffmpeg` 走 PATH 调的。** 它自己不认 Homebrew 的目录，
  所以每条链路开跑前都会先 `ensure_tools_on_path()` 把 `/opt/homebrew/bin` 补进 `os.environ["PATH"]`，
  否则从 GUI 起的进程即使装了 ffmpeg 也会说找不到。

## 验证状态

三条链路都在这台 M4 上真机跑通过：

| 链路 | 状态 | 备注 |
|---|---|---|
| 分离 | ✅ 端到端 | 4 个轨齐全；模型缓存后 12s 素材耗时 26s |
| 训练 | ✅ 端到端 | preprocess → extract → train → index 全过，产出 `<model>_3e_84s.pth` + `.index` |
| 翻唱 | ✅ 端到端 | 分离 → RVC → 混音，成品 44.1kHz 立体声 |

训练和翻唱用的是合成素材 + 3 epoch 的玩具模型，验的是**流程**不是音质。

## 慢

`main.py` 里那套参数（`overlap=8`、VR 的 TTA 开）是按 N 卡调的，M 系列上会明显更慢。
参数原样保留了，嫌慢就把 `mac/main.py` 顶上的 `FAST_MODE` 改成 `True`：
重叠率降到 2、TTA 关掉，快好几倍，画质略降。
