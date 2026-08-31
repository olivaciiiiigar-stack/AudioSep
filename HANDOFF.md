# AudioSep — 交接文档

最后更新：2026-08-30 晚。换新对话后从这里接着看。

---

## 一句话现状

Mac 版三条链路（分离 / 训练 / 翻唱）**全部跑通**，但训出来的音色模型有明显的
**「赛博坦人」金属音 + 电音**。经过一长串对照实验，已排除设备、参数、分离、
音域等因素，**当前唯一未验证的嫌疑是训练素材的房间混响**——正在 Kaggle 上
用去混响后的素材重训验证。

---

## 一、这个项目是什么

音频分离 + RVC 翻唱 + 音色训练。根目录的 5 个 `.py` 是**原 Windows 版，一行都不要改**。
Mac 版全部在 `mac/` 下，两套并存。

```
分离   audio-separator（BS-Roformer → Karaoke → DeEcho-DeReverb）  ← 本地 MPS，工作正常
训练   外部 Applio 3.6.4                                          ← 有问题，见第四节
翻唱   外部 Applio 3.6.4 推理 + ffmpeg 混音                        ← 工作正常
```

### 文件

| 路径 | 说明 |
|---|---|
| `mac/main.py` | 三阶段分离，`separate_track()` |
| `mac/cover.py` | 翻唱链路：分离 → RVC → 混音 |
| `mac/trainer.py` | 训练：preprocess → extract → train → index |
| `mac/train.py` | 训练的命令行入口，改顶部配置区即可 |
| `mac/procutil.py` | 子进程、进程树 kill、环境变量注入 |
| `mac/settings.py` | 路径配置，`mac/config.json` |
| `mac/dataset_info.py` | 看数据集攒了多少 |
| `mac/dereverb_dataset.py` | 整批去混响 |
| `mac/README.md` | Mac 版的坑与差异 |

### 环境（都已装好）

- 项目 venv：torch 2.13.0（MPS ✅）、onnxruntime 1.29（CoreML）、audio-separator 0.41.1
- ffmpeg 9.0.1（`brew install ffmpeg`）—— **硬性依赖，纯分离也要**
- faiss 1.14.3
- Applio 3.6.4 在 `~/workspace/Applio`，`.venv/bin/python` = Python 3.12.13
- 分离模型缓存：`~/Library/Caches/audio-separator-models`（1.7GB）

---

## 二、Mac 移植踩过的坑（都已在代码里绕开）

1. **`--gpu` 必须传 `-`**。Applio 的 `rvc/train/extract/extract.py:236`：
   `devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" ...]`。传 `0` 会拼成 `cuda:0` 崩掉。
   → `mac/trainer.py` 的 `DEFAULT_GPU = "-"`。**云端反过来，要传 `0`。**

2. **`creationflags` 在 POSIX 上传非 0 会抛 ValueError**。已去掉。

3. **只 `terminate()` 杀不干净**。Applio 的 `core.py` 下面还有 `train.py`。
   → `start_new_session=True` + `killpg` 整组。

4. **前置模型不会自动下**。`rvc/models/predictors/rmvpe.pt` 和 HiFi-GAN 底模
   只有开 Applio 的 GUI 才会下载。缺了之后 `extract.py` 用
   `ProcessPoolExecutor` + `wait()`（**不看结果**），worker 崩掉照样打印
   "Pitch extraction completed"，`f0/` 是空的。
   → `trainer.ensure_prerequisites()`。**embedder 不用管**，`load_embedding`
   会按需去 HuggingFace 拉；而且它的目录名是下划线（`chinese_hubert_base`），
   拿模型名（连字符）拼路径会误判。

5. **`assets/config.json` 只有 GUI 会造**（从 `config_template.json` 复制）。
   缺了它训练能跑完，但导出可分发模型那步静默失败。
   → `trainer.ensure_applio_config()`。

6. **推理挂 `.index` 必 SIGSEGV**。faiss-cpu 和 torch 在 macOS 上各带一份
   `libomp.dylib` → `OMP: Error #15`。
   **`KMP_DUPLICATE_LIB_OK=TRUE` 无效（实测 rc=139）**，管用的是
   `OMP_NUM_THREADS=1` → `procutil.APPLIO_INFER_ENV`。只给推理用。

7. **`PYTORCH_ENABLE_MPS_FALLBACK=1` 要自己补**。`run-applio.sh` 会 export，
   我们绕过它直接调 `core.py`。→ `procutil.APPLIO_MPS_ENV`。

8. **Applio 失败时退出码仍然是 0**。每步必须查产物。`extract` 那步光查
   `filelist.txt` 存在不够，**必须查非空**。

9. **切片器阈值 `-42 dBFS`**（`preprocess.py:47`）。素材各条响度差 13.5 dB 时，
   `normalization_mode="none"` 会把轻的段落当静音裁掉：
   `none` → 272 切片，`pre` → 743 切片。**训练一律用 `pre`。**

10. **GPU 任务不能并行**。分离和训练都吃 MPS，同时跑互相拖死。
    RVC 推理走 CPU，可以和训练并行。

---

## 三、素材

浴室录的（为了不扰民），Voice Memos 无损（ALAC 16bit 44.1kHz）、增强录音关、
MacBook 内置三麦阵列。

```
data/original_voice/voice_zyc_cn_ma/       原始，24 个，36.7 分钟
data/original_voice/voice_zyc_cn_ma_dry/   去混响后，24 个，36.7 分钟  ← 当前在用
data/original_voice/_backup/               CBC17 的削顶原件
```

**实测特征：**

- 信噪比 26–46 dB，无工频干扰，无噪声门痕迹 ✅
- 底噪归一化后中位 -56 dBFS（录音室 < -70）
- **比录音室干声暗 5–9 dB**（4–16kHz），AI 输出频谱几乎完全复刻素材（差 0.5–1.4 dB）
- 混响占 2.2%（录音室 0.7%）
- 基频 152–339 Hz（5–95%），中位 232 Hz
- 去混响后：各频段 ±0.5 dB，**高频反升 0.15–0.56 dB**，0 削顶

**录音配置坑：**

- **蓝牙耳机当输入必然 16kHz**（HFP 协议限制，与耳机档次无关）。
  用户曾报「电音一直有」，实为 MOMENTUM 4 蓝牙重连后被拽进 HFP，
  回放变成 16kHz 单声道。排查：`system_profiler SPAudioDataType | grep -A5 "Default Output Device: Yes"`
- Voice Memos 默认是 AAC 63kbps（16kHz 处有砖墙），必须改成**无损**
- 降噪会让声音**明显发闷**，用户听过后否决

---

## 四、核心问题：「赛博坦人」金属音 + 电音

用户描述：*「有点像赛博坦人 自带一点点回声一点点电音」*。

### 已排除的（每条都是对照实验，不是猜测）

| 嫌疑 | 怎么排除的 |
|---|---|
| 分离质量 | 4 个 Karaoke 模型（aufr33/becruily/gaboxv2/gabox）输出听不出差别 |
| 推理混响 | 关掉后句子间隙噪声只从 -69.1 变到 -68.7 dB |
| 音域外推 | 降调 5 半音到素材中位 230Hz，仍是赛博坦 |
| 归一化 | 纯线性缩放，且改善了响度一致性（标准差 3.17→2.26 dB） |
| 素材降噪 | 用户听后：明显更闷，否决 |
| 学习率 | 3e-5 跑 25 epoch，和 1e-4 的 25 epoch 听感一样 |
| **MPS 设备** | **Kaggle CUDA 训的 epoch 75，一样是赛博坦** |
| **数据量** | Windows 用同样的 `none` 配方、更少样本，却是干净的 |

### 决定性证据

**同域测试**：用户自己的录音 → 用户自己的模型（零外推）→ 输出仍有电音。
说明问题在模型权重里，和分离、音域、混响、选歌都无关。

### 训练指标（本地 1e-4，75 epoch）

- `loss/g/mel` 停滞在 19.2（最后 25 个 epoch 几乎不动）
- `loss/g/fm` 从 6.0 涨到 11.8
- `grad/norm_g` 从 204 涨到 451
- → 像 GAN 失衡，但降学习率没解决

### 当前唯一未验证的嫌疑

**训练素材的房间混响**（浴室，从未在训练前去除）。

三次训练的对照：

| | 设备 | 归一化 | 每 epoch 样本 | 素材 | 结果 |
|---|---|---|---|---|---|
| Windows（用户旧机器） | CUDA | none | 少 | 旧素材，已丢失 | **干净** |
| 本地 Mac | MPS | pre | 736 | 浴室原始 | 赛博坦 |
| Kaggle 第一轮 | CUDA | none | 272 | 浴室原始 | 赛博坦 |
| **Kaggle 第二轮** | CUDA | **pre** | **743** | **去混响** | **← 正在跑** |

---

## 五、当前进行中

**Kaggle notebook `notebookd49be7623a`**（账号 RitZhang）

- 数据集 `zyc-voice-dataset-dry`（Private，24 文件 97MB）
- GPU T4 ×2，模型名 `zyc_cn_mandarin_dry`
- 参数：40000 / batch 8 / 200 epoch / save_every 25 / rmvpe /
  chinese-hubert-base / HiFi-GAN / **normalization=pre** / 预训练底模
- 速度约 **10 秒/epoch**，200 轮约 33 分钟
- 跑完第 10 格打包成 `zyc_trained.zip`，在右侧 Output 下载

**Kaggle 注意事项：**

- **必须用 T4，不能用 P100** —— P100 是 sm_60，新版 PyTorch 已放弃支持。
  症状：`torch.cuda.is_available()` 返回 True 但跑不了 kernel，提取零产出。
- **不要重装 torch** —— Kaggle 镜像自带 GPU 版（2.10.0+cu128）。
  Applio 的 `requirements.txt` pin 了 `torch==2.11.0`（PyPI CPU 版），
  必须 `grep -vE '^(torch|torchaudio|torchvision)=='` 剔除后再装。
- 切换加速器**必须先停会话**，否则点了没反应
- GPU 和联网都需要手机验证（已完成）
- 浏览器上传工具单次限 10MB，76MB 的 zip 得用户手动拖

**notebook 源文件**：`Applio_Kaggle_zyc.ipynb`（项目根 + 桌面各一份）

---

## 六、拿到云端模型后怎么验

```bash
# 解压到
~/workspace/Applio/logs/zyc_cn_mandarin_dry/

# 同域测试（最敏感，0.5 分钟）
cd ~/Workspace/AudioSep
uv run python -c "
import sys, os; sys.path.insert(0,'mac')
import settings, cover
A = os.path.expanduser(settings.load()['applio_dir'])
pth, idx = cover.find_trained_model('zyc_cn_mandarin_dry', A, 'models')
cover.rvc_infer('output/_同域测试/输入_我的原声.flac',
                'output/_同域测试/输出_去混响版.flac',
                pth, idx, embedder_model='chinese-hubert-base',
                reverb=False, applio_dir=A)
"
```

对比 `output/_学习率AB/同域_1e-4_旧.flac`（本地 MPS epoch 25）和
`output/心的距离_云端/人声_云端75.flac`（Kaggle 浴室原始素材 epoch 75）。

**判据：**

- **干净了** → 浴室混响是元凶。以后录音优先解决环境，或每批素材都过一遍去混响。
- **还是赛博坦** → 素材这条线走到头。**建议换思路，别再靠排除法**：
  拿一份公开的高质量人声数据集训一个模型。
  - 它干净 → 流程没问题，问题在这批素材某个还没识别的特质
  - 它也赛博坦 → 是 Applio 3.6.4 这个版本或参数组合的问题（用户当时用的
    Applio 版本未知，这是一直无法排除的变量）

---

## 七、已有产物

**本地模型**（`~/workspace/Applio/logs/`）：

| 模型 | 说明 |
|---|---|
| `zyc_cn_mandarin` | 本地 MPS，1e-4，epoch 25/50/75 存档 + 索引 |
| `zyc_lr3e5` | 本地 MPS，3e-5，epoch 25 |
| `zyc_cn_mandarin_cloud` | Kaggle 第一轮 epoch 75（浴室原始素材） |
| `mactest` | 合成音玩具模型，3 epoch，验流程用，可删（3.8GB） |

**音频产物**（`output/`）：分离的 4 轨、各版本翻唱人声与成品、EQ 测试、
四模型对比、学习率 A/B、同域测试、素材清理测试。

**歌曲**（`data/audio/`）：陈奕迅-心的距离、胡彦斌-男人KTV、周杰伦-那天下雨了
（都是 192kHz/24bit）。

---

## 八、待办

1. **等 Kaggle 这轮结果**，按第六节验证
2. `mac/trainer.py` 的 `run_cover()` 写死了用分离出的干声当源。用户想要
   「自己唱一遍、保留自己咬字」的话，需要加 `vocal_file=` 参数跳过分离，
   并接出 `--f0-autotune` / `--proposed-pitch`
3. 输出端 EQ 提亮已验证有效（`treble=g=8:f=3500:t=q:w=0.6` 能补齐 8.5dB 缺口），
   但还没固化进 `mix_final()`
4. 用户可能补录素材到 60 分钟（`data/original_voice/voice_zyc_cn_ma_v2/` 已建好，
   但决定还在浴室录 + 事后去混响，所以两批可以合并）
5. GUI（`app.py`）的 Mac 版没做，用户说「先不做 app」

---

## 九、给下一个会话的提醒

- **根目录 5 个 `.py` 一行都不要改**，那是 Windows 版
- 用户明确要求：**额度到 5-hour limit 就停手，不要切 credits**
- 我在这次排查中**连续三次拿没校准的指标下结论**（双音高检测、HNR、梳状滤波），
  每次都把方向带偏。**下结论前先拿已知信号校准指标**，否则宁可交给用户的耳朵
- 用户听感判断比任何自动指标都可靠，多做 A/B 音频给他听，少编数字
