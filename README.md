# AI 翻唱工坊

音频分离 + RVC 翻唱 + 音色训练，三合一桌面界面。

## 启动

```bash
uv run app.py
```

## 三个功能

**🎚 分离** — 三阶段分离出 4 个轨：纯伴奏 / 主唱干声 / 纯和声 / 被切除的混响。
用 BS-Roformer → Mel-Band Karaoke → UVR-DeEcho-DeReverb 三个模型串联。

**🎤 翻唱** — 分离 → RVC 变声 → 混音。可调升降调（±12 半音）、混响 5 项、
特征检索强度、音高算法、三轨音量。音色模型从下拉框选，也可以「浏览」手选外部 `.pth`。

**🏋 训练** — 预处理 → 特征提取 → 训练 → 生成索引。可调总 epoch、批大小、采样率、
保存间隔，高级设置里还有音高算法、特征提取器、声码器等。训练素材可以选一个文件夹，
也可以散着挑文件。

## 依赖 Applio

翻唱和训练都是调用外部的 [Applio](https://github.com/IAHispano/Applio) 完成的。
默认路径 `C:\workspace\Applio`，可以在界面顶部改，点「保存路径」记到 `config.json`。

## 几个坑（已在代码里绕开，这里记一下原因）

- Applio 的 `preprocess` 只认 `wav/mp3/flac/ogg`。手机录音的 `.m4a` 扔进去只会报
  "No audio files found"，所以训练前会自动转成 FLAC（原文件不动）。
- Applio 的 `core.py` 每一步失败时**退出码仍然是 0**，只 echo 一行英文。
  所以每步跑完都会去查产物在不在，否则失败也会一路显示成功。
- `separate_track()` 的 `clean_output` 默认关闭。开启会 `rmtree` 整个输出目录，
  界面上用户可能随手选了桌面，只有命令行里指定的专用目录才该开。

## 命令行

`main.py` 和 `cover.py` 保留了原来的批量跑法，改顶部配置区即可：

```bash
uv run cover.py
```
