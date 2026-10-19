# LuxTTS 运行与部署指南

本文档提供从零开始安装、模型下载、加载推理到生产部署与参数调优的完整操作手册。

---

## 1. 环境准备与依赖安装

### 1.1 系统与 Python 版本要求
- **操作系统**：macOS (Apple Silicon 推荐)、Linux (Ubuntu 20.04+ 推荐)、Windows (需安装 C++ 编译环境)。
- **Python 版本**：Python 3.10 ~ 3.13。

### 1.2 创建虚拟环境与安装依赖
建议在专属虚拟环境中安装，以避免包版本冲突：

```bash
# 1. 克隆代码仓库
git clone https://github.com/haibing-wang/LuxTTS.git
cd LuxTTS

# 2. 创建并激活虚拟环境 (以 Python 自带 venv 为例)
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# 或者是 Windows PowerShell: .venv\Scripts\Activate.ps1

# 3. 安装项目依赖
pip install -r requirements.txt
```

> **特别提示：**
> - 如果在安装 `piper_phonemize` 遇到问题，请确保使用 `requirements.txt` 中指定的 find-links：
>   ```bash
>   pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html
>   ```
> - 运行代码时出现 `WARNING:root:Failed import k2 with error No module named 'k2'` 是**非致命提示**，系统会自动回退到纯 PyTorch 算子实现，不影响推理与音频生成。

---

## 2. 模型的下载与缓存机制

LuxTTS 涉及两个核心模型族：
1. **LuxTTS 声学与声码器模型**（托管于 Hugging Face 仓库 [`YatharthS/LuxTTS`](https://huggingface.co/YatharthS/LuxTTS)）；
2. **Whisper ASR 语音识别模型**（用于将参考音频转写为文本，GPU/MPS 下默认使用 `openai/whisper-base`，CPU 下默认使用 `openai/whisper-tiny`）。

### 2.1 自动在线下载（推荐）
在首次执行 `LuxTTS()` 初始化时，程序会自动调用 Hugging Face Hub 的 `snapshot_download()` 进行全自动拉取与校验，无需手动操作：

```python
from zipvoice.luxvoice import LuxTTS

# 首次调用会自动下载所有必要模型权重并缓存至本地
lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')
```

#### 本地缓存存储位置：
- 模型默认缓存在操作系统的 Hugging Face 缓存目录：
  - **Linux / macOS**: `~/.cache/huggingface/hub/models--YatharthS--LuxTTS/`
  - **Windows**: `C:\Users\<用户名>\.cache\huggingface\hub\models--YatharthS--LuxTTS\`

### 2.2 离线环境与手动下载部署
在隔离内网或需要提前准备权重的生产服务器上，推荐使用 `huggingface-cli` 提前下载完整目录：

```bash
# 安装 huggingface 官方命令行工具
pip install huggingface_hub

# 下载 LuxTTS 完整权重到本地指定目录（例如 ./checkpoints/LuxTTS）
huggingface-cli download YatharthS/LuxTTS --local-dir ./checkpoints/LuxTTS --local-dir-use-symlinks False

# 提前下载 Whisper 模型（可选，用于离线 ASR）
huggingface-cli download openai/whisper-base --local-dir ./checkpoints/whisper-base --local-dir-use-symlinks False
```

#### 在代码中加载本地离线权重：
```python
from zipvoice.luxvoice import LuxTTS

# 传入本地解压后的绝对路径或相对路径
lux_tts = LuxTTS(model_path='./checkpoints/LuxTTS', device='cuda')
```

### 2.3 权重文件清单与用途解析

| 文件名 | 大小 | 用途说明 |
| :--- | :--- | :--- |
| `model.pt` | ~491 MB | 核心声学模型权重（基于 ZipVoiceDistill 蒸馏架构），供 GPU/MPS 加载。 |
| `config.json` | ~1 KB | 声学模型超参数配置（特征维度、采样率、注意力层数等）。 |
| `tokens.txt` | ~3 KB | Emilia 分词器词汇表，支持多语言拼音/音素映射。 |
| `vocoder/vocos.bin` | ~64 MB | 48kHz Vocos 神经声码器权重。 |
| `vocoder/config.yaml` | ~1 KB | 声码器网络架构与频段超参数配置。 |
| `text_encoder.onnx` | ~17 MB | ONNX 格式的文本编码器，供 CPU 模式推理。 |
| `fm_decoder.onnx` | ~478 MB | ONNX 格式的流匹配解码器，供 CPU 模式推理。 |
| `text_encoder_int8.onnx` | ~5.5 MB | INT8 量化文本编码器（超轻量 CPU 备选）。 |
| `fm_decoder_int8.onnx` | ~125 MB | INT8 量化流匹配解码器（超轻量 CPU 备选）。 |

---

## 3. 模型加载与设备模式配置

`LuxTTS` 类支持三种硬件模式：

### 3.1 NVIDIA GPU (CUDA)
适合 Linux/Windows 服务器，推理速度最快（可达 150x 实时）：
```python
from zipvoice.luxvoice import LuxTTS
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')
```

### 3.2 Apple Silicon (MPS)
专为搭载 M1/M2/M3/M4 系列芯片的 Mac 优化，利用 Unified Memory 和 GPU 核心硬件加速：
```python
from zipvoice.luxvoice import LuxTTS
lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')
```

### 3.3 通用 CPU 模式 (ONNX Runtime)
适合无独立显卡的普通电脑、虚拟机或轻量容器。通过多线程优化保持实时速度：
```python
from zipvoice.luxvoice import LuxTTS
# 可通过 threads 控制 ONNX Runtime 使用的 CPU 线程数（默认 4）
lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=4)
```

> **自动回退机制 (Auto-Fallback)**：  
> 当代码请求 `device='cuda'` 但当前设备没有 NVIDIA 显卡时，代码会自动检测是否有可用的 `mps`；若无则自动回退到 `cpu`，确保代码在任何机器上均可无缝执行。

---

## 4. 运行方式与命令行入口

### 4.1 CLI 命令行入口 ([main.py](../main.py))

项目提供功能完善的命令行入口工具，支持**单文本合成**与**多行文本分段合成+自动合并+时间戳 JSON 导出**：

#### 场景 1：单行文本合成
```bash
python main.py -a audio_file.wav -t "Hello world, this is LuxTTS!" -o output.wav
```

#### 场景 2：多行文本文件合成（按换行分段，自动拼接并输出时间轴）
```bash
python main.py -a audio_file.wav -f script.txt -o merged.wav --time_json timestamps.json
```
- **参数说明**：
  - `-a`, `--prompt_audio`：参考音频文件路径（必需）；
  - `-f`, `--text_file`：待合成的多行文本文件；
  - `-o`, `--output`：输出的整段合并音频文件；
  - `--time_json`：时间戳 JSON 输出路径（默认自动为 `<output>_timestamps.json`）；
  - `--pause`：各行之间的停顿静音间隔（秒，默认 0.3s）；
  - `-l`, `--language`：参考音频语言提示（`en` 或 `zh`），防止 Whisper 语种检测漂移；
  - `--prompt_text`：参考音频的文本内容（若提供则直接跳过 Whisper，极速合成）。

#### 导出的时间戳 JSON 格式示例：
```json
{
  "1": {
    "time": "00:00.00 - 00:04.66",
    "start_seconds": 0.0,
    "end_seconds": 4.66,
    "duration_seconds": 4.66,
    "text": "Welcome to the LuxTTS command line interface."
  },
  "2": {
    "time": "00:04.96 - 00:10.20",
    "start_seconds": 4.96,
    "end_seconds": 10.2,
    "duration_seconds": 5.24,
    "text": "This demonstrates the multi-line batch voice cloning feature."
  }
}
```

### 4.2 Python 代码极简调用 ([demo.py](../demo.py))

```python
from zipvoice.luxvoice import LuxTTS
import soundfile as sf

# 1. 初始化模型 (自动根据系统检测 cuda / mps / cpu)
lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')

# 2. 准备合成文本与参考音频 (支持 wav / mp3，建议 3~5 秒)
text = "你好，欢迎使用 LuxTTS 高保真声音克隆系统！"
prompt_audio = 'audio_file.wav'

# 3. 编码参考音频
encoded_prompt = lux_tts.encode_prompt(prompt_audio, duration=5, rms=0.01)

# 4. 生成语音 (4 步快速采样)
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)

# 5. 保存输出为 48kHz WAV 文件
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)
print("音频已成功生成至 output.wav！")

# 6. 如果在 Jupyter Notebook 环境下预览：
try:
    from IPython.display import Audio, display  # type: ignore
    display(Audio(final_wav, rate=48000))
except (ImportError, NameError):
    pass
```

### 4.2 高级参数调优示例

```python
# 提取音色特征时的参数调控
encoded_prompt = lux_tts.encode_prompt(
    prompt_audio='audio_file.wav',
    duration=5,      # 参考音频截取时长（秒）。建议 3~5 秒；若发现音质有瑕疵可调大至 10 或 1000（整段）
    rms=0.01         # 响度基准值。调高（如 0.015）声音更响亮，调低（如 0.005）更轻柔
)

# 生成阶段的高级参数微调
final_wav = lux_tts.generate_speech(
    text="这是经过精细参数调节生成的语音。",
    encode_dict=encoded_prompt,
    num_steps=4,         # 采样步数：通常 3 或 4 步即可达到高音质；调大到 6~8 步可稍微增加饱满度，但耗时线性增加
    t_shift=0.9,         # 时间调度偏移系数（0.1 ~ 1.0）：值越高整体音色细节表现更生动，但过高可能增加字词发音误差；发音不准时可适当降低至 0.5
    speed=1.0,           # 语速调节：1.0 为标准速度，小于 1.0 放慢语速，大于 1.0 加快语速
    guidance_scale=3.0,  # 无分类器引导 (CFG) 强度：控制生成结果对文本条件的贴合度，推荐 2.5 ~ 3.5
    return_smooth=False  # 平滑开关：若遇到高频金属破音或杂音时，可设置为 True 增强平滑滤波
)
```

---

## 5. 推理核心参数速查表

| 参数名 | 所属方法 | 默认值 | 推荐区间 | 说明与调优建议 |
| :--- | :--- | :--- | :--- | :--- |
| `prompt_audio` | `encode_prompt` | - | 本地路径 | 参考音频路径（WAV/MP3）。建议 3~5 秒清晰单人人声。 |
| `prompt_text` | `encode_prompt` | `None` | 文本字符串 | **可选手动指定参考音频文本**。若传入则跳过 Whisper 识别，速度大幅提升且彻底避免 ASR 幻觉。 |
| `language` | `encode_prompt` | `None` | `'en'` / `'zh'` | **可选强制指定识别语言**。防止 Whisper 自动语言检测误判为韩语/日语等其他语言。 |
| `duration` | `encode_prompt` | `5` | `3` ~ `10` | 参考音频截取秒数。最少需要 3 秒以完整捕获发音人音色。 |
| `rms` | `encode_prompt` | `0.001` | `0.008` ~ `0.02` | 音量/响度控制。推荐 `0.01`，避免声音过小或过载爆音。 |
| `num_steps` | `generate_speech` | `4` | `3` ~ `6` | 流匹配采样步数。由于经过蒸馏，4 步即可达到巅峰质量。 |
| `t_shift` | `generate_speech` | `0.5` | `0.5` ~ `0.9` | 时间步偏移系数。音质不理想可提至 0.9；若有吐字不清可降至 0.5。 |
| `speed` | `generate_speech` | `1.0` | `0.8` ~ `1.3` | 控制生成语速，数值越小语速越慢。 |
| `guidance_scale` | `generate_speech` | `3.0` | `2.0` ~ `4.0` | CFG 引导权重。数值适中有助于提升发音清晰度与稳定性。 |
| `return_smooth` | `generate_speech` | `False` | `True` / `False` | 是否关闭 48kHz 超高频扩展并开启平滑，用于抑制偶现的金属杂音。 |

---

## 6. 文本输入规范与控制技巧

### 6.1 支持的语种范围
- **支持**：**中文（简体/繁体）**、**英文（English）** 以及 **中英混合输入**。
- **不支持**：**日语、韩语、法语等其他语言**。
  - **重要提醒**：请勿输入日语平假名/片假名或韩文。分词器会将这些字符当作非法字符过滤丢弃；日文汉字会被误当成中文发音读出。

### 6.2 停顿控制技巧 (Pauses)
LuxTTS 能够精确响应标点符号的时间延展，建议通过标点调节说话停顿与节奏：
- **短停顿 (Short pause)**：输入逗号 `,` 或全角 `，`。例如：`"Hello, welcome to LuxTTS."`
- **句末停顿 (Sentence pause)**：输入句号 `.` 或全角 `。`。
- **较长停顿 (Long pause)**：输入省略号 `...` 或全角 `……`。
- **语意转折停顿**：输入破折号 `-` 或分号 `;`。

### 6.3 中文声调与多音字精确控制 (`<pinyin>` 语法)
分词器原生支持带 1~5 调的拼音嵌入语法。当遇到多音字（如“朝阳”、“重庆”、“行长”）或需要强制声调时，可直接用尖括号包裹拼音：
```python
# 示例：通过 <chao2><yang2> 显式指定多音字读音与二声/二声声调
text = "这里的风景真好，迎着<chao2><yang2>出发！"
final_wav = lux_tts.generate_speech(text, encoded_prompt)
```
- 声调数字：`1` (阴平/一声), `2` (阳平/二声), `3` (上声/三声), `4` (去声/四声), `5` (轻声)。

### 6.4 语调、音高 (Pitch) 与情绪控制
- **疑问语调**：句尾使用问号 `?`，模型会自动在句子尾部拉升音调，形成自然的疑问升调；
- **感叹语调**：使用感叹号 `!`，会增强句末音量与重音力度；
- **音高与情绪控制机制**：
  - LuxTTS 是零样本（Zero-Shot）声音克隆模型，**不支持**类似 `<prosody pitch="+10%">` 的 SSML 标签；
  - 音色基频（高音/低沉）、语速风格与情绪**完全取决于参考音频（Prompt Audio）**。如需高音调或激昂语气，请选择一段对应语调特征的参考音频进行克隆。

---

## 7. 常见问题排查 (FAQ)

### Q1: 运行 `python main.py` 提示 `No module named 'soundfile'` 或 IDE 代码标红？
- **原因**：依赖安装在了项目的 `.venv` 虚拟环境中，但 IDE 或终端使用了系统默认的 Python。
- **解决**：
  1. 激活虚拟环境：`source .venv/bin/activate`；
  2. 使用项目的 Python 执行：`./.venv/bin/python main.py`；
  3. 项目根目录已提供 `pyrightconfig.json` 与 `.vscode/settings.json`，确保编辑器识别虚拟环境。

### Q2: 提示 `Failed import k2 with error No module named 'k2'` 是否需要处理？
- **解答**：**不需要**。`k2` 仅作为可选的 C++ 加速库。在缺失 `k2` 时，LuxTTS 会自动无缝切换到纯 PyTorch 算子实现，功能与音质完全一致。

### Q3: 为什么输入日文文本或使用日文音频会报错？
- **解答**：LuxTTS 的官方预训练模型及词表（`tokens.txt`）专为**中英双语**构建：
  1. 日文平假名/片假名不存在于词表中，会被直接过滤抛弃；
  2. 若参考音频为日文，Whisper 转写如果得到日文字符，会导致参考 Token 长度为 0，引发底层 `Calculated padded input size (6) < Kernel size (7)` 报错；
  3. 如需测试请使用清晰的中英文音频与中英文字符。

### Q4: 声音克隆效果不够相似或出现吞字、杂音怎么办？
1. **参考音频质量**：确保参考音频长度大于 3 秒，单人说话，无背景混响、杂音或 BGM；
2. **强制指定语言**：调用 `encode_prompt(..., language='en')` 或 `'zh'`，防止 Whisper 自动检测跑偏；
3. **跳过识别直接传文本**：若知道参考音频内容，传入 `prompt_text="..."` 可 100% 避免 ASR 误差并极大提速；
4. **调节 `t_shift`**：若吞字或发音错误，将 `t_shift` 调小（例如从 0.9 降到 0.5）；
5. **消除金属杂音**：将 `return_smooth` 设置为 `True`；
6. **增加采样步数**：将 `num_steps` 设置为 `6`。
