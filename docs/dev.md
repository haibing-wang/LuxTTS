# LuxTTS 命令行工具开发与使用文档 (CLI Dev Guide)

本文档记录 **LuxTTS** 命令行入口工具 [`main.py`](../main.py) 的系统架构、实现细节、参数设计规范与多行批量合成时间轴机制。

---

## 1. 架构背景与设计目标

为了从简单的演示脚本（已归档为 [`demo.py`](../demo.py)）升级为具备工程化可用性的命令行工具，[`main.py`](../main.py) 实现了对生产环境友好的两项核心业务模式：

1. **单文本快速合成 (Single Text Mode)**：直接通过参数传入文本，适合作为 API 管道、自动化构建脚本的调用入口；
2. **多行文本文件批量分段合成 (Multi-line Batch Mode)**：处理长文章或台词剧本，按行独立分段合成，拼接合并为完整音频，并精确导出各行对应的时间轴 JSON 文件，广泛适用于**字幕生成、长篇有声书制作、播客制作以及数字人对口型（Lip-sync）**等场景。

```mermaid
graph TD
    CLI["命令行入口 (main.py)"] --> Parse["参数解析 (argparse)"]
    Parse --> ModelInit["LuxTTS 初始化 (硬件自适应: CUDA / MPS / CPU)"]
    ModelInit --> EncodePrompt["参考音频编码 (encode_prompt)<br/>*一次编码，全流程内存复用*"]
    
    EncodePrompt --> Branch{"判断输入类型"}
    
    Branch -->|单文本 (-t)| Single["单条合成 (synthesize_single)"]
    Single --> SaveSingle["导出 output.wav"]
    
    Branch -->|多行文本文件 (-f)| Multi["多行批量合成 (synthesize_multiline)"]
    Multi --> Loop["按行循环生成音频片段<br/>累加计算各行起始/结束时间戳"]
    Loop --> InsertGap["句间插入可选平滑静音 (--pause)"]
    InsertGap --> Concat["时域合并 (np.concatenate)"]
    Concat --> SaveAudio["导出合并音频 (output.wav)"]
    Concat --> SaveJSON["导出时间轴数据 (*_timestamps.json)"]
```

---

## 2. 命令行核心特性设计

### 2.1 参考音频一次编码，多次复用 (Prompt Caching)
在多行批量合成模式下，参考音频的特征提取和 Whisper ASR 转写耗时相对固定（约数秒）。`main.py` 在进入循环前仅执行一次 `encode_prompt()`，后续各行文本直接共享同一音色条件字典（`encoded_prompt`），将长文本合成的综合耗时降低 80% 以上。

### 2.2 句间平滑静音插入 (`--pause`)
合成多行文本时，若直接将各句波形首尾硬拼接，听感会过于紧凑仓促。`main.py` 支持 `--pause` 参数（默认 `0.3` 秒），在相邻句子间自动插入低噪声静音帧，同时精确计算进时间轴累加中。

### 2.3 异常语言漂移防护
支持 `-l / --language`（`'en'` 或 `'zh'`）与 `--prompt_text` 参数。显式指定可跳过或约束 Whisper 的语种检测，彻底防止语种误判导致下游声码器卷积崩溃。

---

## 3. 时间戳 JSON 数据规范

在多行模式下，程序会自动生成时间轴 JSON 文件（默认为 `<输出文件名>_timestamps.json`）。该文件包含每行台词在整体音频中的绝对时间区间，兼具**直观易读性**与**程序解析友好性**。

### 3.1 JSON 数据结构示例
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
  },
  "3": {
    "time": "00:10.50 - 00:16.96",
    "start_seconds": 10.5,
    "end_seconds": 16.96,
    "duration_seconds": 6.46,
    "text": "Each sentence is synthesized and merged seamlessly with accurate timestamps."
  }
}
```

### 3.2 字段说明
| 字段 | 类型 | 说明 | 应用场景 |
| :--- | :--- | :--- | :--- |
| `time` | `string` | 分:秒.毫秒 格式区间（如 `00:04.96 - 00:10.20`） | 供人工查阅、日志打印、前端播放器时间刻度 |
| `start_seconds` | `float` | 该行在合并音频中的起始时间（秒，保留 2 位小数） | SRT/VTT 字幕对齐、音频剪辑切片 |
| `end_seconds` | `float` | 该行在合并音频中的结束时间（秒，保留 2 位小数） | 字幕结束标记、视频混剪卡点 |
| `duration_seconds` | `float` | 该行单句的有效朗读时长（秒） | 语速统计、停顿分析 |
| `text` | `string` | 该行对应的原文文本 | 字幕文本内容对照 |

---

## 4. 命令行参数完整对照表

| 参数 | 缩写 | 类型 | 默认值 | 是否必需 | 说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `--prompt_audio` | `-a` | `str` | - | **必需** | 参考音频文件路径（WAV/MP3，建议 3~5 秒单人清晰语音）。 |
| `--text` | `-t` | `str` | `None` | 二选一 | 待合成的单条文本字符串。 |
| `--text_file` | `-f` | `str` | `None` | 二选一 | 待合成的多行文本文件路径（按换行符切分）。 |
| `--output` | `-o` | `str` | `output.wav` | 可选 | 输出音频文件的保存路径。 |
| `--time_json` | - | `str` | 自动同名 | 可选 | 时间戳 JSON 文件路径（默认自动为 `<output>_timestamps.json`）。 |
| `--pause` | - | `float` | `0.3` | 可选 | 多行文本合并时各行之间的静音停顿秒数。 |
| `--language` | `-l` | `str` | `None` | 可选 | 参考音频语言提示（`en` 或 `zh`），防止 Whisper 误判。 |
| `--prompt_text` | - | `str` | `None` | 可选 | 参考音频对应台词（传入可彻底跳过 Whisper 转写，极速合成）。 |
| `--device` | `-d` | `str` | 自动探测 | 可选 | 计算设备（`cuda`、`mps` 或 `cpu`）。 |
| `--model_path` | - | `str` | `'YatharthS/LuxTTS'` | 可选 | HuggingFace 仓库名或本地权重解压目录。 |
| `--num_steps` | - | `int` | `4` | 可选 | 流匹配采样步数（推荐 3 ~ 6）。 |
| `--speed` | - | `float` | `1.0` | 可选 | 语速调节倍率（如 1.2 加快，0.8 减慢）。 |
| `--t_shift` | - | `float` | `0.5` | 可选 | 采样时间偏移系数（音色生动度调控）。 |
| `--rms` | - | `float` | `0.01` | 可选 | 参考音频目标 RMS 响度基准。 |
| `--guidance_scale`| - | `float` | `3.0` | 可选 | 无分类器引导权重（CFG 强度）。 |
| `--return_smooth` | - | `bool` | `False` | 可选 | 标志参数，是否启用声码器高频平滑滤波。 |

---

## 5. 测试数据与使用演练

项目根目录已保留标准多行测试数据文件 [`test_multiline.txt`](../test_multiline.txt)，涵盖中英文混合多行场景。

### 5.1 执行多行测试命令
```bash
python main.py \
  -a audio_file.wav \
  -l en \
  -f test_multiline.txt \
  -o test_merged.wav \
  --time_json test_timestamps.json
```

### 5.2 控制台实时反馈输出
```text
 初始化 LuxTTS 模型 (设备: mps, 模型: YatharthS/LuxTTS)...
 正在编码参考音频 'audio_file.wav'...
Transcribed prompt text: This is the most important part of VISPRES.

开始批量合成多行文本，共 6 行...
[1/6] 正在合成: "Welcome to the LuxTTS command line inter..."
[2/6] 正在合成: "This demonstrates the multi-line batch v..."
[3/6] 正在合成: "Each sentence is synthesized and merged ..."
[4/6] 正在合成: "欢迎使用 LuxTTS 命令行多行文本合成功能。..."
[5/6] 正在合成: "这是一段中文语音测试，支持按行分段并精确计算时间轴。..."
[6/6] 正在合成: "祝您使用愉快！..."

============================================================
 多行文本合并音频生成成功！
  • 输出音频: test_merged.wav (总时长: 00:28.45, 28.45s)
  • 时间轴 JSON: test_timestamps.json
============================================================
时间段分配清单:
  [1] 00:00.00 - 00:04.66 : Welcome to the LuxTTS command line 
  [2] 00:04.96 - 00:10.20 : This demonstrates the multi-line ba
  [3] 00:10.50 - 00:16.96 : Each sentence is synthesized and me
  [4] 00:17.26 - 00:21.10 : 欢迎使用 LuxTTS 命令行多行文本合成
  [5] 00:21.40 - 00:26.55 : 这是一段中文语音测试，支持按行分段
  [6] 00:26.85 - 00:28.45 : 祝您使用愉快！
============================================================
```
