#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LuxTTS 命令行入口工具 (CLI)
支持：
  1. 单条文本转语音 (Single Text Mode)
  2. 多行文本文件批量合成并按时间轴合并 (Multi-line Batch Mode)，自动导出时间戳 JSON
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from zipvoice.luxvoice import LuxTTS


def format_timestamp(seconds: float) -> str:
    """将秒数格式化为 分:秒.毫秒 (如 00:05.01 或 01:23.45)"""
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes:02d}:{sec:05.2f}"


def get_default_device() -> str:
    """自动探测推荐设备"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synthesize_single(
    tts: LuxTTS,
    text: str,
    encoded_prompt: dict,
    output_path: str,
    sample_rate: int = 48000,
    num_steps: int = 4,
    t_shift: float = 0.5,
    speed: float = 1.0,
    guidance_scale: float = 3.0,
    return_smooth: bool = False,
) -> float:
    """合成单条文本并保存"""
    print(f"\n[1/1] 正在生成语音: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    wav = tts.generate_speech(
        text=text,
        encode_dict=encoded_prompt,
        num_steps=num_steps,
        t_shift=t_shift,
        speed=speed,
        guidance_scale=guidance_scale,
        return_smooth=return_smooth,
    )
    wav_np = wav.numpy().squeeze()
    sf.write(output_path, wav_np, sample_rate)
    duration = len(wav_np) / sample_rate
    print(f" 音频已保存至: {output_path} (时长: {format_timestamp(duration)}, {duration:.2f}s)")
    return duration


def synthesize_multiline(
    tts: LuxTTS,
    lines: List[str],
    encoded_prompt: dict,
    output_path: str,
    time_json_path: str,
    sample_rate: int = 48000,
    pause_seconds: float = 0.3,
    num_steps: int = 4,
    t_shift: float = 0.5,
    speed: float = 1.0,
    guidance_scale: float = 3.0,
    return_smooth: bool = False,
):
    """批量合成多行文本，合并音频并导出时间戳 JSON"""
    total_lines = len(lines)
    print(f"\n开始批量合成多行文本，共 {total_lines} 行...")

    audio_segments: List[np.ndarray] = []
    timestamps_data: Dict[str, dict] = {}
    current_time = 0.0
    pause_samples = int(sample_rate * pause_seconds)
    silence_gap = np.zeros(pause_samples, dtype=np.float32)

    for idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        print(f"[{idx}/{total_lines}] 正在合成: \"{line[:40]}{'...' if len(line) > 40 else ''}\"")
        wav = tts.generate_speech(
            text=line,
            encode_dict=encoded_prompt,
            num_steps=num_steps,
            t_shift=t_shift,
            speed=speed,
            guidance_scale=guidance_scale,
            return_smooth=return_smooth,
        )
        wav_np = wav.numpy().squeeze()
        duration = len(wav_np) / sample_rate

        start_time = current_time
        end_time = current_time + duration

        # 记录时间区间
        timestamps_data[str(idx)] = {
            "time": f"{format_timestamp(start_time)} - {format_timestamp(end_time)}",
            "start_seconds": round(start_time, 2),
            "end_seconds": round(end_time, 2),
            "duration_seconds": round(duration, 2),
            "text": line,
        }

        audio_segments.append(wav_np)

        # 累加时间与插入停顿（最后一行不加尾部静音）
        if idx < total_lines and pause_seconds > 0:
            audio_segments.append(silence_gap)
            current_time = end_time + pause_seconds
        else:
            current_time = end_time

    # 合并完整波形
    combined_wav = np.concatenate(audio_segments, axis=0)
    sf.write(output_path, combined_wav, sample_rate)
    total_duration = len(combined_wav) / sample_rate

    # 输出时间戳 JSON 文件
    with open(time_json_path, "w", encoding="utf-8") as f:
        json.dump(timestamps_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(" 多行文本合并音频生成成功！")
    print(f"  • 输出音频: {output_path} (总时长: {format_timestamp(total_duration)}, {total_duration:.2f}s)")
    print(f"  • 时间轴 JSON: {time_json_path}")
    print("=" * 60)
    print("时间段分配清单:")
    for idx_str, item in timestamps_data.items():
        print(f"  [{idx_str}] {item['time']} : {item['text'][:35]}")
    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LuxTTS 命令行语音合成与克隆工具 (支持单文本与多行分段合成)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 单行文本合成:
     python main.py -a audio_file.wav -t "Hello world, this is LuxTTS!" -o output.wav

  2. 多行文本文件合成 (自动生成时间戳 JSON):
     python main.py -a audio_file.wav -f script.txt -o merged.wav --time_json time.json

  3. 指定参考语言并调速:
     python main.py -a audio_file.wav -l en -t "Speaking a bit faster." --speed 1.2
        """,
    )

    # 核心输入
    parser.add_argument(
        "-a", "--prompt_audio",
        type=str,
        required=True,
        help="参考音频文件路径 (WAV/MP3，建议 3~5 秒单人清晰语音)",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="可选：参考音频对应的台词文本 (若提供将跳过 Whisper ASR，大幅提速且避免识别误差)",
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        choices=["en", "zh"],
        help="参考音频语言提示 ('en' 或 'zh')，强制指定可杜绝 Whisper 语种检测漂移",
    )

    # 待合成文本（二选一）
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "-t", "--text",
        type=str,
        help="直接传入待合成的单条文本",
    )
    text_group.add_argument(
        "-f", "--text_file",
        type=str,
        help="待合成的多行文本文件路径 (以换行分割，按行合成分段并合并音频)",
    )

    # 输出相关
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.wav",
        help="输出音频文件路径 (默认: output.wav)",
    )
    parser.add_argument(
        "--time_json",
        type=str,
        default=None,
        help="多行模式下输出的时间戳 JSON 文件路径 (默认自动为 <output>_timestamps.json)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.3,
        help="多行合成时各句子之间的静音间隔秒数 (默认: 0.3s)",
    )

    # 模型与生成超参数
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="运行硬件后端 (默认自动检测: cuda > mps > cpu)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="YatharthS/LuxTTS",
        help="HuggingFace 仓库名或本地模型目录 (默认: 'YatharthS/LuxTTS')",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="流匹配采样步数 (默认: 4)",
    )
    parser.add_argument(
        "--t_shift",
        type=float,
        default=0.5,
        help="采样时间偏移系数 (默认: 0.5，希望音色起伏更丰富可设为 0.9)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="语速控制倍率 (默认: 1.0)",
    )
    parser.add_argument(
        "--rms",
        type=float,
        default=0.01,
        help="参考音频目标 RMS 响度 (默认: 0.01)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="无分类器引导权重 (默认: 3.0)",
    )
    parser.add_argument(
        "--return_smooth",
        action="store_true",
        help="是否开启声码器高频平滑 (抑制偶发金属杂音)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 校验输入文件存在性
    if not os.path.exists(args.prompt_audio):
        print(f"❌ 错误: 找不到参考音频文件 '{args.prompt_audio}'", file=sys.stderr)
        sys.exit(1)

    if args.text_file and not os.path.exists(args.text_file):
        print(f"❌ 错误: 找不到文本文件 '{args.text_file}'", file=sys.stderr)
        sys.exit(1)

    device = args.device or get_default_device()
    print(f" 初始化 LuxTTS 模型 (设备: {device}, 模型: {args.model_path})...")
    lux_tts = LuxTTS(model_path=args.model_path, device=device)

    # 2. 编码参考音频
    print(f" 正在编码参考音频 '{args.prompt_audio}'...")
    encoded_prompt = lux_tts.encode_prompt(
        prompt_audio=args.prompt_audio,
        duration=5,
        rms=args.rms,
        prompt_text=args.prompt_text,
        language=args.language,
    )

    # 3. 执行单文本模式或多行文本模式
    if args.text:
        # 单条文本模式
        synthesize_single(
            tts=lux_tts,
            text=args.text,
            encoded_prompt=encoded_prompt,
            output_path=args.output,
            num_steps=args.num_steps,
            t_shift=args.t_shift,
            speed=args.speed,
            guidance_scale=args.guidance_scale,
            return_smooth=args.return_smooth,
        )
    else:
        # 多行文本文件模式
        with open(args.text_file, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f.readlines()]
            lines = [l for l in raw_lines if l]

        if not lines:
            print(f"❌ 错误: 文本文件 '{args.text_file}' 中没有有效非空文本行！", file=sys.stderr)
            sys.exit(1)

        # 确定时间戳 JSON 路径
        time_json_path = args.time_json
        if not time_json_path:
            output_stem = Path(args.output).stem
            time_json_path = str(Path(args.output).parent / f"{output_stem}_timestamps.json")

        synthesize_multiline(
            tts=lux_tts,
            lines=lines,
            encoded_prompt=encoded_prompt,
            output_path=args.output,
            time_json_path=time_json_path,
            pause_seconds=args.pause,
            num_steps=args.num_steps,
            t_shift=args.t_shift,
            speed=args.speed,
            guidance_scale=args.guidance_scale,
            return_smooth=args.return_smooth,
        )


if __name__ == "__main__":
    main()
