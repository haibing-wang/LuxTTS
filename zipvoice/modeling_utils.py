import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors.torch
import torch
import librosa
import torchaudio
from transformers import pipeline
from huggingface_hub import snapshot_download
from lhotse.utils import fix_random_seed

from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import rms_norm

from dataclasses import dataclass, field
from typing import Optional, List

from linacodec.vocoder.vocos import Vocos
from zipvoice.onnx_modeling import OnnxModel
from torch.nn.utils import parametrize


@dataclass
class LuxTTSConfig:
    # Model Setup
    model_dir: Optional[str] = None
    checkpoint_name: str = "model.pt"
    vocoder_path: Optional[str] = None
    trt_engine_path: Optional[str] = None

    # Tokenizer & Language
    tokenizer: str = "emilia"  # choices: ["emilia", "libritts", "espeak", "simple"]
    lang: str = "en-us"

    # Runtime attributes
    device: Optional[torch.device] = None
    sampling_rate: Optional[int] = None
    seed: Optional[int] = None
    onnx_int8: Optional[bool] = None


@torch.inference_mode()
def process_audio(audio, transcriber, tokenizer, feature_extractor, device, target_rms=0.1, duration=4, feat_scale=0.1, prompt_text=None, language=None):
    prompt_wav, sr = librosa.load(audio, sr=24000, duration=duration)
    prompt_wav2, sr = librosa.load(audio, sr=16000, duration=duration)

    if prompt_text is not None and prompt_text.strip():
        print(f"Using provided prompt text: {prompt_text}")
    else:
        gen_kwargs = {"task": "transcribe"}
        if language is not None:
            gen_kwargs["language"] = language
        
        prompt_text = transcriber(prompt_wav2, generate_kwargs=gen_kwargs)["text"].strip()
        
        # 如果自动转录未提取到任何有效 token，且未强制指定语言，尝试使用英文重试
        test_tokens = tokenizer.texts_to_token_ids([prompt_text])
        if (not test_tokens or len(test_tokens[0]) == 0) and language is None:
            logging.info("Initial ASR produced no English/Chinese tokens, retrying with language='en'...")
            prompt_text = transcriber(prompt_wav2, generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()

        print(f"Transcribed prompt text: {prompt_text}")

    prompt_wav = torch.from_numpy(prompt_wav).unsqueeze(0)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=24000
    ).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    if not prompt_tokens or len(prompt_tokens[0]) == 0:
        raise ValueError(
            f"No valid English or Chinese tokens could be extracted from prompt text: '{prompt_text}'. "
            "LuxTTS requires an English or Chinese reference audio. "
            "Please provide a clear English/Chinese audio or pass `prompt_text` / `language` to `encode_prompt()`."
        )

    return prompt_tokens, prompt_features_lens, prompt_features, prompt_rms

def generate(prompt_tokens, prompt_features_lens, prompt_features, prompt_rms, text, model, vocoder, tokenizer, num_step=4, guidance_scale=3.0, speed=1.0, t_shift=0.5, target_rms=0.1):
    tokens = tokenizer.texts_to_token_ids([text])
    device = next(model.parameters()).device  # Auto-detect device

    speed = speed * 1.3

    with torch.inference_mode():
        (pred_features, _, _, _) = model.sample(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_lens=prompt_features_lens,
            speed=speed,
            t_shift=t_shift,
            duration='predict',
            num_step=num_step,
            guidance_scale=guidance_scale,
        )

    # Convert to waveform
    pred_features = pred_features.permute(0, 2, 1) / 0.1
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Volume matching
    if prompt_rms < target_rms:
        wav = wav * (prompt_rms / target_rms)

    return wav

def load_models_gpu(model_path=None, device="cuda"):
    params = LuxTTSConfig()
    if model_path is None:
        model_path = snapshot_download("YatharthS/LuxTTS")

    token_file = f"{model_path}/tokens.txt"
    model_ckpt = f"{model_path}/model.pt"
    model_config = f"{model_path}/config.json"

    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = ZipVoiceDistill(
        **model_config["model"],
        **tokenizer_config,
    )
    load_checkpoint(filename=Path(model_ckpt), model=model, strict=True)
    params.device = torch.device(device, 0)

    model = model.to(params.device).eval()
    feature_extractor = VocosFbank()

    vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').to(device)
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f'{model_path}/vocoder/vocos.bin', map_location=params.device))

    params.sampling_rate = model_config["feature"]["sampling_rate"]
    return model, feature_extractor, vocos, tokenizer, transcriber

def load_models_cpu(model_path = None, num_thread=2):
    params = LuxTTSConfig()
    params.seed = 42

    model_path = snapshot_download('YatharthS/LuxTTS')

    token_file = f"{model_path}/tokens.txt"
    text_encoder_path = f"{model_path}/text_encoder.onnx"
    fm_decoder_path = f"{model_path}/fm_decoder.onnx"
    model_config  = f"{model_path}/config.json"

    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device='cpu')

    tokenizer = EmiliaTokenizer(token_file=token_file)
    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    model = OnnxModel(text_encoder_path, fm_decoder_path, num_thread=num_thread)

    vocos = Vocos.from_hparams(f'{model_path}/vocoder/config.yaml').eval()
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f'{model_path}/vocoder/vocos.bin', map_location=torch.device('cpu')))

    feature_extractor = VocosFbank()

    params.sampling_rate = model_config["feature"]["sampling_rate"]
    params.onnx_int8 = True
    return model, feature_extractor, vocos, tokenizer, transcriber
