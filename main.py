from zipvoice.luxvoice import LuxTTS
import soundfile as sf

# load model on MPS for macs
lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')


text = "Hey, what's up? I'm feeling really great if you ask me honestly!Speech topics for students can range from everyday experiences and personal development to technology, education, society, and more unconventional ideas. Finding a suitable subject, however, can sometimes be more difficult than delivering the speech itself, especially when you want something engaging without making it unnecessarily complicated. This guide brings together 100 ideas across different themes to help students find a subject that feels relevant, interesting, and worth presenting."

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

## encode audio(takes 10s to init because of librosa first time)
## 推荐强制指定参考音频的语言 ('en' 或 'zh')，可彻底避免 Whisper 自动语种检测跑偏产生幻觉
encoded_prompt = lux_tts.encode_prompt(prompt_audio, rms=0.01, language='zh')

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech (if in interactive/notebook environment)
try:
    from IPython.display import Audio, display  # type: ignore
    display(Audio(final_wav, rate=48000))
except (ImportError, NameError):
    pass