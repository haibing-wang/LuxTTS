from zipvoice.luxvoice import LuxTTS
import soundfile as sf

# load model on GPU
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')

# load model on CPU
# lux_tts = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=2)

# load model on MPS for macs
lux_tts = LuxTTS('YatharthS/LuxTTS', device='mps')


text = "Hey, what's up? I'm feeling really great if you ask me honestly!"

## change this to your reference file path, can be wav/mp3
prompt_audio = 'audio_file.wav'

## encode audio(takes 10s to init because of librosa first time)
encoded_prompt = lux_tts.encode_prompt(prompt_audio, rms=0.01)

## generate speech
final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps=4)

## save audio
final_wav = final_wav.numpy().squeeze()
sf.write('output.wav', final_wav, 48000)

## display speech (if in interactive/notebook environment)
try:
    from IPython.display import Audio, display
    display(Audio(final_wav, rate=48000))
except (ImportError, NameError):
    pass