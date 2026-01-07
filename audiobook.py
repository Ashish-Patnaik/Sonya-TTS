import os
import re
import torch
import numpy as np
from scipy.io.wavfile import write
from tts import commons
from tts import utils
from tts.models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

_ESPEAK_LIBRARY = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
if os.path.exists(_ESPEAK_LIBRARY):
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)
    print(f"✅ Found eSpeak-ng: {_ESPEAK_LIBRARY}")
else:
    print("⚠️ eSpeak-ng not found (ok if already working)")


REPO_ID = "PatnaikAshish/Sonya-TTS"
MODEL_FILENAME = "sonya-tts.safetensors"
CONFIG_FILENAME = "config.json"

LOCAL_MODEL_PATH = "checkpoints/sonya-tts.safetensors"
LOCAL_CONFIG_PATH = "checkpoints/config.json"
OUTPUT_WAV_SHORT = "output.wav"
OUTPUT_WAV_LONG  = "audiobook.wav"

USE_LONG_FORM = True  # ← change to False for short text

TEXT = """
A neural network or Artificial Neural Network is a computer system inspired by the human brain, 
using interconnected nodes neurons in layers to recognize complex patterns in data for tasks like 
image recognition, language processing, and prediction
"""

def save_wav_int16(path, audio, sample_rate):
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)
    write(path, sample_rate, audio)


def clean_text_for_vits(text):
    text = text.strip()

   
    text = text.replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")

  
    text = re.sub(r"[()\[\]{}<>]", "", text)

   
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text

def get_text(text, hps):
    text = clean_text_for_vits(text)
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def split_sentences(text):
    text = clean_text_for_vits(text)
    if not text:
        return []
    return re.split(r'(?<=[.!?])\s+', text)


def generate_audiobook(
    net_g,
    hps,
    text,
    device,
    output_file,
    noise_scale=0.5,
    noise_scale_w=0.6,
    length_scale=1.0,
    base_pause=0.4,
):
    print("📖 Long-form audiobook mode enabled")

    sentences = split_sentences(text)
    print(f"🔹 Sentences: {len(sentences)}")

    audio_chunks = []

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue

        stn_tst = get_text(sent, hps)

        with torch.no_grad():
            x = stn_tst.to(device).unsqueeze(0)
            x_len = torch.LongTensor([stn_tst.size(0)]).to(device)

            audio = net_g.infer(
                x,
                x_len,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0].cpu().numpy()

        if sent.endswith("?"):
            pause = base_pause + 0.15
        elif sent.endswith("!"):
            pause = base_pause
        else:
            pause = base_pause + 0.05

        silence = np.zeros(int(hps.data.sampling_rate * pause))

        audio_chunks.append(audio)
        audio_chunks.append(silence)

        print(f"   ✅ Sentence {i+1}/{len(sentences)} done")

    final_audio = np.concatenate(audio_chunks)
    save_wav_int16(output_file, final_audio, hps.data.sampling_rate)

    print(f"🎉 Audiobook saved: {os.path.abspath(output_file)}")


def main():
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_CONFIG_PATH):
        print("✅ Loading Sonya TTS from local checkpoints...")
        model_path = LOCAL_MODEL_PATH
        config_path = LOCAL_CONFIG_PATH
    else:
        print("🌍 Downloading Sonya TTS from Hugging Face...")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
        config_path = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILENAME)


    hps = utils.get_hparams_from_file(config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Load model
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    net_g.eval()

    # Load checkpoint
    state_dict = load_file(model_path)
    net_g.load_state_dict(state_dict)
    print(f"✅ Loaded model: {model_path}")

    
    if USE_LONG_FORM:
        generate_audiobook(
            net_g,
            hps,
            TEXT,
            device,
            OUTPUT_WAV_LONG,
        )
    else:
        print("🗣️ Short-text inference")

        stn_tst = get_text(TEXT, hps)
        with torch.no_grad():
            x = stn_tst.to(device).unsqueeze(0)
            x_len = torch.LongTensor([stn_tst.size(0)]).to(device)

            audio = net_g.infer(
                x,
                x_len,
                noise_scale=0.5,
                noise_scale_w=0.6,
                length_scale=1.0,
            )[0][0, 0].cpu().numpy()

        save_wav_int16(OUTPUT_WAV_SHORT, audio, hps.data.sampling_rate)
        print(f"💾 Saved audio: {os.path.abspath(OUTPUT_WAV_SHORT)}")

if __name__ == "__main__":
    main()
