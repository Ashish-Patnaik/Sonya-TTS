import os
import re
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.io.wavfile import write
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from tts import commons
from tts import utils
from tts.models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

_ESPEAK_LIBRARY = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
if os.path.exists(_ESPEAK_LIBRARY):
    EspeakWrapper.set_library(_ESPEAK_LIBRARY)
    print(f"✅ Found eSpeak at: {_ESPEAK_LIBRARY}")
else:
    print("⚠️ WARNING: eSpeak-ng not found (ok if already working)")


REPO_ID = "PatnaikAshish/Sonya-TTS"

MODEL_FILENAME = "checkpoints/sonya-tts.safetensors"
CONFIG_FILENAME = "checkpoints/config.json"

LOCAL_MODEL_PATH = "checkpoints/sonya-tts.safetensors"
LOCAL_CONFIG_PATH = "checkpoints/config.json"

OUTPUT_WAV = "output.wav"

TEXT = "Hello I am Sonya, an expressive Text to Speech model that can run fast in edge devices."

# --- CONTROLLABLE INFERENCE PARAMETERS ---
ns  = 0.5   # noise_scale
nsw = 0.6   # noise_scale_w
ls  = 1.0   # length_scale


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


def save_wav_int16(path, audio, sample_rate):
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)
    write(path, sample_rate, audio)


def main():
    print("🔄 Initializing Sonya TTS...")

    # --------------------------------------------------------
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

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)

    net_g.eval()

    print("📦 Loading model weights...")
    state_dict = load_file(model_path)
    net_g.load_state_dict(state_dict)
    print("🎉 Sonya TTS model loaded successfully!")

    print(f"🗣️ Generating: '{TEXT}'")
    stn_tst = get_text(TEXT, hps)

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=ns,
            noise_scale_w=nsw,
            length_scale=ls
        )[0][0, 0].cpu().float().numpy()

    save_wav_int16(OUTPUT_WAV, audio, hps.data.sampling_rate)
    print(f"💾 Saved audio to: {os.path.abspath(OUTPUT_WAV)}")

if __name__ == "__main__":
    main()

