import gradio as gr
import os
import re
import torch
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
    print(f"✅ Found eSpeak-ng: {_ESPEAK_LIBRARY}")


REPO_ID = "PatnaikAshish/Sonya-TTS"

MODEL_FILENAME = "sonya-tts.safetensors"
CONFIG_FILENAME = "config.json"

LOCAL_MODEL_PATH = "checkpoints/sonya-tts.safetensors"
LOCAL_CONFIG_PATH = "checkpoints/config.json"

device = "cuda" if torch.cuda.is_available() else "cpu"


def clean_text_for_vits(text):
    text = text.strip()
    text = text.replace("'", "'")
    text = text.replace(""", '"').replace(""", '"')
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


print("🔄 Loading Sonya TTS Model...")

if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_CONFIG_PATH):
    print("✅ Loading Sonya TTS from local checkpoints...")
    model_path = LOCAL_MODEL_PATH
    config_path = LOCAL_CONFIG_PATH
else:
    print("🌍 Downloading Sonya TTS from Hugging Face...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    config_path = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILENAME)

hps = utils.get_hparams_from_file(config_path)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).to(device)

net_g.eval()

state_dict = load_file(model_path)
net_g.load_state_dict(state_dict)
print("🎉 Sonya TTS loaded successfully!")


def infer_short(text, noise_scale, noise_scale_w, length_scale):
    if not text.strip():
        return None
        
    stn_tst = get_text(text, hps)
    
    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        
        audio = net_g.infer(
            x_tst, 
            x_tst_lengths, 
            noise_scale=noise_scale, 
            noise_scale_w=noise_scale_w, 
            length_scale=length_scale
        )[0][0,0].data.cpu().float().numpy()
        
    return (hps.data.sampling_rate, audio)


def infer_long(text, length_scale, noise_scale):
    if not text.strip():
        return None

    sentences = split_sentences(text)
    audio_chunks = []
    
    fixed_noise_w = 0.6 
    base_pause = 0.3

    for sent in sentences:
        if len(sent.strip()) < 2: 
            continue
        
        stn_tst = get_text(sent, hps)
        with torch.no_grad():
            x_tst = stn_tst.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            
            audio = net_g.infer(
                x_tst, 
                x_tst_lengths, 
                noise_scale=noise_scale, 
                noise_scale_w=fixed_noise_w, 
                length_scale=length_scale
            )[0][0,0].data.cpu().float().numpy()
            
        if sent.endswith("?"):
            pause_dur = base_pause + 0.2
        elif sent.endswith("!"):
            pause_dur = base_pause + 0.1
        else:
            pause_dur = base_pause
            
        silence = np.zeros(int(hps.data.sampling_rate * pause_dur))
        
        audio_chunks.append(audio)
        audio_chunks.append(silence)
        
    final_audio = np.concatenate(audio_chunks)
    return (hps.data.sampling_rate, final_audio)


theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="rose",
    neutral_hue="slate"
).set(
    button_primary_background_fill="linear-gradient(90deg, #ff69b4, #ff1493)",
    button_primary_background_fill_hover="linear-gradient(90deg, #ff1493, #c71585)",
    button_primary_text_color="white",
)

custom_css = """
.banner-container {
    width: 100%;
    max-width: 100%;
    margin: 0 auto 20px auto;
    display: flex;
    justify-content: center;
    align-items: center;
}

.banner-container img {
    width: 100%;
    max-width: 1800px;
    max-height: 120px;
    height: auto;
    object-fit: scale-down;  
    object-position: center;
    border-radius: 8px;
}

.main-title {
    text-align: center;
    color: #ff1493;
    font-size: 2em;
    font-weight: 700;
    margin: 15px 0 8px 0;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 1.1em;
    margin-bottom: 25px;
    font-weight: 400;
}

footer {
    display: none !important;
}
"""


with gr.Blocks(theme=theme, css=custom_css, title="Sonya TTS") as app:

    with gr.Row(elem_classes="banner-container"):
        if os.path.exists("logo.png"):
            gr.Image("logo.png", show_label=False, container=False, elem_classes="banner-img")

    gr.HTML("""
        <h1 class="main-title">✨ Sonya TTS — A Beautiful, Expressive Neural Voice Engine</h1>
        <p class="subtitle">High-fidelity AI speech with emotion, rhythm, and audiobook mode</p>
    """)

    with gr.Tabs():

        with gr.TabItem("🎛️ Studio Mode"):
            with gr.Row():
                with gr.Column(scale=2):
                    inp_short = gr.Textbox(
                        label="💬 Input Text", 
                        placeholder="Type something for Sonya to say...", 
                        lines=4,
                        value="Hello! I am Sonya, your AI voice."
                    )

                    with gr.Accordion("⚙️ Voice Controls", open=True):
                        slider_ns = gr.Slider(0.1, 1.0, value=0.4, label="🎭 Emotion", info="Higher = more expressive")
                        slider_nsw = gr.Slider(0.1, 1.0, value=0.5, label="🎵 Rhythm", info="Higher = looser timing")
                        slider_ls = gr.Slider(0.5, 1.5, value=0.97, label="⏱ Speed", info="Lower = faster, Higher = slower")

                    btn_short = gr.Button("✨ Generate Voice", variant="primary", size="lg")

                with gr.Column(scale=1):
                    out_short = gr.Audio(label="🔊 Sonya's Voice", type="numpy")

            btn_short.click(
                infer_short, 
                inputs=[inp_short, slider_ns, slider_nsw, slider_ls], 
                outputs=[out_short]
            )

        with gr.TabItem("📖 Audiobook Mode"):
            gr.Markdown(
                """<p style='text-align: center; color: #666; font-size: 1.05em;'>
                Paste long text. Sonya will read it beautifully with natural pauses.
                </p>""",
                elem_classes="audiobook-description"
            )

            with gr.Row():
                with gr.Column(scale=2):
                    inp_long = gr.Textbox(
                        label="📜 Long Text Input", 
                        placeholder="Paste your story or article here...", 
                        lines=10
                    )

                    with gr.Accordion("⚙️ Narration Settings", open=False):
                        long_ls = gr.Slider(0.5, 1.5, value=1.0, label="⏱ Reading Speed")
                        long_ns = gr.Slider(0.1, 1.0, value=0.5, label="🎭 Tone Variation")

                    btn_long = gr.Button("🎧 Read Aloud", variant="primary", size="lg")

                with gr.Column(scale=1):
                    out_long = gr.Audio(label="📢 Full Narration", type="numpy")

            btn_long.click(
                infer_long, 
                inputs=[inp_long, long_ls, long_ns], 
                outputs=[out_long]
            )

if __name__ == "__main__":
    app.launch()
