<p align="center">
  <img src="logo.png" alt="Sonya TTS Logo" width="800"/>
</p>

<h1 align="center">✨ Sonya TTS</h1>
<h3 align="center">A Beautiful, Expressive Neural Voice Engine</h3>

<p align="center">
  <em>High-fidelity AI speech with emotion, rhythm, and audiobook-quality narration</em>
</p>

<p align="center">
  <a href="https://huggingface.co/PatnaikAshish/Sonya-TTS">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow" alt="Hugging Face"/>
  </a>
  <a href="https://huggingface.co/spaces/PatnaikAshish/Sonya-TTS">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow" alt="Hugging Face Demo"/>
  </a>
  <img src="https://img.shields.io/badge/Language-English-blue" alt="Language"/>
  <img src="https://img.shields.io/badge/Architecture-VITS-green" alt="VITS"/>
  <img src="https://img.shields.io/badge/Python-3.10-brightgreen" alt="Python"/>
</p>

---

## 🎧 Listen to Sonya

Experience the expressive quality of Sonya TTS:

<div align="center">

https://github.com/user-attachments/assets/801a64d9-fd60-4dd2-8be9-c697cf0ea514

</div>

*Extended narration showcasing rhythm control, natural pauses, and consistent tone across paragraphs. More examples in examples folder*

Try Demo at Hugging Space Demo
<a href="https://huggingface.co/spaces/PatnaikAshish/Sonya-TTS">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow" alt="Hugging Face Demo"/>
</a>

---

## 🌸 About Sonya TTS

**Sonya TTS** is a lightweight, expressive **single-speaker English Text-to-Speech model** built on the **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)** architecture.

Trained for approximately **10,000 steps** on a publicly available **expressive voice dataset**, Sonya delivers:

- 🎭 **Natural emotion and intonation** — More human-like speech with genuine expressiveness
- 🎵 **Smooth rhythm and prosody** — Natural flow and timing in speech
- 📖 **Long-form narration** — Perfect for audiobook-style content with consistent quality
- ⚡ **Blazing-fast inference** — Optimized for both **GPU and CPU** deployment

This isn't just a model—it's a complete, production-ready TTS system with a web interface, command-line tools, and audiobook narration capabilities.

---

## ✨ Key Features

### 🎭 Expressive Voice Quality
Unlike monotone TTS models, Sonya produces speech with natural emotion, dynamic intonation, and human-like expressiveness. Trained on an expressive dataset, it captures the nuances that make speech feel alive.

### ⚡ Lightning-Fast Inference
Highly optimized for real-world deployment:
- **GPU**: Extremely fast generation for real-time applications
- **CPU**: Efficient performance for edge devices and local deployments
- Low latency makes it suitable for interactive applications

### 📖 Audiobook Mode
Built for long-form content with:
- Intelligent sentence splitting and paragraph handling
- Natural pauses between sentences
- Consistent voice quality across extended text
- Stable rhythm and pacing throughout

### 🎛️ Fine-Grained Voice Control
Customize speech output with intuitive parameters:
- **Emotion (Noise Scale)** — Control expressiveness and variation
- **Rhythm (Noise Width)** — Adjust timing and flow
- **Speed (Length Scale)** — Modify speaking rate

### 🌍 Open & Accessible
Model weights and configuration files are publicly hosted on Hugging Face:
- 📦 **SafeTensors** format for secure, fast loading
- 🔓 Available for research and experimentation
- 🚀 Easy integration with your projects

---

## ⚠️ Limitations & Transparency

Sonya TTS is a research project and **not a perfect commercial solution**:

- **Word skipping**: Occasionally skips or merges words in complex sentences
- **Pronunciation**: Some uncommon words may be mispronounced
- **Alignment artifacts**: Rare timing issues in very long passages
- **Single speaker**: Currently supports only one English voice
- **Language**: English only at this time

Despite these limitations, Sonya demonstrates strong practical usability and expressive quality.

---

## 🧠 Training Journey

This project was a deep dive into modern speech synthesis:

| Detail | Value |
|--------|-------|
| **Architecture** | VITS (Conditional VAE + GAN) |
| **Training Steps** | ~10,400 |
| **Dataset** | Public expressive speech corpus |
| **Language** | English |
| **Speaker** | Single female voice |
| **Training Focus** | Emotion, prosody, and long-form stability |

### What I Learned
Building Sonya taught me invaluable lessons about:
- Text-to-speech alignment mechanisms and attention
- Prosody control and emotional expressiveness
- Audio generation pipelines and vocoding
- Model optimization for inference speed
- Packaging and deployment of ML models
- Real-world challenges in speech synthesis

---

## 📦 Repository Structure

```
Sonya-TTS/
├── checkpoints/
│   ├── sonya-tts.safetensors    # Model weights (SafeTensors format)
│   └── config.json              # Model configuration
│
├── tts/                         # Core model architecture
│   ├── models.py
│   ├── commons.py
│   └── modules.py
│
├── text/                        # Text processing pipeline
│   ├── symbols.py
│   ├── cleaners.py
│   └── __init__.py
│
├── infer.py                     # CLI for short text synthesis
├── audiobook.py                 # Long-form narration script
├── webui.py                     # Gradio web interface
│
├── examples/
│   ├── short.wav                # Quick speech demo
│   └── long.wav                 # Audiobook demo
│
├── logo.png                     # Project logo
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Conda (recommended) or virtualenv
- eSpeak-NG (for phonemization)

### Step 1: Create Environment

```bash
# Create a new conda environment
conda create -n sonya-tts python=3.10 -y

# Activate the environment
conda activate sonya-tts
```

### Step 2: Install eSpeak-NG

**🪟 Windows**
1. Download the installer from [eSpeak-NG Releases](https://github.com/espeak-ng/espeak-ng/releases)
2. Run the installer and follow the setup wizard
3. Add eSpeak to your system PATH if not done automatically

**🐧 Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install espeak-ng
```

**🍎 macOS**
```bash
# Using Homebrew
brew install espeak-ng
```

### Step 3: Install Dependencies

```bash
# Install all required Python packages
pip install -r requirements.txt
```

### Step 4: Launch Sonya TTS

```bash
# Start the web interface
python webui.py
```

The terminal will display a local URL (typically `http://127.0.0.1:7860`). Open it in your browser to access the interface!

---

## 🎯 Usage Options

Sonya TTS provides three flexible ways to generate speech:

### 1️⃣ `infer.py`

Perfect for generating single audio files from short text:

```bash
python infer.py 
```

**Use Case**: Quick testing, automation scripts, batch processing

### 2️⃣ `audiobook.py` — Long-Form Narration

Designed for extended text with intelligent sentence splitting:

```bash
python audiobook.py 
```

**Features**:
- Automatic paragraph detection
- Natural pauses between sentences
- Consistent voice across long passages
- Perfect for audiobooks, articles, and documentation

### 3️⃣ `webui.py` — Interactive Web Interface

Beautiful Gradio-powered UI with real-time controls:

```bash
python webui.py
```

**Features**:
- Adjustable emotion, rhythm, and speed sliders
- Audiobook mode toggle
- Download generated audio
- No coding required!

---

## 🌍 Model Hosting

All model files are hosted on Hugging Face for easy access:

**🤗 Model Repository**: [PatnaikAshish/Sonya-TTS](https://huggingface.co/PatnaikAshish/Sonya-TTS)

**Files in `checkpoints/` directory**:
- `sonya-tts.safetensors` — Model weights (SafeTensors format)
- `config.json` — Model configuration and hyperparameters

The code **automatically downloads** these files on first run if they're not present locally. No manual setup needed!

---

## 🎛️ Advanced Configuration

You can customize the voice output by adjusting these parameters:

| Parameter | Range | Effect |
|-----------|-------|--------|
| **noise_scale** | 0.1 - 1.0 | Controls emotion and expressiveness (higher = more variation) |
| **noise_scale_w** | 0.1 - 1.0 | Affects rhythm and timing (higher = more natural pauses) |
| **length_scale** | 0.5 - 2.0 | Controls speaking speed (lower = faster, higher = slower) |

Example in code:
```python
    text="Your text here",
    noise_scale=0.667,      # Moderate emotion
    noise_scale_w=0.8,      # Natural rhythm
    length_scale=1.0        # Normal speed
```

---

## 💡 Use Cases

Sonya TTS is versatile and can be used for:

- 📚 **Audiobook Production** — Convert books and articles to speech
- 🎮 **Game Narration** — Dynamic voiceovers for indie games
- 📱 **Accessibility Tools** — Screen readers and assistive technology
- 🎓 **E-Learning** — Educational content narration
- 🤖 **Virtual Assistants** — Expressive voice for chatbots
- 📻 **Podcast Intros** — Quick voiceovers and announcements
- 🎬 **Prototyping** — Rapid audio mockups for videos

---

## 🔧 Technical Details

### VITS Architecture
Sonya uses VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech), which combines:
- **Conditional VAE** for probabilistic acoustic modeling
- **GAN-based training** for high-quality audio generation
- **Normalizing flows** for flexible distribution modeling
- **Stochastic duration prediction** for natural timing

### Performance Benchmarks
- **GPU (NVIDIA RTX 3090)**: ~0.1s for 10 seconds of audio
- **CPU (Intel i7-12700K)**: ~2s for 10 seconds of audio
- Real-time factor: 10x-100x depending on hardware

---

## 📜 License & Citation
The project is MIT License and If you use Sonya TTS in your projects, please credit:

```bibtex
@software{sonya_tts_2026,
  author = {Ashish Patnaik},
  title = {Sonya TTS: An Expressive Neural Voice Engine},
  year = {2026},
  url = {https://huggingface.co/PatnaikAshish/Sonya-TTS}
}
```
Also see the original repo about vits:
```
https://github.com/jaywalnut310/vits
```

---

## 💜 Final Words

Sonya TTS represents countless hours of experimentation, training, debugging, and iteration. It's not perfect—but it's real, it's fast, and it's expressive.

This project taught me that building AI isn't just about achieving perfect metrics; it's about creating something useful, understanding the challenges deeply, and sharing knowledge with the community.

If Sonya helps you in any way—whether for a project, learning, or just exploration—I'd genuinely love to hear about it.

✨ **Thank you for listening to Sonya.**

---

## 👤 Author

**Ashish Patnaik**  
🤗 Hugging Face: [@PatnaikAshish](https://huggingface.co/PatnaikAshish)  
📧 Reach out for collaborations or questions!

---

## 🔗 Quick Links

- [🤗 Model on Hugging Face](https://huggingface.co/PatnaikAshish/Sonya-TTS)
- [📖 VITS Paper](https://arxiv.org/abs/2106.06103)
- [🎤 eSpeak-NG](https://github.com/espeak-ng/espeak-ng)

---

<p align="center">
  <sub>Made with 💜 by Ashish Patnaik</sub>
</p>
