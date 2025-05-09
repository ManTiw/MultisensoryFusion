# AI-Driven Multisensory Fusion

## 🎯 Project Overview

**AI-Driven Multisensory Fusion** is an interactive application that integrates multiple advanced AI models to generate outputs from **text or audio input**. Users can:
- Generate coherent stories using **GPT-2**
- Convert text prompts into images using **Stable Diffusion v2.0**, **Stable Diffusion v1.5**, or **OpenJourney**
- Use **speech recognition** to transcribe spoken input and feed it directly into the AI pipeline

This is implemented using **Streamlit** as a web interface and supports real-time input-output interaction.

---

## 🔧 Features

- 🧠 **GPT-2 (Text Generation)**: Converts text or audio-transcribed prompts into creative stories.
- 🎨 **Stable Diffusion & OpenJourney (Text-to-Image)**: Generates realistic or artistic images based on prompts.
- 🎤 **SpeechRecognition**: Converts recorded or uploaded audio into text.
- ⚡ **Live Testing**: Supports microphone input and file upload with immediate AI processing.

---

## 🛠️ Tech Stack

| Layer         | Technology |
|---------------|------------|
| Frontend      | Streamlit  |
| Backend       | Python 3.10 |
| NLP Model     | Hugging Face GPT-2 (gpt2-large) |
| Image Models  | Stable Diffusion v2.0, Stable Diffusion v1.5, OpenJourney |
| Audio Input   | Google Speech Recognition API |
| Frameworks    | PyTorch, Transformers, Diffusers, PIL, Base64 |

---

## 🧪 How It Works

1. **Input Selection**: User types a text prompt or records/uploads an audio file.
2. **Model Selection**: Choose between GPT-2, Stable Diffusion v1.5, v2.0, or OpenJourney.
3. **Output Generation**:
   - For **GPT-2**: Text prompt (or transcribed audio) → Generated story
   - For **Diffusion Models**: Text prompt → Generated image
4. **Streamlit UI**: Displays output in real-time with user-friendly controls.

---

## 📂 Project Structure
```plaintext
AI-Driven-Multisensory-Fusion/
├── app.py                         # Main Streamlit UI for selecting models and inputs
├── gpt2_model.py                  # GPT-2 model for text/story generation
├── stable_diffusion_v1_5.py       # (Optional) Stable Diffusion v1.5 for realistic image generation
├── stable_diffusion_v2_0.py       # Stable Diffusion v2.0 for high-quality image generation
├── openjourney_model.py           # Artistic image generation using OpenJourney
├── requirements.txt               # List of Python dependencies
├── README.md                      # Project overview and documentation
├── outputs/                       # Folder for storing generated outputs (images/text) [optional]
│   ├── images/
│   └── stories/
├── audio/                         # Uploaded or recorded audio files [temporary]
├── utils/                         # Utility scripts (e.g., file handlers, helper functions) [optional]
├── assets/                        # Static assets like screenshots, logos (used in README or app)
├── test/                          # Testing scripts or mock data for UAT/QA
│   └── test_cases.md
└── appendix/                      # Reports, documentation, or user manual
    ├── report.docx
    └── plagiarism_certificate.pdf
