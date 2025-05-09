# MultisensoryFusion
A Dynamic integragration of various AI's in a single software.

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

