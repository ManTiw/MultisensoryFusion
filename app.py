import streamlit as st
import importlib
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import tempfile
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import speech_recognition as sr
from PIL import Image
import os
import base64

# RTC Configuration for audio processing
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def display_base64_image(base64_str):
    img_html = f'<img src="data:image/png;base64,{base64_str}" alt="Generated Image" style="width:100%; max-width: 400px;" />'
    st.markdown(img_html, unsafe_allow_html=True)

# Speech-to-text function
def convert_speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError:
        return "Speech recognition service is unavailable."

# Streamlit UI
st.title("AI Model Hub")

# Model selection
model_choice = st.selectbox(
    "Choose a model:",
    [
        "Select a model",
        "Stable Diffusion v1.5: Text-to-Image",
        "Stable Diffusion 2.0: Text-to-Image",
        "OpenJourney: Artistic Text-to-Image",
        "GPT-2: Sentence-to-Story",
    ],
)

# Prompt input
text_prompt = st.text_input("Enter your text prompt:", placeholder="Enter your text here...")

# Audio recording
st.subheader("Record Audio:")
audio_processor = webrtc_streamer(
    key="microphone",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=rtc_config,
    media_stream_constraints={"audio": True, "video": False},
)

# File upload
st.subheader("Upload an Audio File:")
uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "webm"])

# Microphone audio recording
recorded_audio_path = None
if audio_processor and audio_processor.audio_receiver:
    recorded_audio_data = b"".join([frame.to_ndarray().tobytes() for frame in audio_processor.audio_receiver])
    if recorded_audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.write(recorded_audio_data)
            recorded_audio_path = temp_audio_file.name
            st.success("Audio recording completed. Ready for transcription.")

# Generate button functionality
if st.button("Generate"):
    if model_choice == "Select a model":
        st.error("Please select a model.")
    else:
        model_file_mapping = {
            "Stable Diffusion v1.5: Text-to-Image": "stable_diffusion_v1_5",
            "Stable Diffusion 2.0: Text-to-Image": "stable_diffusion_v2_0",
            "OpenJourney: Artistic Text-to-Image": "openjourney_model",  # NEW
            "GPT-2: Sentence-to-Story": "gpt2_model",
        }

        selected_model_file = model_file_mapping.get(model_choice)

        if selected_model_file:
            try:
                model_module = importlib.import_module(selected_model_file)

                # Stable Diffusion-based image models
                if model_choice.startswith("Stable Diffusion") or model_choice.startswith("OpenJourney"):
                    if text_prompt:
                        st.subheader("Generated Image:")
                        generated_images = model_module.generate_images(text_prompt)
                        for img_data in generated_images:
                            display_base64_image(img_data)
                    else:
                        st.error("Please enter a text prompt.")

                # GPT-2 story generation
                elif model_choice == "GPT-2: Sentence-to-Story":
                    if text_prompt:
                        st.subheader("Generated Output:")
                        generated_output = model_module.generate_text(text_prompt)
                        st.write(generated_output)
                    elif uploaded_file:
                        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                            temp_audio_file.write(uploaded_file.read())
                            audio_path = temp_audio_file.name
                        transcribed_text = convert_speech_to_text(audio_path)
                        st.write(f"Transcribed Text: {transcribed_text}")
                        st.subheader("Generated Output:")
                        generated_output = model_module.generate_text(transcribed_text)
                        st.write(generated_output)
                    elif recorded_audio_path:
                        transcribed_text = convert_speech_to_text(recorded_audio_path)
                        st.write(f"Transcribed Text: {transcribed_text}")
                        st.subheader("Generated Output:")
                        generated_output = model_module.generate_text(transcribed_text)
                        st.write(generated_output)
                    else:
                        st.error("Please enter a text prompt, upload an audio file, or record audio.")

            except Exception as e:
                st.error(f"Error loading the model: {e}")
