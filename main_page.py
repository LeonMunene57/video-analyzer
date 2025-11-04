import streamlit as st
from moviepy import VideoFileClip
import whisper
import os
import re

st.set_page_config(page_title="Local Video Audio Extractor", layout="centered")
st.title("Extract Audio from Local Video")

def extract_audio_with_moviepy(video_file, output_folder="data"):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        video_path = os.path.join(output_folder, video_file.name)
        with open(video_path, "wb") as out_file:
            out_file.write(video_file.getbuffer())
        clip = VideoFileClip(video_path)
        audio_path = os.path.splitext(video_path)[0] + ".mp3"
        clip.audio.write_audiofile(audio_path)
        clip.close()
        return audio_path
    except Exception as e:
        return f"Error extracting audio: {str(e)}"

def transcribe_audio_with_whisper(audio_path, model_name="tiny"):
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        transcript = result["text"]
        # Split transcript on sentence-ending punctuation and join with newlines
        lines = [s.strip() for s in re.split(r'(?<=[.?!])\s+', transcript) if s.strip()]
        formatted_transcript = "\n".join(lines)
        return formatted_transcript
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

st.subheader("Upload Local Video File")
uploaded_file = st.file_uploader("Choose a video file (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.markdown("**File uploaded successfully!**")
    with st.spinner("Extracting audio..."):
        audio_file_path = extract_audio_with_moviepy(uploaded_file)
    if audio_file_path and not audio_file_path.startswith("Error"):
        st.success(f"Audio extracted and saved as: {audio_file_path}")
        with open(audio_file_path, "rb") as audio_file:
            st.download_button(
                label="Download Extracted Audio",
                data=audio_file,
                file_name=os.path.basename(audio_file_path),
                mime="audio/mp3"
            )
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio_with_whisper(audio_file_path, model_name="tiny")
        if transcript and not transcript.startswith("Error"):
            st.markdown("#### Transcript:")
            st.text_area("Transcript Output", transcript, height=300)
            st.download_button(
                label="Download Transcript",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )
        else:
            st.error(transcript)
    else:
        st.error(audio_file_path)

st.markdown("---")
st.info("After uploading a video file above, its audio will be extracted and transcribed to text, both available for download.")
