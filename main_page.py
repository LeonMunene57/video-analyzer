import streamlit as st
from moviepy import VideoFileClip
import os

st.set_page_config(page_title="Local Video Audio Extractor", layout="centered")
st.title("Extract Audio from Local Video")

def extract_audio_with_moviepy(video_file, output_folder="data"):
    """
    Extracts audio from the uploaded video file and saves it as MP3.
    Returns the path to the saved MP3 file.
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Save uploaded file temporarily
        video_path = os.path.join(output_folder, video_file.name)
        with open(video_path, "wb") as out_file:
            out_file.write(video_file.getbuffer())
        # Use MoviePy to extract audio
        clip = VideoFileClip(video_path)
        audio_path = os.path.splitext(video_path)[0] + ".mp3"
        clip.audio.write_audiofile(audio_path)
        clip.close()
        return audio_path
    except Exception as e:
        return f"Error extracting audio: {str(e)}"

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
    else:
        st.error(audio_file_path)

st.markdown("---")
st.info("After uploading a video file above, its audio will be extracted and made ready for download.")

# Ensure 'moviepy' is installed: pip install moviepy
