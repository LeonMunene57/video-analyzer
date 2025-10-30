import streamlit as st

st.set_page_config(page_title="Video Notes Generator", layout="centered")

st.title("Upload YouTube Video or Local Video")

# Section for YouTube Link
st.subheader("Upload via YouTube URL")
youtube_url = st.text_input("Paste YouTube video link here:")

if youtube_url:
    st.markdown(f"**YouTube Link uploaded:** {youtube_url}")

# Section for Local File Upload
st.subheader("Upload Local Video File")
uploaded_file = st.file_uploader("Choose a video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.markdown("**File uploaded successfully!**")

st.markdown("---")

st.info("After uploading, you will be able to generate notes in the next step.")

# You might connect 'next step' to audio extraction or transcript/summarization as a button later.
