import streamlit as st
from moviepy import VideoFileClip
import whisper
import os
import re
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Ensure NLTK sentence chunking is available
nltk.download('punkt', quiet=True)

@st.cache_resource
def load_fine_tuned_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, t5_model = load_fine_tuned_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)

st.set_page_config(page_title="Video Audio Summarizer", layout="centered")
st.title("Summarize Videos")

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
        lines = [s.strip() for s in re.split(r'(?<=[.?!])\s+', transcript) if s.strip()]
        formatted_transcript = "\n".join(lines)
        return formatted_transcript
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def chunk_text_with_overlap(text, chunk_size=10, overlap=4):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks = []
    start = 0
    while start < len(sentences):
        chunk = sentences[start:start + chunk_size]
        chunks.append(" ".join(chunk))
        start += max(chunk_size - overlap, 1)
    return chunks

def summarize_transcript_chunked(
    transcript,
    chunk_size=10,
    overlap=4,
    chunk_max_length=512,
    chunk_max_summary=160,
    final_max_summary=400
):
    prefix = "summarize only the key ideas from the main topic in full, cohesive detail. Also give a topic relevant introduction before the summary and comprehensive conclusion at the end of the summary: "  # Prompt tuned
    chunks = chunk_text_with_overlap(transcript, chunk_size=chunk_size, overlap=overlap)
    summaries = []
    for ix, chunk in enumerate(chunks):
        input_text = prefix + chunk
        enc = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=chunk_max_length,  # Ensures max context without overflow
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        summary_ids = t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=chunk_max_summary,
            min_length=64,
            length_penalty=1.0,
            num_beams=8,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary.strip())
    # Improved deduplication: substring-based, avoid near-duplicates
    cleaned = []
    for summ in summaries:
        if summ and not any(summ in existing for existing in cleaned):
            cleaned.append(summ)
    # Final merge-and-summarize step
    if len(cleaned) > 1:
        combined_summary = " ".join(cleaned)
        input_text = prefix + combined_summary
        enc = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=chunk_max_length,
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        summary_ids = t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=final_max_summary,
            min_length=180,
            length_penalty=1.0,
            num_beams=8,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return final_summary.strip()
    else:
        return cleaned[0] if cleaned else ""

st.subheader("Upload Video File")
uploaded_file = st.file_uploader(
    "Choose a video file (MP4, MOV, AVI, MKV)",
    type=["mp4", "mov", "avi", "mkv"],
)

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
                mime="audio/mp3",
            )
        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio_with_whisper(audio_file_path, model_name="tiny")
        if transcript and not transcript.startswith("Error"):
            st.markdown("#### Transcript:")
            st.text_area("Transcript", transcript, height=300)
            if st.button("Generate Summary"):
                with st.spinner("Summarizing..."):
                    summary = summarize_transcript_chunked(
                        transcript,
                        chunk_size=10,
                        overlap=4,
                    )
                from nltk.tokenize import sent_tokenize
                if summary:
                    st.markdown("Summary")
                    sentences = sent_tokenize(summary)
                    st.markdown('\n'.join([f"- {sent.strip()}" for sent in sentences if sent.strip()]))
                else:
                    st.warning("Could not generate a summary from the transcript.")
        else:
            st.error(transcript)
    else:
        st.error(audio_file_path)
