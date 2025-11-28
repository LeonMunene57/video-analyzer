# Video Notes Generator

This project is a Streamlit application that processes educational YouTube videos by downloading audio, transcribing it with OpenAI Whisper, and generating concise notes using a fine‑tuned FLAN‑T5‑small model.

---

## Features

- Input YouTube URLs or upload local video/audio files
- Automatic audio extraction from videos (PyTube + MoviePy)
- Speech‑to‑text transcription with OpenAI Whisper
- Abstractive summarization with a LoRA‑fine‑tuned T5 model
- Simple Streamlit UI with progress indicators and downloadable text notes
- GPU/TPU friendly, tested in Google Colab environments

---



