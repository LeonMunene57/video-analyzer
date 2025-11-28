# video-analyzer
This Streamlit application processes educational YouTube videos by downloading audio, transcribing it with OpenAI's Whisper model, and generating concise summaries using a fine-tuned FLAN-T5-small model. Built for NLP engineers working on video analysis, it supports local or cloud deployment and integrates Hugging Face Transformers for efficient model handling.

Features
Upload or input YouTube URLs for automatic audio extraction via PyTube.

Real-time transcription of audio clips using Whisper for accurate speech-to-text conversion.

Abstractive summarization powered by a LoRA-fine-tuned T5 model, optimized for long transcripts up to 2000 words.

Interactive Streamlit interface with progress indicators and downloadable notes in text format.

GPU/TPU compatibility for faster processing, tested on Google Colab environments.

Installation
Clone the repository and navigate to the project directory.

Create a virtual environment: python -m venv venv.

Activate the environment: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows).

Install dependencies: pip install -r requirements.txt.

Ensure FFmpeg is installed for audio processing: sudo apt install ffmpeg (Linux) or download from the official site for other OS.​

For GPU support, install PyTorch with CUDA if available; the app falls back to CPU automatically.​

Usage
Run the app with streamlit run app.py. Enter a YouTube URL or upload an audio/video file. The app extracts audio using MoviePy, transcribes via Whisper, and generates summaries with the T5 model. View results in the browser and download generated notes. Tested with Python 3.10+ and handles batch processing for multiple videos.​

Project Structure
app.py: Main Streamlit application script handling UI, audio extraction, transcription, and summarization.

model_handler.py: Loads and runs the fine-tuned T5 model using Hugging Face Transformers.

audio_processor.py: Manages YouTube download (PyTube), audio extraction (MoviePy), and Whisper transcription.

requirements.txt: Lists all Python dependencies.

data/: Directory for temporary audio files and cached model outputs.

Fine-Tuning Notes
The T5 model was fine-tuned on ~699,000 examples from datasets like CNN/DailyMail using LoRA in Google Colab with TPU v2-8 runtime. Training optimized batch sizes (e.g., 8-16 per device) to reduce runtime from 4.5 hours to under 30 minutes while minimizing validation loss. ROUGE metrics guide evaluation for summarization quality.

Troubleshooting
If Whisper fails, verify FFmpeg installation and audio format compatibility.

For T5 loading errors, ensure sufficient RAM (8GB+ recommended) or use model quantization.

Deployment issues on Streamlit Cloud: Check requirements.txt for version conflicts with transformers and torch.​

Debug training pipelines in Colab by monitoring PyTorch/XLA compatibility.
