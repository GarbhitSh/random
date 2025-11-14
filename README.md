# Polyglot Voice Studio

Streamlit application that lets you:

- Paste extremely large bodies of text
- Translate them into several languages
- Generate natural-sounding speech with four curated female voices (including Hindi)
- Play or pause the narration directly in the browser and download the MP3

Powered by `googletrans` for neural translation and `edge-tts` (Microsoft neural voices) for high-quality TTS.

## Requirements

- Python 3.9+
- Internet access (for translation and speech synthesis)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

This opens `http://localhost:8501` in your browser.

## Usage

1. Paste or type any amount of text. The app auto-detects the source language.
2. (Optional) Choose a target language and click **Translate text**.
3. Pick one of the four female voices:
   - English (US) – Ava
   - English (UK) – Libby
   - English (Australia) – Natasha
   - Hindi – Swara
4. Adjust speech rate / volume and choose whether to read the original or translated text.
5. Click **Create narration**. Once ready you can play, pause, or download the MP3.

### Large text handling

- Text is automatically chunked (≈3k chars) so both translation and narration stay within API limits.
- Chunking happens at sentence boundaries whenever possible so narration remains natural.

## Notes

- The googletrans package relies on Google translate endpoints and may occasionally throttle. Just retry if that happens.
- Speech synthesis uses Microsoft Edge neural voices via `edge-tts`, which does not require API keys.

