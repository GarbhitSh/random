import asyncio
import re
from json import JSONDecodeError
from datetime import datetime
from typing import Dict, List, Optional

import edge_tts
import streamlit as st
from googletrans import LANGUAGES, Translator


st.set_page_config(
    page_title="Polyglot Voice Studio",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VOICE_OPTIONS: List[Dict[str, str]] = [
    {
        "label": "English (US) – Ava",
        "voice_id": "en-US-JennyNeural",
        "lang": "en",
    },
    {
        "label": "English (UK) – Libby",
        "voice_id": "en-GB-LibbyNeural",
        "lang": "en",
    },
    {
        "label": "English (Australia) – Natasha",
        "voice_id": "en-AU-NatashaNeural",
        "lang": "en",
    },
    {
        "label": "Hindi – Swara",
        "voice_id": "hi-IN-SwaraNeural",
        "lang": "hi",
    },
]

TARGET_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
    "Portuguese": "pt",
    "No translation (use original text)": "none",
}

MAX_CHARS_TRANSLATION = 4500
MAX_CHARS_TTS = 3000

def _init_translator(service_urls):
    translator_instance = Translator(service_urls=service_urls)
    if not hasattr(Translator, "raise_Exception"):
        # googletrans 4.0.0rc1 uses `raise_Exception` in one code path.
        # Python 3.13+ removed that attribute, so keep them in sync.
        setattr(
            Translator,
            "raise_Exception",
            property(
                lambda self: getattr(self, "raise_exception", False),
                lambda self, value: setattr(self, "raise_exception", value),
            ),
        )
    if not hasattr(translator_instance, "raise_Exception"):
        translator_instance.raise_Exception = getattr(
            translator_instance, "raise_exception", False
        )
    try:
        translator_instance.raise_Exception = True
    except Exception:
        setattr(
            translator_instance,
            "raise_Exception",
            getattr(translator_instance, "raise_exception", False),
        )
    return translator_instance


PRIMARY_TRANSLATOR = _init_translator(
    ["translate.googleapis.com", "translate.google.com"]
)
FALLBACK_TRANSLATOR = _init_translator(["translate.google.com"])
TRANSLATORS = (PRIMARY_TRANSLATOR, FALLBACK_TRANSLATOR)
GT_LANGUAGES = {code.lower(): name for code, name in LANGUAGES.items()}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_chars: int) -> List[str]:
    """Break text into chunks without cutting sentences in half."""
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text.strip())
    if len(normalized) <= max_chars:
        return [normalized]

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: List[str] = []
    buffer = ""

    for sentence in sentences:
        if not sentence:
            continue
        prospective = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(prospective) <= max_chars:
            buffer = prospective
        else:
            if buffer:
                chunks.append(buffer)
            if len(sentence) > max_chars:
                for start in range(0, len(sentence), max_chars):
                    chunks.append(sentence[start : start + max_chars])
                buffer = ""
            else:
                buffer = sentence

    if buffer:
        chunks.append(buffer)

    return chunks


def rate_to_edge(rate_value: int) -> str:
    return f"{rate_value:+d}%"


def volume_to_edge(volume_value: int) -> str:
    return f"{volume_value:+d}%"


def detect_language_safe(sample: str):
    for client in TRANSLATORS:
        try:
            return client.detect(sample)
        except Exception as exc:
            if 'Unexpected status code "404"' in str(exc):
                continue
            raise
    return None


def translate_chunk_with_fallback(chunk: str, target_lang: str) -> str:
    last_status_exc: Optional[Exception] = None
    for client in TRANSLATORS:
        try:
            result = client.translate(chunk, dest=target_lang)
        except JSONDecodeError as exc:
            raise RuntimeError(
                "Google Translate returned an empty or invalid response. "
                "Streamlit Cloud sometimes blocks the unofficial googletrans backend. "
                "Try redeploying later or provide a paid translation API."
            ) from exc
        except Exception as exc:
            if 'Unexpected status code "404"' in str(exc):
                last_status_exc = exc
                continue
            raise
        else:
            return result.text

    raise RuntimeError(
        "Google Translate returned HTTP 404 for every available endpoint. "
        "Streamlit Cloud may be blocking the unofficial googletrans backend. "
        "Redeploy later or switch to a paid translation provider (e.g. Google Cloud Translation)."
    ) from last_status_exc


@st.cache_data(show_spinner=False)
def translate_large_text(text: str, target_lang: str) -> str:
    if target_lang in ("none", "", None):
        return text

    chunks = chunk_text(text, MAX_CHARS_TRANSLATION)
    translated_segments: List[str] = []

    for chunk in chunks:
        translated_segments.append(translate_chunk_with_fallback(chunk, target_lang))

    return " ".join(translated_segments)


async def synthesize_chunk(
    text: str, *, voice_id: str, rate: str, volume: str
) -> bytes:
    communicator = edge_tts.Communicate(text, voice=voice_id, rate=rate, volume=volume)
    audio = bytearray()

    async for chunk in communicator.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])

    return bytes(audio)


@st.cache_data(show_spinner=False, max_entries=32)
def synthesize_speech(
    text: str,
    voice_id: str,
    rate_value: int,
    volume_value: int,
) -> bytes:
    rate = rate_to_edge(rate_value)
    volume = volume_to_edge(volume_value)

    chunks = chunk_text(text, MAX_CHARS_TTS)
    audio_buffer = bytearray()

    for chunk in chunks:
        chunk_audio = asyncio.run(
            synthesize_chunk(chunk, voice_id=voice_id, rate=rate, volume=volume)
        )
        audio_buffer.extend(chunk_audio)

    return bytes(audio_buffer)


def get_voice_meta(label: str) -> Optional[Dict[str, str]]:
    for voice in VOICE_OPTIONS:
        if voice["label"] == label:
            return voice
    return None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🗣️ Polyglot Voice Studio")
st.caption(
    "Paste or type very large passages, translate them, and turn them into lifelike "
    "speech with play/pause controls and four curated female voices (including Hindi)."
)


if "translated_text" not in st.session_state:
    st.session_state["translated_text"] = ""
if "translation_lang" not in st.session_state:
    st.session_state["translation_lang"] = "none"


with st.container():
    st.subheader("1. Paste or type your script")
    raw_text = st.text_area(
        "Input text",
        height=260,
        placeholder="Paste any length of text here...",
        key="raw_text_input",
    )
    char_count = len(raw_text)
    st.caption(f"Characters: {char_count:,}")
    if raw_text.strip():
        sample = raw_text[:5000]
        detection = detect_language_safe(sample)
        if detection is None:
            st.caption("Detected language: unavailable")
        else:
            lang_code = detection.lang.lower()
            lang_label = GT_LANGUAGES.get(lang_code, lang_code).title()
            confidence = getattr(detection, "confidence", None)
            if confidence is not None:
                st.caption(
                    f"Detected language: {lang_label} "
                    f"(confidence {confidence * 100:.1f}%)"
                )
            else:
                st.caption(f"Detected language: {lang_label}")


col_translate, col_settings = st.columns([2, 1])

with col_translate:
    st.subheader("2. Translation (optional)")
    target_label = st.selectbox("Choose target language", list(TARGET_LANGUAGES.keys()))
    target_code = TARGET_LANGUAGES[target_label]

    if st.button("Translate text", use_container_width=True):
        if not raw_text.strip():
            st.warning("Please paste some text before translating.")
        else:
            with st.spinner("Translating. Large passages may take a moment..."):
                try:
                    translated = translate_large_text(raw_text, target_code)
                except Exception as exc:
                    st.error(f"Translation failed: {exc}")
                else:
                    st.session_state["translated_text"] = translated
                    st.session_state["translation_lang"] = target_code
                    st.success("Translation ready!")

    if st.session_state["translated_text"]:
        st.text_area(
            "Translated text",
            value=st.session_state["translated_text"],
            height=220,
            key="translated_text_view",
        )


with col_settings:
    st.subheader("3. Voice settings")
    selected_voice_label = st.selectbox(
        "Female voice", [voice["label"] for voice in VOICE_OPTIONS]
    )
    speech_rate = st.slider("Speech rate (%)", -30, 30, 0)
    volume = st.slider("Volume (%)", -20, 20, 0)
    source_choice = st.radio(
        "Read from",
        options=("Original text", "Translated text"),
        horizontal=True,
    )


st.subheader("4. Generate narration")
st.markdown(
    "The audio player supports native play / pause controls. For very large texts we "
    "automatically narrate in safe chunks and stitch the audio together for you."
)

if st.button("Create narration", type="primary", use_container_width=True):
    candidate_text = raw_text if source_choice == "Original text" else st.session_state[
        "translated_text"
    ]

    if not candidate_text.strip():
        st.warning(
            "There is nothing to narrate. Paste text or run a translation first."
        )
    else:
        voice_meta = get_voice_meta(selected_voice_label)
        if not voice_meta:
            st.error("Voice selection invalid. Please choose again.")
        else:
            with st.spinner("Rendering narration..."):
                try:
                    audio_bytes = synthesize_speech(
                        candidate_text,
                        voice_id=voice_meta["voice_id"],
                        rate_value=speech_rate,
                        volume_value=volume,
                    )
                except Exception as exc:
                    st.error(f"Speech synthesis failed: {exc}")
                else:
                    st.success("Narration ready! Use the player below.")
                    if not audio_bytes:
                        st.error("No audio bytes were returned. Please try again.")
                    else:
                        st.audio(audio_bytes, format="audio/mp3")

                        file_name = (
                            f"narration_{voice_meta['voice_id']}_"
                            f"{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.mp3"
                        )
                        st.download_button(
                            label="Download MP3",
                            data=audio_bytes,
                            file_name=file_name,
                            mime="audio/mpeg",
                            use_container_width=True,
                        )


with st.expander("How it works & tips", expanded=False):
    st.markdown(
        "- We automatically chunk large passages so translation and narration stay within API limits.\n"
        "- Translation uses Google's neural models; speech uses Microsoft neural female voices.\n"
        "- Volume and rate controls are relative adjustments. 0% keeps the natural voice profile.\n"
        "- Downloaded MP3 files include timestamps so you can catalog multiple takes easily."
    )


st.caption("Built with Streamlit, googletrans, and edge-tts.")

