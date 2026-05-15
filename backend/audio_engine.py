"""
╔══════════════════════════════════════════════════════════════╗
║   KHEDIM IA v8.0 — MOTEUR AUDIO                             ║
║   STT : API Groq Whisper-large-v3                           ║
║   TTS : Edge-TTS → gTTS → silencieux                        ║
║   Fondé par Khedim Benyakhlef (Beny-Joe)                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import asyncio
import tempfile

VOICES = {
    "militaire": "fr-FR-HenriNeural",
    "standard":  "fr-FR-DeniseNeural",
    "pro":       "fr-FR-AlainNeural",
}
DEFAULT_VOICE = "militaire"


def transcribe_audio(audio_path: str, language: str = "fr") -> dict:
    """STT via API Groq Whisper-large-v3 (sans Whisper local)"""
    if not audio_path or not os.path.exists(audio_path):
        return {"text": "", "success": False, "error": "Fichier audio introuvable.", "engine": "none"}
    try:
        import groq as groq_lib
        client = groq_lib.Groq(api_key=os.environ.get("GROQ_API_KEY"))
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3",
                language=language,
            )
        text = transcription.text.strip()
        if text:
            return {"text": text, "success": True, "engine": "groq-whisper-large-v3"}
    except Exception as e:
        return {"text": "", "success": False, "error": str(e), "engine": "none"}
    return {"text": "", "success": False, "error": "Transcription vide.", "engine": "none"}


async def _tts_edge_async(text: str, voice: str, path: str) -> bool:
    try:
        import edge_tts
        comm = edge_tts.Communicate(text[:600], voice)
        await comm.save(path)
        return os.path.exists(path) and os.path.getsize(path) > 100
    except Exception:
        return False


def generate_tts(text: str, voice_style: str = DEFAULT_VOICE, output_path: str = None) -> str | None:
    """TTS — Edge-TTS → gTTS → None"""
    if not text or not text.strip():
        return None
    voice = VOICES.get(voice_style, VOICES[DEFAULT_VOICE])
    clean = text.replace("*", "").replace("#", "").replace("_", "").strip()[:500]
    if output_path is None:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            output_path = tmp.name
            tmp.close()
        except Exception:
            return None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ok = loop.run_until_complete(_tts_edge_async(clean, voice, output_path))
        loop.close()
        if ok:
            return output_path
    except Exception:
        pass
    try:
        from gtts import gTTS
        gTTS(text=clean, lang="fr", slow=False).save(output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            return output_path
    except Exception:
        pass
    return None


def get_audio_info() -> dict:
    info = {"faster_whisper": False}
    try:
        import groq as groq_lib
        info["faster_whisper"] = True  # on réutilise la clé pour l'affichage ✅
    except ImportError:
        pass
    for lib in ["edge_tts", "gtts"]:
        try:
            __import__(lib.replace("-", "_"))
            info[lib] = True
        except ImportError:
            info[lib] = False
    return info
