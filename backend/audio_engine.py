"""
╔══════════════════════════════════════════════════════════════╗
║   GENDARME AI v3.0 — MOTEUR AUDIO CPU LITE                  ║
║   STT : faster-whisper small (CPU) → Google fallback        ║
║   TTS : Edge-TTS → gTTS → silencieux (jamais d'erreur)      ║
║   DÉBRUITAGE : scipy + numpy (filtre passe-haut + norm.)    ║
║   Fondé par Khedim Benyakhlef (Biny-Joe)                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import asyncio
import tempfile

_whisper_model = None

VOICES = {
    "militaire": "fr-FR-HenriNeural",
    "standard":  "fr-FR-DeniseNeural",
    "pro":       "fr-FR-AlainNeural",
}
DEFAULT_VOICE = "militaire"


# ══════════════════════════════════════════════════
#   DÉBRUITAGE AUDIO — réduction du bruit parasite
# ══════════════════════════════════════════════════

def _denoise_audio(audio_path: str) -> str:
    """
    Réduit les bruits parasites avant transcription.
    Retourne TOUJOURS un chemin valide (original si tout échoue).
    """
    try:
        import numpy as np
        from scipy.io import wavfile
        from scipy.signal import butter, filtfilt

        sr, data = wavfile.read(audio_path)

        # Convertir en float32 mono
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        # 1. Normalisation volume
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak * 0.95

        nyq = sr / 2.0

        # 2. Filtre passe-haut 80Hz (coupe grondements)
        low_cut = 80.0 / nyq
        if 0 < low_cut < 1:
            b, a = butter(2, low_cut, btype="high")
            data = filtfilt(b, a, data)

        # 3. Filtre passe-bas 8000Hz (coupe sifflements)
        high_cut = min(8000.0 / nyq, 0.99)
        if 0 < high_cut < 1:
            b, a = butter(2, high_cut, btype="low")
            data = filtfilt(b, a, data)

        # 4. Wiener optionnel
        try:
            from scipy.signal import wiener
            data = wiener(data, mysize=29)
        except Exception:
            pass

        # 5. Renormalisation finale
        peak2 = np.abs(data).max()
        if peak2 > 0:
            data = data / peak2 * 0.92

        # 6. Rééchantillonnage 16kHz si nécessaire
        if sr != 16000:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(int(sr), 16000)
                data = resample_poly(data, 16000 // g, int(sr) // g)
                sr = 16000
            except Exception:
                pass

        # Sauvegarder WAV nettoyé
        out_tmp = tempfile.NamedTemporaryFile(suffix="_clean.wav", delete=False)
        out_path = out_tmp.name
        out_tmp.close()

        data_int16 = (data * 32767).astype(np.int16)
        from scipy.io import wavfile as wf
        wf.write(out_path, sr, data_int16)
        return out_path

    except Exception:
        # scipy absent ou fichier non-WAV → essai pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            try:
                from pydub.effects import normalize
                audio = normalize(audio)
            except Exception:
                pass
            out_tmp = tempfile.NamedTemporaryFile(suffix="_clean.wav", delete=False)
            out_path = out_tmp.name
            out_tmp.close()
            audio.export(out_path, format="wav")
            return out_path
        except Exception:
            pass

    # Aucun traitement possible → fichier original intact
    return audio_path


# ══════════════════════════════════════════════════
#   TRANSCRIPTION STT
# ══════════════════════════════════════════════════

def transcribe_audio(audio_path: str, language: str = "fr") -> dict:
    """
    STT avec débruitage préalable.
    Pipeline : débruitage → faster-whisper base → tiny → Google Speech
    """
    global _whisper_model

    if not audio_path or not os.path.exists(audio_path):
        return {"text": "", "success": False, "error": "Fichier audio introuvable.", "engine": "none"}

    # ── Débruitage avant transcription ──
    clean_path = _denoise_audio(audio_path)

    # ── Tentative 1 : faster-whisper (base > tiny pour meilleure précision) ──
    for model_size in ["base", "tiny"]:
        try:
            from faster_whisper import WhisperModel
            if _whisper_model is None or getattr(_whisper_model, "_model_size", "") != model_size:
                _whisper_model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    download_root="/tmp/whisper_models"
                )
                _whisper_model._model_size = model_size

            segs, info = _whisper_model.transcribe(
                clean_path,
                language=language,
                beam_size=5,                   # ↑ meilleure précision (était 1)
                best_of=5,                     # ↑ sélectionne la meilleure
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 300,   # ignore silences courts
                    "speech_pad_ms": 200,             # garde du contexte
                    "threshold": 0.35,                # seuil VAD plus sensible (était défaut 0.5)
                },
                condition_on_previous_text=False,     # évite hallucinations
                temperature=0.0,                      # mode déterministe
                no_speech_threshold=0.5,              # filtre audio vide
            )
            text = " ".join(s.text for s in segs).strip()
            # Filtrer les hallucinations classiques de Whisper
            hallucinations = [
                "merci", "sous-titres", "sous titres", "abonnez", "likez",
                "transcription", "www.", ".com", "youtube", "musique",
                "♪", "♫", "[", "]", "(musique)", "(bruit)",
            ]
            text_lower = text.lower()
            is_hallucination = any(h in text_lower for h in hallucinations) and len(text) < 30
            if text and not is_hallucination:
                return {
                    "text": text,
                    "language": info.language,
                    "success": True,
                    "engine": f"faster-whisper-{model_size}",
                    "denoised": clean_path != audio_path,
                }
        except Exception:
            continue

    # ── Tentative 2 : SpeechRecognition Google (avec fichier débruité) ──
    try:
        import speech_recognition as sr
        rec = sr.Recognizer()
        # Réduire la sensibilité au bruit ambiant
        rec.energy_threshold = 200
        rec.dynamic_energy_threshold = True
        with sr.AudioFile(clean_path) as src:
            rec.adjust_for_ambient_noise(src, duration=0.5)
            audio = rec.record(src)
        text = rec.recognize_google(audio, language="fr-FR")
        if text.strip():
            return {"text": text, "success": True, "engine": "google-speech"}
    except Exception:
        pass

    # ── Nettoyage fichier temporaire ──
    if clean_path != audio_path:
        try:
            os.unlink(clean_path)
        except Exception:
            pass

    return {"text": "", "success": False, "error": "Transcription impossible — bruit trop élevé ou micro trop faible.", "engine": "none"}


# ══════════════════════════════════════════════════
#   TTS — SYNTHÈSE VOCALE
# ══════════════════════════════════════════════════

async def _tts_edge_async(text: str, voice: str, path: str) -> bool:
    try:
        import edge_tts
        comm = edge_tts.Communicate(text[:600], voice)
        await comm.save(path)
        return os.path.exists(path) and os.path.getsize(path) > 100
    except Exception:
        return False


def generate_tts(text: str, voice_style: str = DEFAULT_VOICE, output_path: str = None) -> str | None:
    """TTS — Edge-TTS → gTTS → None (jamais d'exception)"""
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

    # ── Edge-TTS ──
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ok = loop.run_until_complete(_tts_edge_async(clean, voice, output_path))
        loop.close()
        if ok:
            return output_path
    except Exception:
        pass

    # ── gTTS ──
    try:
        from gtts import gTTS
        gTTS(text=clean, lang="fr", slow=False).save(output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            return output_path
    except Exception:
        pass

    return None


def get_audio_info() -> dict:
    info = {}
    for lib in ["faster_whisper", "whisper", "edge_tts", "gtts", "speech_recognition", "scipy", "pydub"]:
        try:
            __import__(lib.replace("-", "_"))
            info[lib] = True
        except ImportError:
            info[lib] = False
    return info
