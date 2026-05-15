"""
╔══════════════════════════════════════════════════════════════╗
║       GENDARME AI — MOTEUR VISION                           ║
║       Analyse d'images contextuelle                         ║
║       Fondé par Khedim Benyakhlef (Biny-Joe)               ║
╚══════════════════════════════════════════════════════════════╝
"""

import base64
import io
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def image_to_base64(image_path: str, max_size: int = 1280) -> str:
    """
    Convertit, améliore et redimensionne une image en base64 JPEG.
    Applique : contraste + netteté pour mieux voir les visages.
    """
    try:
        if HAS_PIL:
            img = Image.open(image_path)

            # Convertir en RGB (supporte RGBA, P, L, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # ── Amélioration contraste (utile caméra sombre/parasitée) ──
            try:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.3)   # +30% contraste
            except Exception:
                pass

            # ── Amélioration netteté ──
            try:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.4)   # +40% netteté
            except Exception:
                pass

            # ── Amélioration luminosité si image sombre ──
            try:
                import numpy as np
                arr = np.array(img)
                mean_brightness = arr.mean()
                if mean_brightness < 80:   # image sombre
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.4)
            except Exception:
                pass

            # ── Redimensionnement (max 1280px) ──
            img.thumbnail((max_size, max_size), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode()

        else:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()

    except Exception:
        return ""


def analyze_image_with_groq(image_path: str, user_prompt: str, groq_engine) -> dict:
    """
    Analyse une image via GROQ vision.
    Retourne {"description": str, "success": bool}
    """
    b64 = image_to_base64(image_path)
    if not b64:
        return {"description": "Impossible de lire l'image.", "success": False}

    system = """Tu es GENDARME AI, unité IA tactique fondée par Khedim Benyakhlef.
Tu analyses les images avec précision comme un agent sur le terrain.

RÈGLES :
- Décris ce que tu vois RÉELLEMENT dans l'image (personnes, objets, environnement)
- Si tu vois un visage humain, décris-le (genre, âge approximatif, expression)
- Si la question concerne une personne visible, réponds à propos de CETTE personne
- Commence par : "RAPPORT VISUEL — "
- Sois précis, professionnel, utile
- Si l'image est floue ou sombre, dis-le mais essaie quand même de décrire"""

    prompt = user_prompt or "Décris en détail ce que tu vois dans cette image."

    result = groq_engine.chat_vision(prompt, b64, system)
    return {
        "description": result.get("content", "Analyse indisponible."),
        "success": result.get("success", False)
    }


def get_image_info(image_path: str) -> dict:
    """Retourne les métadonnées d'une image"""
    try:
        if HAS_PIL:
            img = Image.open(image_path)
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format or Path(image_path).suffix.upper(),
                "size_kb": round(Path(image_path).stat().st_size / 1024, 1)
            }
    except Exception:
        pass
    return {"width": 0, "height": 0, "mode": "?", "format": "?", "size_kb": 0}
