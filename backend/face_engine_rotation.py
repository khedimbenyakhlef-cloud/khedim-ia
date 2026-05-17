"""
╔══════════════════════════════════════════════════════════════╗
║   KHEDIM IA — Moteur Reconnaissance Faciale v8.1            ║
║   Rotation automatique des bibliothèques biométriques        ║
║   InsightFace → face_recognition → DeepFace → Auto-ID       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
from typing import Optional

# ══════════════════════════════════════════════
#   STOCKAGE SESSION (personnes non enregistrées)
# ══════════════════════════════════════════════

_session_db = {}   # { embedding_hash: {"id": "Inconnu_001", "embedding": np.array, "hits": int, "last_seen": float} }
_session_counter = [0]
_detection_cache = {}   # { frame_hash: (result_dict, timestamp) }
CACHE_TTL = 1.5  # secondes — évite de recalculer sur des frames similaires


def _next_session_id() -> str:
    _session_counter[0] += 1
    return f"Inconnu_{_session_counter[0]:03d}"


def _cosine_sim(a, b) -> float:
    try:
        a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0
    except Exception:
        return 0.0


def _frame_hash(img_rgb: np.ndarray) -> str:
    """Hash rapide d'une frame pour le cache."""
    try:
        small = img_rgb[::8, ::8].flatten()[:256]
        return str(hash(small.tobytes()))
    except Exception:
        return ""


# ══════════════════════════════════════════════
#   MOTEURS BIOMÉTRIQUES — ROTATION AUTOMATIQUE
# ══════════════════════════════════════════════

class EngineResult:
    def __init__(self, engine: str, embedding: Optional[np.ndarray],
                 name: Optional[str], confidence: float, faces_count: int = 0):
        self.engine = engine
        self.embedding = embedding
        self.name = name
        self.confidence = confidence
        self.faces_count = faces_count
        self.success = embedding is not None


def _try_insightface(img_rgb: np.ndarray) -> EngineResult:
    """InsightFace — plus précis, 512-dim."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        faces = app.get(img_rgb)
        if not faces:
            return EngineResult("InsightFace", None, None, 0.0, 0)
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb = face.embedding
        conf = float(getattr(face, "det_score", 0.85))
        return EngineResult("InsightFace", emb, None, conf, len(faces))
    except Exception as e:
        print(f"[InsightFace] Erreur: {e}")
        return EngineResult("InsightFace", None, None, 0.0, 0)


def _try_face_recognition(img_rgb: np.ndarray) -> EngineResult:
    """face_recognition (dlib) — rapide, 128-dim."""
    try:
        import face_recognition as fr
        locations = fr.face_locations(img_rgb, model="hog")
        if not locations:
            return EngineResult("face_recognition", None, None, 0.0, 0)
        encodings = fr.face_encodings(img_rgb, locations)
        if not encodings:
            return EngineResult("face_recognition", None, None, 0.0, 0)
        # Prendre le plus grand visage
        idx = 0
        if len(locations) > 1:
            areas = [(b-t) * (r-l) for (t, r, b, l) in locations]
            idx = int(np.argmax(areas))
        emb = encodings[idx]
        return EngineResult("face_recognition", emb, None, 0.80, len(locations))
    except Exception as e:
        print(f"[face_recognition] Erreur: {e}")
        return EngineResult("face_recognition", None, None, 0.0, 0)


def _try_deepface(img_rgb: np.ndarray) -> EngineResult:
    """DeepFace — plusieurs modèles en fallback."""
    models = ["VGG-Face", "Facenet", "ArcFace"]
    for model_name in models:
        try:
            from deepface import DeepFace
            import cv2
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            result = DeepFace.represent(
                img_path=img_bgr,
                model_name=model_name,
                enforce_detection=False,
                detector_backend="opencv"
            )
            if result and isinstance(result, list):
                emb = np.array(result[0]["embedding"], dtype=np.float32)
                conf = float(result[0].get("face_confidence", 0.75))
                return EngineResult(f"DeepFace/{model_name}", emb, None, conf, 1)
        except Exception as e:
            print(f"[DeepFace/{model_name}] Erreur: {e}")
            continue
    return EngineResult("DeepFace", None, None, 0.0, 0)


# Ordre de rotation des moteurs
ENGINES_ORDER = [
    ("InsightFace",      _try_insightface),
    ("face_recognition", _try_face_recognition),
    ("DeepFace",         _try_deepface),
]

# Seuils de similarité par moteur
THRESHOLDS = {
    "InsightFace":      0.50,
    "face_recognition": 0.55,
    "DeepFace":         0.45,
}


# ══════════════════════════════════════════════
#   MATCHING — BASE MONGO + SESSION
# ══════════════════════════════════════════════

def _match_in_mongo(embedding: np.ndarray, engine_name: str, threshold: float) -> Optional[dict]:
    """Cherche l'embedding dans MongoDB (visages enregistrés)."""
    try:
        from backend.face_engine import _get_collection
        col = _get_collection()
        if col is None:
            return None
        all_docs = list(col.find({}, {"name": 1, "section": 1, "embeddings": 1}))
        best_name, best_sim, best_section = None, 0.0, ""
        for doc in all_docs:
            embs = doc.get("embeddings", {})
            # Les embeddings sont stockés par moteur
            eng_embs = embs.get(engine_name, embs.get("InsightFace", embs.get("face_recognition", [])))
            if not eng_embs:
                # Chercher dans tous les moteurs disponibles
                for v in embs.values():
                    if isinstance(v, list) and v:
                        eng_embs = v
                        break
            if not isinstance(eng_embs, list):
                eng_embs = [eng_embs]
            for stored_emb in eng_embs:
                if stored_emb is None:
                    continue
                sim = _cosine_sim(embedding, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_name = doc.get("name", "?")
                    best_section = doc.get("section", "")
        if best_sim >= threshold and best_name:
            return {"name": best_name, "section": best_section, "confidence": best_sim, "source": "mongodb"}
    except Exception as e:
        print(f"[match_mongo] Erreur: {e}")
    return None


def _match_in_session(embedding: np.ndarray, threshold: float = 0.52) -> Optional[dict]:
    """Cherche l'embedding dans la session (personnes non enregistrées)."""
    best_key, best_sim = None, 0.0
    for key, entry in _session_db.items():
        sim = _cosine_sim(embedding, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_key = key
    if best_sim >= threshold and best_key:
        entry = _session_db[best_key]
        entry["hits"] += 1
        entry["last_seen"] = time.time()
        return {"name": entry["id"], "section": "Session", "confidence": best_sim, "source": "session"}
    return None


def _register_session(embedding: np.ndarray) -> str:
    """Enregistre un nouveau visage inconnu en session."""
    new_id = _next_session_id()
    key = str(id(embedding)) + str(time.time())
    _session_db[key] = {
        "id": new_id,
        "embedding": embedding,
        "hits": 1,
        "last_seen": time.time()
    }
    print(f"[Session] Nouveau visage enregistré : {new_id}")
    return new_id


# ══════════════════════════════════════════════
#   MOTEUR PRINCIPAL — ROTATION AUTOMATIQUE
# ══════════════════════════════════════════════

def identify_face_rotation(img_rgb: np.ndarray) -> dict:
    """
    Identification faciale en rotation automatique.
    Ordre : InsightFace → face_recognition → DeepFace
    Pour chaque moteur : cherche d'abord dans MongoDB, puis session.
    Si aucun résultat → crée un ID de session temporaire.
    
    Retourne un dict compatible avec l'ancien identify_face() :
    {
        "connu": bool,
        "nom": str,
        "section": str,
        "confiance": float,
        "engine": str,
        "nb_visages": int,
        "source": "mongodb" | "session" | "nouveau",
        "personnes": [...]
    }
    """
    # Vérifier le cache
    fhash = _frame_hash(img_rgb)
    if fhash and fhash in _detection_cache:
        cached_result, cached_time = _detection_cache[fhash]
        if time.time() - cached_time < CACHE_TTL:
            return cached_result

    t_start = time.time()
    result = None

    for engine_name, engine_fn in ENGINES_ORDER:
        print(f"[Rotation] Essai moteur: {engine_name}")
        eng_result = engine_fn(img_rgb)

        if not eng_result.success:
            print(f"[Rotation] {engine_name}: aucun visage détecté")
            continue

        threshold = THRESHOLDS.get(engine_name.split("/")[0], 0.50)
        embedding = eng_result.embedding
        nb_visages = eng_result.faces_count

        # 1. Chercher dans MongoDB
        mongo_match = _match_in_mongo(embedding, engine_name, threshold)
        if mongo_match:
            elapsed = round(time.time() - t_start, 2)
            print(f"[Rotation] ✅ MongoDB — {mongo_match['name']} ({engine_name}) en {elapsed}s")
            result = {
                "connu": True,
                "nom": mongo_match["name"],
                "section": mongo_match["section"],
                "confiance": mongo_match["confidence"],
                "engine": engine_name,
                "nb_visages": nb_visages,
                "source": "mongodb",
                "temps": elapsed,
                "personnes": [{
                    "nom": mongo_match["name"],
                    "connu": True,
                    "confiance": mongo_match["confidence"],
                    "section": mongo_match["section"]
                }]
            }
            break

        # 2. Chercher dans la session
        session_match = _match_in_session(embedding)
        if session_match:
            elapsed = round(time.time() - t_start, 2)
            print(f"[Rotation] 🔄 Session — {session_match['name']} ({engine_name}) en {elapsed}s")
            result = {
                "connu": False,
                "nom": session_match["name"],
                "section": "Session",
                "confiance": session_match["confidence"],
                "engine": engine_name,
                "nb_visages": nb_visages,
                "source": "session",
                "temps": elapsed,
                "personnes": [{
                    "nom": session_match["name"],
                    "connu": False,
                    "confiance": session_match["confidence"],
                    "section": "Session"
                }]
            }
            break

        # 3. Nouveau visage — enregistrer en session et continuer
        # On continue la rotation pour essayer les autres moteurs d'abord
        # Si c'est le dernier moteur, on l'enregistre
        if engine_name == ENGINES_ORDER[-1][0]:
            new_id = _register_session(embedding)
            elapsed = round(time.time() - t_start, 2)
            print(f"[Rotation] 🆕 Nouveau visage: {new_id} ({engine_name}) en {elapsed}s")
            result = {
                "connu": False,
                "nom": new_id,
                "section": "Inconnu",
                "confiance": eng_result.confidence,
                "engine": engine_name,
                "nb_visages": nb_visages,
                "source": "nouveau",
                "temps": elapsed,
                "personnes": [{
                    "nom": new_id,
                    "connu": False,
                    "confiance": eng_result.confidence,
                    "section": "Inconnu"
                }]
            }

    # Si aucun moteur n'a détecté de visage
    if result is None:
        elapsed = round(time.time() - t_start, 2)
        result = {
            "connu": False,
            "nom": "Aucun visage",
            "section": "",
            "confiance": 0.0,
            "engine": "Aucun",
            "nb_visages": 0,
            "source": "none",
            "temps": elapsed,
            "personnes": []
        }

    # Mettre en cache
    if fhash:
        _detection_cache[fhash] = (result, time.time())
        # Nettoyer le cache si trop grand
        if len(_detection_cache) > 50:
            oldest = min(_detection_cache, key=lambda k: _detection_cache[k][1])
            del _detection_cache[oldest]

    return result


def get_session_summary() -> list:
    """Retourne la liste des visages inconnus détectés en session."""
    return [
        {
            "id": e["id"],
            "hits": e["hits"],
            "last_seen": round(time.time() - e["last_seen"], 1)
        }
        for e in _session_db.values()
    ]


def reset_session():
    """Réinitialise la session des visages inconnus."""
    _session_db.clear()
    _session_counter[0] = 0
    _detection_cache.clear()
    print("[Session] Réinitialisée")
