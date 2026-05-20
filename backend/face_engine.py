"""
╔══════════════════════════════════════════════════════════════╗
║   KHEDIM IA v8.0 — MOTEUR BIOMÉTRIE MULTI-ENGINES           ║
║   Rotation : InsightFace → face_recognition → DeepFace      ║
║   Base de données : MongoDB Atlas (persistant)               ║
║   Fondé par Khedim Benyakhlef (Beny-Joe)                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import time
import numpy as np
from pymongo import MongoClient

# ══════════════════════════════════════════════
#   CONNEXION MONGODB ATLAS
# ══════════════════════════════════════════════

_mongo_client = None
_mongo_collection = None

def _get_collection():
    global _mongo_client, _mongo_collection
    if _mongo_collection is not None:
        return _mongo_collection
    try:
        uri = os.getenv("MONGO_URI", "mongodb+srv://khedimbenyakhlef_db_user:5q7RcqsusonfGeSv@cluster0.logfyqe.mongodb.net/atare_db")
        db_name = os.getenv("DB_NAME", "atare_db")
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _mongo_collection = _mongo_client[db_name]["faces"]
        print(f"✅ MongoDB connecté : {db_name}/faces")
        return _mongo_collection
    except Exception as e:
        print(f"❌ MongoDB erreur : {e}")
        return None

# ══════════════════════════════════════════════
#   BASE DE DONNÉES VISAGES (MongoDB)
# ══════════════════════════════════════════════

_db_cache = None
_db_cache_time = 0
_DB_CACHE_TTL = 60

def _load_db() -> dict:
    global _db_cache, _db_cache_time
    import time as _t
    now = _t.time()
    if _db_cache is not None and (now - _db_cache_time) < _DB_CACHE_TTL:
        return _db_cache
    col = _get_collection()
    if col is None:
        return _db_cache or {}
    try:
        db = {}
        for doc in col.find({}, {"_id": 0}):
            name = doc.get("name")
            if name:
                db[name] = doc
        _db_cache = db
        _db_cache_time = now
        print(f"[Cache] {len(db)} visages charges")
        return db
    except Exception as e:
        print(f"❌ _load_db erreur : {e}")
        return _db_cache or {}

def _invalidate_cache():
    global _db_cache_time
    _db_cache_time = 0

def _save_person(name: str, data: dict):
    col = _get_collection()
    if col is None:
        return
    try:
        col.update_one({"name": name}, {"$set": data}, upsert=True)
        _invalidate_cache()
    except Exception as e:
        print(f"❌ _save_person erreur : {e}")

def get_all_persons() -> list:
    col = _get_collection()
    if col is None:
        return []
    try:
        return [doc["name"] for doc in col.find({}, {"name": 1, "_id": 0})]
    except Exception:
        return []

def delete_person(name: str) -> bool:
    col = _get_collection()
    if col is None:
        return False
    try:
        result = col.delete_one({"name": name})
        return result.deleted_count > 0
    except Exception:
        return False

# ══════════════════════════════════════════════
#   ENGINE 1 : InsightFace
# ══════════════════════════════════════════════

_insight_app = None

def _get_insight():
    global _insight_app
    if _insight_app is not None:
        return _insight_app
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        _insight_app = app
        return app
    except Exception:
        return None

def _embed_insightface(img_rgb: np.ndarray) -> np.ndarray | None:
    app = _get_insight()
    if app is None:
        return None
    try:
        faces = app.get(img_rgb)
        if faces:
            return faces[0].embedding
    except Exception:
        pass
    return None

def _cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0

# ══════════════════════════════════════════════
#   ENGINE 2 : face_recognition (dlib)
# ══════════════════════════════════════════════

def _embed_face_recognition(img_rgb: np.ndarray) -> np.ndarray | None:
    try:
        import face_recognition
        encs = face_recognition.face_encodings(img_rgb)
        if encs:
            return encs[0]
    except Exception:
        pass
    return None

def _match_face_recognition(embedding, db: dict, threshold=0.5) -> tuple[str, float]:
    best_name, best_dist = "Inconnu", 1.0
    for name, data in db.items():
        for stored in data.get("embeddings_dlib", []):
            dist = float(np.linalg.norm(np.array(embedding) - np.array(stored)))
            if dist < best_dist:
                best_dist = dist
                best_name = name
    confidence = max(0.0, 1.0 - best_dist / threshold)
    return (best_name, round(confidence * 100)) if best_dist < threshold else ("Inconnu", 0)

# ══════════════════════════════════════════════
#   ENGINE 3 : DeepFace
# ══════════════════════════════════════════════

def _embed_deepface(img_rgb: np.ndarray) -> np.ndarray | None:
    try:
        from deepface import DeepFace
        import cv2
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        result = DeepFace.represent(
            img_path=img_bgr,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv"
        )
        if result:
            return np.array(result[0]["embedding"])
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════
#   ENREGISTREMENT MULTI-ENGINE
# ══════════════════════════════════════════════

def register_face(img_rgb: np.ndarray, name: str, section: str = "Manuel") -> dict:
    """Enregistre un visage avec tous les moteurs disponibles."""
    db = _load_db()
    if name not in db:
        db[name] = {
            "name": name,
            "embeddings_insight": [],
            "embeddings_dlib": [],
            "embeddings_deepface": [],
            "sections": [],
            "count": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M")
        }

    registered_engines = []

    # InsightFace
    emb = _embed_insightface(img_rgb)
    if emb is not None:
        db[name]["embeddings_insight"].append(emb.tolist())
        registered_engines.append("InsightFace")

    # face_recognition (dlib)
    emb2 = _embed_face_recognition(img_rgb)
    if emb2 is not None:
        db[name]["embeddings_dlib"].append(emb2.tolist())
        registered_engines.append("face_recognition")

    # DeepFace
    emb3 = _embed_deepface(img_rgb)
    if emb3 is not None:
        db[name]["embeddings_deepface"].append(emb3.tolist())
        registered_engines.append("DeepFace")

    if not registered_engines:
        return {"success": False, "message": "❌ Aucun visage détecté par aucun moteur."}

    db[name]["count"] += 1
    if section not in db[name]["sections"]:
        db[name]["sections"].append(section)
    db[name]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M")

    # ✅ Sauvegarde MongoDB au lieu de JSON
    _save_person(name, db[name])

    return {
        "success": True,
        "name": name,
        "engines": registered_engines,
        "total_samples": db[name]["count"],
        "message": f"✅ {name} enregistré via {', '.join(registered_engines)} ({db[name]['count']} échantillons)"
    }

# ══════════════════════════════════════════════
#   RECONNAISSANCE MULTI-ENGINE AVEC ROTATION
# ══════════════════════════════════════════════


# ══ ROTATION ENGINE — ajouté automatiquement ═══════════════════
try:
    from backend.face_engine_rotation import identify_face_rotation
    _ROTATION_AVAILABLE = True
    print("✅ Moteur rotation faciale chargé")
except Exception as _rot_err:
    _ROTATION_AVAILABLE = False
    print(f"⚠️ Moteur rotation non disponible: {_rot_err}")
# ═══════════════════════════════════════════════════════════════

def identify_face(img_rgb: np.ndarray) -> dict:
    """
    Tente la reconnaissance dans l'ordre :
    InsightFace → face_recognition → DeepFace
    Retourne le premier résultat positif.
    """
    db = _load_db()
    if not db:
        return {"name": "Inconnu", "confidence": 0, "engine": "none", "message": "Base vide"}

    results = []

    # ── Engine 1 : InsightFace ──
    emb = _embed_insightface(img_rgb)
    if emb is not None:
        best_name, best_score = "Inconnu", 0.0
        for name, data in db.items():
            for stored in data.get("embeddings_insight", []):
                sim = _cosine_sim(emb, stored)
                if sim > best_score:
                    best_score = sim
                    best_name = name
        confidence = round(best_score * 100)
        if best_score >= 0.35 and best_name != "Inconnu":
            return {
                "name": best_name,
                "confidence": confidence,
                "engine": "InsightFace",
                "message": f"✅ Reconnu : {best_name} ({confidence}%) via InsightFace"
            }
        results.append(("InsightFace", best_name, confidence))

    # ── Engine 2 : face_recognition ──
    emb2 = _embed_face_recognition(img_rgb)
    if emb2 is not None:
        name2, conf2 = _match_face_recognition(emb2, db)
        if name2 != "Inconnu":
            return {
                "name": name2,
                "confidence": conf2,
                "engine": "face_recognition",
                "message": f"✅ Reconnu : {name2} ({conf2}%) via face_recognition"
            }
        results.append(("face_recognition", name2, conf2))

    # ── Engine 3 : DeepFace ──
    emb3 = _embed_deepface(img_rgb)
    if emb3 is not None:
        best_name, best_score = "Inconnu", 0.0
        for name, data in db.items():
            for stored in data.get("embeddings_deepface", []):
                sim = _cosine_sim(emb3, stored)
                if sim > best_score:
                    best_score = sim
                    best_name = name
        confidence = round(best_score * 100)
        if best_score >= 0.40 and best_name != "Inconnu":
            return {
                "name": best_name,
                "confidence": confidence,
                "engine": "DeepFace",
                "message": f"✅ Reconnu : {best_name} ({confidence}%) via DeepFace"
            }
        results.append(("DeepFace", best_name, confidence))

    if results:
        best = max(results, key=lambda x: x[2])
        return {
            "name": "Inconnu",
            "confidence": best[2],
            "engine": best[0],
            "message": f"❓ Non reconnu (meilleur: {best[2]}% via {best[0]})"
        }

    return {"name": "Inconnu", "confidence": 0, "engine": "none", "message": "❌ Aucun visage détecté"}

# ══════════════════════════════════════════════
#   ANALYZE FRAME (interface publique)
# ══════════════════════════════════════════════

def analyze_frame(frame, section: str = "Caméra") -> dict:
    """Point d'entrée principal pour l'analyse d'une frame."""
    import cv2

    if frame is None:
        return _empty_result("Frame nulle")

    if isinstance(frame, np.ndarray):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
    else:
        try:
            frame_rgb = np.array(frame)
        except Exception:
            return _empty_result("Format invalide")

    if 'frame_rgb' not in dir():
        frame_rgb = frame

    personnes = []
    nb_visages = 0
    image_annotee = frame.copy()

    app = _get_insight()
    if app is not None:
        try:
            faces = app.get(frame_rgb)
            nb_visages = len(faces)
            for face in faces:
                box = face.bbox.astype(int)
                x1, y1, x2, y2 = box
                result = identify_face(frame_rgb)
                name = result["name"]
                conf = result["confidence"]
                color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
                cv2.rectangle(image_annotee, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({conf}%)"
                cv2.putText(image_annotee, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                personnes.append({
                    "name": name,
                    "confidence": conf,
                    "engine": result["engine"],
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        except Exception:
            pass

    if nb_visages == 0:
        result = identify_face(frame_rgb)
        if result["name"] != "Inconnu":
            nb_visages = 1
            personnes.append(result)

    msg = _build_message(nb_visages, personnes, section)

    return {
        "nb_visages": nb_visages,
        "personnes": personnes,
        "message": msg,
        "image_annotee": image_annotee,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M"),
        "mode": "multi-engine",
    }

def _empty_result(reason: str) -> dict:
    return {
        "nb_visages": 0, "personnes": [], "message": reason,
        "image_annotee": None, "timestamp": time.strftime("%Y-%m-%dT%H:%M"), "mode": ""
    }

def _build_message(nb: int, personnes: list, section: str) -> str:
    if nb == 0:
        return f"[{section}] Aucun visage détecté."
    noms = [f"{p['name']} ({p.get('confidence', 0)}%)" for p in personnes]
    return f"[{section}] {nb} visage(s) : {', '.join(noms)}"

# ══════════════════════════════════════════════
#   UTILITAIRES (compatibilité app.py)
# ══════════════════════════════════════════════

def sanitize_state(state: dict) -> dict:
    return {
        "nb": state.get("nb", 0),
        "personnes": state.get("personnes", []),
        "message": state.get("message", ""),
        "timestamp": state.get("timestamp", ""),
        "mode": state.get("mode", ""),
    }

def numpy_to_pil(img):
    if img is None:
        return None
    try:
        from PIL import Image
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img)
    except Exception:
        return None

def faces_db():
    return _load_db()

def get_system_info() -> dict:
    info = {}
    for lib in ["insightface", "face_recognition", "deepface", "cv2"]:
        try:
            __import__(lib)
            info[lib] = True
        except ImportError:
            info[lib] = False
    info["engines"] = [k for k, v in info.items() if v]
    info["db_count"] = len(_load_db())
    info["opencv"] = info.get("cv2", False)
    info["faces_enregistres"] = info.get("db_count", 0)
    info["insightface"] = info.get("insightface", False)
    info["face_recognition"] = info.get("face_recognition", False)
    info["deepface"] = info.get("deepface", False)
    return info

shared_memory = {}

def get_detection_log(limit: int = 50) -> list:
    return []

# ══════════════════════════════════════════════
#   MÉMOIRE DE SESSION
# ══════════════════════════════════════════════

class SessionMemory:
    def __init__(self):
        self._storage = {}

    def add(self, key, value):
        self._storage[key] = value

    def get(self, key, default=None):
        return self._storage.get(key, default)

    def get_context_text(self) -> str:
        if not self._storage:
            return "Aucune donnée en mémoire."
        lines = []
        for k, v in list(self._storage.items())[-5:]:
            v_str = str(v)[:100]
            lines.append(f"{k}: {v_str}")
        return "\n".join(lines) if lines else "Mémoire vide."

session_memory = SessionMemory()

class SharedMemory:
    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value

    def __contains__(self, key):
        return key in self._data
    def est_connu(self, key) -> bool:
        return key in self._data
    def get_contexte(self, key, default="") -> str:
        return str(self._data.get(key, default))
    def est_connu(self, key) -> bool:
        return key in self._data
    def get_contexte(self, key, default="") -> str:
        return str(self._data.get(key, default))
    def get_resume(self) -> str:
        if not self._data:
            return ""
        return " | ".join(f"{k}: {v}" for k, v in self._data.items())

shared_memory = SharedMemory()

class _SessionMemoryPatched(SessionMemory):
    def get_recent_alerts(self, limit: int = 15) -> list:
        alerts = self._storage.get("alerts", [])
        return alerts[-limit:] if alerts else []

    def add_alert(self, alert: str):
        if "alerts" not in self._storage:
            self._storage["alerts"] = []
        self._storage["alerts"].append(alert)

session_memory = _SessionMemoryPatched()
