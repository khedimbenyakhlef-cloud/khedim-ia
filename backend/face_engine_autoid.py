"""
Auto-ID intelligent : identifie les inconnus par embeddings ArcFace
et leur assigne des designations militaires automatiques.
"""
import time
import numpy as np
import os

# Base session : embeddings des inconnus
_session_db = []
_counter = [0]
_LABELS = [
    "ALPHA","BRAVO","CHARLIE","DELTA","ECHO",
    "FOXTROT","GOLF","HOTEL","INDIA","JULIET",
    "KILO","LIMA","MIKE","NOVEMBER","OSCAR"
]

# Singleton InsightFace
_app_cache = [None]

def _get_app():
    if _app_cache[0] is None:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(160, 160))
            _app_cache[0] = app
        except Exception as e:
            print(f"InsightFace non disponible: {e}")
    return _app_cache[0]

def _next_label():
    idx = _counter[0] % len(_LABELS)
    num = _counter[0] // len(_LABELS)
    _counter[0] += 1
    suffix = f"-{num+1}" if num > 0 else ""
    return f"SUJET-{_LABELS[idx]}{suffix}"

def _cosine_sim(a, b):
    try:
        a, b = np.array(a, dtype=float), np.array(b, dtype=float)
        n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
    except:
        return 0.0

def _get_embedding(img_rgb: np.ndarray):
    """Extrait embedding 512-dim via InsightFace buffalo_sc"""
    # Methode 1 : InsightFace ArcFace 512-dim
    try:
        app = _get_app()
        if app is not None:
            import cv2
            small = cv2.resize(img_rgb, (0,0), fx=0.5, fy=0.5)
            faces = app.get(small)
            if faces:
                return faces[0].embedding, "ArcFace-512"
    except Exception as e:
        pass
    # Methode 2 : face_recognition 128-dim
    try:
        import face_recognition as fr
        import cv2
        small = cv2.resize(img_rgb, (0,0), fx=0.5, fy=0.5)
        locs = fr.face_locations(small, model="hog")
        if locs:
            encs = fr.face_encodings(small, locs)
            if encs:
                return encs[0], "FaceNet-128"
    except Exception as e:
        pass
    return None, None

def identify_or_register(img_rgb: np.ndarray, threshold: float = 0.45) -> str:
    emb, method = _get_embedding(img_rgb)
    if emb is None:
        return "Aucun visage detecte"

    best_label, best_sim = None, 0.0
    for entry in _session_db:
        sim = _cosine_sim(emb, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_label = entry["label"]

    if best_sim >= threshold and best_label:
        for entry in _session_db:
            if entry["label"] == best_label:
                entry["last_seen"] = time.time()
                entry["count"] = entry.get("count", 1) + 1
        conf = int(best_sim * 100)
        return f"{best_label} ({conf}% — {method})"
    else:
        new_label = _next_label()
        _session_db.append({
            "label": new_label,
            "embedding": emb,
            "method": method,
            "first_seen": time.time(),
            "last_seen": time.time(),
            "count": 1
        })
        return f"[NOUVEAU] {new_label} — identite inconnue, profil cree"

def get_session_faces() -> list:
    return [{"id": f["label"], "method": f.get("method","?"), "count": f.get("count",1), "last_seen": f["last_seen"]} for f in _session_db]

def reset_session():
    _session_db.clear()
    _counter[0] = 0
    _app_cache[0] = None
