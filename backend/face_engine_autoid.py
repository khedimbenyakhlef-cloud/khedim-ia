"""
Auto-ID temporaire : détecte les visages et leur donne un ID de session
sans base de données préenregistrée.
"""
import time
import numpy as np

# Stockage session : embeddings + IDs temporaires
_session_faces = []  # liste de {"id": "Personne_001", "embedding": np.array, "last_seen": timestamp}
_counter = [0]

def _next_id() -> str:
    _counter[0] += 1
    return f"Personne_{_counter[0]:03d}"

def _cosine_sim(a, b) -> float:
    try:
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0
    except Exception:
        return 0.0

def _get_embedding(img_rgb: np.ndarray):
    """Essaie InsightFace → face_recognition → None"""
    # InsightFace
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        faces = app.get(img_rgb)
        if faces:
            return faces[0].embedding
    except Exception:
        pass
    # face_recognition
    try:
        import face_recognition
        encs = face_recognition.face_encodings(img_rgb)
        if encs:
            return encs[0]
    except Exception:
        pass
    return None

def identify_or_register(img_rgb: np.ndarray, threshold: float = 0.55) -> str:
    """
    Retourne l'ID temporaire de la personne détectée.
    Si inconnue → crée un nouvel ID de session.
    Si aucun visage → retourne 'Aucun visage détecté'.
    """
    emb = _get_embedding(img_rgb)
    if emb is None:
        return "Aucun visage détecté"

    # Chercher dans la session
    best_id, best_sim = None, 0.0
    for entry in _session_faces:
        sim = _cosine_sim(emb, entry["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_id = entry["id"]

    if best_sim >= threshold and best_id:
        # Mise à jour last_seen
        for entry in _session_faces:
            if entry["id"] == best_id:
                entry["last_seen"] = time.time()
        return best_id
    else:
        # Nouveau visage → nouvel ID
        new_id = _next_id()
        _session_faces.append({
            "id": new_id,
            "embedding": emb,
            "last_seen": time.time()
        })
        return f"🆕 {new_id} (nouveau)"

def get_session_faces() -> list:
    return [{"id": f["id"], "last_seen": f["last_seen"]} for f in _session_faces]

def reset_session():
    _session_faces.clear()
    _counter[0] = 0
