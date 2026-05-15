"""
╔══════════════════════════════════════════════════════════════╗
║   KHEDIM IA — MOTEUR RECONNAISSANCE FACIALE v8.0            ║
║   InsightFace Multi-angle + Mémoire Partagée Inter-Sections  ║
║   Multi-encodages par personne — Profil/Frontal/Nuit        ║
║   Fondé par Khedim Benyakhlef (Biny-Joe)                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, json
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy.spatial.distance import cosine as cosine_distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ══ CHEMINS ══
MEMORY_DIR       = Path("memory")
FACES_DB_PATH    = MEMORY_DIR / "faces_db.json"
DETECTION_LOG    = MEMORY_DIR / "detection_log.json"
SESSION_MEM_PATH = MEMORY_DIR / "session_memory.json"
SHARED_ID_PATH   = MEMORY_DIR / "identifications_partagees.json"
MEMORY_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════
#   MÉMOIRE PARTAGÉE INTER-SECTIONS (singleton)
# ══════════════════════════════════════════════════

class SharedIdentificationMemory:
    """
    Mémoire centrale partagée entre toutes les sections.
    Une identification dans n'importe quelle section (Chat, Audio,
    Caméra, Vision, Multimodal) est immédiatement disponible partout.
    """
    def __init__(self):
        self._data = self._load()

    def _load(self):
        if SHARED_ID_PATH.exists():
            try:
                with open(SHARED_ID_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"identifications": {}, "derniere_maj": None}

    def _save(self):
        SHARED_ID_PATH.parent.mkdir(exist_ok=True)
        self._data["derniere_maj"] = datetime.now().isoformat()
        with open(SHARED_ID_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def enregistrer(self, nom: str, section: str, confiance: float):
        if nom not in self._data["identifications"]:
            self._data["identifications"][nom] = {
                "nom": nom,
                "premiere_detection": datetime.now().isoformat(),
                "nb_total": 0,
                "sections_vues": [],
                "confiance_max": 0.0,
                "derniere_section": None,
                "derniere_detection": None,
            }
        e = self._data["identifications"][nom]
        e["nb_total"] += 1
        e["derniere_detection"] = datetime.now().isoformat()
        e["derniere_section"] = section
        if confiance > e.get("confiance_max", 0):
            e["confiance_max"] = confiance
        if section not in e["sections_vues"]:
            e["sections_vues"].append(section)
        self._save()

    def est_connu(self, nom: str) -> bool:
        return nom in self._data["identifications"]

    def get_contexte(self, nom: str) -> str:
        if nom not in self._data["identifications"]:
            return ""
        e = self._data["identifications"][nom]
        secs = ", ".join(e.get("sections_vues", []))
        return (f"{nom} — {e['nb_total']} détection(s) — Sections: [{secs}] — "
                f"Confiance max: {int(e.get('confiance_max', 0) * 100)}%")

    def get_resume(self) -> str:
        ids = self._data["identifications"]
        if not ids:
            return "Aucune identification partagée."
        lines = []
        for nom, e in sorted(ids.items(), key=lambda x: x[1].get("nb_total", 0), reverse=True):
            secs = ", ".join(e.get("sections_vues", []))
            ts = e.get("derniere_detection", "")[:16]
            lines.append(
                f"• {nom}  |  {e['nb_total']}×  |  [{secs}]  |  "
                f"{int(e.get('confiance_max', 0) * 100)}%  |  {ts}"
            )
        return "\n".join(lines[:30])

    def get_noms(self) -> list:
        return list(self._data["identifications"].keys())

    def effacer(self, nom: str) -> bool:
        if nom in self._data["identifications"]:
            del self._data["identifications"][nom]
            self._save()
            return True
        return False

    def tout_effacer(self):
        self._data["identifications"] = {}
        self._save()


shared_memory = SharedIdentificationMemory()


# ══ INSIGHTFACE ══

_insight_app = None

def _get_insight_app():
    global _insight_app
    if _insight_app is not None:
        return _insight_app
    if not HAS_INSIGHTFACE:
        return None
    try:
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"],
                           allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        _insight_app = app
        return _insight_app
    except Exception as e:
        print(f"[InsightFace] {e}")
        return None


# ══ CASCADES OPENCV MULTI-ANGLE ══

_face_cascade = None
_profile_cascade = None

def _get_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    if not HAS_CV2:
        return None
    for p in [cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
              "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"]:
        if os.path.exists(p):
            _face_cascade = cv2.CascadeClassifier(p)
            return _face_cascade
    try:
        import urllib.request
        dest = "/tmp/haar_frontal.xml"
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            dest)
        _face_cascade = cv2.CascadeClassifier(dest)
    except Exception:
        pass
    return _face_cascade

def _get_profile_cascade():
    global _profile_cascade
    if _profile_cascade is not None:
        return _profile_cascade
    if not HAS_CV2:
        return None
    for p in [cv2.data.haarcascades + "haarcascade_profileface.xml",
              "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml"]:
        if os.path.exists(p):
            _profile_cascade = cv2.CascadeClassifier(p)
            return _profile_cascade
    return None


# ══ BASE BIOMÉTRIQUE ══

class FacesDatabase:
    """Base biométrique KHEDIM IA — InsightFace 512-dim multi-encodages."""

    def __init__(self):
        self.db = self._load()

    def _load(self):
        FACES_DB_PATH.parent.mkdir(exist_ok=True)
        if FACES_DB_PATH.exists():
            try:
                with open(FACES_DB_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for nom, info in data.get("personnes", {}).items():
                    if "encoding" in info and isinstance(info["encoding"], list):
                        info["encoding"] = np.array(info["encoding"], dtype=np.float32)
                    if "encodings_multiples" in info:
                        info["encodings_multiples"] = [
                            np.array(e, dtype=np.float32) if isinstance(e, list) else e
                            for e in info["encodings_multiples"]
                        ]
                return data
            except Exception:
                pass
        return {"personnes": {}, "total_detections": 0, "version": "KHEDIM-8.0"}

    def _save(self):
        FACES_DB_PATH.parent.mkdir(exist_ok=True)
        ser = {"personnes": {}, "total_detections": self.db.get("total_detections", 0), "version": "KHEDIM-8.0"}
        for nom, info in self.db["personnes"].items():
            entry = dict(info)
            if "encoding" in entry and isinstance(entry["encoding"], np.ndarray):
                entry["encoding"] = entry["encoding"].tolist()
            if "encodings_multiples" in entry:
                entry["encodings_multiples"] = [
                    e.tolist() if isinstance(e, np.ndarray) else e for e in entry["encodings_multiples"]
                ]
            ser["personnes"][nom] = entry
        with open(FACES_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(ser, f, ensure_ascii=False, indent=2)

    def add_person(self, nom: str, encoding=None, grade: str = "", unite: str = "", notes: str = "") -> bool:
        existing = self.db["personnes"].get(nom, {})
        enc_list = list(existing.get("encodings_multiples", []))
        if encoding is not None:
            enc_list.append(encoding)
        self.db["personnes"][nom] = {
            "nom": nom,
            "grade": grade,
            "unite": unite,
            "notes": notes,
            "date_enregistrement": existing.get("date_enregistrement", datetime.now().isoformat()),
            "nb_detections": existing.get("nb_detections", 0),
            "encoding": enc_list[0] if enc_list else None,
            "encodings_multiples": enc_list,
            "a_encodage_reel": len(enc_list) > 0,
            "nb_angles": len(enc_list),
        }
        self._save()
        return True

    def update_info(self, nom: str, grade: str = "", unite: str = "", notes: str = "") -> bool:
        if nom not in self.db["personnes"]:
            return False
        if grade:
            self.db["personnes"][nom]["grade"] = grade
        if unite:
            self.db["personnes"][nom]["unite"] = unite
        if notes:
            self.db["personnes"][nom]["notes"] = notes
        self._save()
        return True

    def delete_person(self, nom: str) -> bool:
        if nom in self.db["personnes"]:
            del self.db["personnes"][nom]
            self._save()
            return True
        return False

    def get_all_names(self) -> list:
        return sorted(self.db["personnes"].keys())

    def get_encodings(self) -> list:
        result = []
        for nom, info in self.db["personnes"].items():
            multi = info.get("encodings_multiples", [])
            if multi:
                for enc_raw in multi:
                    enc = np.array(enc_raw, dtype=np.float32) if isinstance(enc_raw, list) else enc_raw
                    if isinstance(enc, np.ndarray) and enc.size > 0:
                        result.append((nom, enc))
            else:
                enc = info.get("encoding")
                if enc is not None:
                    if isinstance(enc, list):
                        enc = np.array(enc, dtype=np.float32)
                    if isinstance(enc, np.ndarray) and enc.size > 0:
                        result.append((nom, enc))
        return result

    def increment_detection(self, nom: str):
        if nom in self.db["personnes"]:
            self.db["personnes"][nom]["nb_detections"] = self.db["personnes"][nom].get("nb_detections", 0) + 1
            self.db["total_detections"] = self.db.get("total_detections", 0) + 1
            self.db["personnes"][nom]["derniere_detection"] = datetime.now().isoformat()
            self._save()

    def get_person_card(self, nom: str) -> str:
        if nom not in self.db["personnes"]:
            return f"'{nom}' non trouvé."
        info = self.db["personnes"][nom]
        grade = info.get("grade", "—")
        unite = info.get("unite", "—")
        notes = info.get("notes", "—")
        nb = info.get("nb_detections", 0)
        enc = "✅ Biométrique" if info.get("a_encodage_reel") else "⚠️ Nom seul"
        angles = info.get("nb_angles", 0)
        ts = info.get("derniere_detection", "jamais")[:16]
        return (f"┌─ FICHE PERSONNEL ─────────────\n"
                f"│ Nom    : {nom}\n"
                f"│ Grade  : {grade}\n"
                f"│ Unité  : {unite}\n"
                f"│ Notes  : {notes}\n"
                f"│ Déts   : {nb}× | Dernière: {ts}\n"
                f"│ Encodage: {enc} ({angles} angle(s))\n"
                f"└────────────────────────────────")


faces_db = FacesDatabase()


# ══ SESSION MEMORY ══

class SessionMemory:
    def __init__(self):
        self.data = self._load()

    def _load(self):
        SESSION_MEM_PATH.parent.mkdir(exist_ok=True)
        if SESSION_MEM_PATH.exists():
            try:
                with open(SESSION_MEM_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"visages_recents": [], "alertes": [], "resume": ""}

    def _save(self):
        SESSION_MEM_PATH.parent.mkdir(exist_ok=True)
        with open(SESSION_MEM_PATH, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def log_detection(self, noms: list, nb_inconnus: int):
        self.data["visages_recents"].append({
            "timestamp": datetime.now().isoformat(), "connus": noms, "nb_inconnus": nb_inconnus})
        self.data["visages_recents"] = self.data["visages_recents"][-50:]
        if nb_inconnus > 0:
            self.data["alertes"].append({
                "timestamp": datetime.now().isoformat(),
                "message": f"⚠️ {nb_inconnus} inconnu(s) détecté(s)"})
            self.data["alertes"] = self.data["alertes"][-30:]
        self._save()

    def get_context_text(self) -> str:
        recent = self.data["visages_recents"][-5:]
        if not recent:
            return "Aucune détection récente."
        return "\n".join(
            f"[{e['timestamp'][:16]}] Connus: {', '.join(e['connus']) or '—'} | Inconnus: {e['nb_inconnus']}"
            for e in recent)

    def get_recent_alerts(self, n=10) -> list:
        return self.data["alertes"][-n:]


session_memory = SessionMemory()


# ══ JOURNAL ══

def _log_detection(personnes: list):
    DETECTION_LOG.parent.mkdir(exist_ok=True)
    log = []
    if DETECTION_LOG.exists():
        try:
            with open(DETECTION_LOG, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append({"timestamp": datetime.now().isoformat(),
                 "personnes": [{"nom": p["nom"], "connu": p["connu"],
                                "confiance": round(p.get("confiance", 0), 3)} for p in personnes]})
    log = log[-300:]
    with open(DETECTION_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def get_detection_log(n=10) -> list:
    if not DETECTION_LOG.exists():
        return []
    try:
        with open(DETECTION_LOG, "r", encoding="utf-8") as f:
            return json.load(f)[-n:]
    except Exception:
        return []


# ══ COMPARAISON ══

SIMILARITY_THRESHOLD = 0.45

def _compare_encoding(enc: np.ndarray, known_list: list) -> tuple:
    if not known_list:
        return None, 0.0
    best_name, best_dist = None, float("inf")
    for nom, known_enc in known_list:
        if HAS_SCIPY:
            dist = float(cosine_distance(enc, known_enc))
        else:
            norm = float(np.linalg.norm(enc) * np.linalg.norm(known_enc))
            dist = 1.0 - float(np.dot(enc, known_enc)) / norm if norm > 0 else 1.0
        if dist < best_dist:
            best_dist, best_name = dist, nom
    if best_dist < SIMILARITY_THRESHOLD:
        return best_name, round(max(0.0, 1.0 - best_dist / SIMILARITY_THRESHOLD), 2)
    return None, 0.0


# ══ PRÉTRAITEMENT ══

def _preprocess(img: np.ndarray) -> np.ndarray:
    if not HAS_CV2:
        return img
    h, w = img.shape[:2]
    if w < 320 or h < 240:
        scale = max(320 / w, 240 / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    try:
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(yuv[:, :, 0])
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    except Exception:
        pass
    return img

def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if HAS_CV2 else img[:, :, ::-1]


def _detect_fallback(img: np.ndarray) -> list:
    results = []
    c = _get_cascade()
    if not c or not HAS_CV2:
        return results
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    for sf, mn in [(1.05, 3), (1.08, 2)]:
        f = c.detectMultiScale(gray, sf, mn, minSize=(25, 25))
        if len(f) > 0:
            results.extend(f.tolist())
            break
    pc = _get_profile_cascade()
    if pc:
        pf = pc.detectMultiScale(gray, 1.05, 3, minSize=(25, 25))
        if len(pf) > 0:
            results.extend(pf.tolist())
        gf = cv2.flip(gray, 1)
        pf2 = pc.detectMultiScale(gf, 1.05, 3, minSize=(25, 25))
        if len(pf2) > 0:
            w_img = gray.shape[1]
            results.extend([[w_img - x - fw, y, fw, fh] for (x, y, fw, fh) in pf2])
    return results


# ══ ANNOTATION KHEDIM IA ══

def _annotate(img: np.ndarray, personnes: list) -> np.ndarray:
    if not HAS_CV2:
        return img
    out = img.copy()
    for p in personnes:
        pos = p.get("position")
        if pos is None:
            continue
        x, y, w, h = int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])
        conf = p.get("confiance", 0)
        if p["connu"]:
            color = (60, 210, 90) if conf > 0.65 else (200, 165, 25)
        else:
            color = (215, 50, 50)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cl = min(w, h) // 5
        for cx, cy, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(out, (cx, cy), (cx + dx * cl, cy), color, 3)
            cv2.line(out, (cx, cy), (cx, cy + dy * cl), color, 3)
        nom = p["nom"]
        info = faces_db.db["personnes"].get(nom, {})
        grade = info.get("grade", "")
        label = f"{grade} {nom}".strip() if grade else nom
        if p["connu"] and conf > 0:
            label += f" {int(conf * 100)}%"
        if p["connu"] and shared_memory.est_connu(nom):
            label += " ✓"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(out, label, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
    return out


# ══ API PUBLIQUE ══

def analyze_frame(frame, section: str = "Caméra") -> dict:
    ts = datetime.now().isoformat()
    if frame is None:
        return {"nb_visages": 0, "personnes": [], "message": "Aucune image.", "image_annotee": None, "timestamp": ts, "mode": "N/A"}
    if HAS_PIL and isinstance(frame, Image.Image):
        frame = np.array(frame)
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return {"nb_visages": 0, "personnes": [], "message": "Format invalide.", "image_annotee": None, "timestamp": ts, "mode": "N/A"}

    frame = _preprocess(frame)
    known_list = faces_db.get_encodings()
    app = _get_insight_app()
    personnes = []

    if app is not None:
        try:
            bgr = _to_bgr(frame)
            faces = app.get(bgr)
            if len(faces) == 0 and HAS_CV2:
                h, w = frame.shape[:2]
                big = cv2.resize(bgr, (w * 2, h * 2))
                faces2 = app.get(big)
                if len(faces2) > 0:
                    for f2 in faces2:
                        f2.bbox = f2.bbox / 2.0
                    faces = faces2
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                pos = (x1, y1, x2 - x1, y2 - y1)
                enc = face.normed_embedding if hasattr(face, "normed_embedding") else None
                nom_trouve, confiance = None, 0.0
                if enc is not None and known_list:
                    nom_trouve, confiance = _compare_encoding(enc, known_list)
                if nom_trouve:
                    faces_db.increment_detection(nom_trouve)
                    shared_memory.enregistrer(nom_trouve, section, confiance)
                    personnes.append({"nom": nom_trouve, "connu": True, "confiance": confiance, "position": pos})
                else:
                    personnes.append({"nom": "Inconnu", "connu": False, "confiance": 0.0, "position": pos})
        except Exception as e:
            print(f"[face_engine] {e}")
            for (x, y, w, h) in _detect_fallback(frame):
                personnes.append({"nom": "Inconnu", "connu": False, "confiance": 0.0, "position": (x, y, w, h)})
        mode = "InsightFace 512-dim"
    else:
        for (x, y, w, h) in _detect_fallback(frame):
            personnes.append({"nom": "Inconnu", "connu": False, "confiance": 0.0, "position": (x, y, w, h)})
        mode = "OpenCV Haar Multi-angle"

    annotated = _annotate(frame, personnes)
    connus = [p["nom"] for p in personnes if p["connu"]]
    nb_inc = sum(1 for p in personnes if not p["connu"])
    if personnes:
        session_memory.log_detection(connus, nb_inc)
        _log_detection(personnes)

    nb = len(personnes)
    if nb == 0:
        msg = "Aucun visage détecté."
    else:
        parts = []
        if connus:
            parts.append(f"✅ {', '.join(connus)}")
        if nb_inc > 0:
            parts.append(f"⚠️ Inconnus: {nb_inc}")
        msg = f"{nb} visage(s) — " + " | ".join(parts)

    return {"nb_visages": nb, "personnes": personnes, "message": msg,
            "image_annotee": annotated, "timestamp": ts, "mode": mode}


def register_face(image, nom: str, grade: str = "", unite: str = "", notes: str = "", section: str = "Enregistrement") -> dict:
    if image is None:
        return {"success": False, "message": "❌ Image requise."}
    if not nom.strip():
        return {"success": False, "message": "❌ Nom requis."}
    if HAS_PIL and isinstance(image, Image.Image):
        image = np.array(image)
    if not isinstance(image, np.ndarray) or image.size == 0:
        return {"success": False, "message": "❌ Image invalide."}
    image = _preprocess(image)
    app = _get_insight_app()
    encoding = None
    if app is not None:
        try:
            bgr = _to_bgr(image)
            faces = app.get(bgr)
            if len(faces) == 0:
                h, w = image.shape[:2]
                big = cv2.resize(image, (w * 2, h * 2)) if HAS_CV2 else image
                faces = app.get(_to_bgr(big))
            if len(faces) == 0:
                return {"success": False, "message": "❌ Aucun visage détecté. Photo plus claire SVP."}
            if len(faces) > 1:
                return {"success": False, "message": f"❌ {len(faces)} visages — 1 seule personne par photo SVP."}
            encoding = faces[0].normed_embedding
        except Exception as e:
            return {"success": False, "message": f"❌ Erreur InsightFace: {e}"}
    else:
        if not _detect_fallback(image):
            return {"success": False, "message": "❌ Aucun visage détecté (mode léger)."}

    existing = nom.strip() in faces_db.db["personnes"]
    faces_db.add_person(nom.strip(), encoding, grade=grade, unite=unite, notes=notes)
    shared_memory.enregistrer(nom.strip(), section, 1.0)

    action = "mis à jour" if existing else "enregistré"
    note = "✅ Encodage biométrique 512-dim." if encoding else "⚠️ Mode léger — sans encodage biométrique."
    nb_angles = faces_db.db["personnes"][nom.strip()].get("nb_angles", 1)
    return {
        "success": True,
        "message": (f"✅ '{nom}' {action} avec succès.\n"
                    f"{note}\n"
                    f"📐 {nb_angles} angle(s) enregistré(s)\n"
                    f"🔗 Mémoire partagée inter-sections activée.")
    }


def numpy_to_pil(image_np: np.ndarray):
    if not HAS_PIL:
        return image_np
    return Image.fromarray(image_np.astype(np.uint8))


def get_system_info() -> dict:
    app = _get_insight_app()
    return {
        "opencv": HAS_CV2,
        "insightface": HAS_INSIGHTFACE,
        "insightface_actif": app is not None,
        "face_recognition": False,
        "mode": "InsightFace 512-dim" if app else "OpenCV Haar Multi-angle",
        "faces_enregistres": len(faces_db.get_all_names()),
        "faces_avec_encodage": sum(1 for _, info in faces_db.db["personnes"].items() if info.get("a_encodage_reel")),
        "detection_log_count": len(get_detection_log(300)),
        "identifications_partagees": len(shared_memory.get_noms()),
    }
