"""
PRAY PLUS+ - Module Accessibilite Intelligente
Invente par Khedim Benyakhlef (Beny-Joe)
Integre dans KHEDIM IA v8.0
"""
import os, cv2, numpy as np, tempfile, time

# ═══ 1. OCR ═══
def extraire_texte(img_rgb):
    try:
        import easyocr
        reader = easyocr.Reader(["fr","ar","en"], gpu=False, verbose=False)
        res = reader.readtext(img_rgb, detail=0, paragraph=True)
        t = " ".join(res).strip()
        return t if t else ""
    except: pass
    try:
        import pytesseract
        from PIL import Image
        t = pytesseract.image_to_string(Image.fromarray(img_rgb), config="--oem 3 --psm 6 -l fra+ara+eng").strip()
        return t if t else ""
    except: pass
    return ""

def ocr_audio(img_rgb, lang="fr"):
    texte = extraire_texte(img_rgb)
    if not texte or len(texte) < 3:
        return {"texte": "Aucun texte detecte", "audio": None, "ok": False}
    audio = tts(texte, lang)
    return {"texte": texte, "audio": audio, "ok": True}

# ═══ 2. DESCRIPTION IA ═══
def decrire_scene(img_rgb, question=None, lang="fr"):
    import base64, requests
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode()
    prompt = question if question else f"Decris cette scene en {lang} pour un malvoyant : objets, textes, personnes, obstacles."
    for k in [os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")]:
        if not k: continue
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json"},
                json={"model": "meta-llama/llama-4-scout-17b-16e-instruct",
                      "messages": [{"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                          {"type": "text", "text": prompt}]}],
                      "max_tokens": 400}, timeout=15)
            if r.status_code == 200:
                desc = r.json()["choices"][0]["message"]["content"]
                return {"description": desc, "audio": tts(desc, lang), "ok": True}
        except: continue
    return {"description": "Impossible analyser", "audio": None, "ok": False}

# ═══ 3. DETECTION OBSTACLES ═══
_LAST_ALERT = [0]

def detecter_obstacles(img_rgb, seuil=0.30):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    zone = bgr[int(h*0.3):int(h*0.9), int(w*0.2):int(w*0.8)]
    g = cv2.GaussianBlur(cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY), (5,5), 0)
    edges = cv2.Canny(g, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zh, zw = zone.shape[:2]
    obs = []
    for c in cnts:
        if cv2.contourArea(c) / (zh*zw) > seuil:
            x,y,cw,ch = cv2.boundingRect(c)
            pos = "gauche" if x < zw*0.33 else ("droite" if x > zw*0.66 else "centre")
            obs.append(pos)
    if obs:
        msg = f"ATTENTION obstacle {list(set(obs))[0]}"
        return {"obstacle": True, "message": msg}
    return {"obstacle": False, "message": "Voie libre"}

def alerte_obstacle(img_rgb):
    if time.time() - _LAST_ALERT[0] < 3.0:
        return {"alerte": False}
    r = detecter_obstacles(img_rgb)
    if r["obstacle"]:
        _LAST_ALERT[0] = time.time()
        return {"alerte": True, "message": r["message"], "audio": tts(r["message"], "fr")}
    return {"alerte": False}

# ═══ 4. RECONNAISSANCE PROCHES ═══
_PROCHES = {}
_APP = [None]

def _emb(img_rgb):
    try:
        from insightface.app import FaceAnalysis
        if _APP[0] is None:
            app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(160,160))
            _APP[0] = app
        s = cv2.resize(img_rgb, (0,0), fx=0.5, fy=0.5)
        f = _APP[0].get(s)
        if f: return f[0].embedding
    except: pass
    try:
        import face_recognition as fr
        s = cv2.resize(img_rgb, (0,0), fx=0.5, fy=0.5)
        l = fr.face_locations(s, model="hog")
        e = fr.face_encodings(s, l)
        if e: return e[0]
    except: pass
    return None

def _sim(a, b):
    try:
        a,b = np.array(a,dtype=float), np.array(b,dtype=float)
        n1,n2 = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a,b)/(n1*n2)) if n1>0 and n2>0 else 0.0
    except: return 0.0

def enregistrer_proche(img_rgb, nom):
    e = _emb(img_rgb)
    if e is None: return {"ok": False, "msg": f"Aucun visage pour {nom}"}
    if nom not in _PROCHES: _PROCHES[nom] = []
    _PROCHES[nom].append(e)
    return {"ok": True, "msg": f"{nom} enregistre ({len(_PROCHES[nom])} echantillons)"}

def reconnaitre_proche(img_rgb):
    if not _PROCHES: return {"reconnu": False, "msg": "Base vide"}
    e = _emb(img_rgb)
    if e is None: return {"reconnu": False, "msg": "Aucun visage"}
    best_nom, best_sim = None, 0.0
    for nom, embs in _PROCHES.items():
        for em in embs:
            s = _sim(e, em)
            if s > best_sim: best_sim, best_nom = s, nom
    if best_sim >= 0.50:
        msg = f"Proche reconnu : {best_nom} {int(best_sim*100)} pourcent"
        return {"reconnu": True, "nom": best_nom, "conf": int(best_sim*100), "msg": msg, "audio": tts(msg,"fr")}
    return {"reconnu": False, "msg": "Personne inconnue"}

def lister_proches():
    return [{"nom": n, "echantillons": len(e)} for n,e in _PROCHES.items()]

# ═══ TTS ═══
def tts(texte, lang="fr"):
    if not texte or len(texte.strip()) < 2: return None
    try:
        import asyncio, edge_tts
        voix = {"fr":"fr-FR-DeniseNeural","ar":"ar-DZ-AminaNeural","en":"en-US-JennyNeural"}.get(lang,"fr-FR-DeniseNeural")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        async def _s():
            await edge_tts.Communicate(texte[:500], voix).save(tmp.name)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_s())
        loop.close()
        if os.path.getsize(tmp.name) > 100: return tmp.name
    except: pass
    try:
        from gtts import gTTS
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        gTTS(text=texte[:500], lang=lang, slow=False).save(tmp.name)
        return tmp.name
    except: pass
    return None

# ═══ ANALYSE COMPLETE ═══
def analyser(img_rgb, mode="complet", lang="fr"):
    res = {}
    if mode in ("ocr","complet"): res["ocr"] = ocr_audio(img_rgb, lang)
    if mode in ("scene","complet"): res["scene"] = decrire_scene(img_rgb, lang=lang)
    if mode in ("obstacle","complet"): res["obstacle"] = alerte_obstacle(img_rgb)
    if mode in ("proche","complet"): res["proche"] = reconnaitre_proche(img_rgb)
    return res
