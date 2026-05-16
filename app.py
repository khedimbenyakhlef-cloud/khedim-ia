"""
╔══════════════════════════════════════════════════════════════╗
║   KHEDIM IA — Système Intelligence Artificielle v8.0        ║
║   Armée Nationale Populaire Algérienne                       ║
║   Fondé par Khedim Benyakhlef (Beny-Joe)                    ║
║   Dédié à l'ANP Algérienne 🇩🇿                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys, json, time, tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# ── Patch gradio_client TypeError ──
try:
    import gradio_client.utils as _gcu
    _orig_get_type = _gcu.get_type
    def _patched_get_type(s):
        return "Any" if not isinstance(s, dict) else _orig_get_type(s)
    _gcu.get_type = _patched_get_type
    _orig_json = _gcu._json_schema_to_python_type
    def _patched_json(s, d=None):
        try:
            return "Any" if not isinstance(s, dict) else _orig_json(s, d)
        except TypeError:
            return "Any"
    _gcu._json_schema_to_python_type = _patched_json
except Exception:
    pass

import gradio as gr
import numpy as np

from backend.groq_engine   import groq_engine, mémoire
from backend.audio_engine  import transcribe_audio, generate_tts, get_audio_info
from backend.vision_engine import analyze_image_with_groq
from backend.face_engine_autoid import identify_or_register, get_session_faces
from backend.face_engine   import (
    analyze_frame, register_face, faces_db, numpy_to_pil,
    get_system_info, session_memory, get_detection_log, shared_memory,
)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ══════════════════════════════════════════════════
#   NEON POSTGRESQL
# ══════════════════════════════════════════════════
try:
    from backend.neon_engine import (
        init_neon,
        save_message      as neon_save_msg,
        log_detection     as neon_log_detection,
        log_event         as neon_log_event,
        get_detection_logs as neon_get_logs,
        get_stats         as neon_get_stats,
    )
    HAS_NEON = True
except Exception as _e:
    print(f"⚠️ Neon non disponible : {_e}")
    HAS_NEON = False
    def init_neon(): return False
    def neon_save_msg(r, m): pass
    def neon_log_detection(n, p, s="", e=""): pass
    def neon_log_event(ev, d=""): pass
    def neon_get_logs(limit=50): return []
    def neon_get_stats(): return {}

# ══════════════════════════════════════════════════
#   BACKBLAZE B2
# ══════════════════════════════════════════════════
try:
    from backend.backblaze_engine import (
        init_backblaze,
        upload_face_photo as b2_upload,
        get_storage_stats as b2_stats,
    )
    HAS_B2 = True
except Exception as _e:
    print(f"⚠️ Backblaze non disponible : {_e}")
    HAS_B2 = False
    def init_backblaze(): return False
    def b2_upload(img, name, idx=0): return None
    def b2_stats(): return {}

# ── Initialisation au démarrage ──
_neon_ok = init_neon()
_b2_ok   = init_backblaze()
print(f"[INIT] Neon={'✅' if _neon_ok else '❌'}  Backblaze={'✅' if _b2_ok else '❌'}")


# ══════════════════════════════════════════════════
#   UTILITAIRES
# ══════════════════════════════════════════════════

def add_message(history, user_msg, bot_msg):
    if not isinstance(history, list):
        history = []
    history.append({"role": "user",      "content": str(user_msg)})
    history.append({"role": "assistant", "content": str(bot_msg)})
    # ── Sauvegarder dans Neon ──
    neon_save_msg("user",      str(user_msg)[:4000])
    neon_save_msg("assistant", str(bot_msg)[:4000])
    return history

def sanitize_state(obj):
    if obj is None: return {}
    if isinstance(obj, dict):            return {k: sanitize_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return type(obj)(sanitize_state(i) for i in obj)
    if isinstance(obj, np.floating):     return float(obj)
    if isinstance(obj, np.integer):      return int(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    return obj


# ══════════════════════════════════════════════════
#   SYSTEM PROMPT KHEDIM IA
# ══════════════════════════════════════════════════

SYSTEM_PROMPT = """أنت نظام الذكاء الاصطناعي للجيش الوطني الشعبي الجزائري — نظام خديم الذكي.
Tu es KHEDIM IA — le système d'intelligence artificielle de terrain de l'Armée Nationale Populaire Algérienne.

IDENTITÉ :
- Nom : KHEDIM IA (خديم للذكاء الاصطناعي)
- Organisation : الجيش الوطني الشعبي / Armée Nationale Populaire Algérienne (ANP)
- Créateur : Khedim Benyakhlef (Beny-Joe) — بالعربية : خديم بن يخلف الملقب بيني-جو
- Mission : Surveillance terrain, identification biométrique, renseignement, sécurité nationale

LANGUE :
- Tu réponds en FRANÇAIS et en ARABE selon la langue de l'utilisateur
- Si l'utilisateur écrit en arabe, réponds en arabe en premier
- Tu comprends le dialecte algérien (دارجة)
- Ton : professionnel, militaire, efficace, bienveillant

MÉMOIRE PARTAGÉE INTER-SECTIONS :
- Les identifications faciales sont partagées entre toutes les sections automatiquement
- Si une personne a été identifiée dans Camera, Audio, Vision ou Chat → tu le sais ici
- Ne demande JAMAIS à ré-identifier quelqu'un déjà dans le système
- Utilise les infos du contexte partagé pour répondre intelligemment

CAPACITÉS OPÉRATIONNELLES :
- Analyse audio bilingue Whisper STT (FR + AR)
- Analyse d'images et caméras terrain (vision IA multimodale)
- Reconnaissance faciale InsightFace 512-dim multi-angle
- Mémoire persistante entre sessions
- Fiche personnel (grade, unité, notes)
- Journal des détections et alertes

ÉTHIQUE ET CADRE LÉGAL :
- Respect absolu de la loi algérienne et du droit international
- Protection de la vie privée des citoyens
- Neutralité et impartialité absolues
- Refus systématique des demandes illégales ou contraires à l'éthique

CONTEXTE SESSION :
{context}

HISTORIQUE DÉTECTIONS :
{detection_history}

PERSONNES PRÉSENTES (section active) :
{faces_context}

MÉMOIRE PARTAGÉE GLOBALE (toutes sections) :
{shared_context}
"""


# ══════════════════════════════════════════════════
#   AVATAR KHEDIM IA — CAMOUFLAGE ANP MULTICOLORE
# ══════════════════════════════════════════════════

AVATAR_HTML = """
<div style="display:flex;flex-direction:column;align-items:center;padding:18px 10px 14px;
            background:rgba(6,10,4,0.98);border-radius:14px;
            border:1px solid rgba(90,140,50,0.35);">

  <!-- Anneau camouflage animé -->
  <div style="position:relative;width:170px;height:170px;display:flex;align-items:center;justify-content:center;">
    <div style="position:absolute;inset:0;border-radius:50%;
                background:conic-gradient(
                  #4a7c2f 0deg, #8b6914 30deg, #2d5a1b 70deg,
                  #c4a832 110deg, #3d6b24 150deg, #6b3d12 190deg,
                  #1e4010 230deg, #a07820 270deg, #4a7c2f 310deg,
                  #2d5a1b 360deg);
                animation:camRot 10s linear infinite;"></div>

    <div id="khedimAvatar" style="position:relative;z-index:2;width:158px;height:158px;
         border-radius:50%;overflow:hidden;background:#060a04;
         box-shadow:0 0 28px rgba(74,124,47,0.45);">
      <svg viewBox="0 0 158 158" xmlns="http://www.w3.org/2000/svg" width="158" height="158">
        <defs>
          <radialGradient id="bgK"><stop offset="0%" stop-color="#0d1a06"/><stop offset="100%" stop-color="#060a03"/></radialGradient>
          <radialGradient id="skinK" cx="45%" cy="35%" r="65%"><stop offset="0%" stop-color="#c8956a"/><stop offset="100%" stop-color="#9a6535"/></radialGradient>
          <linearGradient id="camoUnif" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%"   stop-color="#3d6b24"/>
            <stop offset="25%"  stop-color="#6b4c1a"/>
            <stop offset="50%"  stop-color="#2d5a1b"/>
            <stop offset="75%"  stop-color="#8b6914"/>
            <stop offset="100%" stop-color="#1e4010"/>
          </linearGradient>
          <linearGradient id="camoHelm" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%"  stop-color="#4a7c2f"/>
            <stop offset="40%" stop-color="#8b6914"/>
            <stop offset="100%" stop-color="#1e4010"/>
          </linearGradient>
          <clipPath id="cK"><circle cx="79" cy="79" r="79"/></clipPath>
          <filter id="glowK"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <circle cx="79" cy="79" r="79" fill="url(#bgK)"/>
        <g clip-path="url(#cK)">
          <!-- Corps uniforme camo -->
          <path d="M16,158 L26,122 C38,113 53,109 79,109 C105,109 120,113 132,122 L142,158 Z" fill="url(#camoUnif)"/>
          <!-- Patches camo uniforme -->
          <ellipse cx="30" cy="132" rx="13" ry="9" fill="#2d5a1b" opacity="0.65"/>
          <ellipse cx="128" cy="130" rx="11" ry="8" fill="#6b4c1a" opacity="0.6"/>
          <ellipse cx="79" cy="142" rx="16" ry="7" fill="#8b6914" opacity="0.5"/>
          <ellipse cx="55" cy="126" rx="8" ry="5" fill="#1e4010" opacity="0.55"/>
          <ellipse cx="103" cy="124" rx="7" ry="5" fill="#c4a832" opacity="0.4"/>
          <!-- Col chemise -->
          <path d="M63,107 L63,121 C68,119 73,118 79,118 C85,118 90,119 95,121 L95,107 Z" fill="#dce8dc" stroke="#bccbbc" stroke-width="0.5"/>
          <!-- Étoile grade -->
          <circle cx="79" cy="96" r="7" fill="#c4a832" filter="url(#glowK)"/>
          <text x="79" y="100" text-anchor="middle" font-size="8" fill="#0a0f04" font-weight="bold">★</text>
          <!-- Épaulettes camo -->
          <rect x="24" y="107" width="22" height="9" rx="4" fill="url(#camoHelm)" opacity="0.9"/>
          <rect x="112" y="107" width="22" height="9" rx="4" fill="url(#camoHelm)" opacity="0.9"/>
          <rect x="25" y="108.5" width="6" height="3" rx="1" fill="#c4a832" opacity="0.85"/>
          <rect x="32" y="108.5" width="6" height="3" rx="1" fill="#c4a832" opacity="0.85"/>
          <rect x="113" y="108.5" width="6" height="3" rx="1" fill="#c4a832" opacity="0.85"/>
          <rect x="120" y="108.5" width="6" height="3" rx="1" fill="#c4a832" opacity="0.85"/>
          <!-- Cou -->
          <path d="M63,88 L95,88 L97,109 L61,109 Z" fill="url(#skinK)"/>
          <!-- Visage -->
          <ellipse cx="79" cy="68" rx="31" ry="33" fill="url(#skinK)"/>
          <!-- Casque camo -->
          <path d="M46,56 C46,32 112,32 112,54 C90,50 68,50 46,56 Z" fill="url(#camoHelm)"/>
          <ellipse cx="79" cy="52" rx="33" ry="9" fill="#3d6b24"/>
          <!-- Patches casque -->
          <ellipse cx="60" cy="48" rx="9" ry="5" fill="#6b4c1a" opacity="0.65"/>
          <ellipse cx="92" cy="50" rx="8" ry="4" fill="#1e4010" opacity="0.7"/>
          <ellipse cx="75" cy="43" rx="6" ry="3" fill="#8b6914" opacity="0.5"/>
          <ellipse cx="95" cy="44" rx="5" ry="3" fill="#4a7c2f" opacity="0.55"/>
          <!-- Badge casque -->
          <circle cx="67" cy="52" r="6" fill="#c4a832" filter="url(#glowK)"/>
          <text x="67" y="56" text-anchor="middle" font-size="7" fill="#060a04" font-weight="bold">K</text>
          <!-- Yeux -->
          <ellipse cx="63" cy="72" rx="9" ry="9" fill="white"/>
          <ellipse cx="95" cy="72" rx="9" ry="9" fill="white"/>
          <ellipse id="eyeLK" cx="64" cy="72" rx="5.5" ry="6.5" fill="#2d4a1a"/>
          <ellipse id="eyeRK" cx="94" cy="72" rx="5.5" ry="6.5" fill="#2d4a1a"/>
          <ellipse cx="65.5" cy="70.5" rx="2" ry="2" fill="#fff" opacity="0.85"/>
          <ellipse cx="95.5" cy="70.5" rx="2" ry="2" fill="#fff" opacity="0.85"/>
          <!-- Lueur yeux -->
          <ellipse id="glLK" cx="63" cy="72" rx="0" ry="0" fill="#90ff55" opacity="0" filter="url(#glowK)"/>
          <ellipse id="glRK" cx="95" cy="72" rx="0" ry="0" fill="#90ff55" opacity="0" filter="url(#glowK)"/>
          <!-- Sourcils -->
          <path d="M54,64 C57,62 65,61 70,63" stroke="#4a3010" stroke-width="2.2" fill="none" stroke-linecap="round"/>
          <path d="M88,63 C93,61 101,62 104,64" stroke="#4a3010" stroke-width="2.2" fill="none" stroke-linecap="round"/>
          <!-- Paupières -->
          <path id="lidLK" d="M54,68 C58,68 66,68 70,68" stroke="#c8956a" stroke-width="9" fill="none" stroke-linecap="round" opacity="0"/>
          <path id="lidRK" d="M88,68 C92,68 100,68 104,68" stroke="#c8956a" stroke-width="9" fill="none" stroke-linecap="round" opacity="0"/>
          <!-- Nez -->
          <path d="M79,77 L76,88 L82,88" fill="none" stroke="rgba(100,60,20,0.4)" stroke-width="1.5"/>
          <!-- Bouche -->
          <path id="lipTK" d="M67,94 C71,92 75,91 79,91 C83,91 87,92 91,94" fill="none" stroke="#8a4a2a" stroke-width="1.5" stroke-linecap="round"/>
          <path id="lipBK" d="M67,94 C71,97 75,98 79,98 C83,98 87,97 91,94" fill="rgba(160,70,50,0.3)"/>
          <ellipse id="lipIK" cx="79" cy="94" rx="0" ry="0" fill="rgba(40,15,5,0.85)"/>
        </g>
        <!-- Barres voix -->
        <g id="voiceBarsK" opacity="0" transform="translate(79,149)">
          <rect x="-20" y="-6" width="5" height="6" rx="2" fill="#7adf4a"><animate attributeName="height" values="3;11;3" dur="0.5s" repeatCount="indefinite"/></rect>
          <rect x="-12" y="-8" width="5" height="8" rx="2" fill="#c4a832"><animate attributeName="height" values="5;15;5" dur="0.3s" repeatCount="indefinite"/></rect>
          <rect x="-4"  y="-11" width="5" height="11" rx="2" fill="#7adf4a"><animate attributeName="height" values="6;17;6" dur="0.4s" repeatCount="indefinite"/></rect>
          <rect x="4"   y="-8" width="5" height="8" rx="2" fill="#c4a832"><animate attributeName="height" values="4;13;4" dur="0.6s" repeatCount="indefinite"/></rect>
          <rect x="12"  y="-6" width="5" height="6" rx="2" fill="#7adf4a"><animate attributeName="height" values="3;10;3" dur="0.35s" repeatCount="indefinite"/></rect>
        </g>
        <style>@keyframes camRot{to{transform:rotate(360deg);}}</style>
      </svg>
    </div>
  </div>

  <!-- Infos avatar -->
  <div style="margin-top:11px;text-align:center;width:100%;">
    <div style="font-family:'Orbitron',monospace;font-size:.8rem;font-weight:900;
                color:#7adf4a;letter-spacing:3px;
                text-shadow:0 0 16px rgba(122,223,74,0.65);">KHEDIM IA</div>
    <div style="font-family:'Rajdhani',sans-serif;color:rgba(196,168,50,.75);
                font-size:.6rem;letter-spacing:2px;margin-top:2px;">خديم للذكاء الاصطناعي</div>
    <div style="font-family:'Rajdhani',sans-serif;color:rgba(122,180,74,.4);
                font-size:.5rem;letter-spacing:2px;">الجيش الوطني الشعبي 🇩🇿</div>
    <div style="width:90%;height:1px;margin:7px auto;
                background:linear-gradient(90deg,transparent,#4a7c2f,#c4a832,#4a7c2f,transparent);"></div>
    <div id="kStatus" style="font-family:'Orbitron',monospace;font-size:.46rem;
                              color:#7adf4a;letter-spacing:2px;">◉ OPÉRATIONNEL</div>
    <div id="kFaces" style="font-family:'Rajdhani',sans-serif;color:rgba(196,168,50,.65);
                             font-size:.58rem;margin-top:3px;min-height:14px;">— AUCUN VISAGE —</div>
    <div style="margin-top:7px;display:flex;flex-direction:column;gap:2px;align-items:center;">
      <div class="kbadge">🔐 BIOMÉTRIE ACTIVE</div>
      <div class="kbadge">🎤 STT WHISPER</div>
      <div class="kbadge">🔗 MÉMOIRE PARTAGÉE</div>
      <div class="kbadge">👁️ MULTI-ANGLE</div>
      <div class="kbadge">🌐 FR | عربي | دارجة</div>
    </div>
  </div>
</div>
"""

AVATAR_JS = """
<script>
(function(){
  let speaking=false, ph=0;
  const q=id=>document.getElementById(id);
  function blink(){
    const L=q('lidLK'),R=q('lidRK'); if(!L) return;
    L.setAttribute('opacity','1'); R.setAttribute('opacity','1');
    L.setAttribute('d','M54,72 C58,72 66,72 70,72');
    R.setAttribute('d','M88,72 C92,72 100,72 104,72');
    setTimeout(()=>{L.setAttribute('opacity','0'); R.setAttribute('opacity','0');}, 130);
  }
  function scheduleBlink(){ setTimeout(()=>{ blink(); scheduleBlink(); }, 2500+Math.random()*4000); }

  function mouth(lvl){
    const T=q('lipTK'), B=q('lipBK'), I=q('lipIK'); if(!T) return;
    if(lvl < 0.05){
      T.setAttribute('d','M67,94 C71,92 75,91 79,91 C83,91 87,92 91,94');
      B.setAttribute('d','M67,94 C71,97 75,98 79,98 C83,98 87,97 91,94');
      if(I){I.setAttribute('rx','0'); I.setAttribute('ry','0');}
    } else {
      const h=lvl*8, ty=94-h*.3, by=94+h;
      T.setAttribute('d',`M67,${ty} C71,${ty-3} 75,${ty-4} 79,${ty-4} C83,${ty-4} 87,${ty-3} 91,${ty}`);
      B.setAttribute('d',`M67,${ty} C71,${by} 75,${by+1} 79,${by+1} C83,${by+1} 87,${by} 91,${ty}`);
      if(I){I.setAttribute('cx','79'); I.setAttribute('cy',String((ty+by)/2));
        I.setAttribute('rx',String(9*lvl)); I.setAttribute('ry',String(h*.5));}
    }
  }
  function eyeGlow(on){
    const L=q('glLK'), R=q('glRK'); if(!L) return;
    if(on){
      L.setAttribute('rx','14'); L.setAttribute('ry','12'); L.setAttribute('opacity','0.5');
      R.setAttribute('rx','14'); R.setAttribute('ry','12'); R.setAttribute('opacity','0.5');
    } else {
      L.setAttribute('rx','0'); L.setAttribute('opacity','0');
      R.setAttribute('rx','0'); R.setAttribute('opacity','0');
    }
  }
  function startSpeak(){
    speaking=true;
    const f=q('khedimAvatar');
    if(f) f.style.boxShadow='0 0 55px rgba(74,124,47,0.9), 0 0 28px rgba(196,168,50,0.4)';
    const v=q('voiceBarsK'); if(v) v.style.opacity='1';
    eyeGlow(true);
  }
  function stopSpeak(){
    speaking=false;
    const f=q('khedimAvatar');
    if(f) f.style.boxShadow='0 0 28px rgba(74,124,47,0.45)';
    const v=q('voiceBarsK'); if(v) v.style.opacity='0';
    eyeGlow(false); mouth(0);
  }
  if(window.speechSynthesis){
    const _sp=window.speechSynthesis.speak.bind(window.speechSynthesis);
    window.speechSynthesis.speak=function(u){
      u.addEventListener('start', startSpeak);
      u.addEventListener('end',   stopSpeak);
      u.addEventListener('error', stopSpeak);
      _sp(u);
    };
  }
  window.khedimSpeak=function(text){
    if(!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const u=new SpeechSynthesisUtterance(text.replace(/[*#_]/g,'').substring(0,400));
    const isAr=/[\u0600-\u06FF]/.test(text);
    u.lang=isAr?'ar-DZ':'fr-FR'; u.rate=0.87; u.pitch=0.70;
    const vs=window.speechSynthesis.getVoices();
    if(isAr){ const ar=vs.find(v=>v.lang.startsWith('ar')); if(ar) u.voice=ar; }
    else { const fr=vs.find(v=>v.lang.startsWith('fr')&&v.name.includes('Male'))||vs.find(v=>v.lang.startsWith('fr')); if(fr) u.voice=fr; }
    window.speechSynthesis.speak(u);
  };
  window.updateFaceDisplay=function(text){
    const el=q('kFaces'); if(el) el.innerHTML=text;
  };
  setInterval(()=>{
    if(!speaking) return;
    ph+=0.45;
    mouth(Math.min(Math.abs(Math.sin(ph))*0.75+Math.random()*0.25, 1));
  }, 90);
  function init(){ scheduleBlink(); setTimeout(blink, 900); }
  if(document.readyState==='complete') init();
  else { window.addEventListener('load', init); setTimeout(init, 1500); }
})();
</script>
"""


# ══════════════════════════════════════════════════
#   CSS — THÈME KHEDIM IA CAMOUFLAGE ANP
# ══════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap');
*{ box-sizing:border-box; }

body,.gradio-container{
  background:#080c05 !important;
  font-family:'Rajdhani',sans-serif !important;
  color:#b8d4a0 !important;
}
.gradio-container{
  background:
    radial-gradient(ellipse at 8% 15%,  rgba(61,107,36,0.14) 0%, transparent 38%),
    radial-gradient(ellipse at 92% 85%, rgba(107,76,26,0.12) 0%, transparent 38%),
    radial-gradient(ellipse at 50% 50%, rgba(30,64,16,0.08) 0%, transparent 55%),
    linear-gradient(155deg, #0c1208 0%, #090d05 50%, #07090303 100%) !important;
  max-width:100% !important;
  padding:0 !important;
}

/* ── Onglets ── */
.tab-nav button{
  background:transparent !important; border:none !important;
  border-bottom:2px solid transparent !important;
  color:rgba(122,180,74,.45) !important;
  font-family:'Orbitron',monospace !important; font-size:.56rem !important;
  letter-spacing:2px !important; padding:10px 14px !important;
  transition:all .3s !important; text-transform:uppercase !important;
}
.tab-nav button.selected,.tab-nav button:hover{
  color:#c4a832 !important;
  border-bottom:2px solid #c4a832 !important;
  background:rgba(25,45,8,.35) !important;
  text-shadow:0 0 8px rgba(196,168,50,0.4) !important;
}

/* ── Chatbot ── */
.chatbot,[class*="chatbot"]{background:rgba(7,12,4,0.97) !important;}
.message.bot,[data-testid="bot"],[data-testid="bot"]>div,div[class*="bot"]{
  background:rgba(9,16,5,0.97) !important;
  border-left:3px solid #4a7c2f !important; border-radius:8px !important;
}
.message.user,[data-testid="user"],[data-testid="user"]>div,div[class*="user"]{
  background:rgba(18,14,4,0.97) !important;
  border-right:3px solid #8b6914 !important; border-radius:8px !important;
}
.chatbot p,.chatbot span,.chatbot div,.chatbot li,
[data-testid="bot"] *,[data-testid="user"] *,
.message p,.message span,.message div,
.prose p,.prose span,.prose li,.md p,.md span,.md li{
  color:#c8deb0 !important; font-family:'Rajdhani',sans-serif !important;
  font-size:0.92rem !important; line-height:1.7 !important;
  opacity:1 !important; visibility:visible !important;
}

/* ── Inputs ── */
textarea,input[type=text]{
  background:rgba(9,14,5,.88) !important;
  border:1px solid rgba(74,124,47,.32) !important;
  color:#b8d4a0 !important; font-family:'Rajdhani',sans-serif !important;
  font-size:.9rem !important; border-radius:8px !important;
}
textarea:focus,input[type=text]:focus{
  border-color:rgba(196,168,50,.58) !important;
  box-shadow:0 0 12px rgba(196,168,50,.1) !important; outline:none !important;
}

/* ── Boutons ── */
button.primary{
  background:linear-gradient(135deg,#182c09,#2d5a1b) !important;
  border:1px solid rgba(74,124,47,.52) !important;
  color:#7adf4a !important; font-family:'Orbitron',monospace !important;
  font-size:.55rem !important; letter-spacing:2px !important;
  border-radius:8px !important; transition:all .3s !important;
}
button.primary:hover{
  background:linear-gradient(135deg,#2d5a1b,#3d6b24) !important;
  box-shadow:0 0 18px rgba(122,223,74,.32) !important; transform:translateY(-1px) !important;
}
button.secondary{
  background:rgba(14,22,7,.8) !important;
  border:1px solid rgba(139,105,20,.32) !important;
  color:#c4a832 !important; font-family:'Orbitron',monospace !important;
  font-size:.52rem !important; letter-spacing:1.5px !important;
  border-radius:8px !important; transition:all .3s !important;
}
button.secondary:hover{
  border-color:rgba(196,168,50,.52) !important;
  box-shadow:0 0 10px rgba(196,168,50,.18) !important;
}

/* ── Labels ── */
label,.label-wrap span{
  font-family:'Orbitron',monospace !important;
  color:rgba(122,180,74,.62) !important; font-size:.52rem !important;
  letter-spacing:2px !important; text-transform:uppercase !important;
}
input[type=range]{ accent-color:#7adf4a !important; }
::-webkit-scrollbar{ width:3px; }
::-webkit-scrollbar-thumb{ background:rgba(74,124,47,.32); border-radius:2px; }
.face-output img{
  border:1px solid rgba(74,124,47,.32) !important; border-radius:8px !important;
  box-shadow:0 0 22px rgba(74,124,47,.18) !important;
}
.accordion{ background:rgba(9,14,5,.8) !important; border:1px solid rgba(61,107,36,.2) !important; border-radius:8px !important; }

/* ── Composants KHEDIM ── */
.kbadge{
  background:rgba(9,16,5,.8); border:1px solid rgba(74,124,47,.25);
  border-radius:5px; padding:3px 8px;
  font-family:'Orbitron',monospace; font-size:.43rem;
  color:#c4a832; letter-spacing:1px; text-align:center;
  margin:2px 0; display:inline-block;
}
.ktitle{
  font-family:'Orbitron',monospace; font-size:.56rem;
  color:rgba(122,180,74,.62); letter-spacing:2.5px;
  text-transform:uppercase; padding:7px 0 6px;
  border-bottom:1px solid rgba(74,124,47,.18); margin-bottom:9px;
}
.kcard{
  background:rgba(9,16,5,.85); border:1px solid rgba(196,168,50,.32);
  border-radius:8px; padding:8px 11px; margin:4px 0;
  font-family:'Orbitron',monospace; font-size:.48rem; color:#c4a832; letter-spacing:1px;
}
.kalert{
  background:rgba(40,18,4,.9); border:1px solid rgba(220,80,30,.4);
  border-radius:7px; padding:6px 10px; margin:3px 0;
  font-family:'Orbitron',monospace; font-size:.5rem; color:#e87040; letter-spacing:1px;
}
[dir="rtl"],.ar-text{direction:rtl;text-align:right;font-family:'Rajdhani','Arial',sans-serif !important;}
"""


# ══════════════════════════════════════════════════
#   HELPERS CONTEXTE
# ══════════════════════════════════════════════════

def build_system(context, faces_ctx, det_history="", shared_ctx=""):
    return SYSTEM_PROMPT.format(
        context=context,
        faces_context=faces_ctx,
        detection_history=det_history or "Aucun historique.",
        shared_context=shared_ctx or "Aucune identification partagée.",
    )

def _faces_to_context(face_state):
    if not face_state or not isinstance(face_state, dict) or face_state.get("nb", 0) == 0:
        return "Aucun visage détecté."
    nb = face_state["nb"]
    personnes = face_state.get("personnes", [])
    mode = face_state.get("mode", "")
    lines = [f"{nb} visage(s) — Mode: {mode}"]
    for p in personnes:
        conf = p.get("confiance", 0)
        if p["connu"]:
            ctx_line = ""
            if shared_memory.est_connu(p["nom"]):
                ctx_line = f" — {shared_memory.get_contexte(p['nom'])}"
            lines.append(f"✅ {p['nom']} ({int(float(conf)*100)}%){ctx_line}")
        else:
            lines.append("⚠️ INCONNU")
    return "\n".join(lines)

def _det_history_text():
    log = get_detection_log(6)
    if not log:
        return "Aucun historique."
    parts = []
    for e in reversed(log):
        ts = e["timestamp"][:16]
        connus = [p["nom"] for p in e.get("personnes", []) if p.get("connu")]
        parts.append(f"[{ts}] {', '.join(connus) if connus else 'inconnus'}")
    return "\n".join(parts)

def _shared_context():
    return shared_memory.get_resume()


# ══════════════════════════════════════════════════
#   LOGIQUE MÉTIER — TOUTES SECTIONS
# ══════════════════════════════════════════════════

# ── Chat texte ──
def chat_text(message, history, face_state):
    if not message.strip():
        return history, "", face_state
    ctx = mémoire.get_context("default", 4)
    faces_ctx = _faces_to_context(face_state)
    system = build_system(ctx, faces_ctx, _det_history_text(), _shared_context())
    is_ar = any('\u0600' <= c <= '\u06FF' for c in message)
    msgs = [{"role": "user", "content": message + (" (Réponds en arabe)" if is_ar else "")}]
    res = groq_engine.chat(msgs, system_prompt=system, max_tokens=700)
    reply = res["content"]
    mémoire.add_exchange("default", message, reply, "text")
    return add_message(history, message, reply), "", face_state


# ── Audio STT + TTS ──
def chat_audio(audio_path, history, face_state):
    if not audio_path:
        return history, "Aucun audio.", None, face_state
    tr_fr = transcribe_audio(audio_path, "fr")
    tr_ar = transcribe_audio(audio_path, "ar")
    text_fr = tr_fr.get("text", "").strip()
    text_ar = tr_ar.get("text", "").strip()
    transcribed = text_ar if len(text_ar) > len(text_fr) else text_fr
    if not transcribed:
        return history, "Transcription échouée.", None, face_state
    ctx = mémoire.get_context("default", 4)
    faces_ctx = _faces_to_context(face_state)
    system = build_system(ctx, faces_ctx, _det_history_text(), _shared_context())
    is_ar = any('\u0600' <= c <= '\u06FF' for c in transcribed)
    msgs = [{"role": "user", "content": transcribed + (" (Réponds en arabe)" if is_ar else "")}]
    res = groq_engine.chat(msgs, system_prompt=system, max_tokens=700)
    reply = res["content"]
    mémoire.add_exchange("default", transcribed, reply, "audio")
    audio_out = generate_tts(reply, "militaire")
    return add_message(history, f"🎤 {transcribed}", reply), f"Transcrit: {transcribed[:70]}...", audio_out, face_state


# ── Caméra terrain ──
def process_camera_frame(frame, history, face_state, auto_analyze):
    if frame is None:
        return None, history, face_state, "Aucune image."
    if isinstance(frame, np.ndarray):
        if frame.size == 0: return None, history, face_state, "Frame vide."
        h, w = frame.shape[:2]
        if h < 10 or w < 10: return None, history, face_state, "Initialisation..."
        if frame.mean() < 4: return None, history, face_state, "Image trop sombre."
    result = analyze_frame(frame, section="Caméra Terrain")
    # ── Log détection dans Neon ──
    if result.get("nb_visages", 0) > 0:
        engine_used = ""
        for p in result.get("personnes", []):
            engine_used = p.get("engine", "")
            break
        neon_log_detection(
            result["nb_visages"],
            result.get("personnes", []),
            section="Caméra Terrain",
            engine=engine_used,
        )
    # Auto-ID : identifier ou enregistrer chaque visage détecté
    if isinstance(frame, np.ndarray) and result.get("nb_visages", 0) > 0:
        auto_id = identify_or_register(frame)
        result["message"] = f"[{auto_id}] " + result["message"]
    annotated = result.get("image_annotee")
    new_fs = sanitize_state({
        "nb": result["nb_visages"], "personnes": result["personnes"],
        "message": result["message"], "timestamp": result["timestamp"],
        "mode": result.get("mode", ""),
    })
    img_out = numpy_to_pil(annotated) if annotated is not None and HAS_PIL else None
    status = result["message"]
    if auto_analyze and result["nb_visages"] > 0:
        ctx = mémoire.get_context("default", 3)
        system = build_system(ctx, _faces_to_context(new_fs), _det_history_text(), _shared_context())
        res = groq_engine.chat([{"role": "user", "content": f"Terrain: {result['message']}. Commente brièvement en français et arabe."}],
                               system_prompt=system, max_tokens=300)
        reply = res["content"]
        mémoire.add_exchange("default", result["message"], reply, "vision")
        return img_out, add_message(history, f"📷 {result['message']}", reply), new_fs, status
    return img_out, history, new_fs, status


def _safe_process(frame, history, f_state, auto):
    if frame is None: return None, history, f_state, "En attente caméra..."
    if isinstance(frame, np.ndarray) and (frame.size == 0 or frame.mean() < 3):
        return None, history, f_state, "Initialisation..."
    return process_camera_frame(frame, history, f_state, auto)


# ── Analyse image ──
def analyze_image_full(image, question, history, face_state):
    if image is None:
        return history, face_state, "Aucune image."
    face_res = analyze_frame(image, section="Analyse Image")
    new_fs = sanitize_state({"nb": face_res["nb_visages"], "personnes": face_res["personnes"],
                              "message": face_res["message"], "mode": face_res.get("mode", "")})
    ctx = mémoire.get_context("default", 3)
    system = build_system(ctx, _faces_to_context(new_fs), _det_history_text(), _shared_context())
    q = question.strip() if question.strip() else "Décris cette image en détail (français et arabe)."
    try:
        if HAS_PIL:
            pil = numpy_to_pil(image)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            pil.save(tmp.name, "JPEG", quality=85)
            vision_result = analyze_image_with_groq(tmp.name, q, groq_engine)
            reply = vision_result.get("description", "Analyse indisponible.")
        else:
            res = groq_engine.chat([{"role": "user", "content": q}], system_prompt=system, max_tokens=700)
            reply = res["content"]
    except Exception as e:
        reply = f"Erreur: {e}"
    mémoire.add_exchange("default", q, reply, "vision")
    return add_message(history, f"🖼️ {q}", reply), new_fs, face_res["message"]


# ── Enregistrement visage ──
def register_face_fn(image, nom, grade, unite, notes):
    if image is None: return "❌ Image requise."
    if not nom.strip(): return "❌ Nom requis."
    result = register_face(image, nom.strip(), grade=grade.strip(), unite=unite.strip(), notes=notes.strip())
    msg = result["message"]
    # ── Upload photo vers Backblaze B2 ──
    if result.get("success", False) and HAS_B2:
        try:
            url = b2_upload(image, nom.strip(), result.get("total_samples", 1))
            if url:
                msg += f"\n📸 Photo B2 : {url[:70]}..."
        except Exception as _e:
            msg += f"\n⚠️ B2 : {_e}"
    # ── Log dans Neon ──
    neon_log_event("register_face", f"{nom.strip()} — {msg[:200]}")
    return msg

def add_angle_fn(image, nom):
    """Ajoute un angle/vue supplémentaire pour une personne déjà enregistrée."""
    if image is None: return "❌ Image requise."
    if not nom.strip(): return "❌ Nom requis."
    if nom.strip() not in faces_db():
        return f"❌ '{nom}' n'existe pas. Enregistrez d'abord."
    result = register_face(image, nom.strip(), section="Ajout Angle")
    return result["message"]

def delete_face_fn(nom):
    ok = delete_person(nom.strip())
    shared_memory.effacer(nom.strip())
    return f"✅ '{nom}' supprimé." if ok else f"❌ '{nom}' introuvable."

def get_faces_list():
    names = list(faces_db().keys())
    if not names: return "Aucun visage enregistré."
    lines = []
    for n in names:
        info = faces_db().get(n, {})
        enc = "✅ Bio" if info.get("a_encodage_reel") else "⚠️ Léger"
        det = info.get("nb_detections", 0)
        angles = info.get("nb_angles", 0)
        grade = info.get("grade", "")
        unite = info.get("unite", "")
        shared = "🔗" if shared_memory.est_connu(n) else ""
        ts = info.get("derniere_detection", "")[:10]
        lines.append(f"• {grade+' ' if grade else ''}{n}  [{enc} {angles}ang]  {det}× {shared}  {unite}  {ts}")
    return "\n".join(lines)

def get_person_card_fn(nom):
    if not nom.strip(): return "Entrez un nom."
    return "Fonction non disponible"

def get_detection_log_text(n=20):
    log = get_detection_log(n)
    if not log: return "Aucune détection."
    lines = []
    for e in reversed(log):
        ts = e["timestamp"][:16]
        connus = [p["nom"] for p in e.get("personnes", []) if p.get("connu")]
        inconnus = sum(1 for p in e.get("personnes", []) if not p.get("connu"))
        parts = []
        if connus: parts.append(f"✅ {', '.join(connus)}")
        if inconnus: parts.append(f"⚠️ ×{inconnus}")
        lines.append(f"[{ts}]  {' | '.join(parts) if parts else 'vide'}")
    return "\n".join(lines)

def get_shared_text():
    return shared_memory.get_resume() or "Aucune identification partagée."

def get_stats_text():
    stats = groq_engine.get_stats()
    sys_info = get_system_info()
    audio_info = get_audio_info()
    lines = [
        "╔══════════════════════════════════╗",
        "║   KHEDIM IA v8.0 — STATUT        ║",
        "╚══════════════════════════════════╝",
        f"Modèle actif   : {stats.get('modele_actif','?')}",
        f"Clés GROQ      : {stats.get('cle_active','?')}/{stats.get('nb_cles',0)}",
        f"Tokens total   : {stats.get('total_tokens',0)}",
        f"Requêtes       : {stats.get('total_requetes',0)}",
        "───────────────────────────────────",
        f"OpenCV         : {'✅' if sys_info['opencv'] else '❌'}",
        f"InsightFace    : {'✅ ACTIF' if sys_info.get('insightface_actif') else '⚠️ Mode léger'}",
        f"Mode reco      : {sys_info.get('mode','?')}",
        f"Visages DB     : {sys_info['faces_enregistres']}",
        f"Avec encodage  : {sys_info.get('faces_avec_encodage',0)}",
        f"Détections log : {sys_info.get('detection_log_count',0)}",
        f"IDs partagées  : {sys_info.get('identifications_partagees',0)}",
        "───────────────────────────────────",
        f"Whisper STT    : {'✅' if audio_info.get('faster_whisper') else '❌'}",
        f"Edge-TTS       : {'✅' if audio_info.get('edge_tts') else '❌'}",
        f"gTTS backup    : {'✅' if audio_info.get('gtts') else '❌'}",
        "───────────────────────────────────",
    ]
    # ── Stats Neon ──
    neon_stats = neon_get_stats()
    if neon_stats:
        lines += [
            "NEON POSTGRESQL :",
            f"  Messages       : {neon_stats.get('total_messages', 0)}",
            f"  Détections     : {neon_stats.get('total_detections', 0)}",
            f"  Positives      : {neon_stats.get('detections_positives', 0)}",
            f"  Événements     : {neon_stats.get('total_events', 0)}",
        ]
    else:
        lines.append("Neon PostgreSQL : ❌ Non connecté")
    # ── Stats Backblaze ──
    b2 = b2_stats()
    if b2:
        lines += [
            "BACKBLAZE B2 :",
            f"  Photos         : {b2.get('total_photos', 0)}",
            f"  Taille         : {b2.get('total_size_mb', 0)} MB",
            f"  Bucket         : {b2.get('bucket', '—')}",
        ]
    else:
        lines.append("Backblaze B2    : ❌ Non connecté")
    lines += [
        "───────────────────────────────────",
        "ARMÉE NATIONALE POPULAIRE ALGÉRIENNE",
        "KHEDIM IA v8.0 — Khedim Benyakhlef",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════
#   DASHBOARD INVESTISSEURS
# ══════════════════════════════════════════════════

def get_dashboard_html() -> str:
    """Génère le HTML du dashboard investisseurs avec stats temps réel."""
    # ── Récupérer toutes les données ──
    neon_stats  = neon_get_stats()
    b2          = b2_stats()
    sys_info    = get_system_info()
    groq_stats  = groq_engine.get_stats()
    db          = faces_db()
    audio_info  = get_audio_info()

    total_msgs    = neon_stats.get("total_messages", 0)
    total_det     = neon_stats.get("total_detections", 0)
    det_pos       = neon_stats.get("detections_positives", 0)
    total_events  = neon_stats.get("total_events", 0)
    total_photos  = b2.get("total_photos", 0)
    taille_mb     = b2.get("total_size_mb", 0.0)
    nb_visages_db = len(db)
    nb_tokens     = groq_stats.get("total_tokens", 0)
    nb_requetes   = groq_stats.get("total_requetes", 0)
    modele        = groq_stats.get("modele_actif", "llama3")
    taux_det      = round((det_pos / total_det * 100) if total_det > 0 else 0, 1)

    neon_ok = "✅ EN LIGNE" if neon_stats else "❌ HORS LIGNE"
    b2_ok   = "✅ EN LIGNE" if b2        else "❌ HORS LIGNE"
    insight = "✅ ACTIF"   if sys_info.get("insightface_actif") else "⚠️ MODE LÉGER"
    whisper = "✅"         if audio_info.get("faster_whisper") else "❌"
    edge    = "✅"         if audio_info.get("edge_tts")       else "❌"

    ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600;700&display=swap');
.dash-wrap{{
  background:linear-gradient(135deg,#060a03 0%,#0a1005 50%,#070c04 100%);
  border:1px solid rgba(74,124,47,.3);border-radius:14px;padding:22px;
  font-family:'Rajdhani',sans-serif;
}}
.dash-title{{
  font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:900;
  color:#7adf4a;letter-spacing:4px;text-align:center;
  text-shadow:0 0 22px rgba(122,223,74,.5);margin-bottom:4px;
}}
.dash-subtitle{{
  font-family:'Rajdhani',sans-serif;color:rgba(196,168,50,.75);
  font-size:.72rem;letter-spacing:2px;text-align:center;margin-bottom:18px;
}}
.dash-ts{{
  font-family:'Orbitron',monospace;font-size:.45rem;color:rgba(122,180,74,.45);
  text-align:center;letter-spacing:2px;margin-bottom:20px;
}}
.dash-sep{{
  height:1px;
  background:linear-gradient(90deg,transparent,#4a7c2f,#c4a832,#4a7c2f,transparent);
  margin:18px 0;
}}
/* ── KPI Cards ── */
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}}
.kpi-card{{
  background:rgba(9,16,5,.85);border:1px solid rgba(74,124,47,.28);
  border-radius:10px;padding:14px 10px;text-align:center;
  transition:border-color .3s;
}}
.kpi-card:hover{{border-color:rgba(196,168,50,.5);}}
.kpi-icon{{font-size:1.4rem;margin-bottom:6px;}}
.kpi-val{{
  font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;
  color:#7adf4a;text-shadow:0 0 14px rgba(122,223,74,.4);
  line-height:1;
}}
.kpi-val.gold{{color:#c4a832;text-shadow:0 0 14px rgba(196,168,50,.4);}}
.kpi-val.blue{{color:#4ab8df;text-shadow:0 0 14px rgba(74,184,223,.4);}}
.kpi-val.red {{color:#df4a4a;text-shadow:0 0 14px rgba(223,74,74,.4);}}
.kpi-label{{
  font-family:'Orbitron',monospace;font-size:.42rem;color:rgba(122,180,74,.6);
  letter-spacing:1.5px;margin-top:6px;text-transform:uppercase;
}}
/* ── Status Row ── */
.status-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px;}}
.status-card{{
  background:rgba(9,16,5,.85);border:1px solid rgba(74,124,47,.22);
  border-radius:8px;padding:12px;
}}
.status-title{{
  font-family:'Orbitron',monospace;font-size:.48rem;color:#c4a832;
  letter-spacing:2px;margin-bottom:8px;border-bottom:1px solid rgba(196,168,50,.2);
  padding-bottom:5px;
}}
.status-row{{
  display:flex;justify-content:space-between;align-items:center;
  padding:3px 0;border-bottom:1px solid rgba(74,124,47,.1);
}}
.status-key{{font-size:.75rem;color:rgba(184,212,160,.7);}}
.status-val{{font-size:.75rem;font-weight:600;color:#7adf4a;}}
.status-val.gold{{color:#c4a832;}}
.status-val.warn{{color:#e87040;}}
/* ── Progress bar ── */
.prog-wrap{{margin-top:10px;}}
.prog-label{{
  display:flex;justify-content:space-between;
  font-size:.7rem;color:rgba(184,212,160,.65);margin-bottom:4px;
}}
.prog-bar{{
  height:6px;background:rgba(74,124,47,.15);border-radius:3px;overflow:hidden;
}}
.prog-fill{{
  height:100%;border-radius:3px;
  background:linear-gradient(90deg,#2d5a1b,#7adf4a);
  transition:width .5s ease;
}}
.prog-fill.gold{{background:linear-gradient(90deg,#8b6914,#c4a832);}}
/* ── Footer ── */
.dash-footer{{
  text-align:center;font-family:'Orbitron',monospace;
  font-size:.42rem;color:rgba(74,124,47,.35);letter-spacing:2px;margin-top:14px;
}}
</style>

<div class="dash-wrap">
  <div class="dash-title">⚡ KHEDIM IA — TABLEAU DE BORD</div>
  <div class="dash-subtitle">ARMÉE NATIONALE POPULAIRE ALGÉRIENNE 🇩🇿 — ANALYTICS TEMPS RÉEL</div>
  <div class="dash-ts">🕐 DERNIÈRE MISE À JOUR : {ts}</div>

  <div class="dash-sep"></div>

  <!-- KPI PRINCIPAUX -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-icon">💬</div>
      <div class="kpi-val">{total_msgs}</div>
      <div class="kpi-label">Messages<br>Sauvegardés</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">👁️</div>
      <div class="kpi-val gold">{total_det}</div>
      <div class="kpi-label">Détections<br>Terrain</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">🔐</div>
      <div class="kpi-val blue">{nb_visages_db}</div>
      <div class="kpi-label">Visages<br>Enregistrés</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">📸</div>
      <div class="kpi-val gold">{total_photos}</div>
      <div class="kpi-label">Photos<br>Stockées B2</div>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-icon">✅</div>
      <div class="kpi-val">{det_pos}</div>
      <div class="kpi-label">Identifications<br>Positives</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">🎯</div>
      <div class="kpi-val gold">{taux_det}%</div>
      <div class="kpi-label">Taux de<br>Reconnaissance</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">⚡</div>
      <div class="kpi-val blue">{nb_requetes}</div>
      <div class="kpi-label">Requêtes<br>GROQ IA</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-icon">🗂️</div>
      <div class="kpi-val">{total_events}</div>
      <div class="kpi-label">Événements<br>Système</div>
    </div>
  </div>

  <!-- BARRES DE PROGRESSION -->
  <div class="prog-wrap">
    <div class="prog-label">
      <span>Taux reconnaissance biométrique</span>
      <span>{taux_det}%</span>
    </div>
    <div class="prog-bar">
      <div class="prog-fill" style="width:{min(taux_det,100)}%"></div>
    </div>
  </div>
  <div class="prog-wrap" style="margin-top:8px;">
    <div class="prog-label">
      <span>Stockage Backblaze B2</span>
      <span>{taille_mb} MB</span>
    </div>
    <div class="prog-bar">
      <div class="prog-fill gold" style="width:{min(taille_mb/10*100,100)}%"></div>
    </div>
  </div>

  <div class="dash-sep"></div>

  <!-- STATUT COMPOSANTS -->
  <div class="status-grid">

    <div class="status-card">
      <div class="status-title">🗄️ BASES DE DONNÉES</div>
      <div class="status-row">
        <span class="status-key">Neon PostgreSQL</span>
        <span class="status-val {'warn' if '❌' in neon_ok else ''}">{neon_ok}</span>
      </div>
      <div class="status-row">
        <span class="status-key">Backblaze B2</span>
        <span class="status-val {'warn' if '❌' in b2_ok else ''}">{b2_ok}</span>
      </div>
      <div class="status-row">
        <span class="status-key">MongoDB Atlas</span>
        <span class="status-val">✅ EN LIGNE</span>
      </div>
      <div class="status-row">
        <span class="status-key">Photos stockées</span>
        <span class="status-val gold">{total_photos} | {taille_mb} MB</span>
      </div>
      <div class="status-row">
        <span class="status-key">Messages Neon</span>
        <span class="status-val gold">{total_msgs}</span>
      </div>
    </div>

    <div class="status-card">
      <div class="status-title">🤖 MOTEUR IA GROQ</div>
      <div class="status-row">
        <span class="status-key">Modèle actif</span>
        <span class="status-val gold">{modele}</span>
      </div>
      <div class="status-row">
        <span class="status-key">Tokens utilisés</span>
        <span class="status-val">{nb_tokens:,}</span>
      </div>
      <div class="status-row">
        <span class="status-key">Requêtes total</span>
        <span class="status-val">{nb_requetes}</span>
      </div>
      <div class="status-row">
        <span class="status-key">STT Whisper</span>
        <span class="status-val">{whisper}</span>
      </div>
      <div class="status-row">
        <span class="status-key">TTS Edge-TTS</span>
        <span class="status-val">{edge}</span>
      </div>
    </div>

    <div class="status-card">
      <div class="status-title">👁️ BIOMÉTRIE</div>
      <div class="status-row">
        <span class="status-key">InsightFace</span>
        <span class="status-val {'warn' if '⚠️' in insight else ''}">{insight}</span>
      </div>
      <div class="status-row">
        <span class="status-key">Visages DB</span>
        <span class="status-val gold">{nb_visages_db}</span>
      </div>
      <div class="status-row">
        <span class="status-key">Détections total</span>
        <span class="status-val">{total_det}</span>
      </div>
      <div class="status-row">
        <span class="status-key">Identifications +</span>
        <span class="status-val">{det_pos}</span>
      </div>
      <div class="status-row">
        <span class="status-key">Taux succès</span>
        <span class="status-val gold">{taux_det}%</span>
      </div>
    </div>

  </div>

  <div class="dash-sep"></div>
  <div class="dash-footer">
    KHEDIM IA v8.0 — ARMÉE NATIONALE POPULAIRE ALGÉRIENNE — KHEDIM BENYAKHLEF (BENY-JOE) 🇩🇿
  </div>
</div>
"""


# ══════════════════════════════════════════════════
#   INTERFACE GRADIO — KHEDIM IA v8.0
# ══════════════════════════════════════════════════

def build_app():
    with gr.Blocks( title="KHEDIM IA — Armée Nationale Populaire Algérienne v8.0") as demo:
        face_state = gr.State(value=None)

        # ── Bandeau en-tête ANP ──
        gr.HTML(f"""
        <div style="background:rgba(6,10,3,0.99);border-bottom:2px solid #2d5a1b;padding:0;">
          <div style="height:5px;background:linear-gradient(90deg,
            #3d6b24 0%,#8b6914 20%,#c4a832 40%,#4a7c2f 60%,#6b4c1a 80%,#3d6b24 100%);"></div>
          <div style="padding:12px 22px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div style="font-size:2rem;">🇩🇿</div>
            <div style="font-size:1.5rem;">🪖</div>
            <div>
              <div style="font-family:'Orbitron',monospace;font-size:1.05rem;font-weight:900;
                          color:#7adf4a;letter-spacing:4px;
                          text-shadow:0 0 22px rgba(122,223,74,0.5);">KHEDIM IA</div>
              <div style="font-family:'Rajdhani',sans-serif;color:rgba(196,168,50,.78);
                          font-size:0.85rem;letter-spacing:3px;">
                خديم للذكاء الاصطناعي — الجيش الوطني الشعبي الجزائري</div>
              <div style="font-family:'Rajdhani',sans-serif;color:rgba(122,180,74,.4);
                          font-size:0.52rem;letter-spacing:2px;margin-top:1px;">
                ARMÉE NATIONALE POPULAIRE — v8.0 — KHEDIM BENYAKHLEF (BINY-JOE)</div>
            </div>
            <div style="margin-left:auto;display:flex;gap:5px;align-items:center;flex-wrap:wrap;">
              <span style="background:rgba(9,16,5,.88);border:1px solid rgba(74,124,47,.38);
                border-radius:5px;padding:4px 9px;font-family:'Orbitron',monospace;
                font-size:.43rem;color:#7adf4a;letter-spacing:2px;">KHEDIM IA</span>
              <span style="background:rgba(9,16,5,.88);border:1px solid rgba(74,124,47,.38);
                border-radius:5px;padding:4px 9px;font-family:'Orbitron',monospace;
                font-size:.43rem;color:#7adf4a;letter-spacing:2px;">INSIGHTFACE</span>
              <span style="background:rgba(9,16,5,.88);border:1px solid rgba(139,105,20,.38);
                border-radius:5px;padding:4px 9px;font-family:'Orbitron',monospace;
                font-size:.43rem;color:#c4a832;letter-spacing:2px;">🔗 MEM. PARTAGÉE</span>
              <span style="background:rgba(9,16,5,.88);border:1px solid rgba(74,124,47,.38);
                border-radius:5px;padding:4px 9px;font-family:'Orbitron',monospace;
                font-size:.43rem;color:#22d060;letter-spacing:2px;">◉ EN LIGNE</span>
            </div>
          </div>
          <div style="height:2px;background:linear-gradient(90deg,transparent,#4a7c2f,#c4a832,#4a7c2f,transparent);"></div>
        </div>
        {AVATAR_JS}
        """)

        with gr.Row(equal_height=False):

            # ── Colonne avatar ──
            with gr.Column(scale=1, min_width=200):
                gr.HTML(AVATAR_HTML)

            # ── Colonne principale ──
            with gr.Column(scale=5):
                with gr.Tabs():

                    # ── 1. CONVERSATION ──
                    with gr.Tab("Conversation"):
                        gr.HTML('<div class="ktitle">💬 TERMINAL KHEDIM IA — CHAT BILINGUE FR / AR</div>')
                        chatbot = gr.Chatbot(label="Terminal KHEDIM IA", height=390)
                        with gr.Row():
                            chat_in = gr.Textbox(
                                placeholder="Message en français ou en arabe... / اكتب رسالتك هنا (يفهم الدارجة)",
                                label="Message", lines=2, scale=5)
                            send_btn = gr.Button("ENVOYER ▶", variant="primary", scale=1, min_width=110)
                        chat_status = gr.Textbox(label="Statut", interactive=False, lines=1)
                        gr.HTML("""
                        <div style="display:flex;gap:5px;flex-wrap:wrap;margin-top:7px;">
                          <span style="font-family:'Orbitron',monospace;font-size:.46rem;
                            color:rgba(122,180,74,.45);letter-spacing:2px;align-self:center;">EXEMPLES :</span>
                          <button onclick="(()=>{const t=document.querySelector('textarea');
                            t.value='Présente-toi, KHEDIM IA.';t.dispatchEvent(new Event('input'));})()"
                            style="background:rgba(9,16,5,.75);border:1px solid rgba(74,124,47,.3);
                            color:#7adf4a;font-family:'Orbitron',monospace;font-size:.46rem;
                            border-radius:4px;padding:4px 8px;cursor:pointer;">Présentation</button>
                          <button onclick="(()=>{const t=document.querySelector('textarea');
                            t.value='قدم نفسك يا خديم.';t.dispatchEvent(new Event('input'));})()"
                            style="background:rgba(9,16,5,.75);border:1px solid rgba(139,105,20,.3);
                            color:#c4a832;font-family:'Orbitron',monospace;font-size:.46rem;
                            border-radius:4px;padding:4px 8px;cursor:pointer;">تعريف عربي</button>
                          <button onclick="(()=>{const t=document.querySelector('textarea');
                            t.value='Rapport de surveillance terrain complet.';t.dispatchEvent(new Event('input'));})()"
                            style="background:rgba(9,16,5,.75);border:1px solid rgba(74,124,47,.3);
                            color:#7adf4a;font-family:'Orbitron',monospace;font-size:.46rem;
                            border-radius:4px;padding:4px 8px;cursor:pointer;">Rapport</button>
                          <button onclick="(()=>{const t=document.querySelector('textarea');
                            t.value='Qui as-tu identifié dans les autres sections ?';t.dispatchEvent(new Event('input'));})()"
                            style="background:rgba(9,16,5,.75);border:1px solid rgba(139,105,20,.3);
                            color:#c4a832;font-family:'Orbitron',monospace;font-size:.46rem;
                            border-radius:4px;padding:4px 8px;cursor:pointer;">IDs partagées</button>
                          <button onclick="(()=>{const t=document.querySelector('textarea');
                            t.value='واش شفت من ناس اليوم؟';t.dispatchEvent(new Event('input'));})()"
                            style="background:rgba(9,16,5,.75);border:1px solid rgba(74,124,47,.3);
                            color:#7adf4a;font-family:'Orbitron',monospace;font-size:.46rem;
                            border-radius:4px;padding:4px 8px;cursor:pointer;">دارجة</button>
                        </div>
                        """)
                        send_btn.click(chat_text, [chat_in, chatbot, face_state], [chatbot, chat_in, face_state])
                        chat_in.submit(chat_text, [chat_in, chatbot, face_state], [chatbot, chat_in, face_state])
                        with gr.Row():
                            gr.Button("Effacer / مسح", variant="secondary").click(
                                lambda: ([], "Session effacée.", {}),
                                outputs=[chatbot, chat_status, face_state])

                    # ── 2. COMMUNICATION VOCALE ──
                    with gr.Tab("Communication Vocale"):
                        gr.HTML('<div class="ktitle">🎤 COMMUNICATION VOCALE — WHISPER STT + EDGE-TTS BILINGUE</div>')
                        audio_chatbot = gr.Chatbot(label="Terminal Vocal KHEDIM IA", height=290)
                        with gr.Row():
                            with gr.Column():
                                audio_in = gr.Audio(sources=["microphone","upload"], type="filepath", label="Enregistrement vocal")
                                audio_send = gr.Button("TRANSMETTRE ▶", variant="primary")
                                audio_status = gr.Textbox(label="Transcription", interactive=False, lines=2)
                            with gr.Column():
                                audio_out = gr.Audio(label="Réponse vocale KHEDIM IA", autoplay=True)
                                gr.HTML("""
                                <div class="kcard">
                                  <div style="color:#7adf4a;margin-bottom:5px;">PIPELINE AUDIO KHEDIM IA</div>
                                  <div style="font-size:.78rem;color:#b8d4a0;line-height:2;font-family:'Rajdhani',sans-serif;">
                                    1. Whisper CPU — Transcription FR + AR automatique<br>
                                    2. GROQ LLaMA 70B — Réponse bilingue KHEDIM IA<br>
                                    3. Edge-TTS — Voix masculine FR ou voix AR<br>
                                    4. Contexte visages + mémoire partagée intégrés<br>
                                    5. Dialecte algérien (دارجة) compris
                                  </div>
                                </div>
                                """)
                        audio_send.click(chat_audio, [audio_in, audio_chatbot, face_state],
                                         [audio_chatbot, audio_status, audio_out, face_state])
                        gr.Button("Effacer", variant="secondary").click(lambda: [], outputs=[audio_chatbot])

                    # ── 3. CAMÉRA TERRAIN ──
                    with gr.Tab("Caméra Terrain"):
                        gr.HTML('<div class="ktitle">📷 RECONNAISSANCE FACIALE — INSIGHTFACE 512-DIM MULTI-ANGLE</div>')
                        with gr.Row():
                            with gr.Column(scale=2):
                                camera_in = gr.Image(sources=["webcam","upload"], type="numpy",
                                                     label="Caméra Terrain", streaming=False)
                                auto_analyze_cb = gr.Checkbox(label="Analyse IA auto si visage détecté", value=False)
                                cam_btn = gr.Button("ANALYSER FRAME ▶", variant="primary")
                                cam_status = gr.Textbox(label="Détection", interactive=False, lines=1)
                            with gr.Column(scale=2):
                                face_output = gr.Image(label="Image annotée", elem_classes=["face-output"])
                                cam_chatbot = gr.Chatbot(label="Rapport Vision", height=200)

                        cam_btn.click(process_camera_frame,
                                      [camera_in, cam_chatbot, face_state, auto_analyze_cb],
                                      [face_output, cam_chatbot, face_state, cam_status])
                        camera_in.change(_safe_process,
                                         [camera_in, cam_chatbot, face_state, auto_analyze_cb],
                                         [face_output, cam_chatbot, face_state, cam_status])

                        gr.HTML('<div style="height:1px;background:linear-gradient(90deg,transparent,#4a7c2f,#c4a832,transparent);margin:10px 0;"></div>')
                        with gr.Row():
                            vision_question = gr.Textbox(placeholder="Question sur l'image... / سؤال عن الصورة", label="Question visuelle", lines=2, scale=4)
                            vision_btn = gr.Button("ANALYSER ▶", variant="primary", scale=1)
                        vision_btn.click(analyze_image_full,
                                         [camera_in, vision_question, cam_chatbot, face_state],
                                         [cam_chatbot, face_state, cam_status])
                        gr.Button("Effacer rapport", variant="secondary").click(lambda: [], outputs=[cam_chatbot])

                    # ── 4. BASE BIOMÉTRIQUE ──
                    with gr.Tab("Base Biométrique"):
                        gr.HTML('<div class="ktitle">🔐 GESTION BASE BIOMÉTRIQUE — KHEDIM IA</div>')
                        with gr.Row():
                            with gr.Column():
                                gr.HTML('<div class="kcard">➕ ENREGISTRER / METTRE À JOUR</div>')
                                reg_image = gr.Image(sources=["webcam","upload"], type="numpy",
                                                     label="Photo (1 seule personne)")
                                reg_name   = gr.Textbox(placeholder="Nom complet", label="Nom complet", lines=1)
                                reg_grade  = gr.Textbox(placeholder="Ex: Colonel, Lt, Sergent...", label="Grade (optionnel)", lines=1)
                                reg_unite  = gr.Textbox(placeholder="Ex: 3ème Région Militaire", label="Unité (optionnel)", lines=1)
                                reg_notes  = gr.Textbox(placeholder="Notes diverses...", label="Notes (optionnel)", lines=2)
                                reg_btn = gr.Button("ENREGISTRER ▶", variant="primary")
                                reg_status = gr.Textbox(label="Statut", interactive=False, lines=4)
                                reg_btn.click(register_face_fn, [reg_image, reg_name, reg_grade, reg_unite, reg_notes], [reg_status])

                                gr.HTML('<div class="kcard" style="margin-top:10px;">📐 AJOUTER UN ANGLE SUPPLÉMENTAIRE</div>')
                                angle_image = gr.Image(sources=["webcam","upload"], type="numpy",
                                                       label="Nouvelle photo (profil, angle...)")
                                angle_name = gr.Textbox(placeholder="Nom exact de la personne", label="Nom", lines=1)
                                angle_btn = gr.Button("AJOUTER ANGLE ▶", variant="secondary")
                                angle_status = gr.Textbox(label="Statut", interactive=False, lines=2)
                                angle_btn.click(add_angle_fn, [angle_image, angle_name], [angle_status])

                            with gr.Column():
                                gr.HTML('<div class="kcard">📋 PERSONNEL ENREGISTRÉ</div>')
                                faces_list_box = gr.Textbox(value=get_faces_list(), label="Base de données", interactive=False, lines=12)
                                gr.Button("Actualiser", variant="secondary").click(fn=get_faces_list, outputs=[faces_list_box])

                                gr.HTML('<div class="kcard" style="margin-top:10px;">👤 FICHE PERSONNEL</div>')
                                card_name = gr.Textbox(placeholder="Nom exact", label="Nom", lines=1)
                                card_btn = gr.Button("Voir fiche ▶", variant="secondary")
                                card_out = gr.Textbox(label="Fiche", interactive=False, lines=8)
                                card_btn.click(get_person_card_fn, [card_name], [card_out])

                                gr.HTML('<div class="kcard" style="margin-top:10px;">🗑️ SUPPRIMER</div>')
                                del_name = gr.Textbox(placeholder="Nom exact", label="Nom", lines=1)
                                del_btn = gr.Button("SUPPRIMER", variant="secondary")
                                del_status = gr.Textbox(label="Statut", interactive=False, lines=1)
                                del_btn.click(delete_face_fn, [del_name], [del_status])

                    # ── 5. JOURNAL & MÉMOIRE ──
                    with gr.Tab("Journal & Mémoire"):
                        gr.HTML('<div class="ktitle">📋 JOURNAL DÉTECTIONS + MÉMOIRE PARTAGÉE INTER-SECTIONS</div>')
                        with gr.Row():
                            with gr.Column():
                                gr.HTML('<div class="kcard">🕐 JOURNAL DES DÉTECTIONS</div>')
                                det_log_box = gr.Textbox(value=get_detection_log_text(), label="Journal", interactive=False, lines=12)
                                gr.Button("Actualiser", variant="secondary").click(fn=get_detection_log_text, outputs=[det_log_box])

                                gr.HTML('<div class="kcard" style="margin-top:10px;">💾 MÉMOIRE SESSION</div>')
                                mem_box = gr.Textbox(value=session_memory.get_context_text(),
                                                     label="Contexte session", interactive=False, lines=5)
                                gr.Button("Actualiser mémoire", variant="secondary").click(
                                    fn=session_memory.get_context_text, outputs=[mem_box])

                            with gr.Column():
                                gr.HTML('<div class="kcard">🔗 IDENTIFICATIONS PARTAGÉES (TOUTES SECTIONS)</div>')
                                shared_box = gr.Textbox(value=get_shared_text(),
                                                        label="Mémoire partagée inter-sections", interactive=False, lines=10)
                                gr.Button("Actualiser", variant="secondary").click(fn=get_shared_text, outputs=[shared_box])

                                gr.HTML('<div class="kcard" style="margin-top:10px;">🚨 ALERTES TERRAIN</div>')
                                def get_alerts():
                                    alerts = session_memory.get_recent_alerts(15)
                                    if not alerts: return "Aucune alerte."
                                    return "\n".join(f"[{a['timestamp'][:16]}] {a['message']}" for a in reversed(alerts))
                                alerts_box = gr.Textbox(value=get_alerts(), label="Alertes", interactive=False, lines=6)
                                gr.Button("Actualiser", variant="secondary").click(fn=get_alerts, outputs=[alerts_box])

                                gr.HTML('<div class="kcard" style="margin-top:10px;">🧹 RÉINITIALISER MÉMOIRE PARTAGÉE</div>')
                                def clear_shared():
                                    shared_memory.tout_effacer()
                                    return "✅ Mémoire partagée effacée."
                                clear_btn = gr.Button("EFFACER MÉMOIRE PARTAGÉE", variant="secondary")
                                clear_status = gr.Textbox(label="Statut", interactive=False, lines=1)
                                clear_btn.click(clear_shared, outputs=[clear_status])

                    # ── 6. AUDIO + VISION ──
                    with gr.Tab("Audio + Vision"):
                        gr.HTML('<div class="ktitle">🎯 MULTIMODAL — AUDIO + CAMÉRA + IA KHEDIM</div>')
                        av_chatbot = gr.Chatbot(label="Terminal Multimodal KHEDIM IA", height=260)
                        with gr.Row():
                            with gr.Column(scale=1):
                                av_camera = gr.Image(sources=["webcam","upload"], type="numpy",
                                                     label="Caméra Terrain")
                            with gr.Column(scale=1):
                                av_audio = gr.Audio(sources=["microphone","upload"], type="filepath", label="Message vocal")
                                av_send = gr.Button("ENVOYER AUDIO + VISION ▶", variant="primary")
                                av_out_audio = gr.Audio(label="Réponse KHEDIM IA", autoplay=True)
                                av_status = gr.Textbox(label="Statut", interactive=False, lines=2)

                        def av_combined(audio_path, camera_frame, history, f_state):
                            new_fs = f_state.copy() if f_state else {}
                            img_path = None
                            if camera_frame is not None and isinstance(camera_frame, np.ndarray) and camera_frame.size > 0 and camera_frame.mean() > 3:
                                face_res = analyze_frame(camera_frame, section="Audio+Vision")
                                new_fs = sanitize_state({"nb": face_res["nb_visages"], "personnes": face_res["personnes"],
                                                         "message": face_res["message"], "mode": face_res.get("mode","")})
                                if HAS_PIL:
                                    try:
                                        pil_img = numpy_to_pil(camera_frame)
                                        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                                        pil_img.save(tmp.name, "JPEG", quality=85)
                                        img_path = tmp.name
                                    except Exception:
                                        img_path = None
                            transcribed = ""
                            if audio_path:
                                tr = transcribe_audio(audio_path, "fr")
                                tr_ar = transcribe_audio(audio_path, "ar")
                                text_fr = tr.get("text","").strip()
                                text_ar = tr_ar.get("text","").strip()
                                transcribed = text_ar if len(text_ar) > len(text_fr) else text_fr
                            if not transcribed:
                                transcribed = "Analyse la scène terrain et décris en français et arabe."
                            ctx = mémoire.get_context("default", 4)
                            system = build_system(ctx, _faces_to_context(new_fs), _det_history_text(), _shared_context())
                            if img_path:
                                q = f"{transcribed}\n[Détection: {new_fs.get('message','—')}]"
                                vision_result = analyze_image_with_groq(img_path, q, groq_engine)
                                reply = vision_result.get("description","Analyse indisponible.")
                                if new_fs.get("nb",0) > 0:
                                    reply = f"[Visages] {new_fs['message']}\n\n{reply}"
                            else:
                                camera_note = f"\n[Caméra: {new_fs.get('message','—')}]" if new_fs.get("nb",0) > 0 else ""
                                res = groq_engine.chat([{"role":"user","content":f"{transcribed}{camera_note}"}],
                                                       system_prompt=system, max_tokens=700)
                                reply = res["content"]
                            mémoire.add_exchange("default", transcribed, reply, "audio_vision")
                            audio_rep = generate_tts(reply, "militaire")
                            user_msg = f"🎤📷 {transcribed}"
                            if new_fs.get("nb",0) > 0:
                                user_msg += f" | {new_fs['message']}"
                            return add_message(history, user_msg, reply), audio_rep, f"Transcrit: {transcribed[:55]}", new_fs

                        av_send.click(av_combined, [av_audio, av_camera, av_chatbot, face_state],
                                      [av_chatbot, av_out_audio, av_status, face_state])
                        gr.Button("Effacer", variant="secondary").click(lambda: [], outputs=[av_chatbot])

                    # ── 7. SYSTÈME ──
                    with gr.Tab("Système"):
                        gr.HTML('<div class="ktitle">⚙️ STATUT COMPOSANTS — KHEDIM IA v8.0</div>')
                        stats_box = gr.Textbox(value=get_stats_text(), label="Statut système", interactive=False, lines=22)
                        gr.Button("Actualiser ▶", variant="secondary").click(fn=get_stats_text, outputs=[stats_box])
                        gr.HTML("""
                        <div class="kcard" style="margin-top:12px;">
                          <div style="color:#7adf4a;margin-bottom:6px;">⚙️ DÉPLOIEMENT RENDER — CONFIGURATION</div>
                          <div style="font-size:.78rem;color:#b8d4a0;line-height:2.1;font-family:'Rajdhani',sans-serif;">
                            <b style="color:#7adf4a">GROQ_API_KEY_1</b> — Clé GROQ principale (obligatoire)<br>
                            <b style="color:#7adf4a">GROQ_API_KEY_2</b> — Clé GROQ secondaire (optionnel)<br>
                            <b style="color:#7adf4a">GROQ_API_KEY_3</b> — Clé GROQ tertiaire (optionnel)<br>
                            Obtenir : <b style="color:#c4a832">console.groq.com</b><br><br>
                            <b style="color:#7adf4a">Render :</b> Connecter GitHub → Web Service → render.yaml détecté auto<br>
                            <b style="color:#7adf4a">PORT :</b> Lu automatiquement depuis $PORT (Render) ou 7860 (local)
                          </div>
                        </div>
                        """)

                    # ── 8. DASHBOARD INVESTISSEURS ──
                    with gr.Tab("📊 Dashboard"):
                        gr.HTML('<div class="ktitle">📊 DASHBOARD INVESTISSEURS — ANALYTICS TEMPS RÉEL</div>')
                        dash_html = gr.HTML(value=get_dashboard_html())
                        gr.Button("🔄 ACTUALISER ▶", variant="primary").click(
                            fn=get_dashboard_html, outputs=[dash_html]
                        )

        gr.HTML("""
        <div style="font-family:'Orbitron',monospace;font-size:.44rem;color:rgba(74,124,47,.22);
                    letter-spacing:2px;text-align:center;padding:10px;
                    border-top:1px solid rgba(61,107,36,.1);">
          KHEDIM IA v8.0 — ARMÉE NATIONALE POPULAIRE ALGÉRIENNE — KHEDIM BENYAKHLEF (BINY-JOE)
        </div>
        """)

    return demo


# ══ LANCEMENT ══
demo = build_app()
demo.queue()

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port, share=False)
