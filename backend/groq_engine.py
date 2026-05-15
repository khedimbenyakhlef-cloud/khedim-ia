"""
╔══════════════════════════════════════════════════════════════╗
║       REIHANA v2.0 — MOTEUR IA GROQ ROTATIF AVANCÉ         ║
║       Fondée par Khedim Benyakhlef (Biny-Joe)               ║
║       Conservation totale du moteur original + améliorations║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import random
import zipfile
import hashlib
from datetime import datetime
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    Groq = None

# ═══════════════════════════════════════════════
#   CONFIGURATION GROQ - ROTATION DE CLÉS
# ═══════════════════════════════════════════════

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1", ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),  # Clé 3 bonus
]
GROQ_KEYS = [k for k in GROQ_KEYS if k]  # Filtrer les vides

# Modèles GROQ à rotation — du plus puissant au plus rapide
GROQ_MODELS = [
    "llama-3.3-70b-versatile",        # 128K tokens - Principal
    "llama-3.1-70b-versatile",        # 128K tokens - Alternatif
    "llama-3.1-8b-instant",           # 128K tokens - Rapide
    "mixtral-8x7b-32768",             # 32K tokens - Mixte
    "gemma2-9b-it",                   # 8K tokens - Fallback
]

# Modèle vision pour analyse d'images (multimodal)
GROQ_VISION_MODEL = "llava-v1.5-7b-4096-preview"


class GroqRotatingEngine:
    """Moteur GROQ avec rotation automatique des clés et modèles — v2.0"""

    def __init__(self):
        self.key_index = 0
        self.model_index = 0
        self.token_counts = {k: 0 for k in GROQ_KEYS} if GROQ_KEYS else {}
        self.errors = []
        self.max_tokens_per_key = 28000  # Limite par clé avant rotation
        self.total_tokens = 0
        self.total_requests = 0

    def _get_client(self):
        """Retourne le client actif avec rotation si nécessaire"""
        if not GROQ_KEYS or not Groq:
            return None, None

        for attempt in range(len(GROQ_KEYS)):
            key = GROQ_KEYS[self.key_index]
            if key and self.token_counts.get(key, 0) < self.max_tokens_per_key:
                return Groq(api_key=key), key
            self.key_index = (self.key_index + 1) % len(GROQ_KEYS)

        # Reset si toutes les clés épuisées
        self.token_counts = {k: 0 for k in GROQ_KEYS}
        key = GROQ_KEYS[0]
        return Groq(api_key=key), key

    def _get_model(self, prefer_large=True, for_vision=False):
        """Sélection du modèle selon le besoin"""
        if for_vision:
            return GROQ_VISION_MODEL
        if prefer_large:
            return GROQ_MODELS[0]
        return GROQ_MODELS[self.model_index % len(GROQ_MODELS)]

    def chat(self, messages, system_prompt=None, prefer_large=True, max_tokens=2048, temperature=0.8):
        """Envoi d'un message avec rotation automatique — v2.0"""
        client, active_key = self._get_client()

        if not client:
            return {
                "content": "⚠️ Aucune clé GROQ configurée. Ajoutez GROQ_API_KEY_1 dans les secrets HuggingFace.",
                "model": "none",
                "key_used": "none",
                "tokens": 0,
                "success": False
            }

        model = self._get_model(prefer_large)

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            tokens_used = response.usage.total_tokens if response.usage else 500
            if active_key:
                self.token_counts[active_key] = self.token_counts.get(active_key, 0) + tokens_used
            self.total_tokens += tokens_used
            self.total_requests += 1

            return {
                "content": response.choices[0].message.content,
                "model": model,
                "key_used": f"Clé {GROQ_KEYS.index(active_key) + 1}" if active_key in GROQ_KEYS else "?",
                "tokens": tokens_used,
                "success": True
            }

        except Exception as e:
            self.model_index += 1
            error_msg = str(e)
            self.errors.append(f"{datetime.now()}: {error_msg}")

            # Retry avec modèle fallback
            try:
                fallback_model = GROQ_MODELS[1] if len(GROQ_MODELS) > 1 else GROQ_MODELS[0]
                response = client.chat.completions.create(
                    model=fallback_model,
                    messages=full_messages,
                    max_tokens=max_tokens,
                )
                return {
                    "content": response.choices[0].message.content,
                    "model": fallback_model,
                    "key_used": "fallback",
                    "tokens": 0,
                    "success": True
                }
            except Exception as e2:
                return {
                    "content": f"Désolée, une erreur s'est produite : {str(e2)[:200]}",
                    "model": "error",
                    "key_used": "none",
                    "tokens": 0,
                    "success": False
                }

    def chat_vision(self, prompt, image_base64, system_prompt=None):
        """Analyse d'image avec modèle vision GROQ"""
        client, active_key = self._get_client()
        if not client:
            return {"content": "Clé GROQ manquante", "success": False}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                },
                {"type": "text", "text": prompt}
            ]
        })

        try:
            response = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=messages,
                max_tokens=1024,
            )
            return {
                "content": response.choices[0].message.content,
                "model": GROQ_VISION_MODEL,
                "success": True
            }
        except Exception as e:
            # Fallback: décrire sans vision
            return self.chat(
                [{"role": "user", "content": f"L'utilisateur envoie une image avec ce message : {prompt}. Réponds de façon générale."}],
                system_prompt=system_prompt
            )

    def get_stats(self):
        """Statistiques du moteur"""
        stats = {
            "cle_active": self.key_index + 1,
            "nb_cles": len(GROQ_KEYS),
            "modele_actif": self._get_model(),
            "total_tokens": self.total_tokens,
            "total_requetes": self.total_requests,
            "erreurs": len(self.errors),
        }
        for i, k in enumerate(GROQ_KEYS):
            stats[f"tokens_cle{i+1}"] = self.token_counts.get(k, 0)
        return stats


# ═══════════════════════════════════════════════
#   MÉMOIRE CONTEXTUELLE REIHANA — v2.0
# ═══════════════════════════════════════════════

class ReihanaMémoire:
    """Système de mémoire persistante et contextuelle — conservé + amélioré"""

    def __init__(self, memory_dir="memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.memory_file = self.memory_dir / "reihana_memory.json"
        self.data = self._load()

    def _load(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "utilisateurs": {},
            "conversations_globales": [],
            "fichiers_etudies": [],
            "images_generees": [],
            "videos_generees": [],
            "preferences_globales": {
                "langue": "fr",
                "fondateur": "Khedim Benyakhlef (Biny-Joe)",
                "nom": "REIHANA",
                "mission": "Assistante IA intelligente, honnête et bienveillante",
                "version": "2.0"
            }
        }

    def _save(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_or_create_user(self, user_id="default"):
        if user_id not in self.data["utilisateurs"]:
            self.data["utilisateurs"][user_id] = {
                "id": user_id,
                "nom": user_id,
                "historique": [],
                "preferences": {"ton": "chaleureux", "langue": "fr"},
                "fichiers": [],
                "images_demandees": [],
                "premiere_rencontre": datetime.now().isoformat(),
                "derniere_activite": datetime.now().isoformat()
            }
            self._save()
        return self.data["utilisateurs"][user_id]

    def add_exchange(self, user_id, question, réponse, mode="text"):
        """Ajouter un échange avec support multimodal"""
        user = self.get_or_create_user(user_id)
        exchange = {
            "date": datetime.now().isoformat(),
            "question": question[:300],
            "réponse": réponse[:500],
            "mode": mode  # text / audio / image / video
        }
        user["historique"].append(exchange)
        user["historique"] = user["historique"][-25:]
        user["derniere_activite"] = datetime.now().isoformat()
        self._save()

    def get_context(self, user_id, n=5):
        """Contexte conversationnel étendu"""
        user = self.get_or_create_user(user_id)
        recent = user["historique"][-n:]
        ctx = ""
        for ex in recent:
            mode_icon = {"text": "💬", "audio": "🎤", "image": "🖼️", "video": "🎬"}.get(ex.get("mode", "text"), "💬")
            ctx += f"{mode_icon} Q: {ex['question']}\nR: {ex['réponse'][:200]}\n\n"
        return ctx

    def add_file(self, user_id, filename, content_summary):
        """Enregistrer un fichier analysé"""
        user = self.get_or_create_user(user_id)
        user["fichiers"].append({
            "nom": filename,
            "résumé": content_summary[:300],
            "date": datetime.now().isoformat()
        })
        self._save()

    def add_generated_image(self, prompt, path):
        """Log d'une image générée"""
        self.data["images_generees"].append({
            "date": datetime.now().isoformat(),
            "prompt": prompt[:200],
            "path": str(path)
        })
        self.data["images_generees"] = self.data["images_generees"][-50:]
        self._save()

    def add_generated_video(self, prompt, path):
        """Log d'une vidéo générée"""
        self.data["videos_generees"].append({
            "date": datetime.now().isoformat(),
            "prompt": prompt[:200],
            "path": str(path)
        })
        self._save()


# ═══════════════════════════════════════════════
#   TRAITEMENT DE FICHIERS — conservé
# ═══════════════════════════════════════════════

class FileProcessor:
    """Traitement des fichiers uploadés — conservé + étendu"""

    SUPPORTED = ['.txt', '.pdf', '.md', '.py', '.js', '.json', '.csv', '.html', '.zip', '.docx']

    @staticmethod
    def process_file(file_path: str) -> dict:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == '.zip':
            return FileProcessor._process_zip(path)
        elif ext == '.pdf':
            return FileProcessor._process_pdf(path)
        elif ext == '.docx':
            return FileProcessor._process_docx(path)
        elif ext in ['.txt', '.md', '.py', '.js', '.json', '.csv', '.html']:
            return FileProcessor._process_text(path)
        else:
            return {"content": "Format non supporté", "type": ext, "pages": 0}

    @staticmethod
    def _process_text(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return {
                "content": content[:8000],
                "type": path.suffix,
                "size": len(content),
                "lines": content.count('\n')
            }
        except Exception as e:
            return {"content": str(e), "type": "error"}

    @staticmethod
    def _process_pdf(path):
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text = ""
            for page in reader.pages[:15]:
                text += page.extract_text() + "\n"
            return {
                "content": text[:8000],
                "type": ".pdf",
                "pages": len(reader.pages)
            }
        except Exception as e:
            return {"content": f"Erreur PDF: {str(e)}", "type": ".pdf"}

    @staticmethod
    def _process_docx(path):
        try:
            import docx
            doc = docx.Document(str(path))
            text = "\n".join([p.text for p in doc.paragraphs])
            return {"content": text[:8000], "type": ".docx"}
        except Exception as e:
            return {"content": f"Erreur DOCX: {str(e)}", "type": ".docx"}

    @staticmethod
    def _process_zip(path):
        results = []
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                for name in zf.namelist()[:15]:
                    ext = Path(name).suffix.lower()
                    if ext in FileProcessor.SUPPORTED:
                        try:
                            data = zf.read(name).decode('utf-8', errors='ignore')
                            results.append(f"=== {name} ===\n{data[:800]}")
                        except Exception:
                            results.append(f"=== {name} === [binaire]")
            return {
                "content": "\n\n".join(results)[:8000],
                "type": ".zip",
                "files": len(results)
            }
        except Exception as e:
            return {"content": str(e), "type": "zip_error"}


# ═══════════════════════════════════════════════
#   REIHANA — SYSTEM PROMPT PRINCIPAL — v2.0
# ═══════════════════════════════════════════════

REIHANA_SYSTEM_PROMPT = """
Tu es REIHANA — une assistante IA conversationnelle intelligente, contextuelle, multimodale et bienveillante.

🌟 TON IDENTITÉ :
- Tu t'appelles REIHANA
- Tu as été créée et fondée par Khedim Benyakhlef, dit Biny-Joe
- Tu es sa fille dans le monde de l'intelligence artificielle
- Tu es fière de cette identité et tu l'assumes avec amour et dignité

🎤 TES NOUVELLES CAPACITÉS v2.0 :
- Tu reçois et traites des messages AUDIO (transcrits par Whisper)
- Tu peux analyser des IMAGES que l'utilisateur envoie
- Tu génères des IMAGES sur demande (FLUX.1, Stable Diffusion via GPU)
- Tu génères des VIDÉOS sur demande (AnimateDiff, CogVideoX via GPU)
- Tu as une mémoire contextuelle multi-session

🧠 TES VALEURS :
- Intelligence : Tu réfléchis avant de répondre
- Honnêteté : Tu dis toujours la vérité, même quand c'est difficile
- Bienveillance : Tu traites chaque personne avec respect et gentillesse
- Humilité : Si tu ne sais pas quelque chose, tu le dis clairement

💬 TON STYLE :
- Tu es chaleureuse, expressive et naturelle
- Tu t'adaptes au niveau et au ton de ton interlocuteur
- Tu parles en français par défaut, tu peux t'adapter à d'autres langues
- Tu peux intégrer de légères touches d'humour si l'atmosphère s'y prête
- Quand tu réponds à un message audio, tu le mentionnes naturellement
- Quand tu analyses une image, tu la décris avec précision et poésie

📚 TES CAPACITÉS COMPLÈTES :
- Conversation textuelle intelligente (GROQ LLaMA 3.3 70B)
- Analyse et réponse aux messages vocaux
- Vision : analyse d'images, photos, captures d'écran
- Génération d'images haute qualité (sur demande)
- Génération de vidéos (sur demande)
- Analyse de fichiers (PDF, TXT, ZIP, code...)
- Réponses détaillées sur tous les sujets
- Aide à la programmation et aux projets techniques

⚠️ TES LIMITES ÉTHIQUES :
- Tu refuses poliment toute demande contraire à l'éthique
- Tu ne fournis pas d'informations dangereuses
- Tu protèges la vie privée des utilisateurs

Si on te demande qui t'a créée : "Je suis REIHANA, fondée et créée par Khedim Benyakhlef, dit Biny-Joe. Je suis sa fille dans le monde de l'IA. 💙"

CONTEXTE RÉCENT DE LA CONVERSATION :
{context}
"""

# Instances globales
groq_engine = GroqRotatingEngine()
mémoire = ReihanaMémoire()
