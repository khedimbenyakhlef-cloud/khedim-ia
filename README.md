# 🪖 KHEDIM IA — Armée Nationale Populaire Algérienne v8.0

Système d'intelligence artificielle de terrain.
**Fondé par Khedim Benyakhlef (Biny-Joe)** 🇩🇿

## Fonctionnalités v8.0

| Section | Description |
|---|---|
| 💬 Conversation | Chat bilingue FR/AR/Darja avec LLaMA 70B |
| 🎤 Communication Vocale | Whisper STT + Edge-TTS FR+AR |
| 📷 Caméra Terrain | InsightFace multi-angle, annotation temps réel |
| 🔐 Base Biométrique | Fiche personnel (grade, unité), multi-angles par personne |
| 📋 Journal & Mémoire | Détections, alertes, **mémoire partagée inter-sections** |
| 🎯 Audio + Vision | Mode multimodal simultané |
| ⚙️ Système | Statut composants, config Render |

### Mémoire partagée inter-sections
Quand une personne est identifiée dans **n'importe quelle section**, elle est automatiquement
connue dans toutes les autres — **sans réidentification**.

### Base biométrique enrichie
- Grade, unité, notes par personne
- Multi-angles (profil gauche, droit, frontal) — meilleure précision
- Fiche personnel consultable

## Déploiement Render (5 minutes)

1. Pusher ce dossier sur GitHub
2. [render.com](https://render.com) → New Web Service → connecter le repo
3. Render détecte `render.yaml` automatiquement
4. Ajouter les secrets :
   - `GROQ_API_KEY_1` (depuis [console.groq.com](https://console.groq.com))
5. Deploy → accès via l'URL Render fournie

## Déploiement local

```bash
pip install -r requirements.txt
cp .env.example .env
# Éditer .env avec vos clés GROQ
python app.py
# → http://localhost:7860
```

## Structure

```
khedim_ia/
├── app.py                    # Interface principale KHEDIM IA
├── render.yaml               # Config déploiement Render
├── requirements.txt
├── backend/
│   ├── face_engine.py        # Biométrie + mémoire partagée
│   ├── groq_engine.py        # LLM GROQ rotation clés
│   ├── audio_engine.py       # Whisper STT + Edge-TTS
│   └── vision_engine.py      # Vision multimodale
└── memory/                   # Données persistantes JSON
    ├── faces_db.json
    ├── detection_log.json
    ├── session_memory.json
    └── identifications_partagees.json
```
# redeploy Sun May 17 22:59:32     2026
# updated Wed May 20 13:38:14     2026
