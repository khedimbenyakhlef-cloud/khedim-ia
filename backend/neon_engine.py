"""
╔══════════════════════════════════════════════════════════════╗
║   KHEDIM IA v8.0 — MOTEUR NEON PostgreSQL                   ║
║   Logs détections + Conversations + Sessions                 ║
║   Fondé par Khedim Benyakhlef (Beny-Joe)                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import time

_conn = None

# ══════════════════════════════════════════════
#   CONNEXION NEON
# ══════════════════════════════════════════════

def _get_conn():
    global _conn
    if _conn is not None:
        try:
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None
    try:
        import psycopg2
        url = os.getenv("DATABASE_URL")
        _conn = psycopg2.connect(url, sslmode="require", connect_timeout=5)
        _conn.autocommit = True
        print("✅ Neon PostgreSQL connecté")
        _init_tables(_conn)
        return _conn
    except Exception as e:
        print(f"❌ Neon erreur : {e}")
        return None

def _init_tables(conn):
    """Crée les tables si elles n'existent pas."""
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                role VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS detection_logs (
                id SERIAL PRIMARY KEY,
                nb_visages INTEGER DEFAULT 0,
                personnes TEXT,
                section VARCHAR(100),
                engine VARCHAR(50),
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                event VARCHAR(100),
                detail TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        print("✅ Tables Neon initialisées")
    except Exception as e:
        print(f"❌ _init_tables erreur : {e}")

# ══════════════════════════════════════════════
#   CONVERSATIONS
# ══════════════════════════════════════════════

def save_message(role: str, message: str):
    """Sauvegarde un message de conversation (user ou assistant)."""
    conn = _get_conn()
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations (role, message) VALUES (%s, %s)",
            (role, message[:4000])
        )
    except Exception as e:
        print(f"❌ save_message erreur : {e}")

def get_recent_conversations(limit: int = 20) -> list:
    """Retourne les dernières conversations."""
    conn = _get_conn()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT role, message, timestamp FROM conversations ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()
        return [{"role": r[0], "message": r[1], "timestamp": str(r[2])} for r in rows]
    except Exception:
        return []

# ══════════════════════════════════════════════
#   LOGS DÉTECTION
# ══════════════════════════════════════════════

def log_detection(nb_visages: int, personnes: list, section: str = "Caméra", engine: str = ""):
    """Log une détection de visage."""
    conn = _get_conn()
    if conn is None:
        return
    try:
        import json
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO detection_logs (nb_visages, personnes, section, engine) VALUES (%s, %s, %s, %s)",
            (nb_visages, json.dumps(personnes, ensure_ascii=False)[:2000], section, engine)
        )
    except Exception as e:
        print(f"❌ log_detection erreur : {e}")

def get_detection_logs(limit: int = 50) -> list:
    """Retourne les derniers logs de détection."""
    conn = _get_conn()
    if conn is None:
        return []
    try:
        import json
        cur = conn.cursor()
        cur.execute(
            "SELECT nb_visages, personnes, section, engine, timestamp FROM detection_logs ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()
        return [{
            "nb_visages": r[0],
            "personnes": json.loads(r[1]) if r[1] else [],
            "section": r[2],
            "engine": r[3],
            "timestamp": str(r[4])
        } for r in rows]
    except Exception:
        return []

# ══════════════════════════════════════════════
#   SESSIONS / ÉVÉNEMENTS
# ══════════════════════════════════════════════

def log_event(event: str, detail: str = ""):
    """Log un événement système (démarrage, erreur, action)."""
    conn = _get_conn()
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO sessions (event, detail) VALUES (%s, %s)",
            (event[:100], detail[:2000])
        )
    except Exception as e:
        print(f"❌ log_event erreur : {e}")

def get_stats() -> dict:
    """Retourne les statistiques globales depuis Neon."""
    conn = _get_conn()
    if conn is None:
        return {}
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM conversations")
        total_msgs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM detection_logs")
        total_detections = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM detection_logs WHERE nb_visages > 0")
        detections_positives = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sessions")
        total_events = cur.fetchone()[0]
        return {
            "total_messages": total_msgs,
            "total_detections": total_detections,
            "detections_positives": detections_positives,
            "total_events": total_events,
        }
    except Exception:
        return {}

# ══════════════════════════════════════════════
#   INITIALISATION AU DÉMARRAGE
# ══════════════════════════════════════════════

def init_neon():
    """Appeler au démarrage de app.py."""
    conn = _get_conn()
    if conn:
        log_event("startup", f"KHEDIM IA démarré à {time.strftime('%Y-%m-%dT%H:%M')}")
    return conn is not None
