"""
╔══════════════════════════════════════════════════════════════╗
║   KHEDIM IA v8.0 — MOTEUR BACKBLAZE B2                      ║
║   Stockage photos visages (S3 compatible)                    ║
║   Fondé par Khedim Benyakhlef (Beny-Joe)                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import io
import time

# ══════════════════════════════════════════════
#   CONNEXION BACKBLAZE B2
# ══════════════════════════════════════════════

_s3_client = None

def _get_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client
    try:
        import boto3
        from botocore.config import Config
        key_id   = os.getenv("B2_KEY_ID")
        app_key  = os.getenv("B2_APP_KEY")
        endpoint = os.getenv("B2_ENDPOINT")
        if not all([key_id, app_key, endpoint]):
            print("❌ Backblaze : variables manquantes (B2_KEY_ID, B2_APP_KEY, B2_ENDPOINT)")
            return None
        _s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{endpoint}",
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            config=Config(signature_version="s3v4"),
        )
        print("✅ Backblaze B2 connecté")
        return _s3_client
    except Exception as e:
        print(f"❌ Backblaze erreur : {e}")
        return None

def _get_bucket() -> str:
    return os.getenv("B2_BUCKET_NAME", "KHEDIM-AI")

# ══════════════════════════════════════════════
#   UPLOAD PHOTO VISAGE
# ══════════════════════════════════════════════

def upload_face_photo(img_rgb, name: str, sample_index: int = 0) -> str | None:
    """
    Upload une photo de visage vers Backblaze B2.
    Retourne l'URL publique ou None si échec.
    """
    client = _get_client()
    if client is None:
        return None
    try:
        import cv2
        import numpy as np
        # Convertir RGB → JPEG en mémoire
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_bytes = buf.tobytes()
        # Clé S3 : visages/nom/nom_001_timestamp.jpg
        ts = time.strftime("%Y%m%d_%H%M%S")
        key = f"visages/{name}/{name}_{sample_index:03d}_{ts}.jpg"
        bucket = _get_bucket()
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=img_bytes,
            ContentType="image/jpeg",
        )
        endpoint = os.getenv("B2_ENDPOINT")
        url = f"https://{endpoint}/file/{bucket}/{key}"
        print(f"✅ Photo uploadée : {url}")
        return url
    except Exception as e:
        print(f"❌ upload_face_photo erreur : {e}")
        return None

# ══════════════════════════════════════════════
#   LISTER LES PHOTOS D'UNE PERSONNE
# ══════════════════════════════════════════════

def list_face_photos(name: str) -> list:
    """Retourne la liste des URLs des photos d'une personne."""
    client = _get_client()
    if client is None:
        return []
    try:
        bucket = _get_bucket()
        endpoint = os.getenv("B2_ENDPOINT")
        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"visages/{name}/"
        )
        urls = []
        for obj in response.get("Contents", []):
            url = f"https://{endpoint}/file/{bucket}/{obj['Key']}"
            urls.append(url)
        return urls
    except Exception:
        return []

# ══════════════════════════════════════════════
#   SUPPRIMER LES PHOTOS D'UNE PERSONNE
# ══════════════════════════════════════════════

def delete_face_photos(name: str) -> int:
    """Supprime toutes les photos d'une personne. Retourne le nombre supprimé."""
    client = _get_client()
    if client is None:
        return 0
    try:
        bucket = _get_bucket()
        response = client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"visages/{name}/"
        )
        objects = response.get("Contents", [])
        if not objects:
            return 0
        delete_list = [{"Key": obj["Key"]} for obj in objects]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": delete_list}
        )
        print(f"✅ {len(delete_list)} photos supprimées pour {name}")
        return len(delete_list)
    except Exception as e:
        print(f"❌ delete_face_photos erreur : {e}")
        return 0

# ══════════════════════════════════════════════
#   STATS STOCKAGE
# ══════════════════════════════════════════════

def get_storage_stats() -> dict:
    """Retourne les stats du bucket Backblaze."""
    client = _get_client()
    if client is None:
        return {}
    try:
        bucket = _get_bucket()
        response = client.list_objects_v2(Bucket=bucket, Prefix="visages/")
        objects = response.get("Contents", [])
        total_size = sum(obj.get("Size", 0) for obj in objects)
        return {
            "total_photos": len(objects),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "bucket": bucket,
        }
    except Exception:
        return {}

# ══════════════════════════════════════════════
#   INITIALISATION
# ══════════════════════════════════════════════

def init_backblaze() -> bool:
    """Appeler au démarrage de app.py."""
    client = _get_client()
    return client is not None
