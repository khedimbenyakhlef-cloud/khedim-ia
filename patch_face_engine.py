"""
Script de patch automatique — Exécuter avec:
  python patch_face_engine.py
dans le dossier khedim_ia/
"""
import os
import shutil

face_engine_path = "backend/face_engine.py"
rotation_src = "face_engine_rotation.py"
rotation_dst = "backend/face_engine_rotation.py"

# 1. Copier face_engine_rotation.py dans backend/
if os.path.exists(rotation_src):
    shutil.copy(rotation_src, rotation_dst)
    print(f"✅ {rotation_dst} copié")
else:
    print(f"❌ {rotation_src} introuvable — copie manuelle nécessaire")

# 2. Patcher face_engine.py
content = open(face_engine_path, "r", encoding="utf-8").read()

import_patch = """
# ══ ROTATION ENGINE — ajouté automatiquement ═══════════════════
try:
    from backend.face_engine_rotation import identify_face_rotation
    _ROTATION_AVAILABLE = True
    print("✅ Moteur rotation faciale chargé")
except Exception as _rot_err:
    _ROTATION_AVAILABLE = False
    print(f"⚠️ Moteur rotation non disponible: {_rot_err}")
# ═══════════════════════════════════════════════════════════════

"""

old_func = "def identify_face(img_rgb: np.ndarray) -> dict:"
rotation_call = """def identify_face(img_rgb: np.ndarray) -> dict:
    # ── Rotation automatique InsightFace → face_recognition → DeepFace ──
    if _ROTATION_AVAILABLE:
        return identify_face_rotation(img_rgb)
    # ── Fallback code original ────────────────────────────────────────────
"""

patched = False

# Ajouter import si pas encore là
if "_ROTATION_AVAILABLE" not in content:
    content = content.replace(old_func, import_patch + old_func)
    print("✅ Import rotation ajouté dans face_engine.py")
    patched = True
else:
    print("ℹ️  Import déjà présent")

# Patcher la fonction
if "identify_face_rotation" not in content:
    if old_func in content:
        content = content.replace(old_func, rotation_call)
        print("✅ Fonction identify_face() patchée")
        patched = True
    else:
        print("❌ Signature identify_face() non trouvée — vérifier face_engine.py")
else:
    print("ℹ️  Fonction déjà patchée")

if patched:
    # Backup avant modification
    shutil.copy(face_engine_path, face_engine_path + ".bak")
    open(face_engine_path, "w", encoding="utf-8").write(content)
    print("✅ face_engine.py sauvegardé (.bak créé)")

print("\n═══ VÉRIFICATION ═══")
content_check = open(face_engine_path, "r", encoding="utf-8").read()
print("_ROTATION_AVAILABLE dans face_engine.py:", "_ROTATION_AVAILABLE" in content_check)
print("identify_face_rotation dans face_engine.py:", "identify_face_rotation" in content_check)
print("face_engine_rotation.py dans backend/:", os.path.exists(rotation_dst))
print("\n✅ PATCH TERMINÉ")
print("\nCommandes Git à exécuter:")
print('  git add backend/face_engine.py backend/face_engine_rotation.py')
print('  git commit -m "feat: rotation auto InsightFace→face_recognition→DeepFace"')
print('  git push origin main')
