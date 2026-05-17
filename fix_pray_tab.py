import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Trouver la fonction build_app et le bloc gr.Blocks
match = re.search(r'(def build_app\(\):.*?)(with gr\.Blocks\([^)]*\) as demo:)(.*?)(return demo)', content, re.DOTALL)
if not match:
    print("❌ Structure de app.py non reconnue")
    exit(1)

before_blocks = match.group(1)      # code avant le with gr.Blocks
blocks_line = match.group(2)        # la ligne "with gr.Blocks(...) as demo:"
inside_blocks = match.group(3)      # contenu à l'intérieur du with
after_blocks = match.group(4)       # "return demo"

# Vérifier si l'onglet est déjà présent (éviter duplicata)
if '♿ PRAY PLUS+' in inside_blocks:
    print("⚠️ Onglet déjà présent, aucune modification")
    exit(0)

# Code de l'onglet avec l'indentation correcte (8 espaces = 4 pour with + 4 pour l'onglet)
pray_tab = '''
        with gr.Tab("♿ PRAY PLUS+"):
            gr.Markdown("""### 🤖 PRAY PLUS+ — Accessibilité Intelligente
            **Inventé par Khedim Benyakhlef (Beny-Joe)**
            *OCR · Description IA · Détection obstacles · Reconnaissance proches*""")

            with gr.Row():
                with gr.Column(scale=1):
                    source = gr.Radio(["Webcam", "Upload image"], label="Source", value="Webcam")
                    webcam = gr.Image(source="webcam", streaming=True, label="Caméra", visible=True)
                    upload = gr.Image(type="numpy", label="Télécharger une image", visible=False)
                    mode = gr.Dropdown(
                        choices=["complet", "ocr", "scene", "obstacle", "proche"],
                        value="complet",
                        label="Mode PRAY"
                    )
                    with gr.Row():
                        analyse_btn = gr.Button("📸 Analyser maintenant", variant="primary")
                        enreg_btn = gr.Button("💾 Enregistrer ce visage", variant="secondary")
                    nom_proche = gr.Textbox(label="Nom du proche (pour enregistrement)", placeholder="ex: Maman, Karim...")
                
                with gr.Column(scale=1):
                    resultat = gr.Textbox(label="Résultat", lines=8)
                    audio_out = gr.Audio(label="Audio", type="filepath")
            
            # Gestion de la visibilité webcam/upload
            def toggle_source(choice):
                return {webcam: gr.update(visible=(choice=="Webcam")),
                        upload: gr.update(visible=(choice=="Upload image"))}
            source.change(toggle_source, inputs=source, outputs=[webcam, upload])
            
            def analyser_pray(img, mode_choisi, nom):
                if img is None:
                    return "❌ Aucune image disponible", None
                try:
                    from backend.pray_plus_engine import analyser
                    if mode_choisi == "proche" and nom and nom.strip():
                        from backend.pray_plus_engine import enregistrer_proche
                        res_enr = enregistrer_proche(img, nom.strip())
                        if res_enr["ok"]:
                            return f"✅ {res_enr['msg']}\\n\\nMaintenant, utilise le mode 'proche' sans nom pour reconnaître.", None
                        else:
                            return f"❌ {res_enr['msg']}", None
                    res = analyser(img, mode=mode_choisi, lang="fr")
                    texte = ""
                    audio_file = None
                    if "ocr" in res and res["ocr"].get("ok"):
                        texte += f"📖 OCR : {res['ocr']['texte']}\\n\\n"
                        audio_file = res["ocr"].get("audio")
                    if "scene" in res and res["scene"].get("ok"):
                        texte += f"🖼️ Description : {res['scene']['description']}\\n\\n"
                        audio_file = res["scene"].get("audio") or audio_file
                    if "obstacle" in res and res["obstacle"].get("alerte"):
                        texte += f"⚠️ {res['obstacle']['message']}\\n\\n"
                        audio_file = res["obstacle"].get("audio") or audio_file
                    if "proche" in res and res["proche"].get("reconnu"):
                        texte += f"👤 {res['proche']['msg']}\\n\\n"
                        audio_file = res["proche"].get("audio") or audio_file
                    if not texte:
                        texte = "Aucune information détectée (mode = " + mode_choisi + ")"
                    return texte.strip(), audio_file
                except Exception as e:
                    return f"⚠️ Erreur PRAY : {str(e)}", None
            
            analyse_btn.click(analyser_pray, inputs=[webcam, mode, nom_proche], outputs=[resultat, audio_out])
            upload.change(lambda img, mode_choisi, nom: analyser_pray(img, mode_choisi, nom), inputs=[upload, mode, nom_proche], outputs=[resultat, audio_out])
            enreg_btn.click(lambda img, nom: (lambda: (lambda res: (res["msg"], None))(enregistrer_proche(img, nom.strip())) if img is not None and nom and nom.strip() else ("❌ Image ou nom manquant", None))(), inputs=[webcam, nom_proche], outputs=[resultat, audio_out])
'''

# Insérer l'onglet avant le return demo (à la fin de inside_blocks)
new_inside = inside_blocks.rstrip() + '\n' + pray_tab + '\n        '
new_content = before_blocks + blocks_line + new_inside + after_blocks

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Onglet PRAY PLUS+ inséré correctement à l'intérieur de gr.Blocks")
