from pymongo import MongoClient
import requests
import time
import base64

MONGO_URI = 'mongodb+srv://khedimbenyakhlef_db_user:5q7RcqsusonfGeSv@cluster0.logfyqe.mongodb.net/atare_db'
col = MongoClient(MONGO_URI)['atare_db']['faces']

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# Noms alternatifs pour Wikipedia (nom_original -> [essais])
NOMS_ALTERNATIFS = {
    "Ahmed Ben Bella":       ["Ahmed Ben Bella", "Mohammed Ben Bella"],
    "Ahmed Gaid Salah":      ["Ahmed Gaid Salah", "Gaid Salah"],
    "Said Chengriha":        ["Said Chengriha", "Saïd Chengriha"],
    "Islam Slimani":         ["Islam Slimani"],
    "Yacine Brahimi":        ["Yacine Brahimi"],
    "Sofiane Feghouli":      ["Sofiane Feghouli"],
    "Madjid Bougherra":      ["Madjid Bougherra"],
    "Nabil Bentaleb":        ["Nabil Bentaleb"],
    "Baghdad Bounedjah":     ["Baghdad Bounedjah"],
    "Ismail Bennacer":       ["Ismail Bennacer", "Ismaël Bennacer"],
    "Djamel Belmadi":        ["Djamel Belmadi"],
    "Khaled Hadj Ibrahim":   ["Khaled", "Khaled (singer)"],
    "Cheb Mami":             ["Cheb Mami"],
    "Idir":                  ["Idir (singer)", "Idir musician"],
    "Soolking":              ["Soolking", "Soolking rapper"],
    "Lotfi Double Kanon":    ["Lotfi Double Kanon"],
    "Warda Al Jazairia":     ["Warda Al-Jazairia", "Warda (singer)"],
    "Cheikha Rimitti":       ["Cheikha Rimitti", "Rimitti"],
    "Mohamed Fellag":        ["Fellag", "Mohamed Fellag"],
    "Biyouna":               ["Biyouna actress"],
    "Lyes Salem":            ["Lyes Salem"],
    "Kateb Yacine":          ["Kateb Yacine"],
    "Assia Djebar":          ["Assia Djebar"],
    "Yasmina Khadra":        ["Yasmina Khadra"],
    "Mouloud Mammeri":       ["Mouloud Mammeri"],
    "Larbi Ben Mhidi":       ["Larbi Ben M'Hidi", "Larbi Ben Mhidi"],
    "Abane Ramdane":         ["Abane Ramdane", "Ramdane Abane"],
}

def get_photo_wiki(titre):
    """Télécharge photo Wikipedia avec retry"""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": titre,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": 400,
            "redirects": 1
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("ns") == -1:
                continue
            thumb = page.get("thumbnail", {})
            if thumb and "source" in thumb:
                img_url = thumb["source"].replace("50px", "300px").replace("100px", "300px")
                img = requests.get(img_url, headers=HEADERS, timeout=20)
                if img.status_code == 200 and len(img.content) > 5000:
                    return base64.b64encode(img.content).decode("utf-8")
    except Exception as e:
        print(f"    erreur: {e}")
    return None

def get_photo_commons(nom):
    """Cherche sur Wikimedia Commons"""
    try:
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{nom} portrait",
            "srnamespace": 6,
            "format": "json",
            "srlimit": 3
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        data = r.json()
        results = data.get("query", {}).get("search", [])
        for result in results:
            titre_fichier = result["title"]
            # Obtenir l'URL du fichier
            info_params = {
                "action": "query",
                "titles": titre_fichier,
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "iiurlwidth": 300,
                "format": "json"
            }
            r2 = requests.get(url, params=info_params, headers=HEADERS, timeout=20)
            data2 = r2.json()
            pages2 = data2.get("query", {}).get("pages", {})
            for p in pages2.values():
                ii = p.get("imageinfo", [{}])[0]
                mime = ii.get("mime", "")
                if "image" in mime and "svg" not in mime:
                    img_url = ii.get("thumburl") or ii.get("url", "")
                    if img_url:
                        img = requests.get(img_url, headers=HEADERS, timeout=20)
                        if img.status_code == 200 and len(img.content) > 5000:
                            return base64.b64encode(img.content).decode("utf-8")
    except Exception as e:
        print(f"    Commons erreur: {e}")
    return None

# Traitement des visages sans photo
docs_sans_photo = list(col.find({
    "nationalite": "DZ",
    "photo_b64": {"$exists": False}
}))

print(f"{len(docs_sans_photo)} visages sans photo\n")

mis_a_jour = 0

for doc in docs_sans_photo:
    nom = doc["name"]
    print(f"🔍 {nom}")
    photo = None

    # Essai 1 : Wikipedia avec noms alternatifs
    variantes = NOMS_ALTERNATIFS.get(nom, [nom])
    for variante in variantes:
        photo = get_photo_wiki(variante)
        if photo:
            print(f"  ✅ Wikipedia OK ({variante})")
            break
        time.sleep(0.5)

    # Essai 2 : Wikimedia Commons
    if not photo:
        photo = get_photo_commons(nom)
        if photo:
            print(f"  ✅ Commons OK")

    if photo:
        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"photo_b64": photo}}
        )
        mis_a_jour += 1
    else:
        print(f"  ❌ Introuvable")

    time.sleep(1)

print(f"\n✅ Photos ajoutées: {mis_a_jour} / {len(docs_sans_photo)}")
print(f"Total avec photo: {col.count_documents({'photo_b64': {'$exists': True}})}")
