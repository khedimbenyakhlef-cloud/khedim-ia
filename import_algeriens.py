from pymongo import MongoClient
import requests
import time
import base64
from io import BytesIO

MONGO_URI = 'mongodb+srv://khedimbenyakhlef_db_user:5q7RcqsusonfGeSv@cluster0.logfyqe.mongodb.net/atare_db'
col = MongoClient(MONGO_URI)['atare_db']['faces']

print('Connexion OK ! Visages actuels:', col.count_documents({}))

NOMS_ALGERIENS = [
    'Abdelmadjid Tebboune', 'Abdelaziz Bouteflika', 'Liamine Zeroual',
    'Ahmed Ouyahia', 'Houari Boumediene', 'Ahmed Ben Bella',
    'Chadli Bendjedid', 'Mohamed Boudiaf', 'Hocine Ait Ahmed',
    'Ahmed Gaid Salah', 'Said Chengriha',
    'Riyad Mahrez', 'Islam Slimani', 'Yacine Brahimi', 'Sofiane Feghouli',
    'Madjid Bougherra', 'Nabil Bentaleb', 'Faouzi Ghoulam',
    'Baghdad Bounedjah', 'Youcef Atal', 'Ismail Bennacer',
    'Djamel Belmadi', 'Rabah Madjer', 'Lakhdar Belloumi',
    'Khaled Hadj Ibrahim', 'Cheb Mami', 'Faudel', 'Idir',
    'Souad Massi', 'Rachid Taha', 'Soolking', 'Lotfi Double Kanon',
    'Warda Al Jazairia', 'Cheikha Rimitti',
    'Mohamed Fellag', 'Biyouna', 'Lyes Salem',
    'Kateb Yacine', 'Assia Djebar', 'Yasmina Khadra', 'Mouloud Mammeri',
    'Djamila Bouhired', 'Larbi Ben Mhidi', 'Mostefa Ben Boulaïd',
    'Krim Belkacem', 'Abane Ramdane', 'Ferhat Abbas',
]

def get_photo_wikipedia(nom):
    """Télécharge la photo depuis Wikipedia"""
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": nom,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": 300
        }
        r = requests.get(search_url, params=params, timeout=10)
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {})
            if thumb and "source" in thumb:
                img_url = thumb["source"]
                img_data = requests.get(img_url, timeout=10).content
                return base64.b64encode(img_data).decode("utf-8")
    except Exception as e:
        print(f"  ⚠️ Photo non trouvée pour {nom}: {e}")
    return None

importes = 0
deja = 0

for nom in NOMS_ALGERIENS:
    existing = col.find_one({'name': nom})
    if not existing:
        print(f"📥 Import: {nom}")
        photo_b64 = get_photo_wikipedia(nom)
        doc = {
            'name': nom,
            'embeddings_insight': [],
            'embeddings_dlib': [],
            'embeddings_deepface': [],
            'source': 'Algerien',
            'nationalite': 'DZ',
            'count': 0,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M')
        }
        if photo_b64:
            doc['photo_b64'] = photo_b64
            print(f"  ✅ Photo OK")
        else:
            print(f"  ⚠️ Sans photo")
        col.insert_one(doc)
        importes += 1
        time.sleep(0.5)
    else:
        deja += 1

print(f'\nImportes: {importes} | Deja existants: {deja}')
print(f'TOTAL MongoDB: {col.count_documents({})}')
