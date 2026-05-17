from pymongo import MongoClient
import requests
import time
import base64

MONGO_URI = 'mongodb+srv://khedimbenyakhlef_db_user:5q7RcqsusonfGeSv@cluster0.logfyqe.mongodb.net/atare_db'
col = MongoClient(MONGO_URI)['atare_db']['faces']

print('Connexion OK ! Visages:', col.count_documents({}))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def get_photo_duckduckgo(nom):
    """Cherche une photo via DuckDuckGo"""
    try:
        # Étape 1 : obtenir le token vqd
        r = requests.get(
            f"https://duckduckgo.com/?q={nom.replace(' ', '+')}&iax=images&ia=images",
            headers=HEADERS, timeout=10
        )
        vqd = ""
        for line in r.text.split("\n"):
            if "vqd=" in line:
                vqd = line.split("vqd='")[1].split("'")[0] if "vqd='" in line else ""
                break

        if not vqd:
            # Essayer avec l'API directe
            r2 = requests.get(
                f"https://duckduckgo.com/i.js?q={nom.replace(' ', '+')}&o=json",
                headers=HEADERS, timeout=10
            )
            data = r2.json()
            results = data.get("results", [])
            if results:
                img_url = results[0].get("image", "")
                if img_url:
                    img = requests.get(img_url, headers=HEADERS, timeout=10)
                    if img.status_code == 200:
                        return base64.b64encode(img.content).decode("utf-8")
        return None
    except Exception as e:
        print(f"    DDG erreur: {e}")
        return None

def get_photo_wikipedia_v2(nom):
    """Wikipedia avec bons headers"""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": nom,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": 300,
            "redirects": 1
        }
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {})
            if thumb and "source" in thumb:
                img = requests.get(thumb["source"], headers=HEADERS, timeout=15)
                if img.status_code == 200:
                    return base64.b64encode(img.content).decode("utf-8")
    except Exception as e:
        print(f"    Wiki erreur: {e}")
    return None

def get_photo_wikidata(nom):
    """Wikidata SPARQL - alternative"""
    try:
        query = f"""
        SELECT ?image WHERE {{
          ?item wikibase:label {{ bd:serviceParam wikibase:language "en" }}
          ?item rdfs:label "{nom}"@en .
          ?item wdt:P18 ?image .
        }} LIMIT 1
        """
        r = requests.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=15
        )
        data = r.json()
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            img_url = bindings[0]["image"]["value"]
            img = requests.get(img_url, headers=HEADERS, timeout=15)
            if img.status_code == 200:
                return base64.b64encode(img.content).decode("utf-8")
    except Exception as e:
        print(f"    Wikidata erreur: {e}")
    return None

# Tous les visages algériens sans photo
docs_sans_photo = list(col.find({
    "nationalite": "DZ",
    "photo_b64": {"$exists": False}
}))

print(f"\n{len(docs_sans_photo)} visages sans photo à traiter\n")

mis_a_jour = 0
for doc in docs_sans_photo:
    nom = doc["name"]
    print(f"🔍 {nom}")

    photo = None

    # Essai 1 : Wikipedia
    photo = get_photo_wikipedia_v2(nom)
    if photo:
        print(f"  ✅ Wikipedia OK")

    # Essai 2 : Wikidata
    if not photo:
        photo = get_photo_wikidata(nom)
        if photo:
            print(f"  ✅ Wikidata OK")

    # Essai 3 : DuckDuckGo
    if not photo:
        photo = get_photo_duckduckgo(nom)
        if photo:
            print(f"  ✅ DuckDuckGo OK")

    if photo:
        col.update_one(
            {"_id": doc["_id"]},
            {"$set": {"photo_b64": photo}}
        )
        mis_a_jour += 1
    else:
        print(f"  ❌ Aucune photo trouvée")

    time.sleep(1)

print(f"\n✅ Photos ajoutées: {mis_a_jour} / {len(docs_sans_photo)}")
