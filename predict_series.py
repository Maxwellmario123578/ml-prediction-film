import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
import wikipediaapi
import spacy
import time
import re
from urllib.parse import quote
import praw 

# Clé API TMDB
API_KEY = ""

# Clés API Reddit
REDDIT_CLIENT_ID = ""
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT = ""

# Liste des pays et termes afrodescendants
afro_countries = [
    'Nigeria', 'Ghana', 'Kenya', 'South Africa', 'Ethiopia', 'Jamaica', 'Trinidad', 'Haiti', 'Barbados',
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde',
    'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea',
    'Eritrea', 'Eswatini', 'Gabon', 'Gambia', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Lesotho',
    'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique',
    'Namibia', 'Niger', 'Rwanda', 'Sao Tome', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia',
    'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe',
    'Bahamas', 'Dominica', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Antigua and Barbuda',
    'Saint Kitts and Nevis', 'Grenada'
]
afro_keywords = [
    'african', 'afro', 'black', 'african-american', 'afro-american', 'afro-caribbean', 'african descent', 
    'black descent', 'nigerian', 'ghanaian', 'kenyan', 'jamaican', 'trinidadian', 'haitian', 'barbadian',
    'afro-british', 'afro-latino', 'afro-canadian', 'born in', 'parents from', 'heritage', 'ancestry',
    'ancestors', 'cape verdean', 'brava'
]

# Initialiser spaCy pour NLP
nlp = spacy.load("en_core_web_sm")

# Initialiser Wikipédia
wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': 'MyApp/1.0 (contact@example.com)'})

# Fonction pour nettoyer les mots-clés TMDB
def clean_keywords(keywords):
    cleaned = []
    for keyword in keywords:
        # Remplacer & par "and", supprimer caractères spéciaux, limiter à 2 mots
        keyword = re.sub(r'[^\w\s]', ' ', keyword.lower()).replace('&', 'and').strip()
        keyword = ' '.join(keyword.split()[:2])  # Prendre les 2 premiers mots
        if keyword and re.search(r'[a-zA-Z0-9]', keyword):  # Vérifier qu'il y a des lettres/chiffres
            cleaned.append(keyword)
    return list(dict.fromkeys(cleaned))[:5]  # Limiter à 5 mots-clés uniques

# Fonction pour obtenir une version courte du titre
def get_short_title(title):
    if len(title) <= 50:
        return title
    doc = nlp(title)
    significant_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and token.text.lower() not in ['the', 'a', 'an', 'of', 'and']]
    short_title = ' '.join(significant_words[:3])  # Prendre jusqu'à 3 mots significatifs
    return short_title if short_title else title[:50]

# Fonction pour estimer la popularité sur Reddit (MODIFIÉE)
def get_reddit_popularity(title, keywords):
    try:
        # Authentification Reddit
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                             client_secret=REDDIT_CLIENT_SECRET,
                             user_agent=REDDIT_USER_AGENT)
        
        # Subreddits à scraper
        subreddits = ['BlackFilmmakers', 'movies', 'askblackpeople', 'blackladies']
        engagement_score = 0
        total_posts = 0
        total_comments = 0
        
        # Nettoyer les mots-clés et inclure le titre court
        cleaned_keywords = clean_keywords(keywords)
        short_title = get_short_title(title)
        filter_keywords = cleaned_keywords + [short_title.lower()]
        query = f"{short_title} {' OR '.join(cleaned_keywords)}" if cleaned_keywords else short_title
        
        # Recherche dans chaque subreddit
        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            posts_found = 0
            comments_found = 0
            subreddit_score = 0
            
            for submission in subreddit.search(query, sort='relevance', limit=10):
                if any(keyword in submission.title.lower() or keyword in submission.selftext.lower() 
                       for keyword in filter_keywords):
                    subreddit_score += submission.score
                    posts_found += 1
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments:
                        if any(keyword in comment.body.lower() for keyword in filter_keywords):
                            subreddit_score += comment.score
                            comments_found += 1
            
            engagement_score += subreddit_score
            total_posts += posts_found
            total_comments += comments_found
            print(f"Subreddit {subreddit_name} : {posts_found} posts pertinents, {comments_found} commentaires pertinents, score = {subreddit_score}")
        
        print(f"Requête utilisée : {query}")
        print(f"Mots-clés de filtrage : {filter_keywords}")
        print(f"Score d'engagement total pour '{title}' : {engagement_score} ({total_posts} posts, {total_comments} commentaires)")
        
        # Retourner à la fois la catégorie et le score numérique
        if engagement_score > 10000000:
            return "Élevée", engagement_score
        elif engagement_score > 1000000:
            return "Moyenne", engagement_score
        else:
            return "Faible", engagement_score
    except Exception as e:
        print(f"Erreur lors de la récupération de la popularité Reddit : {e}")
        return "Faible", 0
    
# Fonction pour récupérer les données TMDB (MODIFIÉE)
def get_tmdb_names(tmdb_id, category):
    try:
        if category.lower() not in ['movie', 'tv']:
            raise ValueError("Catégorie doit être 'movie' ou 'tv'")
        
        acteurs = []
        genres = []
        vote_average = None
        vote_count = None
        title = None
        language = None
        country = None
        release_year = None
        runtime = None
        episode_count = None
        episode_runtime = None
        
        if category.lower() == 'tv':
            endpoint = f"tv/{tmdb_id}/aggregate_credits"
            url = f"https://api.themoviedb.org/3/{endpoint}?api_key={API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            acteurs = sorted(
                data.get('cast', []),
                key=lambda x: x.get('total_episode_count', 0),
                reverse=True
            )
            acteurs = [actor['name'] for actor in acteurs[:10] if actor.get('name')]  # Limiter à 10 acteurs
            
            endpoint_info = f"tv/{tmdb_id}"
        else:  # category == 'movie'
            endpoint_info = f"movie/{tmdb_id}"
            url = f"https://api.themoviedb.org/3/{endpoint_info}/credits?api_key={API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data_credits = response.json()
            acteurs = [actor['name'] for actor in data_credits.get('cast', [])[:10] if actor.get('name')]  # Limiter à 10 acteurs
        
        url_info = f"https://api.themoviedb.org/3/{endpoint_info}?api_key={API_KEY}"
        response = requests.get(url_info)
        response.raise_for_status()
        data_info = response.json()
        
        genres = [genre['name'] for genre in data_info.get('genres', [])]
        vote_average = data_info.get('vote_average', None)
        vote_count = data_info.get('vote_count', None)
        title = data_info.get('name' if category.lower() == 'tv' else 'title', 'Unknown')
        language = data_info.get('original_language', 'Unknown')

        # Récupération des informations de durée
        if category.lower() == 'tv':
            runtime = None  # Les séries n'ont pas de runtime global
            episode_count = data_info.get('number_of_episodes', 0)
            episode_runtime = data_info.get('episode_run_time', [0])[0] if data_info.get('episode_run_time') else 0
        else:
            runtime = data_info.get('runtime', 0)
            episode_count = None
            episode_runtime = None

        if category.lower() == 'tv':
            country = data_info.get('origin_country', ['Unknown'])[0]
            release_year = data_info.get('first_air_date', '0000')[:4]
            createurs = [creator['name'] for creator in data_info.get('created_by', [])]
        else:
            country = data_info.get('production_countries', [{'name': 'Unknown'}])[0]['name']
            release_year = data_info.get('release_date', '0000')[:4]
            createurs = [crew['name'] for crew in data_info.get('credits', {}).get('crew', []) if crew.get('job') == 'Director']
        
        return acteurs, createurs, genres, vote_average, vote_count, title, language, country, release_year, runtime, episode_count, episode_runtime
    except Exception as e:
        print(f"Erreur TMDB : {e}")
        return [], [], [], None, None, 'Unknown', 'Unknown', 'Unknown', '0000', None, None, None

def get_ethnicity(name):
    try:
        url = f"https://en.wikipedia.org/wiki/{quote(name)}"
        response = requests.get(url, headers={'User-Agent': 'MyApp/1.0 (contact@example.com)'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup.find_all(string=re.compile(r'\[\d+\]')):
            tag.replace_with(re.sub(r'\[\d+\]', '', tag))

        infobox = soup.find('table', {'class': 'infobox'})
        if infobox:
            for field in infobox.find_all('tr'):
                header = field.find('th')
                value = field.find('td')
                if header and value and ('born' in header.text.lower() or 'nationality' in header.text.lower() or 'ethnicity' in header.text.lower()):
                    text = value.text.lower()
                    text = re.sub(r'\[\d+\]', '', text)
                    if any(keyword in text for keyword in afro_keywords) or any(country.lower() in text for country in afro_countries):
                        return f"{name}: Afrodescendant (Infobox) - {text[:200]}..."

        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.text.lower()
            text = re.sub(r'\[\d+\]', '', text)
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'GPE' and ent.text.lower() in [c.lower() for c in afro_countries]:
                    for token in doc:
                        if token.text.lower() in ['ancestors', 'heritage', 'parents', 'family'] and ent.text.lower() in text:
                            return f"{name}: Afrodescendant (NLP - {ent.text}) - {text[:200]}..."
            if any(keyword in text for keyword in afro_keywords):
                return f"{name}: Afrodescendant (Contenu - Keywords) - {text[:200]}..."

        url = f"https://ethnicelebs.com/{name.lower().replace(' ', '-')}"
        response = requests.get(url, headers={'User-Agent': 'MyApp/1.0 (contact@example.com)'})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('div', {'class': 'entry-content'})
            if content:
                text = re.sub(r'\[\d+\]', '', content.text.lower())
                if any(keyword in text for keyword in afro_keywords) or any(country.lower() in text for country in afro_countries):
                    return f"{name}: Afrodescendant (Ethnicelebs) - {text[:200]}..."

        return f"{name}: Non afrodescendant (ou non précisé)"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"{name}: Non afrodescendant (ou non précisé, page non trouvée)"
        return f"{name}: Erreur réseau - {e}"
    except Exception as e:
        return f"{name}: Erreur générale - {e}"
    finally:
        time.sleep(1)

def calculate_afrodescendant_percentage(results, acteurs):
    total = len([r for r in results if r.split(':')[0] in acteurs and not r.startswith('Erreur')])
    afrodescendant_count = len([r for r in results if r.split(':')[0] in acteurs and 'Afrodescendant' in r])
    if total > 0:
        percentage = (afrodescendant_count / total) * 100
        return percentage
    return 0.0

def check_creator_afrodescendant(results, createurs):
    for result in results:
        name = result.split(':')[0]
        if name in createurs:
            return 'Oui' if 'Afrodescendant' in result else 'Non'
    return 'Non'

def get_keywords(tmdb_id, category):
    try:
        if category.lower() == "tv":
            url = f"https://api.themoviedb.org/3/tv/{tmdb_id}/keywords?api_key={API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return [kw['name'] for kw in data.get('results', [])]
        elif category.lower() == "movie":
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/keywords?api_key={API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return [kw['name'] for kw in data.get('keywords', [])]
        else:
            raise ValueError("Catégorie doit être 'movie' ou 'tv'")
    except Exception as e:
        print(f"Erreur récupération keywords : {e}")
        return []

def get_platforms(tmdb_id, category, country_codes=["FR", "US"]):
    try:
        result = {}
        if category.lower() == "tv":
            url_info = f"https://api.themoviedb.org/3/tv/{tmdb_id}?api_key={API_KEY}"
            response = requests.get(url_info)
            response.raise_for_status()
            data_info = response.json()
            result["Original Network"] = [n['name'] for n in data_info.get('networks', [])]
            url_providers = f"https://api.themoviedb.org/3/tv/{tmdb_id}/watch/providers?api_key={API_KEY}"
            response = requests.get(url_providers)
            response.raise_for_status()
            data_providers = response.json()
            for code in country_codes:
                if code in data_providers.get("results", {}):
                    providers = [p['provider_name'] for p in data_providers["results"][code].get("flatrate", [])]
                    if providers:
                        result[code] = providers
        elif category.lower() == "movie":
            url_providers = f"https://api.themoviedb.org/3/movie/{tmdb_id}/watch/providers?api_key={API_KEY}"
            response = requests.get(url_providers)
            response.raise_for_status()
            data_providers = response.json()
            for code in country_codes:
                if code in data_providers.get("results", {}):
                    providers = [p['provider_name'] for p in data_providers["results"][code].get("flatrate", [])]
                    if providers:
                        result[code] = providers
        all_platforms = []
        for plats in result.values():
            all_platforms.extend(plats)
        return list(dict.fromkeys(all_platforms))  # Supprimer doublons
    except Exception as e:
        print(f"Erreur récupération plateformes : {e}")
        return []

def calculate_completion_rate(engagement_score, runtime, episode_count, episode_runtime, category):
    if (category.lower() == 'movie' and (runtime is None or runtime == 0)) or \
       (category.lower() == 'tv' and (episode_count is None or episode_count == 0 or episode_runtime is None or episode_runtime == 0)):
        return 50.0
    
    if category.lower() == 'movie':
        total_duration = runtime
    else:
        total_duration = episode_count * episode_runtime
    
    max_score = 10000000
    min_score = 0
    base_completion = 30.0
    max_completion = 95.0
    
    if engagement_score >= max_score:
        return max_completion
    elif engagement_score <= min_score:
        return base_completion
    
    completion_rate = base_completion + (max_completion - base_completion) * (engagement_score / max_score)
    
    if total_duration > 0:
        duration_penalty = max(0.8, 1.0 - (total_duration / 240.0))
        completion_rate *= duration_penalty
    
    return min(max(completion_rate, 0.0), 100.0)

# Fonction fetch_tmdb_data MODIFIÉE
def fetch_tmdb_data():
    tmdb_id = input("Entrez l'ID TMDB : ")
    category = input("Entrez la catégorie (movie ou tv) : ")
    
    # Récupérer toutes les informations
    acteurs, createurs, genres, vote_average, vote_count, title, language, country, release_year, runtime, episode_count, episode_runtime = get_tmdb_names(tmdb_id, category)
    keywords = get_keywords(tmdb_id, category)
    platforms = get_platforms(tmdb_id, category)
    
    if not acteurs and not createurs:
        print("Aucune personne récupérée depuis TMDB. Vérifiez l'ID ou la catégorie.")
        return None
    
    print(f"\nActeurs analysés ({len(acteurs)} acteurs) : {acteurs}")
    print(f"Créateurs analysés : {createurs}")
    
    results = []
    for personne in acteurs + createurs:
        result = get_ethnicity(personne)
        print(result)
        results.append(result)
    
    afro_percentage = calculate_afrodescendant_percentage(results, acteurs)
    creator_afro = check_creator_afrodescendant(results, createurs)
    
    # Récupérer à la fois la catégorie et le score d'engagement
    social_popularity, engagement_score = get_reddit_popularity(title, keywords)
    
    # Calculer le taux de completion
    completion_rate = calculate_completion_rate(
        engagement_score, 
        runtime, 
        episode_count, 
        episode_runtime, 
        category
    )
    
    print(f"Genre : {genres}")
    if vote_average is not None:
        print(f"Note TMDB : {vote_average}/10 ({vote_count} votes)")
    print("Plateformes :", " / ".join(platforms))
    print(f"Mots-clés : {keywords}")
    print(f"Popularité réseaux estimée (Reddit) : {social_popularity} (score: {engagement_score})")
    print(f"Taux de completion calculé : {completion_rate:.2f}%")
    print("\nRésultat final :")
    print(f"Pourcentage afrodescendant (acteurs) : {afro_percentage:.2f}%")
    print(f"Createur_afrodescendant : {creator_afro}")
    
    return {
        'Titre': title,
        'Acteurs_afrodescendants_%': afro_percentage,
        'Createur_afrodescendant': creator_afro,
        'Themes_culturels_afro': ', '.join(keywords),
        'Genre': genres[0] if genres else 'Unknown',
        'Langue': language,
        'Pays_origine': country,
        'Annee_sortie': int(release_year) if release_year.isdigit() else 0,
        'Plateforme': platforms[0] if platforms else 'Unknown',
        'Saisons': 0,  # Valeur par défaut, à ajuster si nécessaire
        'Note_moyenne_afro': vote_average if vote_average is not None else 0.0,
        'Popularite_reseaux': social_popularity,
        'Taux_completion_%': completion_rate,  # Utiliser le taux calculé
        'Budget_millions': 4,  # Valeur par défaut, à ajuster si nécessaire
        'Recompenses': ''  # Valeur par défaut, à ajuster si nécessaire
    }

# Charger le modèle entraîné
model = joblib.load('logistic_model.pkl')

# Définir une fonction pour préparer les nouvelles données
def prepare_new_series(data_dict):
    new_series = pd.DataFrame([data_dict])
    new_series["Createur_afrodescendant"] = new_series["Createur_afrodescendant"].map({"Oui": 1, "Non": 0})
    new_series["Popularite_reseaux"] = new_series["Popularite_reseaux"].map({"Faible": 0, "Moyenne": 1, "Élevée": 2})
    cat_cols = ["Genre", "Langue", "Pays_origine", "Plateforme", "Recompenses"]
    new_series = pd.get_dummies(new_series, columns=cat_cols, drop_first=True)
    df = pd.read_excel("series_afrodescendants.xlsx")
    df["Createur_afrodescendant"] = df["Createur_afrodescendant"].map({"Oui": 1, "Non": 0})
    df["Popularite_reseaux"] = df["Popularite_reseaux"].map({"Faible": 0, "Moyenne": 1, "Élevée": 2})
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    missing_cols = set(df.columns) - set(new_series.columns) - {'Titre', 'Themes_culturels_afro', 'Plait_au_public_afro'}
    for col in missing_cols:
        new_series[col] = 0
    new_series = new_series.drop(columns=['Titre', 'Themes_culturels_afro'], errors='ignore')
    new_series = new_series[df.drop(columns=['Titre', 'Themes_culturels_afro', 'Plait_au_public_afro']).columns]
    return new_series

# Fonction principale
def main():
    new_series_data = fetch_tmdb_data()
    if new_series_data is None:
        return
    X_new = prepare_new_series(new_series_data)
    prediction = model.predict(X_new)
    probability = model.predict_proba(X_new)[:, 1]
    print(f"\nPrédiction pour '{new_series_data['Titre']}': {'Oui' if prediction[0] == 1 else 'Non'}")
    print(f"Probabilité de plaire au public afro-descendant : {probability[0]:.2%}")

if __name__ == "__main__":
    main()
