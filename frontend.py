import dash
from dash import dcc, html, Input, Output, State
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
import threading
from datetime import datetime

# Initialisation de l'application
app = dash.Dash(__name__)

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
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
    print("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")

# Initialiser Wikipédia
wiki = wikipediaapi.Wikipedia('en', headers={'User-Agent': 'MyApp/1.0 (contact@example.com)'})

# Charger le modèle entraîné
try:
    model = joblib.load('logistic_model.pkl')
except:
    model = None
    print("Modèle non trouvé. Veuillez exécuter ml.py d'abord.")

# Styles CSS centralisés
styles = {
    'container': {
        'maxWidth': '800px',
        'margin': '0 auto',
        'padding': '20px',
        'fontFamily': "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        'backgroundColor': '#f8f9fa',
        'minHeight': '100vh'
    },
    'card': {
        'backgroundColor': 'white',
        'borderRadius': '12px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
        'padding': '25px',
        'margin': '20px 0'
    },
    'title': {
        'textAlign': 'center',
        'color': '#007bff',
        'marginBottom': '25px',
        'fontWeight': 'bold',
        'fontSize': '2.2rem'
    },
    'label': {
        'fontWeight': 'bold',
        'marginBottom': '8px',
        'color': '#333',
        'fontSize': '1rem'
    },
    'input': {
        'width': '100%',
        'padding': '12px',
        'border': '2px solid #e9ecef',
        'borderRadius': '8px',
        'fontSize': '1rem',
        'marginBottom': '20px'
    },
    'button': {
        'width': '100%',
        'padding': '14px',
        'backgroundColor': '#007bff',
        'color': 'white',
        'border': 'none',
        'borderRadius': '8px',
        'fontSize': '1.1rem',
        'fontWeight': 'bold',
        'cursor': 'pointer',
        'transition': 'all 0.3s ease'
    },
    'result': {
        'borderRadius': '8px',
        'padding': '15px',
        'marginTop': '20px',
        'fontSize': '1rem'
    },
    'success': {
        'backgroundColor': '#d4edda',
        'border': '1px solid #c3e6cb',
        'color': '#155724'
    },
    'error': {
        'backgroundColor': '#f8d7da',
        'border': '1px solid #f5c6cb',
        'color': '#721c24'
    },
    'footer': {
        'textAlign': 'center',
        'marginTop': '40px',
        'color': '#6c757d',
        'fontSize': '0.9rem'
    },
    'progress': {
        'width': '100%',
        'backgroundColor': '#e9ecef',
        'borderRadius': '10px',
        'height': '20px',
        'margin': '10px 0'
    },
    'progressBar': {
        'height': '100%',
        'backgroundColor': '#007bff',
        'borderRadius': '10px',
        'textAlign': 'center',
        'color': 'white',
        'lineHeight': '20px',
        'transition': 'width 0.3s ease'
    }
}

# Layout
app.layout = html.Div([
    html.H1("🎬 Recherche Film/Série", style=styles['title']),

    html.Div([
        html.H3("📋 Informations de recherche",
                style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#495057'}),

        # Champ ID
        html.Label("ID TMDB:", style=styles['label']),
        dcc.Input(
            id="input-id",
            type="text",
            placeholder="Entrez l'ID TMDB (ex: 12345)",
            style=styles['input']
        ),

        # Sélecteur catégorie
        html.Label("Catégorie:", style=styles['label']),
        dcc.RadioItems(
            id="category-selector",
            options=[
                {'label': ' 🎬 Film', 'value': 'movie'},
                {'label': ' 📺 Série/TV', 'value': 'tv'}
            ],
            value='movie',
            inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
            labelStyle={'display': 'block', 'marginBottom': '10px'}
        ),

        # Bouton recherche
        html.Button("🔍 Rechercher", id="submit-button", n_clicks=0, style=styles['button'])
    ], style=styles['card']),

    # Barre de progression
    html.Div(id="progress-container", style={'display': 'none'}, children=[
        html.Div("Récupération des données...", id="progress-text", style={'textAlign': 'center', 'marginBottom': '10px'}),
        html.Div(style=styles['progress'], children=[
            html.Div(id="progress-bar", style={**styles['progressBar'], 'width': '0%'})
        ])
    ]),

    # Zone résultat
    html.Div(id="result-output"),

    # Footer
    html.Div([
        html.Hr(),
        html.P("Application de recherche Film/Série - Interface responsive")
    ], style=styles['footer']),
    
    # Stockage des données de session
    dcc.Store(id='session-data', data={}),
    dcc.Interval(id='progress-update', interval=1000, n_intervals=0, disabled=True)
], style=styles['container'])

# Fonctions backend intégrées
def clean_keywords(keywords):
    cleaned = []
    for keyword in keywords:
        keyword = re.sub(r'[^\w\s]', ' ', keyword.lower()).replace('&', 'and').strip()
        keyword = ' '.join(keyword.split()[:2])
        if keyword and re.search(r'[a-zA-Z0-9]', keyword):
            cleaned.append(keyword)
    return list(dict.fromkeys(cleaned))[:5]

def get_short_title(title):
    if len(title) <= 50:
        return title
    if nlp:
        doc = nlp(title)
        significant_words = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and token.text.lower() not in ['the', 'a', 'an', 'of', 'and']]
        short_title = ' '.join(significant_words[:3])
        return short_title if short_title else title[:50]
    return title[:50]

def get_reddit_popularity(title, keywords):
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                             client_secret=REDDIT_CLIENT_SECRET,
                             user_agent=REDDIT_USER_AGENT)
        
        subreddits = ['BlackFilmmakers', 'movies', 'askblackpeople', 'blackladies']
        engagement_score = 0
        total_posts = 0
        total_comments = 0
        
        cleaned_keywords = clean_keywords(keywords)
        short_title = get_short_title(title)
        filter_keywords = cleaned_keywords + [short_title.lower()]
        query = f"{short_title} {' OR '.join(cleaned_keywords)}" if cleaned_keywords else short_title
        
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
        
        if engagement_score > 10000000:
            return "Élevée", engagement_score
        elif engagement_score > 1000000:
            return "Moyenne", engagement_score
        else:
            return "Faible", engagement_score
    except Exception as e:
        print(f"Erreur lors de la récupération de la popularité Reddit : {e}")
        return "Faible", 0

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
            acteurs = [actor['name'] for actor in acteurs[:10] if actor.get('name')]
            
            endpoint_info = f"tv/{tmdb_id}"
        else:
            endpoint_info = f"movie/{tmdb_id}"
            url = f"https://api.themoviedb.org/3/{endpoint_info}/credits?api_key={API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            data_credits = response.json()
            acteurs = [actor['name'] for actor in data_credits.get('cast', [])[:10] if actor.get('name')]
        
        url_info = f"https://api.themoviedb.org/3/{endpoint_info}?api_key={API_KEY}"
        response = requests.get(url_info)
        response.raise_for_status()
        data_info = response.json()
        
        genres = [genre['name'] for genre in data_info.get('genres', [])]
        vote_average = data_info.get('vote_average', None)
        vote_count = data_info.get('vote_count', None)
        title = data_info.get('name' if category.lower() == 'tv' else 'title', 'Unknown')
        language = data_info.get('original_language', 'Unknown')

        if category.lower() == 'tv':
            runtime = None
            episode_count = data_info.get('number_of_episodes', 0)
            episode_runtime = data_info.get('episode_run_time', [0])[0] if data_info.get('episode_run_time') else 0
            country = data_info.get('origin_country', ['Unknown'])[0]
            release_year = data_info.get('first_air_date', '0000')[:4]
            createurs = [creator['name'] for creator in data_info.get('created_by', [])]
        else:
            runtime = data_info.get('runtime', 0)
            episode_count = None
            episode_runtime = None
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
            if nlp:
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
        return list(dict.fromkeys(all_platforms))
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

def prepare_new_series(data_dict):
    new_series = pd.DataFrame([data_dict])
    new_series["Createur_afrodescendant"] = new_series["Createur_afrodescendant"].map({"Oui": 1, "Non": 0})
    new_series["Popularite_reseaux"] = new_series["Popularite_reseaux"].map({"Faible": 0, "Moyenne": 1, "Élevée": 2})
    cat_cols = ["Genre", "Langue", "Pays_origine", "Plateforme", "Recompenses"]
    new_series = pd.get_dummies(new_series, columns=cat_cols, drop_first=True)
    
    # Charger le dataset original pour s'assurer d'avoir les mêmes colonnes
    try:
        df = pd.read_excel("series_afrodescendants.xlsx")
        df["Createur_afrodescendant"] = df["Createur_afrodescendant"].map({"Oui": 1, "Non": 0})
        df["Popularite_reseaux"] = df["Popularite_reseaux"].map({"Faible": 0, "Moyenne": 1, "Élevée": 2})
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        missing_cols = set(df.columns) - set(new_series.columns) - {'Titre', 'Themes_culturels_afro', 'Plait_au_public_afro'}
        for col in missing_cols:
            new_series[col] = 0
        new_series = new_series.drop(columns=['Titre', 'Themes_culturels_afro'], errors='ignore')
        new_series = new_series[df.drop(columns=['Titre', 'Themes_culturels_afro', 'Plait_au_public_afro']).columns]
    except:
        print("Avertissement: Fichier series_afrodescendants.xlsx non trouvé. Utilisation des colonnes disponibles.")
    
    return new_series

# Variable globale pour suivre la progression
progress_data = {
    'current': 0,
    'total': 0,
    'message': '',
    'result': None
}

# Fonction pour exécuter le traitement en arrière-plan
def process_data_in_background(tmdb_id, category):
    global progress_data
    
    try:
        progress_data['current'] = 0
        progress_data['total'] = 10
        progress_data['message'] = "Récupération des données TMDB..."
        
        # Récupérer toutes les informations
        acteurs, createurs, genres, vote_average, vote_count, title, language, country, release_year, runtime, episode_count, episode_runtime = get_tmdb_names(tmdb_id, category)
        keywords = get_keywords(tmdb_id, category)
        platforms = get_platforms(tmdb_id, category)
        
        progress_data['current'] = 2
        progress_data['message'] = f"Analyse de {title}..."
        
        if not acteurs and not createurs:
            progress_data['result'] = {
                'error': "Aucune personne récupérée depuis TMDB. Vérifiez l'ID ou la catégorie."
            }
            return
        
        progress_data['current'] = 3
        progress_data['message'] = f"Analyse des ethnies des {len(acteurs) + len(createurs)} personnes..."
        progress_data['total'] = 3 + len(acteurs) + len(createurs)
        
        results = []
        for i, personne in enumerate(acteurs + createurs):
            result = get_ethnicity(personne)
            results.append(result)
            progress_data['current'] = 3 + i + 1
            progress_data['message'] = f"Analyse de {personne} ({i+1}/{len(acteurs)+len(createurs)})..."
        
        afro_percentage = calculate_afrodescendant_percentage(results, acteurs)
        creator_afro = check_creator_afrodescendant(results, createurs)
        
        progress_data['current'] = progress_data['total'] + 1
        progress_data['message'] = "Analyse de la popularité sur Reddit..."
        progress_data['total'] += 3
        
        social_popularity, engagement_score = get_reddit_popularity(title, keywords)
        
        completion_rate = calculate_completion_rate(
            engagement_score, 
            runtime, 
            episode_count, 
            episode_runtime, 
            category
        )
        
        progress_data['current'] = progress_data['total']
        progress_data['message'] = "Préparation des données pour la prédiction..."
        
        result_data = {
            'Titre': title,
            'Acteurs_afrodescendants_%': afro_percentage,
            'Createur_afrodescendant': creator_afro,
            'Themes_culturels_afro': ', '.join(keywords),
            'Genre': genres[0] if genres else 'Unknown',
            'Langue': language,
            'Pays_origine': country,
            'Annee_sortie': int(release_year) if release_year.isdigit() else 0,
            'Plateforme': platforms[0] if platforms else 'Unknown',
            'Saisons': 0,
            'Note_moyenne_afro': vote_average if vote_average is not None else 0.0,
            'Popularite_reseaux': social_popularity,
            'Taux_completion_%': completion_rate,
            'Budget_millions': 4,
            'Recompenses': ''
        }
        
        X_new = prepare_new_series(result_data)
        
        if model is not None:
            prediction = model.predict(X_new)
            probability = model.predict_proba(X_new)[:, 1]
            result_data['Prediction'] = 'Oui' if prediction[0] == 1 else 'Non'
            result_data['Probabilite'] = probability[0]
        else:
            result_data['Prediction'] = 'Modèle non disponible'
            result_data['Probabilite'] = 0
        
        progress_data['result'] = result_data
        progress_data['message'] = "Analyse terminée!"
        
    except Exception as e:
        progress_data['result'] = {
            'error': f"Erreur lors du traitement: {str(e)}"
        }

# Callbacks
@app.callback(
    [Output('progress-container', 'style'),
     Output('progress-update', 'disabled'),
     Output('session-data', 'data')],
    Input('submit-button', 'n_clicks'),
    [State('input-id', 'value'),
     State('category-selector', 'value')],
    prevent_initial_call=True
)
def start_processing(n_clicks, input_id, category):
    if not input_id or not input_id.strip():
        return {'display': 'none'}, True, {}
    
    # Réinitialiser les données de progression
    global progress_data
    progress_data = {
        'current': 0,
        'total': 10,
        'message': 'Démarrage du traitement...',
        'result': None
    }
    
    # Démarrer le traitement en arrière-plan
    thread = threading.Thread(target=process_data_in_background, args=(input_id.strip(), category))
    thread.daemon = True
    thread.start()
    
    # Afficher la barre de progression
    return {'display': 'block'}, False, {'start_time': datetime.now().isoformat()}

@app.callback(
    [Output('progress-bar', 'style'),
     Output('progress-text', 'children'),
     Output('result-output', 'children')],
    [Input('progress-update', 'n_intervals'),
     Input('session-data', 'data')]
)
def update_progress(n_intervals, session_data):
    global progress_data
    
    # Calculer le pourcentage de progression
    if progress_data['total'] > 0:
        progress_percent = min(100, int((progress_data['current'] / progress_data['total']) * 100))
    else:
        progress_percent = 0
    
    progress_style = {**styles['progressBar'], 'width': f'{progress_percent}%'}
    progress_text = f"{progress_data['message']} ({progress_percent}%)"
    
    # Vérifier si le traitement est terminé
    result_output = None
    if progress_data['result'] is not None:
        if 'error' in progress_data['result']:
            result_output = html.Div([
                html.H4("❌ Erreur", style={'marginBottom': '10px'}),
                html.P(progress_data['result']['error'])
            ], style={**styles['result'], **styles['error']})
        else:
            result_data = progress_data['result']
            prediction_color = '#28a745' if result_data['Prediction'] == 'Oui' else '#dc3545'
            
            result_output = html.Div([
                html.H4("✅ Analyse terminée!", style={'marginBottom': '15px'}),
                html.H5(f"Titre: {result_data['Titre']}", style={'marginBottom': '10px'}),
                html.P([html.Strong("Prédiction: "), 
                       html.Span(result_data['Prediction'], 
                                style={'color': prediction_color, 'fontWeight': 'bold'})]),
                html.P([html.Strong("Probabilité: "), 
                       f"{result_data['Probabilite']*100:.2f}%"]),
                html.Hr(),
                html.H5("Détails de l'analyse:", style={'marginBottom': '10px'}),
                html.P([html.Strong("Pourcentage d'acteurs afrodescendants: "), 
                       f"{result_data['Acteurs_afrodescendants_%']:.2f}%"]),
                html.P([html.Strong("Créateur afrodescendant: "), 
                       result_data['Createur_afrodescendant']]),
                html.P([html.Strong("Popularité sur les réseaux: "), 
                       result_data['Popularite_reseaux']]),
                html.P([html.Strong("Taux de complétion estimé: "), 
                       f"{result_data['Taux_completion_%']:.2f}%"]),
                html.P([html.Strong("Genre: "), result_data['Genre']]),
                html.P([html.Strong("Langue: "), result_data['Langue']]),
                html.P([html.Strong("Pays d'origine: "), result_data['Pays_origine']]),
                html.P([html.Strong("Année de sortie: "), str(result_data['Annee_sortie'])]),
                html.P([html.Strong("Plateforme: "), result_data['Plateforme']])
            ], style={**styles['result'], **styles['success']})
    
    return progress_style, progress_text, result_output

# Responsive CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Recherche Film/Série</title>
        {%favicon%}
        {%css%}
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { box-sizing: border-box; }
            body { margin: 0; background-color: #f8f9fa; }
            
            @media (max-width: 768px) {
                h1 { font-size: 1.8rem !important; }
            }
            @media (max-width: 576px) {
                h1 { font-size: 1.6rem !important; }
                input, button { font-size: 1rem !important; padding: 10px !important; }
            }
            button:hover {
                background-color: #0056b3 !important;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            button:active { transform: translateY(0); }
            input:focus, button:focus {
                outline: 2px solid #007bff;
                outline-offset: 2px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)