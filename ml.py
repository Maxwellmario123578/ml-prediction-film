import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve

# Chargement du fichier Excel
df = pd.read_excel("series_afrodescendants.xlsx")

# Affichage des premières lignes
print(df.head())

# Copie du DataFrame original
df_model = df.copy()

# Encodage de la variable cible : Oui → 1, Non → 0
df_model["Plait_au_public_afro"] = df_model["Plait_au_public_afro"].map({"Oui": 1, "Non": 0})

# Encodage booléen pour 'Createur_afrodescendant'
df_model["Createur_afrodescendant"] = df_model["Createur_afrodescendant"].map({"Oui": 1, "Non": 0})

# Conversion en numérique
df_model["Note_moyenne_afro"] = pd.to_numeric(df_model["Note_moyenne_afro"], errors='coerce')

# Encodage texte de 'Popularite_reseaux' (faible < moyenne < élevée)
pop_map = {"Faible": 0, "Moyenne": 1, "Élevée": 2}
df_model["Popularite_reseaux"] = df_model["Popularite_reseaux"].map(pop_map)

# Colonnes catégorielles à encoder avec one-hot encoding
cat_cols = ["Genre", "Langue", "Pays_origine", "Plateforme", "Recompenses"]
df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

# Supprimer les colonnes non exploitables pour le modèle
df_model.drop(columns=["Titre", "Themes_culturels_afro"], inplace=True)

# Affichage de la structure finale
print(df_model.head())

# Séparation des variables explicatives (X) et cible (y)
X = df_model.drop("Plait_au_public_afro", axis=1)
y = df_model["Plait_au_public_afro"]

# Calcul corrélation + p-value
corr_results = []
for col in X.columns:
    corr, pval = pearsonr(X[col], y)
    corr_results.append({
        'Feature': col,
        'Correlation': corr,
        'p_value': pval
    })

# Conversion en DataFrame et tri
corr_df = pd.DataFrame(corr_results)
corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)

# Nouvelle colonne : pertinence
corr_df["Pertinent"] = np.where(corr_df["p_value"] < 0.001, "Oui", "Non")

print("\nCorrélations avec la variable cible :")
print(corr_df)

# Split du dataset en train/test (90% / 10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Création du modèle
model = LogisticRegression(max_iter=1000)

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Sauvegarder le modèle entraîné
joblib.dump(model, 'logistic_model.pkl')
print("Modèle sauvegardé sous logistic_model.pkl")

# Évaluation
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Distribution des probabilités prédites pour la classe positive
y_proba = model.predict_proba(X_test)[:, 1]

# DataFrame pour visualiser
proba_df = pd.DataFrame({
    'Probabilité': y_proba,
    'Vraie_classe': y_test.values
})

plt.figure(figsize=(8, 5))
sns.histplot(
    data=proba_df,
    x='Probabilité',
    hue='Vraie_classe',
    kde=True,
    stat="density",
    common_norm=False
)
plt.title("Distribution des probabilités prédites par le modèle")
plt.xlabel("Probabilité prédite (classe positive)")
plt.ylabel("Densité")
plt.show()

# Importance des variables dans le modèle
coefficients = model.coef_[0]
feature_names = X.columns
importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
importance = importance.reindex(importance['Coefficient'].abs().sort_values(ascending=False).index)

print("\nVariables les plus influentes :")
print(importance.head(10))

# Validation croisée pour log loss et calibration
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_losses = []
y_proba_cv = np.zeros(len(y))

for train_idx, test_idx in cv.split(X, y):
    X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
    y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
    
    model_cv = LogisticRegression(max_iter=1000)
    model_cv.fit(X_train_cv, y_train_cv)
    
    proba_pred = model_cv.predict_proba(X_test_cv)[:, 1]
    y_proba_cv[test_idx] = proba_pred
    
    # Log loss pour ce pli
    from sklearn.metrics import log_loss
    log_losses.append(log_loss(y_test_cv, proba_pred))

print(f"\nLog Loss (validation croisée) : {np.mean(log_losses):.4f} ± {np.std(log_losses):.4f}")
print(f"Log Loss sur tout le dataset : {log_loss(y, y_proba_cv):.4f}")

# Calcul de la courbe de calibration
prob_true, prob_pred = calibration_curve(y, y_proba_cv, n_bins=10)

# Tracé de la courbe de calibration
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Modèle calibré')
plt.plot([0, 1], [0, 1], linestyle='--', label='Référence parfaite')
plt.title("Courbe de calibration (validation croisée manuelle)")
plt.xlabel("Probabilité prédite moyenne")
plt.ylabel("Fréquence vraie")
plt.legend()
plt.grid()
plt.show()
