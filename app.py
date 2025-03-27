import os  # Pour interagir avec le système de fichiers (créer des dossiers, etc.)
import uuid  # Pour générer des identifiants uniques (pour les sessions, les fichiers, etc.)
import pandas as pd  # Pour manipuler et analyser les données (lire les fichiers CSV, etc.)
import joblib  # Pour sauvegarder et charger les modèles de machine learning
import numpy as np  # Pour les opérations numériques (calculs, etc.)
import matplotlib.pyplot as plt  # Pour créer des graphiques
import seaborn as sns  # Pour des graphiques plus esthétiques
from flask import Flask, render_template, request, send_file, session  # Pour l'application web
from sklearn.model_selection import train_test_split  # Pour diviser les données en ensembles d'entraînement et de test
from sklearn.linear_model import LinearRegression  # Un modèle de régression linéaire
from sklearn.svm import SVR  # Un modèle de support vector regression
from sklearn.ensemble import RandomForestRegressor  # Un modèle de forêt aléatoire pour la régression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Pour évaluer les performances des modèles
from sklearn.impute import SimpleImputer  # Pour gérer les valeurs manquantes
from werkzeug.utils import secure_filename  # Pour sécuriser les noms de fichiers
import logging  # Pour enregistrer les erreurs et les événements importants


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.secret_key = 'clé_secrète_unique'  # Nécessaire pour les sessions
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limite à 50MB

def verifier_fichier(fichier):
    """Validation améliorée du fichier."""
    return (
        fichier and 
        '.' in fichier.filename and
        fichier.filename.rsplit('.', 1)[1].lower() == 'csv' and
        fichier.content_length <= app.config['MAX_CONTENT_LENGTH']
    )

@app.route('/')
def home():
    session['unique_id'] = str(uuid.uuid4())[:8]  # ID unique par session
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            raise ValueError("Veuillez sélectionner un fichier.")
        
        fichier = request.files['file']
        
        if not verifier_fichier(fichier):
            raise ValueError("Fichier CSV valide requis (max 50MB).")
        
        nom_securise = f"{session['unique_id']}_{secure_filename(fichier.filename)}"
        chemin_fichier = os.path.join(app.config['UPLOAD_FOLDER'], nom_securise)
        fichier.save(chemin_fichier)
        
        df = pd.read_csv(chemin_fichier)
        if df.empty or len(df.columns) < 2:
            raise ValueError("Fichier CSV structurellement invalide.")
            
        return render_template('index.html', filename=nom_securise, tables=[df.head().to_html(classes='data', index=False)])

    except Exception as e:
        logging.error(f"Erreur lors du téléchargement : {e}")
        return render_template('erreur.html', message=str(e)), 400

@app.route('/train', methods=['POST'])
def train():
    try:
        filename = request.form['filename']
        model_choice = request.form['model']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError("Fichier introuvable.")
        
        df = pd.read_csv(filepath)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
            df['jours_ecoules'] = (df['date'] - df['date'].min()).dt.days
            df = df.drop(columns=['date'])
        
        if df.isnull().values.any():
            df = df.fillna(df.mean(numeric_only=True))
        
        df = pd.get_dummies(df)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "linear_regression": LinearRegression(),
            "svm": SVR(),
            "random_forest": RandomForestRegressor(n_estimators=100)
        }
        model = models.get(model_choice, LinearRegression())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metriques = {
            'mse': round(mean_squared_error(y_test, y_pred), 4),
            'mae': round(mean_absolute_error(y_test, y_pred), 4),
            'r2': round(r2_score(y_test, y_pred), 4)
        }
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', color='red')  # Ligne de référence
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Prédictions")
        plot_path = os.path.join(STATIC_FOLDER, f"plot_{session['unique_id']}.png")
        plt.savefig(plot_path)
        model_path = os.path.join(MODEL_FOLDER, f"model_{session['unique_id']}.pkl")
        joblib.dump(model, model_path)
        return render_template("result.html", model=model_choice, metriques=metriques, plot_url=plot_path, model_path=model_path)

    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement : {e}")
        return render_template('erreur.html', message=str(e)), 500

@app.route('/download_model')
def download_model():
    """Route pour télécharger le modèle entraîné."""
    try:
        # Récupère le chemin du modèle à partir des arguments de la requête
        model_path = request.args.get('model_path')
        
        # Vérifie si le chemin du modèle est fourni et si le fichier existe
        if model_path and os.path.exists(model_path):
            # Envoie le fichier en tant que pièce jointe
            return send_file(model_path, as_attachment=True)
        
        # Si le modèle n'est pas disponible, lève une exception
        raise ValueError("Modèle non disponible.")
    
    except Exception as e:
        # Enregistre l'erreur et affiche un message d'erreur à l'utilisateur
        logging.error(f"Erreur lors du téléchargement du modèle : {e}")
        return render_template('erreur.html', message=str(e)), 404

if __name__ == '__main__':
    app.run(debug=False) # Désactiver le mode debug en production.