import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from preprocessing import preprocess_data
from models import get_model

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialisation de l'application Flask
app = Flask(__name__)

# Définition des dossiers pour stocker les fichiers
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
STATIC_FOLDER = 'static'

# Création des dossiers si inexistants
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Limite de 5MB pour les fichiers uploadés

# Vérification de l'extension du fichier
def allowed_file(filename):
    return filename.lower().endswith('.csv')

# Route principale
@app.route('/')
def home():
    return render_template('index.html')

# Téléchargement et aperçu du fichier CSV
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return "Fichier non valide.", 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    try:
        df = pd.read_csv(filepath)
        return render_template('index.html', filename=filename, tables=[df.head().to_html(classes='data')], titles=df.columns.values)
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier : {e}")
        return "Erreur lors du traitement du fichier.", 500

# Entraînement du modèle
@app.route('/train', methods=['POST'])
def train():
    filename = request.form.get('filename')
    model_choice = request.form.get('model')
    
    if not filename or not model_choice:
        return "Paramètres manquants.", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return "Fichier introuvable.", 404
    
    try:
        df = pd.read_csv(filepath)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        model = get_model(model_choice)
        model.fit(X_train, y_train)
        
        # Évaluation du modèle
        y_pred = model.predict(X_test)
        metrics = {
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }
        
        # Génération du graphique
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.xlabel("Valeurs réelles")
        plt.ylabel("Prédictions")
        plt.title(f"Prédictions vs Réalité ({model_choice})")
        plot_path = os.path.join(STATIC_FOLDER, 'plot.png')
        plt.savefig(plot_path)
        
        # Sauvegarde du modèle
        model_path = os.path.join(MODEL_FOLDER, 'model.pkl')
        joblib.dump(model, model_path)
        
        return render_template("result.html", model=model_choice, mse=round(metrics["MSE"], 4), 
                               mae=round(metrics["MAE"], 4), r2=round(metrics["R2"], 4), 
                               plot_url=plot_path, model_path='model.pkl')
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement du modèle : {e}")
        return "Erreur lors de l'entraînement du modèle.", 500

# Téléchargement du modèle sauvegardé
@app.route('/download_model')
def download_model():
    model_path = os.path.join(MODEL_FOLDER, 'model.pkl')
    return send_file(model_path, as_attachment=True) if os.path.exists(model_path) else "Aucun modèle disponible.", 404

# Lancement de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)