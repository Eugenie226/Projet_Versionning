'''import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Vérification des extensions

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Upload et affichage des données
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier trouvé."
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Nom de fichier invalide ou format non supporté."
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    df = pd.read_csv(filepath)
    return render_template('index.html', filename=file.filename, tables=[df.head().to_html(classes='data')], titles=df.columns.values)

# Entraînement du modèle
@app.route('/train', methods=['POST'])
def train():
    filename = request.form['filename']
    model_choice = request.form['model']
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # Vérification de la colonne date et conversion
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_numeric'] = (df['date'] - df['date'].min()).dt.days
        df.drop(columns=['date'], inplace=True)
    
    # Encodage des variables catégoriques
    df = pd.get_dummies(df)
    
    # Supposons que la dernière colonne est la cible (y) et les autres sont les features (X)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Gestion des valeurs manquantes
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Sélection du modèle
    models = {
        "linear_regression": LinearRegression(),
        "svm": SVR(),
        "random_forest": RandomForestRegressor()
    }
    model = models.get(model_choice, LinearRegression())
    
    # Entraînement
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Sauvegarde du modèle
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', f'{model_choice}.pkl')
    joblib.dump(model, model_path)
    
    # Génération d'un graphique
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title(f"Prédictions vs Réalité ({model_choice})")
    plot_path = "static/plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    return render_template("result.html", model=model_choice, mse=round(mse, 4), mae=round(mae, 4), r2=round(r2, 4), plot_url=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
'''

import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, send_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier trouvé."
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Fichier non valide."
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    df = pd.read_csv(filepath)
    return render_template('index.html', filename=file.filename, tables=[df.head().to_html(classes='data')], titles=df.columns.values)

@app.route('/train', methods=['POST'])
def train():
    filename = request.form['filename']
    model_choice = request.form['model']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_numeric'] = (df['date'] - df['date'].min()).dt.days
        df = df.drop(columns=['date'])
    
    df = pd.get_dummies(df)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "linear_regression": LinearRegression(),
        "svm": SVR(),
        "random_forest": RandomForestRegressor()
    }
    model = models.get(model_choice, LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title(f"Prédictions vs Réalité ({model_choice})")
    plot_path = "static/plot.png"
    plt.savefig(plot_path)
    
    model_path = os.path.join(MODEL_FOLDER, 'model.pkl')
    joblib.dump(model, model_path)
    
    return render_template("result.html", model=model_choice, mse=round(mse, 4), mae=round(mae, 4), r2=round(r2, 4), plot_url=plot_path, model_path=model_path)

@app.route('/download_model')
def download_model():
    model_path = os.path.join(MODEL_FOLDER, 'model.pkl')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    return "Aucun modèle disponible."

if __name__ == '__main__':
    app.run(debug=True)
