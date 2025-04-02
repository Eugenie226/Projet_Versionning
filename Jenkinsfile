pipeline {
    agent any

    stages {
        stage('Cloner le repo') {
            steps {
                git 'https://github.com/Eugenie226/Projet_Versionning.git'
            }
        }

        stage('Créer et activer venv') {
            steps {
                script {
                    if (!fileExists('venv')) {
                        bat 'python -m venv venv' 
                        
                    }
                }
            }
        }

        stage('Installer les dépendances') {
            steps {
                bat 'venv\\Scripts\\python -m pip install --upgrade pip'  // Mise à jour de pip
                bat 'venv\\Scripts\\pip install -r requirements.txt'
            }
        }

        stage('Exécuter les tests') {
            steps {
                bat 'venv\\Scripts\\python -m unittest discover tests'
            }
        }

        stage('Déploiement') {
            steps {
                echo ' Déploiement en cours...'
                bat '''
                    docker-compose down
                    docker-compose up -d --build
                '''
            }
        }
    }
}
