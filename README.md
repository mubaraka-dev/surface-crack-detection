# Surface Crack Detection

Projet académique de Master 1 en traitement d'images
et apprentissage automatique.

## Objectif

Classifier des images de routes en deux catégories :

- `crack` : présence d'une fissure de type D00, D10 ou D20 ;
- `no_crack` : absence de ces types dans les annotations.

Cette classification ne constitue pas une détection
avec localisation des fissures.

## Dataset

RDD2022, pays Japan et India.

| Ensemble | Images |
|---|---:|
| Entraînement | 12 748 |
| Validation | 2 732 |
| Test | 2 732 |
| Total | 18 212 |

Les images brutes ne sont pas incluses dans Git.

## Modèles

- HOG + SVM
- CNN personnalisé
- MobileNetV2 avec transfert d'apprentissage

## Résultats sur le test

| Modèle | Accuracy | F1-score | ROC-AUC |
|---|---:|---:|---:|
| CNN personnalisé | 0.7277 | 0.7330 | 0.8076 |
| MobileNetV2 | 0.7174 | 0.7239 | 0.7904 |
| HOG + SVM | 0.6223 | 0.6497 | 0.6675 |

Le CNN personnalisé présente les meilleures performances
observées sur le jeu de test.

## MLOps

- Git et GitHub : versionnement du code
- DVC : suivi des artefacts ML
- GitHub Actions : tests et intégration continue
- MLflow : suivi des expériences (étape suivante)
- FastAPI : service d'inférence (étape suivante)
- Docker : conteneurisation (étape suivante)

Les fichiers `.dvc` référencent les modèles.
Un stockage distant DVC devra être configuré pour
permettre leur récupération depuis un nouvel environnement.

## Organisation

- `src/` : code Python
- `tests/` : tests automatisés
- `reports/` : métriques et graphiques
- `artifacts/` : modèles suivis par DVC
- `notebooks/` : notebooks d'expérimentation

## Limites

Les résultats sont obtenus sur les données Japan et
India de RDD2022. Le SVM a été entraîné sur un
sous-ensemble de 4 000 images, contrairement aux
réseaux neuronaux entraînés sur 12 748 images.

Le modèle retenu doit être validé sur des données
indépendantes avant une utilisation opérationnelle.






---

## Exécution du projet en local

### 1. Présentation

**Surface Crack Detection** est un projet de classification automatique des images de routes basé sur le dataset RDD2022.

Trois approches ont été expérimentées :

- HOG + SVM
- CNN personnalisé
- MobileNetV2 (Transfer Learning)

Le modèle retenu pour le déploiement est le **CNN personnalisé**, qui a obtenu les meilleures performances sur le jeu de test :

| Métrique | Résultat |
|---|---:|
| Accuracy | 72,77 % |
| Precision | 78,24 % |
| Recall | 68,94 % |
| F1-score | 73,30 % |
| ROC-AUC | 0,8076 |

L'application expose une API REST développée avec FastAPI et conteneurisée avec Docker.

L'API permet de classifier une image dans l'une des deux catégories :

- `crack` : présence probable d'une fissure.
- `no_crack` : absence de fissure détectée.

> **Limitation :** l'API effectue une classification d'image. Elle ne localise pas les fissures et ne génère pas de boîtes englobantes.

### 2. Prérequis

Pour exécuter le projet localement, il faut disposer de :

- Python 3.11 ou 3.12 ;
- Git ;
- Docker Engine et Docker Compose (pour l'exécution conteneurisée) ;
- un environnement virtuel Python (pour l'exécution sans Docker) ;
- le modèle entraîné `cnn_best.keras`.

Le modèle est suivi avec DVC et n'est pas directement inclus dans le dépôt GitHub.

### 3. Récupérer le projet

Cloner le dépôt :

```bash
git clone https://github.com/mubaraka-dev/surface-crack-detection.git
```

Accéder au répertoire :

```bash
cd surface-crack-detection
```

Vérifier les fichiers :

```bash
ls -l Dockerfile compose.yaml requirements-docker.txt
```

### 4. Récupérer le modèle CNN

Le modèle entraîné doit être disponible dans :

```text
artifacts/cnn_best.keras
```

Il peut être récupéré depuis le stockage DVC du projet ou depuis une sauvegarde fournie séparément.

#### Option A : avec DVC

Installer DVC :

```bash
pip install dvc
```

Configurer un stockage DVC accessible, si nécessaire, puis récupérer le modèle :

```bash
dvc pull artifacts/cnn_best.keras.dvc
```

> **Important :** le stockage DVC initial utilise un dossier Google Drive monté dans Google Colab. Sur une autre machine, il faut d'abord configurer un stockage accessible. La commande `dvc pull` ne fonctionne pas automatiquement sur un nouvel ordinateur.

#### Option B : copie manuelle

Si le fichier `cnn_best.keras` a été téléchargé séparément :

```bash
mkdir -p artifacts

cp /chemin/vers/cnn_best.keras artifacts/
```

Vérifier :

```bash
ls -lh artifacts/cnn_best.keras
```

Le modèle doit peser environ **5 Mo**.

### 5. Méthode recommandée : lancer avec Docker

Docker permet de lancer l'API sans installer TensorFlow directement sur l'ordinateur.

Vérifier Docker :

```bash
docker --version
docker compose version
docker info
```

Le fichier `compose.yaml` configure le port de l'ordinateur hôte.

Par défaut, le projet utilise :

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

Si le port 8000 est déjà utilisé, remplacer cette ligne par :

```yaml
ports:
  - "127.0.0.1:8085:8000"
```

Le port 8085 doit être libre sur la machine hôte.

Construire et démarrer l'API :

```bash
docker compose up -d --build
```

La première construction peut prendre plusieurs minutes, car Docker installe TensorFlow et les dépendances nécessaires.

Vérifier le conteneur :

```bash
docker compose ps
```

Le conteneur doit être démarré et son état doit devenir `healthy`.

Pour consulter les journaux :

```bash
docker compose logs --tail=50 surface-crack-api
```

### 6. Vérifier l'API

Avec le port 8085 :

```bash
curl http://127.0.0.1:8085/health
```

Réponse attendue :

```json
{
  "status": "ok",
  "model_loaded": true
}
```

Pour obtenir les informations du modèle :

```bash
curl http://127.0.0.1:8085/model/info
```

Documentation interactive Swagger :

http://127.0.0.1:8085/docs

> Si un autre port a été configuré, remplacer `8085` par le port correspondant.

### 7. Tester une image avec Swagger

1. Ouvrir `http://127.0.0.1:8085/docs`.
2. Sélectionner l'endpoint `POST /predict`.
3. Cliquer sur **Try it out**.
4. Dans le champ `file`, cliquer sur **Choose File**.
5. Sélectionner une image routière au format JPG ou PNG.
6. Cliquer sur **Execute**.
7. Consulter le JSON retourné par l'API.

Exemple de réponse obtenue lors d'un test :

```json
{
  "prediction": "crack",
  "class_id": 1,
  "confidence": 0.7799,
  "crack_probability": 0.7799,
  "threshold": 0.5,
  "model": "CNN personnalisé"
}
```

Ce résultat correspond à une image de chaussée présentant des fissures visibles.

Le seuil de classification est fixé à `0.5`. Le score retourné par le CNN n'est pas nécessairement une probabilité statistiquement calibrée.

### 8. Tester une image depuis le terminal

Il est également possible d'envoyer une image directement avec `curl` :

```bash
curl -X POST \
  http://127.0.0.1:8085/predict \
  -F "file=@/chemin/vers/image_route.jpg"
```

Remplacer `/chemin/vers/image_route.jpg` par le chemin réel de l'image.

### 9. Exécution sans Docker

L'API peut également être lancée directement depuis un environnement Python.

Créer et activer un environnement virtuel :

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Installer les dépendances :

```bash
python -m pip install -r requirements-api.txt
```

Vérifier que le modèle est présent :

```bash
ls -lh artifacts/cnn_best.keras
```

Lancer FastAPI avec Uvicorn :

```bash
python -m uvicorn src.api.main:app \
  --host 127.0.0.1 \
  --port 8085
```

L'API sera disponible sur :

http://127.0.0.1:8085/docs

Cette méthode nécessite que les dépendances Python, notamment TensorFlow, soient installées dans l'environnement virtuel.

### 10. Exemple d'installation sur Ubuntu 24.04

Voici un exemple correspondant à une installation locale sous Ubuntu 24.04.

**Environnement utilisé :**

| Élément | Configuration |
|---|---|
| Système | Ubuntu 24.04.5 LTS |
| Python | 3.12.3 |
| Environnement virtuel | `env_ML` |
| Docker Engine | 29.8.2 |
| Docker Compose | 5.1.0 |
| Port de l'API | 8085 |

**Répertoire local :**

```text
/home/mubaraka/Documents/FOMATION/ML/
├── env_ML/
└── surface-crack-detection/
    ├── artifacts/
    │   └── cnn_best.keras
    ├── src/
    │   └── api/
    │       └── main.py
    ├── Dockerfile
    ├── compose.yaml
    └── README.md
```

Accéder au projet :

```bash
cd /home/mubaraka/Documents/FOMATION/ML/surface-crack-detection
```

Activer l'environnement Python existant, si nécessaire :

```bash
source /home/mubaraka/Documents/FOMATION/ML/env_ML/bin/activate
```

Vérifier la présence du modèle :

```bash
ls -lh artifacts/cnn_best.keras
```

Démarrer l'API avec Docker :

```bash
docker compose up -d --build
```

Vérifier son fonctionnement :

```bash
docker compose ps

curl http://127.0.0.1:8085/model/info
```

Dans cet environnement, le conteneur a été démarré avec succès et son état a été confirmé comme `healthy`.

Une image de route présentant des fissures a également été testée avec Swagger. Le CNN a retourné :

```json
{
  "prediction": "crack",
  "class_id": 1,
  "confidence": 0.7799,
  "crack_probability": 0.7799,
  "threshold": 0.5,
  "model": "CNN personnalisé"
}
```

### 11. Exécuter les tests

Pour lancer les tests automatisés de l'API et du projet, installer les dépendances nécessaires dans l'environnement Python :

```bash
python -m pip install pytest httpx python-multipart
```

Puis exécuter :

```bash
python -m pytest tests/test_project.py tests/test_api.py -v
```

Dans Google Colab, les 10 tests ont été exécutés avec succès.

> Les tests FastAPI nécessitent également TensorFlow, NumPy, Pillow et FastAPI, disponibles après installation de `requirements-api.txt`.

### 12. Arrêter ou redémarrer le projet

Pour arrêter le conteneur :

```bash
docker compose stop
```

Pour le redémarrer :

```bash
docker compose start
```

Pour arrêter et supprimer le conteneur du projet :

```bash
docker compose down
```

Pour reconstruire l'image après une modification du code :

```bash
docker compose up -d --build
```

### 13. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Langage principal |
| TensorFlow / Keras | Entraînement et inférence CNN |
| scikit-learn | Modèle SVM et évaluation |
| OpenCV / scikit-image | Traitement et extraction de caractéristiques |
| MobileNetV2 | Transfer Learning |
| MLflow | Suivi des expériences |
| DVC | Versionnement des modèles |
| FastAPI | API REST |
| Uvicorn | Serveur ASGI |
| Docker | Conteneurisation |
| GitHub Actions | Intégration continue |

---
