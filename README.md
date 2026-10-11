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
