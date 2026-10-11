"""
Surface Crack Detection API

Classification binaire d'images routières avec un CNN Keras.

Classes :
    0 : no_crack
    1 : crack

Prétraitement identique à l'évaluation finale :
    - décodage RGB
    - resize 160 x 160
    - float32 [0,255]

Le modèle Keras gère ses éventuelles couches internes
de normalisation.
"""

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = ROOT / "artifacts" / "cnn_best.keras"

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

THRESHOLD = 0.5

CLASS_NAMES = {
    0: "no_crack",
    1: "crack"
}

MODEL_METRICS = {
    "accuracy": 0.727672,
    "precision": 0.782375,
    "recall": 0.689399,
    "f1_score": 0.732950,
    "roc_auc": 0.807551
}

app = FastAPI(
    title="Surface Crack Detection API",
    description=(
        "API de classification d'images routières "
        "avec un réseau de neurones convolutif."
    ),
    version="1.0.0"
)

# ------------------------------------------------------------
# Chargement du modèle : une seule fois par processus
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model():

    model_path = Path(
        os.environ.get(
            "SURFACE_CRACK_MODEL_PATH",
            str(DEFAULT_MODEL)
        )
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=True
    )

    input_shape = model.input_shape
    output_shape = model.output_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    if tuple(input_shape[1:]) != (160, 160, 3):
        raise RuntimeError(
            f"Entrée CNN inattendue : {input_shape}"
        )

    if output_shape[-1] != 1:
        raise RuntimeError(
            f"Sortie CNN inattendue : {output_shape}"
        )

    return model


# ------------------------------------------------------------
# Prétraitement d'image
# ------------------------------------------------------------

def preprocess_image(image_bytes: bytes):

    if not image_bytes:
        raise ValueError("Le fichier image est vide.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            "Image trop volumineuse : maximum 10 Mo."
        )

    # Vérification du format et des dimensions.
    # La conversion PIL n'est pas utilisée pour le
    # tenseur final : TensorFlow reproduit le pipeline
    # employé lors de l'évaluation des modèles.

    try:

        with Image.open(io.BytesIO(image_bytes)) as img:

            if img.format not in ("JPEG", "PNG"):
                raise ValueError(
                    "Format non accepté : JPEG ou PNG requis."
                )

            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "Dimensions de l'image trop importantes."
                )

            img.verify()

    except UnidentifiedImageError as exc:
        raise ValueError(
            "Le fichier fourni n'est pas une image valide."
        ) from exc

    # Même traitement que dans la cellule 9 :
    # tf.io.decode_image -> tf.image.resize -> float32

    try:

        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            [160, 160]
        )

        image = tf.cast(
            image,
            tf.float32
        )

        image = tf.expand_dims(
            image,
            axis=0
        )

    except Exception as exc:

        raise ValueError(
            "Impossible de décoder ou traiter l'image."
        ) from exc

    return image


# ------------------------------------------------------------
# GET /
# ------------------------------------------------------------

@app.get("/")
def home():

    return {
        "application": "Surface Crack Detection",
        "version": "1.0.0",
        "model": "CNN personnalisé",
        "task": "binary_image_classification",
        "classes": ["no_crack", "crack"],
        "documentation": "/docs"
    }


# ------------------------------------------------------------
# GET /health
# ------------------------------------------------------------

@app.get("/health")
def health():

    try:
        model = get_model()

        return {
            "status": "ok",
            "model_loaded": model is not None
        }

    except Exception:

        raise HTTPException(
            status_code=503,
            detail="Modèle indisponible."
        )


# ------------------------------------------------------------
# GET /model/info
# ------------------------------------------------------------

@app.get("/model/info")
def model_info():

    return {
        "model": "CNN personnalisé",
        "architecture": "Convolutional Neural Network",
        "dataset": "RDD2022",
        "countries": ["Japan", "India"],
        "image_size": [160, 160],
        "threshold": THRESHOLD,
        "positive_class": "crack",
        "classes": CLASS_NAMES,
        "test_samples": 2732,
        "test_metrics": MODEL_METRICS
    }


# ------------------------------------------------------------
# POST /predict
# ------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Vérifier le type déclaré.
    # Le contenu sera également vérifié lors du décodage.

    if file.content_type not in (
        "image/jpeg",
        "image/png"
    ):

        raise HTTPException(
            status_code=415,
            detail="Seules les images JPEG et PNG sont acceptées."
        )

    # Lire au maximum 10 Mo + 1 octet.
    image_bytes = await file.read(MAX_FILE_SIZE + 1)

    if len(image_bytes) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="Fichier trop volumineux (maximum 10 Mo)."
        )

    try:
        image_tensor = preprocess_image(image_bytes)

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    try:

        model = get_model()

        output = model.predict(
            image_tensor,
            verbose=0
        )

        probability = float(
            np.asarray(output).reshape(-1)[0]
        )

        if not np.isfinite(probability):
            raise RuntimeError(
                "Prédiction numérique invalide."
            )

        if probability < 0 or probability > 1:
            raise RuntimeError(
                "Score CNN hors de l'intervalle [0,1]."
            )

    except Exception:

        raise HTTPException(
            status_code=503,
            detail="Erreur pendant l'inférence."
        )

    predicted_class = int(
        probability >= THRESHOLD
    )

    confidence = (
        probability
        if predicted_class == 1
        else 1.0 - probability
    )

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "class_id": predicted_class,
        "confidence": round(confidence, 4),
        "crack_probability": round(probability, 4),
        "threshold": THRESHOLD,
        "model": "CNN personnalisé"
    }
