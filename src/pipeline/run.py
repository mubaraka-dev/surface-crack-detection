"""
Surface Crack Detection — Pipeline ML avec DVC.

Étapes :
  prepare
  train
  evaluate

Les modèles historiques restent intacts.
"""

import argparse
import json
import random
import os

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

REPORTS = ROOT / "reports"

DATA_DIR = ROOT / "data/pipeline"

MODEL_DIR = ROOT / "artifacts/pipeline_experiment"

OUTPUT_DIR = ROOT / "reports/pipeline_experiment"

PARAMS = yaml.safe_load(
    (ROOT / "params.yaml").read_text(
        encoding="utf-8"
    )
)

SEED = int(PARAMS["training"]["seed"])

random.seed(SEED)
np.random.seed(SEED)

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2"
)


# ------------------------------------------------------------
# UTILITAIRE : LECTURE DES MANIFESTES
# ------------------------------------------------------------

def load_manifest(split):

    file = REPORTS / f"{split}_manifest.csv"

    if not file.is_file():
        raise FileNotFoundError(file)

    df = pd.read_csv(file)

    path_column = next(
        (
            c for c in (
                "path",
                "image_path",
                "filepath"
            )
            if c in df.columns
        ),
        None
    )

    label_column = next(
        (
            c for c in (
                "binary_label",
                "label"
            )
            if c in df.columns
        ),
        None
    )

    if path_column is None:
        raise ValueError(
            f"Colonne chemin absente : {file}"
        )

    if label_column is None:
        raise ValueError(
            f"Colonne étiquette absente : {file}"
        )

    raw_labels = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "crack": 1,
        "no_crack": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0
    }

    labels = raw_labels.map(mapping)

    if labels.isna().any():
        raise ValueError(
            f"Étiquettes invalides dans {file}"
        )

    result = pd.DataFrame({
        "path": df[path_column].astype(str),
        "label": labels.astype(int)
    })

    if result["path"].isna().any():
        raise ValueError(
            "Chemins manquants."
        )

    if result["path"].duplicated().any():
        raise ValueError(
            f"Doublons dans {split}."
        )

    return result


# ------------------------------------------------------------
# ÉTAPE 1 — PRÉPARATION
# ------------------------------------------------------------

def prepare():

    print("=" * 65)
    print("PIPELINE DVC — PREPARE")
    print("=" * 65)

    expected = {
        "train": int(
            PARAMS["dataset"]["train_size"]
        ),
        "val": int(
            PARAMS["dataset"]["val_size"]
        ),
        "test": int(
            PARAMS["dataset"]["test_size"]
        )
    }

    frames = {}
    statistics = {}

    for split in ("train", "val", "test"):

        df = load_manifest(split)

        if len(df) != expected[split]:
            raise ValueError(
                f"Split {split}: "
                f"{len(df)} au lieu de "
                f"{expected[split]}"
            )

        missing = [
            p for p in df["path"]
            if not Path(p).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images absentes "
                f"pour {split}.\n"
                f"Exemple : {missing[0]}\n"
                "Il faut restaurer les images RDD2022."
            )

        frames[split] = df

        statistics[split] = {
            "images": len(df),
            "crack": int(
                (df["label"] == 1).sum()
            ),
            "no_crack": int(
                (df["label"] == 0).sum()
            )
        }

        print(split, statistics[split])

    for a, b in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test")
    ):

        overlap = (
            set(frames[a]["path"])
            & set(frames[b]["path"])
        )

        if overlap:
            raise ValueError(
                f"Chevauchement entre {a} et {b}."
            )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for split, df in frames.items():

        df.to_csv(
            DATA_DIR / f"{split}.csv",
            index=False
        )

    (DATA_DIR / "summary.json").write_text(
        json.dumps(
            statistics,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("Préparation terminée.")


# ------------------------------------------------------------
# CHARGEMENT DES IMAGES TENSORFLOW
# ------------------------------------------------------------

def create_dataset(df, training=False):

    import tensorflow as tf

    image_size = int(
        PARAMS["training"]["image_size"]
    )

    batch_size = int(
        PARAMS["training"]["batch_size"]
    )

    paths = df["path"].astype(str).to_numpy()

    labels = df["label"].astype(
        np.float32
    ).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    def decode(path, label):

        data = tf.io.read_file(path)

        image = tf.io.decode_image(
            data,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            [image_size, image_size]
        )

        return (
            tf.cast(image, tf.float32),
            tf.cast(label, tf.float32)
        )

    if training:

        ds = ds.shuffle(
            buffer_size=min(len(df), 10000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        decode,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ------------------------------------------------------------
# ÉTAPE 2 — ENTRAÎNEMENT CNN
# ------------------------------------------------------------

def train():

    print("=" * 65)
    print("PIPELINE DVC — TRAIN CNN")
    print("=" * 65)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)

    train_df = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    val_df = pd.read_csv(
        DATA_DIR / "val.csv"
    )

    train_ds = create_dataset(
        train_df,
        training=True
    )

    val_ds = create_dataset(val_df)

    image_size = int(
        PARAMS["training"]["image_size"]
    )

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1)
    ])

    inputs = tf.keras.Input(
        shape=(image_size, image_size, 3)
    )

    x = augmentation(inputs)

    x = tf.keras.layers.Rescaling(
        1.0 / 255.0
    )(x)

    for filters in (32, 64, 128):

        x = tf.keras.layers.Conv2D(
            filters,
            3,
            padding="same",
            activation="relu"
        )(x)

        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="DVC_Pipeline_CNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(
                PARAMS["training"]["learning_rate"]
            )
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(
            PARAMS["training"]["epochs"]
        ),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    model.save(
        MODEL_DIR / "cnn_pipeline.keras"
    )

    (MODEL_DIR / "history.json").write_text(
        json.dumps(
            {
                k: [float(x) for x in values]
                for k, values in history.history.items()
            },
            indent=2
        ),
        encoding="utf-8"
    )

    print("CNN expérimental entraîné.")


# ------------------------------------------------------------
# ÉTAPE 3 — ÉVALUATION
# ------------------------------------------------------------

def evaluate():

    print("=" * 65)
    print("PIPELINE DVC — EVALUATE CNN")
    print("=" * 65)

    import tensorflow as tf

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )

    test_df = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    model = tf.keras.models.load_model(
        MODEL_DIR / "cnn_pipeline.keras",
        compile=False,
        safe_mode=True
    )

    test_ds = create_dataset(test_df)

    scores = model.predict(
        test_ds,
        verbose=1
    ).reshape(-1)

    y_true = test_df["label"].to_numpy(
        dtype=int
    )

    threshold = float(
        PARAMS["evaluation"]["threshold"]
    )

    y_pred = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true, y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, y_pred,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, scores)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    predictions_df = test_df.copy()

    predictions_df["prediction"] = y_pred
    predictions_df["score"] = scores

    predictions_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False
    )

    print("\nMÉTRIQUES :")

    for key, value in metrics.items():

        print(f"{key:12s}: {value}")


# ------------------------------------------------------------
# POINT D'ENTRÉE
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "train",
            "evaluate"
        ]
    )

    args = parser.parse_args()

    if args.stage == "prepare":
        prepare()

    elif args.stage == "train":
        train()

    elif args.stage == "evaluate":
        evaluate()
