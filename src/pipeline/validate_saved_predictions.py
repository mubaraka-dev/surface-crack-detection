"""
Validation scientifique reproductible — RDD2022.

Recalcule les métriques à partir des prédictions
déjà enregistrées dans final_test_predictions.csv.

Cette étape ne charge pas les modèles et ne les
réentraîne pas.
"""

import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = (
    ROOT / "reports" / "final_test_predictions.csv"
)

OUTPUT = (
    ROOT / "reports" / "pipeline_validation"
)

MODELS = {
    "cnn_personnalise": {
        "prediction": "cnn_personnalisé_prediction",
        "score": "cnn_personnalisé_score"
    },
    "mobilenetv2": {
        "prediction": "mobilenetv2_prediction",
        "score": "mobilenetv2_score"
    },
    "hog_svm": {
        "prediction": "hog__svm_prediction",
        "score": "hog__svm_score"
    }
}

EXPECTED_CNN = {
    "accuracy": 0.727672,
    "precision": 0.782375,
    "recall": 0.689399,
    "f1_score": 0.732950,
    "roc_auc": 0.807551
}

print("=" * 65)
print("DVC — VALIDATION SCIENTIFIQUE")
print("=" * 65)

# --------------------------------------------------------------
# 1. Charger et vérifier les données
# --------------------------------------------------------------

df = pd.read_csv(PREDICTIONS)

if len(df) != 2732:
    raise ValueError(
        f"Nombre de lignes incorrect : {len(df)}"
    )

required_columns = ["y_true"]

for columns in MODELS.values():
    required_columns.extend([
        columns["prediction"],
        columns["score"]
    ])

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Colonnes absentes : {missing}"
    )

if df[required_columns].isna().any().any():
    raise ValueError(
        "Des valeurs manquantes ont été détectées."
    )

y_true = df["y_true"].astype(int)

if set(y_true.unique()) != {0, 1}:
    raise ValueError(
        "Les étiquettes réelles doivent être binaires."
    )

print("\nImages :", len(df))
print("No crack :", int((y_true == 0).sum()))
print("Crack :", int((y_true == 1).sum()))

# --------------------------------------------------------------
# 2. Recalculer les métriques
# --------------------------------------------------------------

results = {}

rows = []

for name, columns in MODELS.items():

    y_pred = df[
        columns["prediction"]
    ].astype(int)

    y_score = df[
        columns["score"]
    ].astype(float)

    if not set(y_pred.unique()).issubset({0, 1}):
        raise ValueError(
            f"Prédictions invalides : {name}"
        )

    if not y_score.between(0, 1).all():
        raise ValueError(
            f"Scores invalides : {name}"
        )

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
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_score)
        ),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }

    results[name] = metrics

    rows.append({
        "model": name,
        **metrics
    })

    print(
        f"\n{name:20s}"
        f" Accuracy={metrics['accuracy']:.6f}"
        f" F1={metrics['f1_score']:.6f}"
    )

# --------------------------------------------------------------
# 3. Vérifier les résultats historiques du CNN
# --------------------------------------------------------------

cnn = results["cnn_personnalise"]

for metric, expected in EXPECTED_CNN.items():

    actual = cnn[metric]

    if abs(actual - expected) > 0.0001:
        raise ValueError(
            f"Incohérence sur {metric} : "
            f"{actual} au lieu de {expected}"
        )

print("\nRésultats historiques CNN : COHÉRENTS")

# --------------------------------------------------------------
# 4. Générer les sorties DVC
# --------------------------------------------------------------

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

comparison = pd.DataFrame(rows)

comparison = comparison.sort_values(
    "f1_score",
    ascending=False
)

comparison.to_csv(
    OUTPUT / "comparison.csv",
    index=False
)

# Les valeurs numériques sont directement
# lisibles par dvc metrics show.
(OUTPUT / "metrics.json").write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

report = {
    "dataset": "RDD2022",
    "test_samples": len(df),
    "source": "final_test_predictions.csv",
    "training_reexecuted": False,
    "validation_status": "passed",
    "models": results
}

(OUTPUT / "validation_report.json").write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("\nRapports générés :")
print(" - comparison.csv")
print(" - metrics.json")
print(" - validation_report.json")

print("\nVALIDATION SCIENTIFIQUE : RÉUSSIE")
