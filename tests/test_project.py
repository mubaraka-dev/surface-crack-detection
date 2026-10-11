from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_project_structure():
    assert (ROOT / "src").is_dir()
    assert (ROOT / "reports").is_dir()
    assert (ROOT / ".dvc").is_dir()


def test_final_comparison():
    path = ROOT / "reports/final_model_comparison.csv"

    assert path.exists()

    df = pd.read_csv(path)

    assert len(df) == 3
    assert "f1_score" in df.columns
    assert df["f1_score"].between(0, 1).all()


def test_cnn_performance():
    df = pd.read_csv(
        ROOT / "reports/final_model_comparison.csv"
    )

    cnn = df.loc[
        df["modele"] == "CNN personnalisé"
    ]

    assert len(cnn) == 1
    assert float(cnn.iloc[0]["accuracy"]) > 0.70
    assert float(cnn.iloc[0]["f1_score"]) > 0.70


def test_dvc_model_pointers():
    for name in (
        "cnn_best.keras",
        "hog_svm_model.joblib",
        "mobilenetv2_best.keras",
    ):
        assert (
            ROOT / "artifacts" / f"{name}.dvc"
        ).exists()
