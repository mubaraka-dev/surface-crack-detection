import io

from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from src.api import main as api_module


def create_test_image():

    image = Image.new(
        "RGB",
        (200, 200),
        color=(125, 125, 125)
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG"
    )

    return buffer.getvalue()


def test_home():

    client = TestClient(api_module.app)

    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["model"] == "CNN personnalisé"


def test_model_info():

    client = TestClient(api_module.app)

    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["image_size"] == [160, 160]
    assert data["positive_class"] == "crack"


def test_invalid_file_type():

    client = TestClient(api_module.app)

    response = client.post(
        "/predict",
        files={
            "file": (
                "document.txt",
                b"Hello",
                "text/plain"
            )
        }
    )

    assert response.status_code == 415


def test_invalid_image():

    client = TestClient(api_module.app)

    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.jpg",
                b"invalid-image-content",
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 400


def test_image_preprocessing():

    image_bytes = create_test_image()

    image_tensor = api_module.preprocess_image(
        image_bytes
    )

    assert tuple(image_tensor.shape) == (
        1, 160, 160, 3
    )

    assert image_tensor.dtype.name == "float32"


def test_predict_with_mocked_model(monkeypatch):

    class FakeModel:

        def predict(self, image, verbose=0):

            assert tuple(image.shape) == (
                1, 160, 160, 3
            )

            return np.array(
                [[0.85]],
                dtype=np.float32
            )

    monkeypatch.setattr(
        api_module,
        "get_model",
        lambda: FakeModel()
    )

    client = TestClient(api_module.app)

    response = client.post(
        "/predict",
        files={
            "file": (
                "road.jpg",
                create_test_image(),
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == "crack"
    assert data["class_id"] == 1

    assert abs(
        data["crack_probability"] - 0.85
    ) < 0.001
