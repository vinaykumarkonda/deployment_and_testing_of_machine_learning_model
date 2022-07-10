import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from classification_model import __version__ as model_version
from ml_api import __version__
from ml_api.config import settings


def test_health_endpoint(client: TestClient):
    # When
    response = client.get(f"{settings.API_V1_STR}/health")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == settings.PROJECT_NAME
    assert data["api_version"] == __version__
    assert data["model_version"] == model_version


def test_prediction_endpoint(client: TestClient, pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    input_length = len(X_test)

    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": X_test.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    response = client.post(f"{settings.API_V1_STR}/predict", json=payload)

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert len(prediction_data["predictions"]) == input_length
    assert prediction_data["errors"] is None


@pytest.mark.parametrize("field", ["sibsp", "pclass", "parch"])
def test_prediction_validation(
    client: TestClient, field: str, valid_test_data: dict
) -> None:
    # Given
    test_data = pd.DataFrame(valid_test_data)
    test_data.loc[0, field] = field

    payload = {
        # ensure pydantic plays well with np.nan
        "inputs": test_data.replace({np.nan: None}).to_dict(orient="records")
    }

    # When
    response = client.post(f"{settings.API_V1_STR}/predict", json=payload)

    # Then
    assert response.status_code == 422
    error_message = response.json()["detail"][0]["msg"]
    assert error_message == "value is not a valid integer"
