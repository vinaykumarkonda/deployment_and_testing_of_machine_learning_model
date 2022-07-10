from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset
from classification_model.processing.validation import validate_inputs
from ml_api.main import app


@pytest.fixture(scope="session")
def pipeline_inputs() -> tuple:
    data = load_dataset()

    # validate the data
    validated_data, errors = validate_inputs(input_data=data)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        validated_data.drop(config.model_config.target, axis=1),
        validated_data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}


@pytest.fixture()
def invalid_test_data() -> dict:
    return {
        "pclass": ["pclass"],
        "name": [100.0],
        "sex": [100.0],
        "age": [100.0],
        "sibsp": ["sibsp"],
        "parch": ["parch"],
        "ticket": [1234],
        "fare": [100.0],
        "cabin": [100.0],
        "embarked": [100.0],
        "boat": [1234],
        "body": ["body"],
        "home_dest": [100.0],
    }


@pytest.fixture()
def valid_test_data() -> dict:
    return {
        "pclass": [1],
        "name": [100.0],
        "sex": [100.0],
        "age": [100.0],
        "sibsp": [0],
        "parch": [0],
        "ticket": [1234],
        "fare": [100.0],
        "cabin": [100.0],
        "embarked": [100.0],
        "boat": [1234],
        "body": [135],
        "home_dest": [100.0],
    }
