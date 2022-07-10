import numpy as np
from sklearn.metrics import f1_score

from classification_model.predict import make_prediction


def test_make_prediction(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    expected_first_prediction_value = 0
    expected_no_predictions = 262

    # When
    result = make_prediction(input_data=X_test)

    # Then
    predictions = result.get("predictions")

    assert isinstance(predictions, list)

    assert isinstance(predictions[0], np.int64)

    assert result.get("errors") is None

    assert len(predictions) == expected_no_predictions

    assert predictions[0] == expected_first_prediction_value


def test_prediction_quality_against_random_model(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    # When
    np.random.seed == 42
    random_predictions = np.random.randint(2, size=X_test.shape[0])
    random_f1_score = f1_score(y_test, random_predictions)
    model_predictions = make_prediction(input_data=X_test)
    model_f1_score = f1_score(y_test, model_predictions.get("predictions"))

    # Then
    assert random_f1_score < model_f1_score
