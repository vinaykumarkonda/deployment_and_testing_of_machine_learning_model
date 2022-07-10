import pandas as pd

from classification_model.processing.validation import validate_inputs


def test_validate_inputs(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    inputs_length = 262
    assert X_test.shape[0] == 262

    # When
    validated_data, errors = validate_inputs(input_data=X_test)

    # Then
    # no errors after validation
    assert not errors
    assert len(validated_data) == inputs_length


def test_validate_inputs_identifies_errors(invalid_test_data: dict) -> None:
    # Given
    test_input = pd.DataFrame(invalid_test_data)

    expected_errors = {
        "sibsp": "value is not a valid integer",
        "pclass": "value is not a valid integer",
        "parch": "value is not a valid integer",
        "body": "value is not a valid float",
    }

    # When
    validated_data, errors = validate_inputs(input_data=test_input)

    # Then
    assert errors
    assert sum([errors[k] != v for k, v in expected_errors.items()]) == 0
