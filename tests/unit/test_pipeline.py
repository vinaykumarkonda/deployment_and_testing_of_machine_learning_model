from classification_model.config.core import config
from classification_model.pipeline import titanic_pipe


def test_piepline_extract_first_cabin(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    titanic_pipe.fit(X_test, y_test)
    assert X_test["cabin"].loc[49] == "B51 B53 B55"

    # When
    transformed = titanic_pipe[:-1].transform(X_test)

    # Then
    assert transformed["cabin_M"].loc[49] == 0


def test_pipeline_extract_title_transformer(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    titanic_pipe.fit(X_test, y_test)

    assert X_test[config.model_config.extract_title].iat[0] == "Rintamaki, Mr. Matti"

    # When
    transformed = titanic_pipe[:-1].transform(X_test)

    # Then
    assert transformed[f"{config.model_config.title_feature}_Mr"].iat[0] == 1


def test_pipeline_drop_unecessary_features(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    drop_features = config.model_config.drop_features
    titanic_pipe.fit(X_test, y_test)

    assert (set(drop_features) & set(X_test.columns.to_list())) == set(drop_features)

    # When
    transformed = titanic_pipe[:-1].transform(X_test)

    # Then
    assert (set(drop_features) & set(transformed.columns.to_list())) == set([])


def test_pipeline_extract_letter_transformer(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    titanic_pipe.fit(X_test, y_test)

    assert X_test["cabin"].iat[-1] == "C90"

    # When
    transformed = titanic_pipe[:-1].transform(X_test)

    # Then
    assert transformed["cabin_C"].iat[-1] == 1
