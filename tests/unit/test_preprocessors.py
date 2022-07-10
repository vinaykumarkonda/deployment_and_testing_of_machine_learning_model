from classification_model.config.core import config
from classification_model.processing import preprocessors as pp


def test_extract_first_cabin(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    transformer = pp.ExtractFirstCabin(variables=config.model_config.transform_vars)

    assert X_test["cabin"].loc[49] == "B51 B53 B55"

    # When
    transformed = transformer.fit_transform(X_test)

    # Then
    assert transformed["cabin"].loc[49] == "B51"


def test_extract_title_transformer(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    transformer = pp.ExtractTitleTransformer(
        name=config.model_config.extract_title, title=config.model_config.title_feature
    )

    assert X_test[config.model_config.extract_title].iat[0] == "Rintamaki, Mr. Matti"

    # When
    transformed = transformer.fit_transform(X_test)

    # Then
    assert transformed[config.model_config.title_feature].iat[0] == "Mr"


def test_drop_unecessary_features(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    drop_features = config.model_config.drop_features
    transformer = pp.DropUnecessaryFeatures(variables=drop_features)

    assert (set(drop_features) & set(X_test.columns.to_list())) == set(drop_features)

    # When
    transformed = transformer.fit_transform(X_test)

    # Then
    assert (set(drop_features) & set(transformed.columns.to_list())) == set([])


def test_extract_letter_transformer(pipeline_inputs: tuple) -> None:
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    transformer = pp.ExtractLetterTransformer(
        variables=config.model_config.transform_vars
    )

    assert X_test["cabin"].iat[-1] == "C90"

    # When
    transformed = transformer.fit_transform(X_test)

    # Then
    assert transformed["cabin"].iat[-1] == "C"
