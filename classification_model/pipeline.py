from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification_model.config.core import config
from classification_model.processing import preprocessors as pp

titanic_pipe = Pipeline(
    [
        # ======= Pre-processing ======
        (
            "extract_first_cabin",
            pp.ExtractFirstCabin(variables=config.model_config.transform_vars),
        ),
        (
            "extract_title",
            pp.ExtractTitleTransformer(
                name=config.model_config.extract_title,
                title=config.model_config.title_feature,
            ),
        ),
        (
            "drop_unnecessary_features",
            pp.DropUnecessaryFeatures(variables=config.model_config.drop_features),
        ),
        # ======= IMPUTATION ==========
        # impute categorical variables with string missing
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars,
            ),
        ),
        # add missing indicator to numerical variables
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars),
        ),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=config.model_config.numerical_vars
            ),
        ),
        # Extract letter from cabin
        (
            "extract_letter",
            pp.ExtractLetterTransformer(variables=config.model_config.transform_vars),
        ),
        # === CATEGORICAL ENCODING =====
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=config.model_config.rare_label_tol,
                n_categories=1,
                variables=config.model_config.categorical_vars,
            ),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True, variables=config.model_config.categorical_vars
            ),
        ),
        # scale
        (
            "scaler",
            SklearnTransformerWrapper(
                transformer=StandardScaler(),
                variables=config.model_config.numerical_vars,
            ),
        ),
        (
            "Logit",
            LogisticRegression(
                C=config.model_config.alpha,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
