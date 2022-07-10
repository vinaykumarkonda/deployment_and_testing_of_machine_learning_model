import re
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from classification_model.config.core import config


class ExtractFirstCabin(BaseEstimator, TransformerMixin):
    """retain only the first cabin if"""

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Any:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].apply(self._get_first_cabin)
        return X

    # private method which used by transform method
    def _get_first_cabin(self, row: Any) -> Any:
        try:
            return row.split()[0]
        except AttributeError:
            return np.nan


class ExtractTitleTransformer(BaseEstimator, TransformerMixin):
    """extracts the title (Mr, Ms, etc) from the name variable"""

    def __init__(self, name: str, title: str):
        if not isinstance([name, title], list):
            raise ValueError("title & name should be strings")
        self.title = title
        self.name = name

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Any:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.title] = X[self.name].apply(self._extract_title)
        return X

    # private method to extract the title from name and used in transform method
    def _extract_title(self, row: Any) -> Any:
        if re.search("Mrs", row):
            return "Mrs"
        elif re.search("Mr", row):
            return "Mr"
        elif re.search("Miss", row):
            return "Miss"
        elif re.search("Master", row):
            return "Master"
        else:
            return "Other"


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    """extract first letter of variable"""

    def __init__(self, variables: List[str]):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Any:
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        # so that we do not over-write the original dataframe
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):
    """drop unecessary features"""

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("Variables should be a list")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Any:
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = X.drop(labels=config.model_config.drop_features, axis=1)
        return X
