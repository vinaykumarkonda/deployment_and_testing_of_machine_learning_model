from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config

# from marshmallow import Schema, ValidationError, fields


def replace_interrogation_marks(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """replace interrogation marks by NaN values"""

    replaced_interrogation_data = input_data.copy().replace("?", np.nan)

    return replaced_interrogation_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)

    # replace interrogation marks by NaN values
    validated_data = replace_interrogation_marks(input_data=input_data)

    errors = None

    # cast numerical variables as floats
    for var in config.model_config.numerical_vars:
        validated_data[var] = validated_data[var].astype("float")

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = {e["loc"][-1]: e["msg"] for e in error.errors()}

    return validated_data, errors


class TitanicDataInputSchema(BaseModel):
    pclass: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[float]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[str]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[str]
    body: Optional[float]
    home_dest: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
