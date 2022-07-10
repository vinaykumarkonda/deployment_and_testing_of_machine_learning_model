from typing import Any, List, Optional

from pydantic import BaseModel

from classification_model.processing.validation import TitanicDataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "pclass": 1,
                        "name": "Allison, Master. Hudson Trevor",
                        "sex": "male",
                        "age": "0.9167",
                        "sibsp": 1,
                        "parch": 2,
                        "ticket": "113781",
                        "fare": "151.55",
                        "cabin": "C22 C26",
                        "embarked": "S",
                        "boat": "11",
                        "body": "135",
                        "home_dest": "Montreal, PQ / Chesterville, ON",
                    }
                ]
            }
        }
