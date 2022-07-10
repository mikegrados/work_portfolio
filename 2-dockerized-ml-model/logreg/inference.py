import os
import config
import joblib
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field, ValidationError


BASE_DIR = config.BASE_DIR
TRAINED_MODEL_DIR = config.TRAINED_MODEL_DIR
PIPELINE_SAVE_FILE = config.PIPELINE_SAVE_FILE


class TitleChoices(str, Enum):
    """Establish the only options for the variable Title"""

    mr = "Mr"
    mrs = "Mrs"
    miss = "Miss"
    master = "Master"
    other = "Other"


class LogRegSetting(BaseModel):
    """Schema for the input"""

    pclass: int = Field(gt=0, lt=4, description="Values: 1,2,3")
    sex: str
    age: Optional[float] = Field(gt=0, lt=105, description="Values between 0 and 105")
    sibsp: int = Field(
        gt=-1, lt=21, description="Siblings and parents between 0 and 20"
    )
    parch: int
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]
    title: TitleChoices


def validate_input(input_data: pd.DataFrame,) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values. 

    Parameters
    ----------
    input_data: pd.DataFrame
        Input values for the model
        
    Returns
    -------
    Either an error or a dictionary with the input for the model
    """

    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        dict_input = input_data.replace({np.nan: None}).to_dict(orient="records")[0]
        LogRegSetting(**dict_input)

    except ValidationError as error:
        errors = error.json()

    return input_data, errors


def make_prediction(input_df: pd.DataFrame) -> dict:
    """Make a prediction using a saved model pipeline.

    Parameters
    ----------
    input_df: pd.DataFrame
        Input values for the model

    Returns
    -------
    Dictionary with an error or the output of the model

    """

    PATH_TRAINED_PIPELINE = os.path.join(TRAINED_MODEL_DIR, PIPELINE_SAVE_FILE)
    model = joblib.load(filename=PATH_TRAINED_PIPELINE)
    validated_data, errors = validate_input(input_df)
    results = {"predictions": None, "errors": errors}

    if not errors:
        pred = model.predict(validated_data)
        proba = model.predict_proba(validated_data)
        results["predictions"] = [pred, proba]

    return results


# Example 1
# =========
if __name__ == "__main__":

    # happy_input = {
    #     "pclass": 3,
    #     "sex": "male",
    #     "sibsp": 1,
    #     "parch": 0,
    #     "fare": 31,
    #     "embarked": "Q",
    #     "title": "Other",
    #     "age": 29,
    #     "cabin": None,
    # }

    happy_input = {
        "pclass": 1,
        "sex": "male",
        "sibsp": 3,
        "parch": 2,
        "fare": 26.55,
        "embarked": "S",
        "title": "Mr",
        "age": 47,
        "cabin": None,
    }

    input_data = pd.DataFrame.from_dict(happy_input, orient="index").T
    data, error_msg = validate_input(input_data)

    print(f"data: \n{data} \nerror_messages: \n{error_msg}")
    print("\n")
    print(f"predictions: \n{make_prediction(input_data)}")
