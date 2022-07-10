import config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from features import (
    MissingIndicator,
    ExtractLetters,
    CategoricalImputer,
    NumericalImputer,
    RareLabelCategoricalEncoder,
    OneHotEncoder,
    OrderingFeatures,
)


SEED_MODEL = config.SEED_MODEL

NUMERICAL_VARS = config.NUMERICAL_VARS
NUMERICAL_VARS_WITH_NA = config.NUMERICAL_VARS_WITH_NA

CATEGORICAL_VARS = config.CATEGORICAL_VARS
CATEGORICAL_VARS_WITH_NA = config.CATEGORICAL_VARS_WITH_NA


logreg_pipeline = Pipeline(
    [
        ("missing_indicator", MissingIndicator(variables=NUMERICAL_VARS)),
        ("cabin__letter", ExtractLetters()),
        ("cat_imptr", CategoricalImputer(variables=CATEGORICAL_VARS_WITH_NA)),
        ("median_imptr", NumericalImputer(variables=NUMERICAL_VARS_WITH_NA)),
        ("rare_lab", RareLabelCategoricalEncoder(variables=CATEGORICAL_VARS)),
        ("dummy_vars", OneHotEncoder(variables=CATEGORICAL_VARS)),
        ("aligning_feats", OrderingFeatures()),
        ("scaling", MinMaxScaler()),
        (
            "log_reg",
            LogisticRegression(
                C=0.0005, class_weight="balanced", random_state=SEED_MODEL
            ),
        ),
    ]
)
