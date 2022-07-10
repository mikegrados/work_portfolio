import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import (
    RandomUnderSampler,
    CondensedNearestNeighbour,
    TomekLinks,
    OneSidedSelection,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    NeighbourhoodCleaningRule,
    NearMiss,
    InstanceHardnessThreshold,
)
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE,
)


PATH_DATA_PERF = "../data/performance/"

SEED = 404
DATASETS_LIST = ["ecoli", "thyroid_sick", "arrhythmia", "ozone_level"]
EVAL_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC AUC",
    "FN",
    "TP",
    "FP",
    "TN",
    "Average_Precision",
]

CLF_DICT = {
    "logreg": LogisticRegression(random_state=SEED, max_iter=2500, solver="saga"),
    "randforest": RandomForestClassifier(random_state=SEED),
    "naivebayes": GaussianNB(),
}

PARAMS_RF = {
    "classifier": [CLF_DICT["randforest"]],
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__max_depth": [2, 5, 7, 10],
    "classifier__max_features": [None, "sqrt", "log2"],
}

PARAMS_LR = {
    "classifier": [CLF_DICT["logreg"]],
    "classifier__penalty": ["l2", "none"],
    "classifier__C": [0.001, 0.01, 0.1, 1.0, 10.0],
}

PARAMS_NB = {
    "classifier": [CLF_DICT["naivebayes"]],
    "classifier__var_smoothing": np.logspace(0, -9, num=20),
}

PARAMS = [PARAMS_RF, PARAMS_LR, PARAMS_NB]

UNDERSAMPLER_DICT = {
    "randundersmpl": RandomUnderSampler(
        sampling_strategy="auto", random_state=0, replacement=True
    ),
    "condnneigh": CondensedNearestNeighbour(
        sampling_strategy="auto", random_state=0, n_neighbors=1, n_jobs=4
    ),
    "tomek": TomekLinks(sampling_strategy="auto", n_jobs=3),
    "oss": OneSidedSelection(
        sampling_strategy="auto", random_state=0, n_neighbors=1, n_jobs=4
    ),
    "enn": EditedNearestNeighbours(
        sampling_strategy="auto", n_neighbors=3, kind_sel="all", n_jobs=4
    ),
    "renn": RepeatedEditedNearestNeighbours(
        sampling_strategy="auto", n_neighbors=3, kind_sel="all", n_jobs=4, max_iter=100
    ),
    "allknn": AllKNN(sampling_strategy="auto", n_neighbors=3, kind_sel="all", n_jobs=4),
    "nearmiss1": NearMiss(sampling_strategy="auto", version=1, n_neighbors=3, n_jobs=4),
    "nearmiss2": NearMiss(sampling_strategy="auto", version=2, n_neighbors=3, n_jobs=4),
    "ihardthresh": InstanceHardnessThreshold(
        estimator=RandomForestClassifier(random_state=0),
        sampling_strategy="auto",
        random_state=0,
        n_jobs=4,
        cv=3,
    ),
}

OVERSAMPLER_DICT = {
    "random": RandomOverSampler(sampling_strategy="auto", random_state=0),
    "smote": SMOTE(sampling_strategy="auto", random_state=0, k_neighbors=5, n_jobs=4),
    "adasyn": ADASYN(sampling_strategy="auto", random_state=0, n_neighbors=5, n_jobs=4),
    "border1": BorderlineSMOTE(
        sampling_strategy="auto",
        random_state=0,
        k_neighbors=5,
        m_neighbors=10,
        kind="borderline-1",
        n_jobs=4,
    ),
    "border2": BorderlineSMOTE(
        sampling_strategy="auto",
        random_state=0,
        k_neighbors=5,
        m_neighbors=10,
        kind="borderline-2",
        n_jobs=4,
    ),
    "svmsmote": SVMSMOTE(
        sampling_strategy="auto",
        random_state=0,
        k_neighbors=5,
        m_neighbors=10,
        n_jobs=4,
        svm_estimator=SVC(kernel="linear"),
    ),
}
