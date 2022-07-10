import os
import config
import pandas as pd
from sklearn.model_selection import train_test_split


def data_split():
    """Split the data into train and test sets, and persist the 
       data as .csv in a given path.
    
    Parameters
    ----------
    No parameters

    Returns
    -------
    None

    """

    DATASETS_DIR = config.DATASETS_DIR
    RETRIEVED_DATA = config.RETRIEVED_DATA

    TARGET = config.TARGET
    TEST_SIZE = config.TEST_SIZE
    SEED_SPLIT = config.SEED_SPLIT
    TRAIN_DATA_FILE = config.TRAIN_DATA_FILE
    TEST_DATA_FILE = config.TEST_DATA_FILE

    df = pd.read_csv(os.path.join(DATASETS_DIR, RETRIEVED_DATA))

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(TARGET, axis=1),
        df[TARGET],
        test_size=TEST_SIZE,
        random_state=SEED_SPLIT,
    )

    PATH_YTRAIN_FILE = os.path.join(DATASETS_DIR, "y_train.csv")
    PATH_YTEST_FILE = os.path.join(DATASETS_DIR, "y_test.csv")

    X_train.to_csv(TRAIN_DATA_FILE, index=False)
    y_train.to_csv(PATH_YTRAIN_FILE, index=False, header=True)
    X_test.to_csv(TEST_DATA_FILE, index=False)
    y_test.to_csv(PATH_YTEST_FILE, index=False, header=True)

    return None
