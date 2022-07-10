import os
import joblib
import config
import pandas as pd
from data_ingestion import data_retrieval
from data_segregation import data_split
from data_processing import logreg_pipeline


def model_training():
    """Run the whole pipeline: 1) retrieve the data, 2) split the 
       data into train and test sets and persist them as .csv files,
       3) run the preprocessing steps and train the model through a
       pipeline, and 4) persist the trained pipeline using joblib.dump
    
    Parameters
    ----------
    No parameters.
    
    Returns
    -------
    None

    """

    URL = config.URL
    DATASETS_DIR = config.DATASETS_DIR
    TRAIN_DATA_FILE = config.TRAIN_DATA_FILE
    TRAINED_MODEL_DIR = config.TRAINED_MODEL_DIR
    PIPELINE_SAVE_FILE = config.PIPELINE_SAVE_FILE

    # Step 1: load and split data
    # ===========================
    data_retrieval(URL)
    data_split()

    # Step 2: Read training data and fit logreg
    # =========================================
    X_train = pd.read_csv(TRAIN_DATA_FILE)
    y_train = pd.read_csv(os.path.join(DATASETS_DIR, "y_train.csv"))
    logreg_pipeline.fit(X_train, y_train.values.ravel())

    # Step 3: Save the trained model
    # ==============================
    save_path = os.path.join(TRAINED_MODEL_DIR, PIPELINE_SAVE_FILE)
    object_to_persist = logreg_pipeline

    joblib.dump(object_to_persist, save_path)

    return None


if __name__ == "__main__":
    model_training()
