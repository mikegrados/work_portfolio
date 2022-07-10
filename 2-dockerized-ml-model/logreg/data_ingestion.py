import os
import re
import config
import numpy as np
import pandas as pd


def data_retrieval(url: str) -> None:
    """Retrieve the titanic data set from a URL.

    Parameters
    ----------
    url: str
        String of the URL where the dataset live.

    Returns
    -------
    None
        The data is stored as .csv in a given path.
    """

    # Loading data from specific url
    data = pd.read_csv(url)

    # Uncovering missing data
    data.replace("?", np.nan, inplace=True)
    data["age"] = data["age"].astype("float")
    data["fare"] = data["fare"].astype("float")

    # helper function 1
    def get_first_cabin(row):
        """Obtain the first letter of the cabin.

        Parameters
        ----------
        row: str
            Information of the cabin.

        Returns
        -------
        str
            Either the first letter of the cabin or NaN.

        """
        try:
            return row.split()[0]
        except Exception:
            return np.nan

    # helper function 2
    def get_title(passenger: str) -> str:
        """Obtain the title of the passenger based on the information 
           provided in the column Name of the dataset.

        Parameters
        ----------
        passenger: str
            The name of passenger (column Name in the dataset)

        Returns
        -------
        str
            Mrs, Mr, Miss, Master, Other
        """
        line = passenger
        if re.search("Mrs", line):
            return "Mrs"
        elif re.search("Mr", line):
            return "Mr"
        elif re.search("Miss", line):
            return "Miss"
        elif re.search("Master", line):
            return "Master"
        else:
            return "Other"

    # Keep only one cabin | Extract the title from 'name'
    data["cabin"] = data["cabin"].apply(get_first_cabin)
    data["title"] = data["name"].apply(get_title)

    # Droping irrelevant columns
    data.drop(columns=config.DROP_COLS, inplace=True)

    PATH_SAVE_DATA = os.path.join(config.DATASETS_DIR, config.RETRIEVED_DATA)
    data.to_csv(PATH_SAVE_DATA, index=False)

    return None
