import re
import numpy as np
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class MissingIndicator(BaseEstimator, TransformerMixin):
    """Create binary column indicating missing values for a given columns"""

    def __init__(self, variables: List[str] = None):
        self.variables = variables if isinstance(variables, list) else [variables]

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """This method is just for compatibility with scikit-learn 

        Parameters
        ----------
        X: pd.DataFram
            Matrix of features
            
        y: pd.Series
            (Default value = None)

        Returns
        -------
        The same object 

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Produces the new columns to the given dataset

        Parameters
        ----------
        X: pd.DataFrame :
            Matrix of features

        Returns
        -------
        Augmented dataframe

        """
        X = X.copy()
        for var in self.variables:
            X[var + "_nan"] = X[var].isnull().astype(int)

        return X


class ExtractLetters(BaseEstimator, TransformerMixin):
    """Modify the column Cabin to just the first letter of the string"""

    def __init__(self):
        self.variable = "cabin"

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """This method is just for compatibility with scikit-learn

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features
            
        y: pd.Series :
            (Default value = None)

        Returns
        -------
        The same object

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Changes the column Cabin to only the first letter of the cabin

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features

        Returns
        -------
        DataFrame with the modification in the column Cabin 
        """
        X = X.copy()
        X[self.variable] = X[self.variable].apply(
            lambda x: "".join(re.findall("[a-zA-Z]+", x)) if type(x) == str else x
        )
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Fill missing values of categorical variables"""

    def __init__(self, variables: List[str] = None):
        self.variables = variables if isinstance(variables, list) else [variables]

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """This method is just for compatibility with scikit-learn

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features
            
        y: pd.Series
            (Default value = None)

        Returns
        -------
        The same object

        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill the missing values of categorical columns with the value "Missing".

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features

        Returns
        -------
        DataFrame with no missing data in the categorical columns

        """
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna("Missing")
        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    """Fill missing values of numerical variables"""

    def __init__(self, variables: List[str] = None):
        self.variables = variables if isinstance(variables, list) else [variables]

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """This method learn the median for each numerical column and persist 
           this information in a dictionary.

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features
            
        y: pd.Series :
            (Default value = None)

        Returns
        -------
        The object with a dictionary of the media per numerical column as a 
        property of the object.

        """
        self.median_dict_ = {}
        for var in self.variables:
            self.median_dict_[var] = X[var].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill the missing values of numerical columns with the median.

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features

        Returns
        -------
        DataFrame with no missing data in the numerical columns

        """
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.median_dict_[var])
        return X


class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Relabel and group classes of categorical columns based on their relative frequency"""

    def __init__(self, tol: float = 0.05, variables: List[str] = None):
        self.tol = tol
        self.variables = variables if isinstance(variables, list) else [variables]

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """For each categorical variable, this method selects those classes 
           with low frequency and relabel them as class Rare.

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features
            
        y: pd.Series
            (Default value = None)

        Returns
        -------
        Dictionary with the classes that are going to be relabeled, for each 
        categorical column.
        """
        self.rare_labels_dict = {}
        for var in self.variables:
            t = pd.Series(X[var].value_counts() / np.float(X.shape[0]))
            self.rare_labels_dict[var] = list(t[t < self.tol].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Relabel the classes with low relative frequency to class Rare

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features

        Returns
        -------
        DataFrame with relabeled classes for each categorical variable.

        """
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.rare_labels_dict[var]), "rare", X[var])
        return X


class OneHotEncoder(BaseEstimator, TransformerMixin):
    """Replace the categorical columns with the dummy variables"""

    def __init__(self, variables: List[str] = None):
        self.variables = variables if isinstance(variables, list) else [variables]

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Build the dummy variables for categorical variables

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features
            
        y: pd.Series
            (Default value = None)

        Returns
        -------
        DataFrame with the dummy variables

        """
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Concatenate the dummy variables to the given DataFrame and remove 
           the original categorical variables

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features
        
        Returns
        -------
        DataFrame with dummy variables

        """
        X = X.copy()
        X = pd.concat([X, pd.get_dummies(X[self.variables], drop_first=True)], axis=1)
        X.drop(columns=self.variables, inplace=True)

        # Adding missing dummies, if any
        missing_dummies = [var for var in self.dummies if var not in X.columns]
        if len(missing_dummies) != 0:
            for col in missing_dummies:
                X[col] = 0

        return X


class OrderingFeatures(BaseEstimator, TransformerMixin):
    """Establish an order of the columns"""

    def __init__(self):
        return None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Provides a list of feature names

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features
            
        y: pd.Series :
            (Default value = None)

        Returns
        -------
        List of (ordered) features

        """
        self.ordered_features = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Order the features

        Parameters
        ----------
        X: pd.DataFrame
            Matrix of features
        
        Returns
        -------
        DataFrame with features ordered.

        """
        return X[self.ordered_features]
