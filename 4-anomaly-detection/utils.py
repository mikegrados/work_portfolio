import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)


def plot_data_anomalies(df, target_col="target", dataset="", scale=True):
    idx_inliers = list(df[df[target_col] == 0].index)
    idx_outliers = list(df[df[target_col] == 1].index)

    if scale:
        X = df.drop(columns=target_col)
        scaler = MinMaxScaler().fit(X)
        scaled_data = scaler.transform(X)
    else:
        scaled_data = df.drop(columns=target_col)

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    plt.scatter(
        x=reduced_data[idx_inliers, 0],
        y=reduced_data[idx_inliers, 1],
        label="inliers",
        s=4,
        color="g",
    )
    plt.scatter(
        x=reduced_data[idx_outliers, 0],
        y=reduced_data[idx_outliers, 1],
        label="true outliers",
        marker="x",
        color="red",
    )
    plt.title(
        f"{dataset} | pos class: {df[target_col].value_counts(normalize=True).loc[1].round(3)}"
    )
    plt.legend()


def plot_predictions(X_test, y_test, classifier):

    test_pred = classifier.predict(X_test)

    idx_inliers_test = np.argwhere(y_test.values.ravel() == 0)
    idx_outliers_test = np.argwhere(y_test.values.ravel() == 1)
    idx_outliers_pred = np.argwhere(test_pred == 1)

    pca = PCA(n_components=2)
    X_test_red = pca.fit_transform(X_test)

    plt.scatter(
        x=X_test_red[idx_inliers_test, 0],
        y=X_test_red[idx_inliers_test, 1],
        label="inliers",
        s=5,
        color="g",
    )
    plt.scatter(
        x=X_test_red[idx_outliers_test, 0],
        y=X_test_red[idx_outliers_test, 1],
        label="true outliers",
        marker="x",
        color="red",
    )
    plt.scatter(
        x=X_test_red[idx_outliers_pred, 0],
        y=X_test_red[idx_outliers_pred, 1],
        facecolors="none",
        edgecolors="brown",
        s=80,
        label="pred outliers",
    )

    plt.legend()


def classification_metrics_report(X_test, y_test, classifier):
    """
    Args: The features and the target column; the labels are the categories, sklearn classifier object
    Calculates Classification metrics of interest
    Returns: A dictionary containing the classification metrics
    """

    # Generate the predictions on the test data which forms the basis of the evaluation metrics
    if "predict_proba" in dir(type(classifier)):
        test_pred = classifier.predict(X_test)
        y_scores = classifier.predict_proba(X_test)[:, 1]

    elif "decision_function" in dir(type(classifier)):
        test_pred = classifier.predict(X_test)
        test_pred = test_pred * -1
        test_pred[test_pred == -1] = 0

        decision_score_list = classifier.decision_function(X_test)
        scaled_decision_score_list = MinMaxScaler().fit_transform(
            decision_score_list.reshape(-1, 1)
        )
        y_scores = [
            1 - item for sublist in scaled_decision_score_list for item in sublist
        ]

    ### Confusion Matrix
    confusion_matrix_test_object = confusion_matrix(y_test, test_pred)

    # Initialize a dictionary to store the metrics we are interested in
    metrics_dict = Counter()

    # These are all the basic threshold-dependent metrics
    metrics_dict["Accuracy"] = float(
        "{0:.4f}".format(accuracy_score(y_test, test_pred))
    )

    # The following are more useful than the accuracy
    metrics_dict["Precision"] = float(
        "{0:.4f}".format(precision_score(y_test, test_pred, average="macro"))
    )
    metrics_dict["Recall"] = float(
        "{0:.4f}".format(recall_score(y_test, test_pred, average="macro"))
    )
    metrics_dict["F1"] = float(
        "{0:.4f}".format(f1_score(y_test, test_pred, average="macro"))
    )

    metrics_dict["TN"] = confusion_matrix_test_object[0][0]
    metrics_dict["TP"] = confusion_matrix_test_object[1][1]
    metrics_dict["FN"] = confusion_matrix_test_object[1][0]
    metrics_dict["FP"] = confusion_matrix_test_object[0][1]

    # These two are threshold-invariant metrics
    metrics_dict["ROC AUC"] = float("{0:.4f}".format(roc_auc_score(y_test, y_scores)))
    metrics_dict["Average_Precision"] = float(
        "{0:.4f}".format(
            average_precision_score(
                y_test, y_scores, average="macro", sample_weight=None
            )
        )
    )

    return metrics_dict
