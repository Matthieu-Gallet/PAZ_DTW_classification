import numpy as np
from tslearn.metrics import dtw
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score

import sys

sys.path.append("../")
from utils.helper_functions import *


def confusion_group(y_test, ypred, grb_test, True_group=False):
    """
    Compute the confusion group based on the predicted labels and ground truth labels.

    Parameters
    ----------
    y_test : numpy.ndarray
        The ground truth labels.
    ypred : numpy.ndarray
        The predicted labels.
    grb_test : numpy.ndarray
        The ground truth group values.
    True_group : bool, optional
        Whether to include the true group information in the output. Default is False.

    Returns
    -------
    dict
        A dictionary containing the confusion group information. The keys of the dictionary
        represent the different classes. The values are dictionaries containing the counts of
        different group values for false negatives (GRFN), false positives (GRFP), true
        positives (GRTP), and true negatives (GRTN). If True_group is False, the dictionary
        only contains GRFN and GRFP.
    """
    if True_group:
        infos = {"GRTP": {}, "GRTN": {}, "GRFN": {}, "GRFP": {}}
    else:
        infos = {"GRFN": {}, "GRFP": {}}
    for class_ in np.unique(y_test):
        idx_true = np.where(y_test == class_)[0]
        idx_trueF = np.where(y_test != class_)[0]

        idx_False_neg = idx_true[np.where(ypred[idx_true] != class_)[0]]
        idx_False_pos = idx_trueF[np.where(ypred[idx_trueF] == class_)[0]]
        idx_True_pos = idx_true[np.where(ypred[idx_true] == class_)[0]]
        idx_True_neg = idx_trueF[np.where(ypred[idx_trueF] != class_)[0]]
        grb_FN = grb_test[idx_False_neg]
        grb_FP = grb_test[idx_False_pos]
        grb_TP = grb_test[idx_True_pos]
        grb_TN = grb_test[idx_True_neg]
        if True_group:
            infos["GRTN"][class_] = np.unique(grb_TN, return_counts=True)
            infos["GRTP"][class_] = np.unique(grb_TP, return_counts=True)
        infos["GRFN"][class_] = np.unique(grb_FN, return_counts=True)
        infos["GRFP"][class_] = np.unique(grb_FP, return_counts=True)
    return prepare_dico(infos)


def benchmark_MDPI(x_train, x_test, y_train, y_test, grb_test, s, metric, **kwargs):
    if metric == "dtw":
        clf = KNeighborsClassifier(
            n_neighbors=25,
            metric=dtw,
            weights="uniform",
            n_jobs=-1,
            **kwargs,
        )
        clf.fit(x_train[:, s, :], y_train)
        ypred = clf.predict(x_test[:, s, :])
    else:
        clf = KNeighborsClassifier(
            n_neighbors=25, metric="euclidean", weights="uniform", n_jobs=-1
        )
        clf.fit(x_train[:, s, :], y_train)
        ypred = clf.predict(x_test[:, s, :])

    labels = np.unique(y_test)
    cf = 100 * confusion_matrix(y_test, ypred, normalize="true", labels=labels)
    f1 = 100 * f1_score(y_test, ypred, average=None)
    cd = pd.DataFrame(cf, index=labels, columns=labels)
    info = confusion_group(y_test, ypred, grb_test)
    return (f1, cd, info)
