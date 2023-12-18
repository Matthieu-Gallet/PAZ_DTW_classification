import numpy as np
import tqdm


def prepare_data(train_index, test_index, xs_train, ys_train, grb_train=-1):
    """
    Prepare the data for training and testing.

    Parameters
    ----------
    train_index : array-like
        Indices of the training data.
    test_index : array-like
        Indices of the testing data.
    xs_train : array-like
        Input features for training.
    ys_train : array-like
        Target labels for training.
    grb_train : int or array-like, optional
        Group labels for training. Defaults to -1.

    Returns
    -------
    tuple
        A tuple containing the prepared data:
        - xp_train : array-like
            Input features for training.
        - xp_test : array-like
            Input features for testing.
        - yp_train : array-like
            Target labels for training.
        - yp_test : array-like
            Target labels for testing.
        - grbs_train : array-like, optional
            Group labels for training. Only returned if grb_train is not an integer.
        - grbs_test : array-like, optional
            Group labels for testing. Only returned if grb_train is not an integer.
    """
    xp_train, xp_test = xs_train[train_index], xs_train[test_index]
    yp_train, yp_test = ys_train[train_index], ys_train[test_index]

    idx = np.arange(xp_train.shape[0])
    np.random.shuffle(idx)
    xp_train, yp_train = xp_train[idx], yp_train[idx]
    if type(grb_train) != int:
        grbs_train = grb_train[train_index][idx]

    idx = np.arange(xp_test.shape[0])
    np.random.shuffle(idx)
    xp_test, yp_test = xp_test[idx], yp_test[idx]
    if type(grb_train) != int:
        grbs_test = grb_train[test_index][idx]

    if type(grb_train) != int:
        return xp_train, xp_test, yp_train, yp_test, grbs_train, grbs_test
    else:
        return xp_train, xp_test, yp_train, yp_test


def minmax_norm(data):
    """Normalize data between 0 and 1

    Parameters
    ----------
    data : np.array
        data to normalize

    Returns
    -------
    np.array
        normalized data
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def optimal_balance(y, step=25, seed=42):
    """
    Find the optimal threshold for balancing the data.
    Thresholds are tested from 0 to the maximum number of samples in a group with a step size of 25.
    The optimal threshold is the one that maximizes the number of samples in each class by
    undersampling the groups with the most samples and maximizing the number of groups.

    Parameters
    ----------
    y : numpy.ndarray
        The input data array.
    step : int, optional
        The step size for threshold values. Default is 25.
    seed : int, optional
        The seed value for random number generation. Default is 42.

    Returns
    -------
    numpy.ndarray
        The indices of the balanced data.

    Raises
    ------
    ValueError
        If no optimal threshold is found for balancing the data.
    """
    opti = []
    ncl, ccounts = np.unique(y[:, 0], return_counts=True)
    min_cla = ncl[ccounts.argmin()]
    maxgr_mincl = np.unique(y[y[:, 0] == min_cla, 1], return_counts=True)[1].max()
    thresh = np.arange(0, maxgr_mincl, step)
    for t in tqdm.tqdm(thresh, desc="Balancing data"):
        cl_idx = groupAclass_balanced(y, t, seed)
        sum_sample_class = np.unique(y[cl_idx, 0], return_counts=True)[1]
        check_res = len(set(sum_sample_class)) == 1 and len(sum_sample_class) == len(
            ncl
        )

        if check_res:
            theta = np.sum(sum_sample_class) / len(y)
            opti.append(theta)
    if len(opti) == 0:
        raise ValueError("No optimal threshold found for balancing the data")
    else:
        best_thresh = thresh[np.argmax(opti)]
        best_idx = groupAclass_balanced(y, best_thresh, seed)
        return best_idx


def undersample_idx(y, idxo, s):
    """
    Undersamples the data by randomly selecting a subset of indices for each class.

    Parameters
    ----------
    y : array-like
        The class labels.
    idxo : array-like
        The original indices.
    s : int or np.random.Generator
        The seed or random number generator.

    Returns
    -------
    array-like
        The undersampled indices.
    """
    if type(s) == int:
        rng = np.random.default_rng(s)
    elif type(s) == np.random.Generator:
        rng = s
    idx = []
    ncl, ccounts = np.unique(y, return_counts=True)
    cmin = ccounts.min()
    for cl in ncl:
        idx.extend(rng.choice(np.argwhere(y == cl).flatten(), size=cmin, replace=False))
    return idxo[idx]


def groupAclass_balanced(y, thresh, s):
    """
    Balance the data by undersampling the minority classes.

    Parameters
    ----------
    y : numpy.ndarray
        The input array containing the class labels.
    thresh : int
        The threshold value for minimum class count.
    s : int
        The desired size of the balanced dataset.

    Returns:
        numpy.ndarray: The indices of the balanced dataset.

    """
    idx = np.arange(len(y))
    ngr, gcounts = np.unique(y[:, 1], return_counts=True)
    grmin = ngr[gcounts < thresh]
    idx = np.array([i for i in idx if y[i, 1] not in grmin])
    gr_idx = undersample_idx(y[idx, 1], idx, s)
    cl_idx = undersample_idx(y[gr_idx, 0], gr_idx, s)
    return cl_idx
