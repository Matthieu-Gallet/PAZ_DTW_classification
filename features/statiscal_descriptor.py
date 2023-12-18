from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from skimage.measure import shannon_entropy
from tqdm import tqdm
import numpy as np


def shannon_entropy_tensor(x, axis=0):
    if axis == 0:
        temp = [shannon_entropy(x[i, :]) for i in range(x.shape[axis])]
    elif axis == 1:
        temp = [shannon_entropy(x[:, i]) for i in range(x.shape[axis])]
    else:
        raise ValueError("axis must be 0, 1")
    return np.array(temp)


def optimal_bins(data):
    """Compute the optimal number of bins for a histogram
    using the Freedman-Diaconis rule

    Parameters
    ----------
    data : np.array
        data to compute the optimal number of bins

    Returns
    -------
    int
        optimal number of bins
    """
    n = len(data)
    min_x = np.min(data)
    max_x = np.max(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    hbins = int(np.ceil((1 / (2 * (q3 - q1))) * (n ** (1 / 3)) * (max_x - min_x)))
    return hbins


# @njit
def log_k1(data, axis=0):
    log_dat = np.log(data)
    k1 = np.mean(log_dat, axis=axis)
    return k1, log_dat


# @njit
def log_k2(data, axis=0):
    k1, log_dat = log_k1(data, axis=axis)
    k2 = np.mean(log_dat**2, axis=axis) - k1**2
    return k1, k2, log_dat


# @njit
def log_k3(data, axis=0):
    k1, k2, log_dat = log_k2(data, axis=axis)
    k3 = (
        np.mean(log_dat**3, axis=axis)
        - 3 * k1 * np.mean(log_dat**2, axis=axis)
        + 2 * k1**3
    )
    return k1, k2, k3, log_dat


# @njit
def log_k4(data, axis=0):
    k1, k2, k3, log_dat = log_k3(data, axis=axis)
    m2 = np.mean(log_dat**2, axis=axis)
    k4 = (
        np.mean(log_dat**4, axis=axis)
        - 4 * k1 * np.mean(log_dat**3, axis=axis)
        - 3 * m2**2
        + 12 * k1**2 * m2
        - 6 * k1**4
    )
    return k1, k2, k3, k4, log_dat


def stats_(data, type_stats, axis=0):
    if type_stats == "mean":
        return np.mean(data, axis=0)
    elif type_stats == "std":
        return np.std(data, axis=0)
    elif type_stats == "CV":
        return np.std(data, axis=0) / np.mean(data, axis=0)
    elif type_stats == "skew":
        return skew(data, axis=0)
    elif type_stats == "kurtosis":
        return kurtosis(data, axis=0)
    elif type_stats == "log_k1":
        return log_k1(data)[0]
    elif type_stats == "log_k2":
        return log_k2(data)[1]
    elif type_stats == "log_k3":
        return log_k3(data)[2]
    elif type_stats == "log_k4":
        return log_k4(data)[3]
    elif type_stats == "entropy":
        # return -np.sum(data * np.log2(data), axis=0)
        return shannon_entropy_tensor(data, axis=0)

    else:
        raise ValueError("Unknown type of stats")


def calcul_hist(data, n_bands, bin):
    """Compute the histogram of a list of array or a matrix on the first axis

    Parameters
    ----------
    data : np.array
        data to compute the histogram
    n_bands : int
        number of bands
    bin : int
        number of bins

    Returns
    -------
    np.array
        vector of histogram of each band between 0 and 1
    """

    vec = []
    for j in range(n_bands):
        hst = np.histogram(data[:, j], bins=bin, range=(0, 1))[0]
        vec.append(hst)
    return np.array(vec).flatten()


class Stats_SAR(BaseEstimator, TransformerMixin):
    """Tranform a tensor or a list of matrix into a vector array of the
    selected moments of each band.
    """

    def __init__(self, type_stats="mean", workers=-1) -> None:
        super().__init__()
        self.type_stats = type_stats
        self.workers = workers

    def fit(self, X, y=None):
        return self

    def get_list_stats(self):
        return [
            "mean",
            "std",
            "CV",
            "skew",
            "kurtosis",
            "log_k1",
            "log_k2",
            "log_k3",
            "log_k4",
        ]

    def transform(self, X, y=None):
        self.vectorize_ = []
        self._size_data = X.shape
        if len(self._size_data) == 3:
            self.n_bands = X.shape[-1]
            X_t = X
        elif len(self._size_data) == 4:
            self.n_bands = X.shape[-1]
            X_t = X.reshape(
                self._size_data[0],
                self._size_data[1] * self._size_data[2],
                self.n_bands,
            )
        elif len(self._size_data) == 5:
            self.n_bands = self._size_data[-2]
            X_t = X.reshape(
                self._size_data[0],
                self._size_data[1] * self._size_data[2],
                self.n_bands,
                self._size_data[-1],
            )

        else:
            raise ValueError("Data dimension not supported")
        for tp_stats in self.type_stats:
            self.vectorize_.append(
                Parallel(n_jobs=self.workers)(delayed(stats_)(x, tp_stats) for x in X_t)
            )
        res = np.array(self.vectorize_)
        if len(self._size_data) == 5:
            res = res[0, :, :, :]
            return np.moveaxis(res, -1, -2)
        else:
            return np.moveaxis(res, 0, -2)


class Hist_SAR(BaseEstimator, ClassifierMixin):
    """Tranform a tensor or a list of matrix into a vector array of the
    histogram of each band.
    """

    def __init__(self, nbins=16) -> None:
        super().__init__()
        self.nbins = nbins

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.vectorize_ = []
        self._size_data = X.shape
        if len(self._size_data) == 3:
            self.n_bands = X.shape[-1]
            X_t = X
        elif len(self._size_data) == 4:
            self.n_bands = X.shape[-1]
            X_t = X.reshape(
                self._size_data[0],
                self._size_data[1] * self._size_data[2],
                self.n_bands,
            )
        elif len(self._size_data) == 5:
            self.n_bands = X.shape[-2]
            X_t = X.reshape(
                self._size_data[0],
                self._size_data[1] * self._size_data[2],
                self._size_data[3],
                self.n_bands,
            )
            ax_m = 1
        else:
            raise ValueError("Data dimension not supported")
        if self.nbins == -1:
            self.nbins = optimal_bins(X_t)

        self.vectorize_ = Parallel(n_jobs=-2)(
            delayed(calcul_hist)(x, self.n_bands, self.nbins) for x in tqdm(X_t)
        )
        return np.array(self.vectorize_)
