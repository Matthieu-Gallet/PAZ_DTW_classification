from sklearn.model_selection import StratifiedGroupKFold
from features.statiscal_descriptor import *
from features.balance_data import *
from features.DBA_features import *
from utils.helper_functions import *
import glob, os, pprint
import numpy as np

os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"


def evaluation_one_stats(xs_train, ys_train, grbs_train, stat):
    """
    Evaluate the metrics using a single statistical method, with
    the Euclidean and DTW metrics, using 20 different seeds and 3-fold

    Parameters
    ----------
    xs_train : array-like
        The training data.
    ys_train : array-like
        The training labels.
    grbs_train : array-like
        The training group labels.
    stat : str
        The statistical method to use.

    Returns
    -------
    f1G_euc : ndarray
        The F1 scores using the Euclidean metric.
    f1G_dtw : ndarray
        The F1 scores using the DTW metric.
    cd_euc : ndarray
        The confusion matrix using the Euclidean metric.
    cd_dtw : ndarray
        The confusion matrix using the DTW metric.
    infG0 : ndarray
        The dictionary containing the False Negative and False Positive group counts using the Euclidean metric.
    infG1 : ndarray
        The dictionary containing the False Negative and False Positive group counts using the DTW metric.
    """
    f1G_euc, f1G_dtw, cd_euc, cd_dtw, infG0, infG1 = [], [], [], [], [], []
    for seed in tqdm.tqdm(range(20)):
        gk = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        for train_index, test_index in gk.split(xs_train, ys_train, grbs_train):
            x_train, x_test, y_train, y_test, grb_train, grb_test = prepare_data(
                train_index, test_index, xs_train, ys_train, grbs_train
            )
            if np.unique(y_test).shape[0] < 4:
                pass
            else:
                f1, cd0, inf0 = 0, 0, 0
                f1_s, cd1, inf1 = 0, 0, 0
                f1, cd0, inf0 = benchmark_MDPI(
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    grb_test,
                    stat,
                    metric="dtw",
                    metric_params={
                        "global_constraint": "sakoe_chiba",
                        "sakoe_chiba_radius": 3,
                    },
                )

                f1_s, cd1, inf1 = benchmark_MDPI(
                    x_train, x_test, y_train, y_test, grb_test, stat, metric="euclidean"
                )
                f1G_euc.append(f1_s)
                f1G_dtw.append(f1)
                cd_euc.append(cd1)
                cd_dtw.append(cd0)
                infG0.append(inf0)
                infG1.append(inf1)
    f1G_euc = np.array(f1G_euc)
    f1G_dtw = np.array(f1G_dtw)
    cd_euc = np.array(cd_euc)
    cd_dtw = np.array(cd_dtw)
    return f1G_euc, f1G_dtw, cd_euc, cd_dtw, infG0, infG1


def main(datapath, polari):
    datapath = f"{datapath}/*spat*kfold_{polari}.h5"
    list_h5_0 = glob.glob(datapath)[0]
    (
        x,
        y,
        gr,
        org,
        _,
    ) = load_h5(list_h5_0)

    idx = optimal_balance(np.dstack([y, gr])[0], step=1)
    xb_train, yb_train, grb_train = x[idx], y[idx], gr[idx]
    clasofinterest = ["ICA", "HAG", "ABL", "ACC"]
    idxTR = [i for i in range(xb_train.shape[0]) if yb_train[i] in clasofinterest]
    xtr, ys_train, grb_train = (xb_train[idxTR], yb_train[idxTR], grb_train[idxTR])
    xtr = xtr**0.5
    print(xtr.shape, np.unique(ys_train, return_counts=True))
    if len(xtr.shape) > 4:
        ##################################""
        # CP = (VV - VH) / sqrt(VV^2 + VH^2)
        xtr = (xtr[:, :, :, :, 0] - xtr[:, :, :, :, 1]) / (
            xtr[:, :, :, :, 0] ** 2 + xtr[:, :, :, :, 1] ** 2
        ) ** 0.5
        sstats = [
            "mean",
            "kurtosis",
            "CV",
        ]
    else:
        sstats = [
            "log_k1",
            "mean",
            "kurtosis",
            "CV",
        ]
    xs_train = Stats_SAR(sstats).fit_transform(xtr)

    wins = datapath.split("/")[-2]
    name_file = f"./result_amp/{wins}_{datapath.split('/')[-1].split('.')[0]}_II.txt"
    os.makedirs(os.path.dirname(name_file), exist_ok=True)
    print(xs_train.shape, ys_train.shape, grb_train.shape)

    with open(name_file, "w") as f:
        for m in range(len(sstats)):
            print(sstats[m])
            f1G_euc, f1G_dtw, cd_euc, cd_dtw, inf0, inf1 = evaluation_one_stats(
                xs_train, ys_train, grb_train, m
            )
            f.write(f"Statistics: {sstats[m]} \n")
            f.write(f"{f1G_dtw.shape} \n")
            line = (
                "KNN euc "
                + f"{f1G_euc.mean().round(2)}:"
                + f"{f1G_euc.mean(axis=0).round(2)}"
                + "+/-"
                + f"{f1G_euc.std(axis=0).round(2)}"
                + "\n"
            )
            f.write(line)
            line = (
                "KNN dtw "
                + f"{f1G_dtw.mean().round(2)}:"
                + f"{f1G_dtw.mean(axis=0).round(2)}"
                + "+/-"
                + f"{f1G_dtw.std(axis=0).round(2)}"
                + "\n"
            )
            f.write(line)

            f.write("KNN euc \n")
            f.write(pprint.pformat(aggregate_dico(inf0, proba=True), width=1))
            f.write("\n")
            f.write("KNN dtw \n")
            f.write(pprint.pformat(aggregate_dico(inf1, proba=True), width=1))
            f.write("\n")
            np.save(
                os.path.dirname(name_file) + f"/{wins}_{sstats[m]}_{polari}_euc.npy",
                aggregate_dico(inf0, proba=True),
            )
            np.save(
                os.path.dirname(name_file) + f"/{wins}_{sstats[m]}_{polari}_dtw.npy",
                aggregate_dico(inf1, proba=True),
            )


if __name__ == "__main__":
    ws = 7
    dpt = f"../../dataseth5/winsize_{ws}/"
    pol = ["HH", "HV", "HV&HH"]
    for polari in pol:
        main(dpt, polari)
