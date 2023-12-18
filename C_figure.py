import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

import matplotlib.colors as mcolors
from copy import deepcopy
from A_create_dataset import *

from datetime import datetime
import matplotlib.dates as mdates
from tslearn import metrics

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{cmbright}",
            ]
        ),
    }
)


def prepare_data_multi_date(pa2th, pol):
    outpath = "../data/"
    cdlf = Dataset_tiff2hdf5(
        pa2th, different_group=True, n_jobs=-1, outpath=outpath, extension="spatial"
    )
    cdlf2 = Dataset_tiff2hdf5(
        pa2th, different_group=True, n_jobs=-1, outpath=outpath, extension="spatial"
    )
    cdlf3 = Dataset_tiff2hdf5(
        pa2th, different_group=True, n_jobs=-1, outpath=outpath, extension="spatial"
    )

    cdlf.extract_data(
        f"classe in  ['ICA','HAG','ABL','ACC','FOR','CIT','ROC','PLA'] & date > '2020-01-01' & date < '2020-01-09'",
        polarisation=f"{pol}",
        winsize=32,
        save=False,
    )
    cdlf2.extract_data(
        f"classe in  ['ICA','HAG','ABL','ACC','FOR','CIT','ROC','PLA'] & date > '2020-04-01' & date < '2020-04-09'",
        polarisation=f"{pol}",
        winsize=32,
        save=False,
    )
    cdlf3.extract_data(
        f"classe in  ['ICA','HAG','ABL','ACC','FOR','CIT','ROC','PLA'] & date > '2020-08-01' & date < '2020-08-09'",
        polarisation=f"{pol}",
        winsize=32,
        save=False,
    )

    cla = ["ICA", "HAG", "ABL", "ACC", "FOR", "CIT", "ROC", "PLA"]
    data = []
    data2 = []
    data3 = []
    for i in cla:
        idx = np.where(cdlf.y[:, 0] == i)[0]
        data.append(cdlf.X[idx].reshape(-1) ** 0.5)
        idx = np.where(cdlf2.y[:, 0] == i)[0]
        data2.append(cdlf2.X[idx].reshape(-1) ** 0.5)
        idx = np.where(cdlf3.y[:, 0] == i)[0]
        data3.append(cdlf3.X[idx].reshape(-1) ** 0.5)

    return [data, data2, data3], [cdlf, cdlf2, cdlf3]


def plot_hist(pat2h, pol):
    cla = ["ICA", "HAG", "ABL", "ACC", "FOR", "CIT", "ROC", "PLA"]
    dt, cd = prepare_data_multi_date(pa2th, pol)
    data, data2, data3 = dt
    cdlf, cdlf2, cdlf3 = cd
    l = 1.25
    fig, ax = plt.subplots(2, 4, figsize=(17 / l, 6 / l), sharex=True, sharey=True)
    for cl in range(len(cla)):
        d3, b3 = np.histogram(data3[cl], bins=75, range=(1e-8, 1.5))
        ax[cl // 4, cl % 4].bar(
            b3[:-1],
            d3 / np.sum(d3),
            width=b3[1] - b3[0],
            alpha=0.8,
            label=cdlf3.list_files.date.iloc[0].strftime("%d %B"),
            edgecolor="black",
            linewidth=0.2,
            color="tab:orange",
        )
        d, b = np.histogram(data[cl], bins=75, range=(1e-8, 1.5))
        ax[cl // 4, cl % 4].bar(
            b[:-1],
            d / np.sum(d),
            width=b[1] - b[0],
            alpha=0.75,
            label=cdlf.list_files.date.iloc[0].strftime("%d %B"),
            edgecolor="black",
            linewidth=0.2,
            color="tab:blue",
        )
        d2, b2 = np.histogram(data2[cl], bins=75, range=(1e-8, 1.5))
        ax[cl // 4, cl % 4].bar(
            b2[:-1],
            d2 / np.sum(d2),
            width=b2[1] - b2[0],
            alpha=0.5,
            label=cdlf2.list_files.date.iloc[0].strftime("%d %B"),
            edgecolor="black",
            linewidth=0.2,
            color="tab:green",
        )
        ax[cl // 4, cl % 4].set_title(cla[cl], fontsize=15, fontweight="bold")
        if cl % 4 == 0 and cl // 4 == 0:
            ax[cl // 4, cl % 4].set_ylabel("Frequency (normalized)", fontsize=15)
            ax[cl // 4, cl % 4].yaxis.set_label_coords(-0.175, -0.2)

        if cl // 4 == 1 and cl % 4 == 0:
            ax[cl // 4, cl % 4].set_xlabel("Amplitude (linear $\sigma_0$)", fontsize=15)
            ax[cl // 4, cl % 4].xaxis.set_label_coords(2.25, -0.15)
        ax[cl // 4, cl % 4].set_xlim(1e-8, 1)
        ax[cl // 4, cl % 4].set_ylim(0, 0.22)
        ax[cl // 4, cl % 4].legend()
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.22, wspace=0.1)
    plt.savefig(
        f"{pat2h}/hist_2020_{pol}.pdf", dpi=300, bbox_inches="tight", backend="pgf"
    )


def comparaison_hh_hv(pa2th):
    cla = ["ICA", "HAG", "ABL", "ACC", "FOR", "CIT", "ROC", "PLA"]
    l = 1.25
    f, ax = plt.subplots(2, 4, figsize=(17 / l, 6 / l))  # , sharex=True, sharey=True)
    for c in cla:
        list_files = glob.glob(pa2th.replace("?", c))
        list_files.sort()
        ax[cla.index(c) // 4, cla.index(c) % 4].grid(zorder=0, linewidth=0.5, alpha=0.5)
        ngroup = len(list_files)
        for f in list_files:
            df = pd.read_csv(f)
            ax[cla.index(c) // 4, cla.index(c) % 4].scatter(
                df["k1_HH"],
                df["k1_HV"],
                s=80,
                alpha=0.8,
                label="XXX" + f.split("/")[-1].split(".")[0].split("_")[1][3:],
                edgecolor="black",
                linewidth=0.1,
            )
            ax[cla.index(c) // 4, cla.index(c) % 4].set_title(
                c + f" ({ngroup})", fontsize=14, fontweight="bold"
            )
            if c == "CIT":
                ax[cla.index(c) // 4, cla.index(c) % 4].legend(
                    bbox_to_anchor=(3.375, 1.9),
                    loc="upper left",
                    borderaxespad=0.0,
                    title_fontproperties={"weight": "bold", "size": 15},
                    fontsize=10,
                    handletextpad=0.1,
                    title="Area",
                )
        if cla.index(c) // 4 == 1 and cla.index(c) % 4 == 0:
            ax[cla.index(c) // 4, cla.index(c) % 4].set_xlabel(
                "$\kappa_1$ HH", fontsize=16
            )
            ax[cla.index(c) // 4, cla.index(c) % 4].xaxis.set_label_coords(2.175, -0.15)
        if cla.index(c) % 4 == 0 and cla.index(c) // 4 == 0:
            ax[cla.index(c) // 4, cla.index(c) % 4].set_ylabel(
                "$\kappa_1$ HV", fontsize=16
            )
            ax[cla.index(c) // 4, cla.index(c) % 4].yaxis.set_label_coords(-0.125, -0.2)

    plt.subplots_adjust(hspace=0.3, wspace=0.15)

    plt.savefig("{pa2th}/comparaison_k1.pdf", bbox_inches="tight", backend="pgf")


def illustrate_knn(
    n_samples=150,
    n_features=2,
    n_classes=2,
    n_neighbors=5,
    new_points=[(0, 0)],
    save=False,
):
    np.random.seed(0)
    # Generate a 2D dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_classes=n_classes,
        class_sep=0.75,
        random_state=0,
    )
    X += np.random.normal(scale=1.25, size=X.shape)

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model
    knn.fit(X, y)

    # Create a mesh to plot the decision boundaries
    h = 0.075  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    f, ax = plt.subplots(1, 2, figsize=(15, 4), sharex=True, sharey=True)
    custom_cmap = ListedColormap([plt.cm.tab20c.colors[2], plt.cm.tab20c.colors[6]])
    custom_cmap2 = ListedColormap([plt.cm.tab20c.colors[1], plt.cm.tab20c.colors[5]])
    # Plot the training points
    clas1 = np.argwhere(y == 1)
    ax[0].scatter(
        X[clas1, 0],
        X[clas1, 1],
        # c=y[clas1],
        edgecolors="k",
        color=custom_cmap2.colors[1],
        marker="o",
        s=40,
        label="Class 1",
    )
    clas0 = np.argwhere(y == 0)
    ax[0].scatter(
        X[clas0, 0],
        X[clas0, 1],
        # c=y[clas0],
        edgecolors="k",
        color=custom_cmap2.colors[0],
        marker="o",
        s=40,
        label="Class 2",
    )

    ax[1].pcolormesh(xx, yy, Z, cmap=custom_cmap, alpha=0.75, linewidth=0, zorder=0)
    ax[1].scatter(
        X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=custom_cmap2, marker="o", s=40
    )
    # Add new points, predict their labels, and draw lines to their neighbors
    labels_added = []
    i = 0
    for point in new_points:
        label = knn.predict([point])
        label_text = "Predicted Class 1" if label else "Predicted Class 2"
        if label_text not in labels_added:
            ax[1].scatter(
                *point,
                color="red" if label else "blue",
                edgecolors="k",
                marker="^",
                s=60,
                label=label_text,
            )
            labels_added.append(label_text)
        else:
            ax[1].scatter(
                *point,
                color="red" if label else "blue",
                edgecolors="k",
                marker="^",
                s=60,
            )

        distances, indices = knn.kneighbors([point])
        for index in indices[0]:
            ax[1].plot(
                *zip(point, X[index]),
                color="red" if label else "blue",
                linestyle="--",
                linewidth=0.75,
                label=" Distance to neighbors" if i == 8 else None,
            )
            i += 1
    ax[0].legend(loc="upper right", fontsize=10)
    ax[1].legend(loc="upper right", fontsize=10)
    ax[0].set_title("Training Data", fontsize=15)
    ax[1].set_title("KNN Classification (K = %i)" % n_neighbors, fontsize=15)
    for i in range(2):
        ax[i].set_xlim(-4, 4.4)
        ax[i].set_ylim(-2.5, 5.3)
        ax[i].set_xlabel("Feature 1", fontsize=15)
        if i == 0:
            ax[i].set_ylabel("Feature 2", fontsize=15)
    plt.tight_layout()
    if save:
        plt.savefig("../figure/knn2d.pdf", bbox_inches="tight", transparent=True)
    else:
        plt.show()


def plot_dtw_area20_21_II(cla, p, save=False):
    l = 1.25
    f, ax = plt.subplots(2, len(cla), figsize=(17 / l, 6 / l), sharex=True, sharey=True)

    for n, j in enumerate(cla):
        data20, data21 = [], []
        pa2th = os.path.join(p, f"**/*X*{j[0]}*.csv")
        list_files = glob.glob(pa2th)
        list_files.sort()
        csf = list_files[0]
        df = pd.read_csv(csf)
        data20.append(
            df["k1_HH"].loc[df["timestamp"] < "2020-12-30"].values.reshape(-1, 1)
        )
        pa2th = os.path.join(p, f"**/*X*{j[1]}*.csv")
        list_files = glob.glob(pa2th)
        list_files.sort()
        csf = list_files[0]
        df = pd.read_csv(csf)
        data21.append(
            df["k1_HH"].loc[df["timestamp"] < "2020-12-30"].values.reshape(-1, 1)
        )

        tsstamp = df["timestamp"].loc[df["timestamp"] < "2020-12-30"].values
        parse20 = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in tsstamp]
        parse21 = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in tsstamp]
        day_month_20 = [
            datetime(year=2000, month=date.month, day=date.day) for date in parse20
        ]
        day_month_21 = [
            datetime(year=2000, month=date.month, day=date.day) for date in parse21
        ]

        plot_d20 = mdates.date2num(day_month_20)
        plot_d21 = mdates.date2num(day_month_21)
        data = np.concatenate([data20[0], data21[0]], axis=1)
        dataset_scaled = np.moveaxis(data, 1, 0)[:, :, np.newaxis]

        dtw_path, sim_dtw = metrics.dtw_path(
            dataset_scaled[0, :, 0],
            dataset_scaled[1, :, 0],
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=3,
        )

        dtw_path2, sim_dtw2 = metrics.dtw_path(
            dataset_scaled[0, :, 0],
            dataset_scaled[1, :, 0],
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=0,
        )

        for i in range(2):
            ax[i][n].plot_date(
                plot_d20,
                dataset_scaled[0, :, 0],
                "+-",
                "tab:blue",
                label=j[0],
                markersize=10,
                linewidth=2,
                # markeredgecolor="black",
            )
            ax[i][n].plot_date(
                plot_d21,
                3 + dataset_scaled[1, :, 0],
                "+-",
                "tab:orange",
                label=j[1],
                markersize=10,
                linewidth=2,
                # markeredgecolor="black",
            )
            if i == 0 and n == 0:
                ax[i][n].set_ylabel("$\kappa_1$  HH", fontsize=15)
                ax[i][n].yaxis.set_label_coords(-0.075, -0.12)
                ax[i][n].yaxis.set_label_coords(-0.015, -0.12)
            ax[i][n].grid(True, linewidth=0.5)
            ax[i][n].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
            ax[i][n].xaxis.set_major_locator(mdates.MonthLocator())
        for positions in dtw_path:
            ax[0][n].plot(
                [plot_d20[positions[0]], plot_d21[positions[1]]],
                [
                    dataset_scaled[0, positions[0], 0],
                    3 + dataset_scaled[1, positions[1], 0],
                ],
                color="k",
                linewidth=1.5,
                linestyle=":",
                alpha=0.75,
                label=f"DTW similarity: {sim_dtw.round(2)}"
                if positions == dtw_path[-1]
                else None,
            )
        for positions in dtw_path2:
            ax[1][n].plot(
                [plot_d20[positions[0]], plot_d21[positions[1]]],
                [
                    dataset_scaled[0, positions[0], 0],
                    3 + dataset_scaled[1, positions[1], 0],
                ],
                color="r",
                linewidth=1.5,
                linestyle="--",
                alpha=0.75,
                label=f"Euclidean distance: {sim_dtw2.round(2)}"
                if positions == dtw_path2[-1]
                else None,
            )
        ax[0][n].legend(fontsize=8)
        ax[1][n].legend(fontsize=8)
        ax[0][n].set_title(
            f"{j[0]} vs {j[1]}", font_properties={"weight": "bold", "size": 14}
        )
        if n == 0:
            ax[1][n].set_xlabel("Month", fontsize=15)
            ax[1][n].xaxis.set_label_coords(1.025, -0.15)

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.075)
    if save:
        print(csf.split("/")[-1][2:-4])
        plt.savefig(
            f"../figure/dtw_{csf.split('/')[-1][2:-4]}.pdf",
            backend="pgf",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.1,
        )
    else:
        plt.show()


def plot_temporal_HH(pa2th, tpol, cla, save=False):
    l = 1.25
    f, ax = plt.subplots(2, 4, figsize=(17 / l, 6 / l), sharex=True, sharey=True)
    for c in cla:
        list_files = glob.glob(pa2th.replace("?", c))
        list_files.sort()
        ngroup = len(list_files)
        for f in list_files:
            df = pd.read_csv(f)
            df["date"] = df.timestamp.apply(
                lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            )
            ax[cla.index(c) // 4, cla.index(c) % 4].fill_between(
                df["date"],
                df[f"mean_{tpol}"] - 0,
                df[f"mean_{tpol}"] + df[f"std_{tpol}"],
                alpha=0.25,
            )
            ax[cla.index(c) // 4, cla.index(c) % 4].plot(
                df["date"],
                df[f"mean_{tpol}"],
                # s=60,
                alpha=0.85,
                label="XXX" + f.split("/")[-1].split(".")[0].split("_")[1][3:],
            )
            ax[cla.index(c) // 4, cla.index(c) % 4].set_title(
                c + f" ({ngroup})", fontsize=15, fontweight="bold"
            )
            if c == "CIT":
                ax[cla.index(c) // 4, cla.index(c) % 4].legend(
                    bbox_to_anchor=(3.375, 1.9),
                    loc="upper left",
                    borderaxespad=0.0,
                    title_fontproperties={"weight": "bold", "size": 15},
                    fontsize=10,
                    handletextpad=0.1,
                    title="Area",
                )
        ax[cla.index(c) // 4, cla.index(c) % 4].grid(zorder=0)

        if cla.index(c) % 4 == 0 and cla.index(c) // 4 == 0:
            ax[cla.index(c) // 4, cla.index(c) % 4].set_ylabel(
                f"Amplitude {tpol} (linear $\sigma_0$)",
                fontsize=15,
                # labelpad=10,
            )

            ax[cla.index(c) // 4, cla.index(c) % 4].yaxis.set_label_coords(-0.14, -0.2)
        if cla.index(c) // 4 == 1 and cla.index(c) % 4 == 0:
            ax[cla.index(c) // 4, cla.index(c) % 4].set_xlabel("Month", fontsize=15)
            ax[cla.index(c) // 4, cla.index(c) % 4].xaxis.set_label_coords(2.25, -0.15)
        ax[cla.index(c) // 4, cla.index(c) % 4].set_xlim(
            datetime(2020, 1, 8), datetime(2021, 1, 27)
        )
        ax[cla.index(c) // 4, cla.index(c) % 4].xaxis.set_major_formatter(
            mdates.DateFormatter("%m.", usetex=True)
        )
    plt.subplots_adjust(hspace=0.3, wspace=0.15)
    if save:
        plt.savefig(
            f"{pa2th}/ts_{tpol}.pdf",
            backend="pgf",
            bbox_inches="tight",
            pad_inches=0.1,
        )
    else:
        plt.show()


def matching(gr, org):
    corres = {}
    for i in np.unique(gr):
        corres[f"{i}"] = org[gr == i][0]
    return corres


def rename_data_match(datapath, data):
    list_h5_0 = glob.glob(datapath)[0]
    (
        _,
        _,
        gr,
        org,
        _,
    ) = load_h5(list_h5_0)
    matchs = matching(gr, org)
    new_data = deepcopy(data)
    for k, v in data.items():
        for key, value in v.items():
            for k2, v2 in value.items():
                new_data[k][key][matchs[k2]] = v2
                del new_data[k][key][k2]

    return new_data, matchs


def expand_cmap(cmap_name, n):
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Get the colors from the colormap
    colors = cmap(np.linspace(0, 1, cmap.N))

    # Create a new colormap by interpolating between the colors
    new_cmap = mcolors.LinearSegmentedColormap.from_list("new_cmap", colors, N=n)
    colors = new_cmap(np.linspace(0.4, 1, n))
    return colors


def create_color_list(mat):
    # Get the colormap
    un = list(mat.values())
    clasofinterest = ["ICA", "HAG", "ABL", "ACC"]
    cmaps = ["Blues", "Greens", "Greys", "Purples"]
    idxTR = [i for i in range(len(un)) if un[i][:3] in clasofinterest]
    un = np.array(un)[idxTR]
    color = {}
    count = [0, 0, 0, 0]
    for k, i in enumerate(un):
        m = np.where(np.array(clasofinterest) == i[:3])[0][0]
        count[m] += 1
    cmaps_new = []
    for l in range(len(cmaps)):
        cmaps_new.append(expand_cmap(cmaps[l], count[l]))
    for k, i in enumerate(un):
        m = np.where(np.array(clasofinterest) == i[:3])[0][0]
        color[i] = cmaps_new[m][count[m] - 1]
        count[m] -= 1
    return color


def plot_false_positive(data0, colors, save, name):
    # Prepare the data for the plot

    fig, ax = plt.subplots(2, 1, figsize=(15, 9))
    for k in range(2):
        data = data0[list(data0.keys())[k]]
        classes = list(data.keys())

        # Create the plot
        for i in range(len(classes)):
            xx = list(data[classes[i]].values())
            idx = np.argsort(xx)
            xx = np.array(xx)[idx]
            labels = list(data[classes[i]].keys())
            labels = np.array(labels)[idx]
            for j in range(len(labels)):
                ax[k].bar(
                    classes[i],
                    xx[j],
                    bottom=xx[:j].sum(),
                    color=colors[labels[j]],
                    edgecolor="white",
                    alpha=0.9,
                )
                if xx[j] > 8:
                    ll = f"{labels[j]}"

                    ax[k].text(
                        i,
                        xx[:j].sum() + xx[j] / 2.4,
                        ll,
                        ha="right",
                        va="center",
                        color="white",
                        weight="extra bold",
                        fontsize=13.5,
                    )
                    ax[k].text(
                        i,
                        xx[:j].sum() + xx[j] / 2.45,
                        f"\\hspace{{0.2cm}}({xx[j]:.1f}%)",
                        ha="left",
                        va="center",
                        color="white",
                        weight="bold",
                        fontsize=10,
                    )

        ax[k].spines["right"].set_visible(False)
        ax[k].spines["top"].set_visible(False)
        ax[k].spines["left"].set_visible(False)
        ax[k].spines["bottom"].set_visible(False)
        ax[k].set_yticks([])
        if list(data0.keys())[k] == "GRFN":
            ax[k].set_title("False Negative", fontsize=20, fontweight="bold", y=0.95)
        else:
            ax[k].set_title("False Positive", fontsize=20, fontweight="bold", y=0.95)
        ax[k].set_ylabel("Percentage of group", labelpad=-20, fontsize=15)
        if k == 1:
            cl0 = ["$\mathbf{" + i + "}$" for i in classes]
            ax[k].set_xticklabels(cl0, fontweight="bold", fontsize=13)
        else:
            cl = ["$\overline{\mathbf{" + i + "}}$" for i in classes]
            ax[k].set_xticklabels(cl, fontweight="bold", fontsize=13)
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    if save:
        plt.savefig(
            f"../figure/FPFN_{name}.pdf",
            backend="pgf",
            bbox_inches="tight",
            pad_inches=0.1,
        )
    else:
        plt.show()


if __name__ == "__main__":
    fig0 = 0
    fig1 = 0
    fig2 = 0
    fig3 = 1
    fig4 = 0
    fig5 = 0

    if fig0:
        pa2th = "../../../../../Dataset/PAZ/SAR_X_II/**/*.tif"
        plot_hist(pa2th, "HH")
        plot_hist(pa2th, "HV")

    if fig1:
        pa2th2 = "../../../../../Dataset/PAZ/SAR/**/*X*?*.csv"
        comparaison_hh_hv(pa2th2)

    if fig2:
        illustrate_knn(
            new_points=[(-2, 2.7), (1.3, 0.3), (2.2, 1.15), (-3.2, -0.1)],
            n_neighbors=5,
            save=True,
        )

    if fig3:
        p = f"../../../../../Dataset/PAZ/SAR"
        plot_dtw_area20_21_II(
            [["ICA003", "ICA004"], ["HAG002", "HAG004"]], p, save=True
        )
        plot_dtw_area20_21_II(
            [["ABL001", "ABL005"], ["ACC001", "ACC004"]], p, save=True
        )

    if fig4:
        pa2th = "../../../../../Dataset/PAZ/SAR/**/*X*?*.csv"
        tpol = "HH"
        cla = ["ICA", "HAG", "ABL", "ACC", "FOR", "CIT", "ROC", "PLA"]
        plot_temporal_HH(pa2th, tpol, cla, save=True)
        tpol = "HV"
        plot_temporal_HH(pa2th, tpol, cla, save=True)

    if fig5:
        p = "dev/result_amp/datanpy/winsize_7_*_dtw.npy"
        for p in glob.glob(p):
            name = os.path.basename(p).split(".")[0].split("/")[-1]
            print(name)
            data = np.load(p, allow_pickle=True).item()
            datapath = f"../dataseth5/winsize_5/*spat*kfold_HV.h5"
            data0, mat = rename_data_match(datapath, data)
            colors23 = create_color_list(mat)
            plot_false_positive(data0, colors23, save=True, name=name)
