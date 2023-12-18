#   ______  ______  ______  ______  __    __  ______  __          _____   ______  ______  ______  ______  ______  ______
#  /\  ___\/\  == \/\  __ \/\___  \/\ "-./  \/\  __ \/\ \        /\  __-./\  __ \/\__  _\/\  __ \/\  ___\/\  ___\/\__  _\
#  \ \ \___\ \  _-/\ \  __ \/_/  /_\ \ \-./\ \ \  __ \ \ \____   \ \ \/\ \ \  __ \/_/\ \/\ \  __ \ \___  \ \  __\\/_/\ \/
#   \ \_____\ \_\   \ \_\ \_\/\_____\ \_\ \ \_\ \_\ \_\ \_____\   \ \____-\ \_\ \_\ \ \_\ \ \_\ \_\/\_____\ \_____\ \ \_\
#    \/_____/\/_/    \/_/\/_/\/_____/\/_/  \/_/\/_/\/_/\/_____/    \/____/ \/_/\/_/  \/_/  \/_/\/_/\/_____/\/_____/  \/_/
#     ______________________________________________________________________________________________________________
#    | Monitoring the cryosphere of Mont-Blanc massif (Western European Alps) with X-band PAZ SAR image time-series |
#     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
# *************************************************************************************************************************
# File: create_dataset.py
# Author: Matthieu Gallet
# Collaborators: Christophe Lin-Kwong-Chon,
# Date: October 17, 2023
# Github: https://github.com/Matthieu-Gallet/PAZ-unsupervised
# Institution: USMB, LISTIC
# Copyright: 2023, LISTIC
# License: MIT
#
# Description: This Python script is designed for processing and managing satellite radar (SAR) images
#              from the PAZ satellite. It provides functionality for data extraction, windowing, and
#              conversion to HDF5 format. The script is customizable to handle various polarizations
#              (HH, HV, VV, VH) and classes of land cover. It also allows for random or maximum window
#              extraction. The processed data can be saved in HDF5 format for machine learning applications.
#
# Usage: You can use this script to process PAZ satellite radar images and convert them into a suitable
#        format for machine learning tasks. Customize the input parameters to select specific data and
#        extraction options.
#
# Dependencies: This script relies on GDAL, NumPy, h5py, and other Python libraries. Make sure to
#               install the necessary dependencies to use the script.
#
# *************************************************************************************************************************
from joblib import Parallel, delayed
from skimage.transform import resize
import h5py, glob, tqdm, os
from numba import jit
import pandas as pd
import numpy as np
from osgeo.gdalconst import GA_ReadOnly
from osgeo import gdal
from utils.helper_functions import load_h5


def load_tiff1D(file_name, gdal_driver="GTiff"):
    """
    Load a 1D TIFF file using the GDAL driver.

    Args:
        file_name (str): The path of the TIFF file.
        gdal_driver (str, optional): The GDAL driver to use. Default is "GTiff".

    Returns:
        numpy.ndarray: A NumPy array representing the image from the TIFF file.
    """
    driver = gdal.GetDriverByName(gdal_driver)
    driver.Register()

    inDs = gdal.Open(file_name, GA_ReadOnly)
    # Get the data as a numpy array
    cols = inDs.RasterXSize
    rows = inDs.RasterYSize
    band = inDs.GetRasterBand(1)
    image_array = band.ReadAsArray(0, 0, cols, rows)
    return image_array


@jit(nopython=True, cache=True)
def extract_non_overlapping_windows3D(matrix, k, sr=0, sc=0):
    """
    Extract non-overlapping square-size windows from a 3D matrix.

    Args:
        matrix (numpy.ndarray): The 3D input matrix.
        k (int): The size of the windows (width=height).
        sr (int, optional): The starting row for extraction. Default is 0.
        sc (int, optional): The starting column for extraction. Default is 0.

    Returns:
        list: List of valid non-overlapping windows.
    """
    valid_windows = []
    m, n, l = matrix.shape

    def is_window_valid(r, c):
        window = matrix[max(0, r) : min(m, r + k), max(0, c) : min(n, c + k), :]
        return window.shape == (k, k, l) and not np.isnan(window).any()

    for r in range(sr, m, k):
        for c in range(sc, n, k):
            if is_window_valid(r, c):
                window = matrix[r : r + k, c : c + k, :]
                valid_windows.append(window)
    return valid_windows


@jit(nopython=True, cache=True)
def extract_non_overlapping_windows4D(matrix, k, sr=0, sc=0):
    valid_windows = []
    m, n, l, o = matrix.shape

    def is_window_valid(r, c):
        window = matrix[max(0, r) : min(m, r + k), max(0, c) : min(n, c + k), :, :]
        return window.shape == (k, k, l, o) and not np.isnan(window).any()

    for r in range(sr, m, k):
        for c in range(sc, n, k):
            if is_window_valid(r, c):
                window = matrix[r : r + k, c : c + k, :, :]
                valid_windows.append(window)
    return valid_windows


def extract_max_windows(matrix, k):
    """
    Extract maximum non-overlapping square windows from a 3D matrix.

    Args:
        matrix (numpy.ndarray): The 3D input matrix.
        k (int): The size of the square windows (width=height).

    Returns:
        list: List of maximum non-overlapping windows.
        int: Number of maximum windows extracted.
    """
    best_windows = []
    max_windows = 0
    depth = k // 2
    sz = matrix.shape
    for i in range(depth):
        for j in range(depth):
            if len(sz) == 3:
                extr = extract_non_overlapping_windows3D(matrix, k, i, j)
            elif len(sz) == 4:
                extr = extract_non_overlapping_windows4D(matrix, k, i, j)
            if len(extr) > max_windows:
                best_windows = extr
                max_windows = len(extr)
    return best_windows, max_windows


class Dataset_tiff2hdf5:
    def __init__(
        self,
        path_dir,
        extension="spatial",
        different_group=True,
        outpath="../",
        n_jobs=1,
    ):
        self.path_dir = path_dir
        self.extension = extension  # spatial or temporal
        # spatial: extract windows from each image independently of the date of the image,
        # leading to a larger number of windows than temporal extension
        # temporal: extract windows on a temporal stack of images
        self.diff_group = different_group
        self.outpath = outpath
        self.n_jobs = n_jobs
        self._extract_init_info()

    def _extract_init_info(self):
        """
        Extracts and initializes information about the dataset from the specified directory.

        This method processes the files found in the directory and extracts relevant information
        such as class, date, polarisation, and group. Depending on the 'diff_group' attribute,
        it may also consolidate group labels if set to False.

        Parameters:
            None

        Returns:
            None
        """
        files_HV = glob.glob(self.path_dir, recursive=True)
        df = pd.DataFrame(files_HV, columns=["path"])
        df["classe"] = df["path"].apply(lambda x: x.split("/")[-1].split("_")[-3][:-3])
        df["date"] = df["path"].apply(lambda x: x.split("/")[-1].split("_")[-1][:-4])
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["polarisation"] = df["path"].apply(lambda x: x.split("/")[-1].split("_")[-2])
        df["group"] = df["path"].apply(lambda x: x.split("/")[-1].split("_")[-3][:])
        df["org_group"] = df["path"].apply(lambda x: x.split("/")[-1].split("_")[-3][:])
        df["id"] = df["path"].apply(lambda x: x.split("/")[-1].split("_")[1])
        if self.diff_group:
            _ = [
                df.group.replace(gr, i, inplace=True)
                for i, gr in enumerate(df.group.unique())
            ]
        else:
            df["group"] = df["path"].apply(
                lambda x: x.split("/")[-1].split("_")[-3][-3:]
            )
        self.info_data = (
            df  # Initialize the data attribute with the processed DataFrame
        )

    def _select_data(self, rqt, polarisation):
        """
        Selects data based on provided query and polarization.

        This method filters the dataset based on a given query ('rqt') and the desired
        polarization ('polarisation'). Depending on the 'polarisation' value, it will
        either select the specified polarization or consolidate data for dual polarization
        (HH and HV) with the same date.

        Parameters:
            rqt (str): A query string to filter the dataset.
            polarisation (str): The desired polarization ('HH', 'HV', 'VV', 'VH', or 'dual').

        Returns:
            None
        """
        df_rq = self.info_data.query(rqt)
        if polarisation != "dual":
            self.list_files = df_rq.query("polarisation == @polarisation")
        else:
            HV = df_rq.query("polarisation == 'HV'")
            HH = df_rq.query("polarisation == 'HH'")
            B = HH.query(f"date in {list(HV.date.dt.strftime('%Y-%m-%d').values)}")
            self.list_files = df_rq.query(f"index in @B.index")
            print(
                "Warning: the list of files contains only HH with the same date as HV"
            )

    def _param(self, rqt, polarisation, winsize):
        """
        Sets and stores parameters used in the dataset selection and processing.

        This method collects and stores various parameters and information used during the
        dataset selection and processing. It creates a dictionary named 'store_param' containing
        details such as the directory path, the number of selected samples, the number of selected
        classes, and more.

        Parameters:
            rqt (str): A query string used for filtering the dataset.
            polarisation (str): The desired polarization ('HH', 'HV', 'VV', 'VH', or 'dual').
            winsize (int): The window size for data extraction.

        Returns:
            None
        """
        self.store_param = {
            "path_dir": self.path_dir,
            "n_samples_selected": len(self.list_files),
            "n_classes_selected": len(
                self.info_data.query("path in @self.list_files").classe.unique()
            ),
            "n_groups_selected": len(
                self.info_data.query("path in @self.list_files").group.unique()
            ),
            "different_group": self.diff_group,
            "rqt": rqt,
            "polarisation": polarisation,
            "winsize": winsize,
        }

    def _temporal_stack(self, i, pola="HH"):
        idu = self.list_files.id.unique()
        temp_stack = self.list_files.loc[self.list_files.id == idu[i]].sort_values(
            by="date"
        )
        dates = temp_stack.date.dt.strftime("%Y-%m-%d").values
        dates = "".join([i + "_" for i in dates])[:-1]
        data = [load_tiff1D(f.replace("HH", pola)) for f in temp_stack.path.values]
        shape = [d.shape for d in data]
        mi_sz, ma_sz = np.min(shape, axis=0)
        data = np.dstack([resize(d, (mi_sz, ma_sz)) for d in data])
        y = [
            temp_stack.classe.iloc[0],
            temp_stack.group.iloc[0],
            dates,
            temp_stack.org_group.iloc[0],
        ]
        return data, y

    def _extract_HH(self, i):
        """
        Extracts data for HH polarization from a specific file in the dataset.

        This method loads data for HH polarization from a specified file in the dataset and
        extracts associated class and group information. The data and labels are returned as a tuple.

        Parameters:
            i (int): Index of the file to extract data from.

        Returns:
            tuple: A tuple containing the data for HH polarization and associated labels in the
            format (data, labels).
        """
        if self.extension == "spatial":
            HH_img = load_tiff1D(self.list_files.path.iloc[i])
            HH_img = HH_img[:, :, np.newaxis]
            y = [
                self.list_files.classe.iloc[i],
                self.list_files.group.iloc[i],
                self.list_files.org_group.iloc[i],
                self.list_files.date.iloc[i].strftime("%Y-%m-%d"),
            ]
        elif self.extension == "temporal":
            HH_img, y = self._temporal_stack(i, pola="HH")
        else:
            raise ValueError("Extension must be spatial or temporal")

        return HH_img, y

    def _extract_HV(self, i):
        """
        Extracts data for HV polarization from a specific file in the dataset.

        This method loads data for HV polarization from a specified file in the dataset, which
        corresponds to the complementary polarization of HH. It also extracts associated class
        and group information. The data and labels are returned as a tuple.

        Parameters:
            i (int): Index of the file to extract data from.

        Returns:
            tuple: A tuple containing the data for HV polarization and associated labels in the
            format (data, labels).
        """
        if self.extension == "spatial":
            HV_img = load_tiff1D(self.list_files.path.iloc[i].replace("HH", "HV"))
            HV_img = HV_img[:, :, np.newaxis]
            y = [
                self.list_files.classe.iloc[i],
                self.list_files.group.iloc[i],
                self.list_files.org_group.iloc[i],
                self.list_files.date.iloc[i].strftime("%Y-%m-%d"),
            ]
        elif self.extension == "temporal":
            HV_img, y = self._temporal_stack(i, pola="HV")

        else:
            raise ValueError("Extension must be spatial or temporal")
        return HV_img, y

    def _extract_all(self, i):
        """
        Extracts data for the specified polarization from a specific file in the dataset.

        This method extracts data for the specified polarization (HH, HV, or dual) from a specified
        file in the dataset. It also extracts associated class and group information. The data is
        prepared according to the polarization type and window size, and labels are assigned.

        Parameters:
            i (int): Index of the file to extract data from.

        Returns:
            tuple: A tuple containing the extracted data and associated labels.
        """
        pol = self.store_param["polarisation"]
        if pol == "HH":
            x, y = self._extract_HH(i)
        elif pol == "HV":
            x, y = self._extract_HV(i)
        elif pol == "dual":
            xh, yh = self._extract_HH(i)
            xv, yv = self._extract_HV(i)
            try:
                assert yh == yv
                y = yh
            except AssertionError:
                print("Error in polarisation, different label")
            if self.extension == "spatial":
                x = np.dstack((xh, xv))
            elif self.extension == "temporal":
                x = np.array([xh, xv])
                x = np.moveaxis(x, 0, -1)
        else:
            raise ValueError("Polarisation must be HH, HV or dual")
        data, n_win = extract_max_windows(x, k=self.store_param["winsize"])
        if n_win > 0:
            labels = [y] * n_win
        else:
            labels = [-999]
        return data, labels

    def _clean_data(self, X, y):
        """
        Cleans and organizes the extracted data and labels.

        This method takes the extracted data and labels and performs a cleaning process.
        It removes entries with labels equal to -999, which indicate invalid data. It organizes
        the cleaned data and labels for further processing.

        Parameters:
            X (numpy.ndarray): Extracted data.
            y (list): Extracted labels.

        Returns:
            None
        """
        xx, yy = [], []
        for i in range(len(y)):
            if y[i] != [-999]:
                yy.extend(y[i])
                xx.extend(X[i])
        self.X = np.array(xx)
        self.y = np.array(yy)

    def _check_saving(self):
        """
        Verifies the correctness of saved data.

        This method checks the correctness of the saved data by comparing it with the data
        stored in the output HDF5 file. It compares the loaded data (Xn, Yn, Gn) with the original
        data (self.X, self.y) and raises an error message if discrepancies are found.

        Parameters:
            None

        Returns:
            None
        """
        Xn, Yn, Gn, IGn, Dn = load_h5(self.filename)
        try:
            assert np.all(self.X == Xn)
            assert np.all(self.y[:, 0] == Yn)
            assert np.all(self.y[:, 1].astype(int) == Gn)
            assert np.all(self.y[:, 2] == Dn)
            assert np.all(self.y[:, 3] == IGn)

        except AssertionError:
            print("Error in saving, different data")

    def _save_h5(self):
        """
        Saves the processed data to an HDF5 file.

        This method saves the processed data (self.X), labels, and groups to an HDF5 file.
        It specifies the file format and compression options for data storage.

        Parameters:
            None

        Returns:
            None
        """
        os.makedirs(os.path.dirname(self.outpath), exist_ok=True)
        if ".h5" not in self.outpath:
            self.filename = os.path.join(self.outpath, "dataX_PAZ.h5")
        else:
            self.filename = self.outpath
        labels = self.y[:, 0].astype("S3")
        groups = self.y[:, 1].astype(int)
        org = self.y[:, 3].astype("S6")

        with h5py.File(self.filename, "w") as hf:
            hf.create_dataset("img", np.shape(self.X), compression="gzip", data=self.X)
            hf.create_dataset(
                "labels", np.shape(labels), compression="gzip", data=labels
            )
            hf.create_dataset(
                "groups", np.shape(groups), compression="gzip", data=groups
            )
            hf.create_dataset("org", np.shape(org), compression="gzip", data=org)
            if self.extension == "temporal":
                dates = self.y[:, 2].astype("S")
                hf.create_dataset(
                    "date",
                    np.shape(dates),
                    compression="gzip",
                    data=dates,
                )
            elif self.extension == "spatial":
                dates = self.y[:, 2].astype("S10")
                hf.create_dataset(
                    "date",
                    np.shape(dates),
                    compression="gzip",
                    data=dates,
                )

    def _save_report(self):
        """
        Generates and saves a txt report with metadata about the processed data.

        This method generates a txt report with various metadata information about the processed data,
        such as file details, data shapes, label statistics, and data statistics.
        The report is saved as a text file.

        Parameters:
            None

        Returns:
            None
        """
        self.store_param["filename"] = self.filename
        self.store_param["X_shape"] = self.X.shape
        self.store_param["y_shape"] = self.y.shape
        self.store_param["labels"] = np.unique(
            self.y[:, 0].astype("S3"), return_counts=True
        )
        self.store_param["groups"] = np.unique(
            self.y[:, 1].astype(int), return_counts=True
        )
        self.store_param["X_min"] = self.X.min(axis=(0, 1, 2))
        self.store_param["X_max"] = self.X.max(axis=(0, 1, 2))
        self.store_param["X_mean"] = self.X.mean(axis=(0, 1, 2))
        self.store_param["X_std"] = self.X.std(axis=(0, 1, 2))
        report_path = self.filename.replace(".h5", "_report.txt")
        label_percentages = np.around(
            (self.store_param["labels"][1] / len(self.y)) * 100, 2
        )
        label_percentages_str = ", ".join(map(str, label_percentages))
        with open(report_path, "w") as fichier:
            for cle, valeur in self.store_param.items():
                fichier.write(f"{cle}: {valeur}\n")
            fichier.write(f"label_percentages: [{label_percentages_str}]\n")

    def extract_data(self, rqt, polarisation="dual", winsize=15, save=False):
        """
        Extracts and processes data from the dataset.

        This method extracts and processes data from the dataset based on the specified query string,
        polarization,and window size. It also provides the option to save the processed data to an
        HDF5 file and generate a report.

        Parameters:
            rqt (str): A query string to filter the dataset.
            polarisation (str): The desired polarization ('HH', 'HV', 'VV', 'VH', or 'dual').
            winsize (int): The window size for data extraction.
            save (bool): If True, save the processed data to an HDF5 file and generate a report.

        Returns:
            None
        """
        self._select_data(rqt, polarisation)
        self._param(rqt, polarisation, winsize)
        if self.extension == "spatial":
            X, y = zip(
                *Parallel(n_jobs=self.n_jobs)(
                    delayed(self._extract_all)(i)
                    for i in tqdm.tqdm(
                        range(len(self.list_files)),
                        desc="    Extract data for "
                        + os.path.splitext(os.path.basename(self.outpath))[0],
                        leave=None,
                    )
                )
            )
        elif self.extension == "temporal":
            X, y = zip(
                *Parallel(n_jobs=self.n_jobs)(
                    delayed(self._extract_all)(i)
                    for i in tqdm.tqdm(
                        range(len(self.list_files.id.unique())),
                        desc="    Extract data for "
                        + os.path.splitext(os.path.basename(self.outpath))[0],
                        leave=None,
                    )
                )
            )
        else:
            raise ValueError("Extension must be spatial or temporal")
        self._clean_data(X, y)
        if self.outpath is not None and save:
            self._save_h5()
            self._save_report()
            self._check_saving()


if __name__ == "__main__":
    pa2th = "../../../../../Dataset/PAZ/SAR_X_II/**/*.tif"
    opath = "../dataseth5/"
    winsize = [11]  # 15, 17, 19, 21]
    os.makedirs(opath, exist_ok=True)

    for ws in winsize:
        out = os.path.join(opath, f"winsize_{ws}")
        print("● Extraction data with a size of " + str(ws))

        rqtemp = "classe in ['ICA','HAG','ABL','ACC','FOR','CIT','ROC','PLA'] & date < '2021-01-01'"
        outpath = os.path.join(out, "ax1_temp_20_train_HH.h5")
        cdlf = Dataset_tiff2hdf5(
            pa2th, different_group=True, n_jobs=1, outpath=outpath, extension="temporal"
        )
        cdlf.extract_data(rqtemp, polarisation="HH", winsize=ws, save=True)

        rqtemp = "classe in ['ICA','HAG','ABL','ACC','FOR','CIT','ROC','PLA'] & date > '2021-01-01'"
        outpath = os.path.join(out, "ax1_temp_20_test_HH.h5")
        cdlf = Dataset_tiff2hdf5(
            pa2th, different_group=True, n_jobs=1, outpath=outpath, extension="temporal"
        )
        cdlf.extract_data(rqtemp, polarisation="HH", winsize=ws, save=True)

        rqtemp = "classe in ['ICA','HAG','ABL','ACC','FOR','CIT','ROC','PLA'] & date < '2021-01-01'"
        outpath = os.path.join(out, "ax2_spat_20_kfold_HH.h5")
        cdlf = Dataset_tiff2hdf5(
            pa2th, different_group=True, n_jobs=1, outpath=outpath, extension="temporal"
        )
        cdlf.extract_data(rqtemp, polarisation="HH", winsize=ws, save=True)

        rqtemp = "classe in ['ICA','HAG','ABL','ACC','FOR','CIT','ROC','PLA'] & date < '2021-01-01'"
        outpath = os.path.join(out, "ax3_spat_20_kfold_HV.h5")
        cdlf = Dataset_tiff2hdf5(
            pa2th, different_group=True, n_jobs=1, outpath=outpath, extension="temporal"
        )
        cdlf.extract_data(rqtemp, polarisation="HV", winsize=ws, save=True)

        rqtemp = "classe in ['ICA','HAG','ABL','ACC','FOR','CIT','ROC','PLA'] & date < '2021-01-01'"
        outpath = os.path.join(out, "ax4_spat_20_kfold_HV&HH.h5")
        cdlf = Dataset_tiff2hdf5(
            pa2th, different_group=True, n_jobs=1, outpath=outpath, extension="temporal"
        )
        cdlf.extract_data(rqtemp, polarisation="dual", winsize=ws, save=True)
