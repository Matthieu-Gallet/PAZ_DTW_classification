import h5py
import numpy as np
import pandas as pd


from copy import deepcopy


def prepare_dico(infos):
    """
    Prepare a dictionary by aggregating counts from arrays.

    Parameters
    ----------
    infos : dict
        The input dictionary containing arrays.

    Returns
    -------
    dict
        The updated dictionary with aggregated counts.

    Notes
    -----
    This function takes a dictionary `infos` as input, where each key1 maps to another dictionary.
    Each key2 in the inner dictionary maps to a tuple of two arrays, `array1` and `array2`.
    The function iterates over the arrays, aggregates the counts based on the values in `array1`,
    and updates the dictionary with the aggregated counts.

    The function modifies the input dictionary in-place and returns the updated dictionary.

    Examples
    --------
    >>> infos = {
    ...     'key1': {
    ...         'key2': ([1, 2, 3], [10, 20, 30])
    ...     }
    ... }
    >>> prepare_dico(infos)
    {'key1': {'key2': {1: 10, 2: 20, 3: 30}}}
    """
    for key1 in infos:
        for key2 in infos[key1]:
            # Get the arrays
            array1 = infos[key1][key2][0]
            array2 = infos[key1][key2][1]

            # Create a dictionary to store the counts
            counts = {}

            # Iterate over the arrays
            for i in range(len(array1)):
                # If the id is not in the dictionary, add it
                if array1[i] not in counts:
                    counts[str(array1[i])] = array2[i]
                # If the id is in the dictionary, add the count
                else:
                    counts[array1[i]] += array2[i]

            # Update the dictionary
            infos[key1][key2] = counts
    return infos


def probafromdico(l):
    return (100 * np.array(list(l.values())) / np.sum(list(l.values()))).round(2)


def dicofromproba(v1):
    return {k: v for k, v in zip(v1.keys(), probafromdico(v1))}


def aggregate_dico(infos_list, proba=False):
    sz = len(infos_list)
    total_counts = deepcopy(infos_list[0])

    # Iterate over the list of dictionaries
    for i in infos_list[1:]:
        # Iterate over each dictionary
        for k, v in i.items():
            # Iterate over each key, value pair in each dictionary
            for k1, v1 in v.items():
                # If the key is not present in the total_counts dictionary
                if k1 not in total_counts[k]:
                    # Add the key, value pair
                    total_counts[k][k1] = v1
                else:
                    # Update the value
                    for k2, v2 in v1.items():
                        if k2 not in total_counts[k][k1]:
                            total_counts[k][k1][k2] = v2 / float(sz)
                        else:
                            total_counts[k][k1][k2] += v2 / float(sz)

    if proba:
        for k, v in total_counts.items():
            for k1, v1 in v.items():
                total_counts[k][k1] = dicofromproba(v1)
    return total_counts


def print_u(text):
    """
    Print a text and underline it with overline.

    Args:
        text (str): The text to be printed and underlined.

    Returns:
        None
    """
    print("")
    print(text)
    print("‾" * len(text))


def load_h5(path, format_date=False):
    """
    Load data from an HDF5 (h5) file.

    Args:
        path (str): The path to the HDF5 file.

    Returns:
        numpy.ndarray: Image data as a NumPy array with data type float32.
        numpy.ndarray: Labels data as a NumPy array with data type str.
        numpy.ndarray: Groups data as a NumPy array with data type int.
    """
    with h5py.File(path, "r") as hf:
        img = hf["img"][:].astype(np.float32)
        labels = hf["labels"][:].astype(str)
        groups = hf["groups"][:].astype(int)
        try:
            org = hf["org"][:].astype(str)
        except:
            org = None
        dates = hf["date"][:].astype(str)
        if format_date:
            if "_" in dates[0]:
                dates = [date.split("_") for date in dates]
                dates = [pd.to_datetime(date, format="%Y-%m-%d") for date in dates]
            else:
                dates = pd.to_datetime(dates, format="%Y-%m-%d")
        dates = np.array(dates)
    if org is not None:
        return img, labels, groups, org, dates
    else:
        return img, labels, groups, dates


def dict_depth(d):
    """
    Find the depth of a nested dictionary.

    Recursively calculates the depth of a nested dictionary. The depth is defined as the maximum number of nested levels
    in the dictionary.

    Args:
        d (dict): The nested dictionary for which to find the depth.

    Returns:
        int: The depth of the nested dictionary, where a dictionary with no nested dictionaries has a depth of 1.
    """
    if isinstance(d, dict):
        if not d:
            return 1
        return 1 + max(dict_depth(d[key]) for key in d)
    else:
        return 0
