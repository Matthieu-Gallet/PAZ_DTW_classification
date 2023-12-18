import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os


def plot_time_series_batch(output_path, data_dict, label, column, figsize=(24, 7), fill_color='green', fill_alpha=0.25, fontsize=20, display=False, save=False):
    """
    Plot time series data for a given label and column from a data dictionary.

    This function plots time series data for a specified label and column from a data dictionary. It provides options
    to customize the appearance of the plot and supports saving the plot to a specified output path.

    Parameters:
        output_path (str, optional): The directory path for saving the plot.
        data_dict (dict): A nested dictionary containing the data.
        label (str): The label for the data series to be plotted.
        column (str): The column name for the data to be plotted.
        figsize (tuple, optional): A tuple specifying the figure size (width, height). Default is (24, 7).
        fill_color (str, optional): The color used for filling the area between the minimum and maximum values of the data.
            Default is 'green'.
        fill_alpha (float, optional): The alpha value (transparency) for the filled area. Default is 0.25.
        fontsize (int, optional): The font size for labels and legends. Default is 20.
        display (boolean, optional)

    Returns:
        str: The filename of the saved plot.

    Example:
        >>> plot_time_series_batch(output_path, data_dict, label='ABL', column='mean_HH', figsize=(18, 6), fill_color='blue')
    """
    if label not in data_dict:
        raise ValueError(f"Label '{label}' does not exist in the data dictionary.")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Select the data for the given label and column
    data = data_dict[label][column]['data']
    timestamps = data_dict[label]['timestamp']['data'][0]
    dates = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in timestamps]
    groups = data_dict[label][column]['group']

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(data)):
        plt.plot(dates, data[i], label=groups[i])

    ax.grid(True, linewidth=1, zorder=0)

    # Add axis groups and a margin between the X/Y axis and the label
    ax.set_xlabel('Date', fontsize=fontsize+6)
    ax.set_ylabel(column, fontsize=fontsize+6, rotation=0, labelpad=60)

    ax.tick_params(axis='both', labelsize=fontsize+4)
    ax.axhline(y=0, color='black', linewidth=1)

    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    ax.fill_between(dates, min_data, max_data, color=fill_color, alpha=fill_alpha)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(groups), fontsize=fontsize-4)

    if save:
        # Customize the name of the backup file using the label and the column
        save_filename = os.path.join(output_path, f'{label}_{column}.pdf')
        plt.savefig(save_filename)
        print(f'Graphic saved as: {save_filename}')

    if display:
        plt.show()
