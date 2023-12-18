import os, re
import pandas as pd
import numpy as np
import functools, operator

class CsvManager:
    """
    CsvManager class for managing CSV files in a specified folder and its subfolders.

    This class provides methods to list CSV files within a specified folder and its subfolders
    and filter them based on specific patterns.

    Args:
        folder_path (str): The path to the root folder to start the search.
        subfolder_depth (int): The maximum depth of subfolders to search for CSV files.

    Attributes:
        folder_path (str): The path to the root folder.
        subfolder_depth (int): The maximum subfolder depth to search.

    Methods:
        list_csv_files: Lists CSV files in the specified folder and its subfolders.
        filter_patterns: Filters files based on specific patterns for each label.
        read_csv: Reads a CSV file and returns its contents as a DataFrame.
        read_whole_csv: Reads a CSV file and returns its contents as a DataFrame.
        read_and_extract_data: Reads a CSV file, extracts specific columns, and appends the data to an existing DataFrame.
        extract_data_dict: Extracts data from CSV files and organizes it into a nested dictionary structure.
        transform: Transform an irregular 2D list into a regular one.
    """


    def __init__(self, folder_path, subfolder_depth=1):
        self.folder_path = folder_path
        self.subfolder_depth = subfolder_depth
        self.csv_files = []


    def list_csv_files(self):
        """
        List CSV files in the specified folder and its subfolders.

        Returns:
            None
        """
        csv_files = []
        for root, dirs, files in os.walk(self.folder_path):
            level = root.replace(self.folder_path, '').count(os.sep)
            if level > self.subfolder_depth:
                continue
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    csv_files.append(file_path)
        self.csv_files = csv_files


    def filter_str(self, filter):
        """
        Filter files based on specific patterns list.

        Args:
            filter (list): A list of patterns to filter files.

        Returns:
            list: A list of file paths that match the specified patterns.
        """
        csv_files = self.csv_files

        for pattern in filter:
            csv_files = [
                file for file in csv_files
                if re.search(pattern, file)
            ]

        return csv_files


    def read_whole_csv(self, file_path):
        """
        Reads a CSV file and returns its contents as a DataFrame.

        This method reads the specified CSV file named `file_path` and returns its contents as a DataFrame.

        Args:
            file_path (str): The name of the CSV file to be read.

        Returns:
            pandas.DataFrame: The contents of the CSV file as a DataFrame, or None if the file is not found.
        """
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None


    def read_and_extract_data(self, file_path, columns_to_extract):
        """
        Reads a CSV file, extracts specific columns, and appends the data to an existing DataFrame.

        Args:
            file_path (str): The path to the CSV file to be read and processed.
            columns_to_extract (list): A list of column names to be extracted from the CSV file.

        Returns:
            pandas.DataFrame: A DataFram3e containing the extracted data.

        Raises:
            FileNotFoundError: If the specified `file_path` does not exist.

        Example:
            >>> csv_manager = CsvManager("SAR_X_II", subfolder_depth)
            >>> file_path = "SAR_X_II/ABL_proc/X_ABL001.csv"
            >>> columns_to_extract = ["timestamp", "mean_HH", "mean_HV"]
            >>> extracted_data = csv_manager.read_and_extract_data(file_path, columns_to_extract)
        """
        data = self.read_whole_csv(file_path)
        data = data[columns_to_extract]

        return data


    def extract_data_dict(self, labels, query):
        """
        Extracts data from CSV files and organizes it into a nested dictionary structure.

        This method extracts data from CSV files based on labels and columns specified in the query, 
        and organizes it into a nested dictionary structure. Each column and label is associated 
        with a dictionary containing 'data' and 'name' lists to store extracted data and file names.

        Args:
            labels (list): A list of labels for filtering files.
            query (list): A list of column group to extract from the CSV files.

        Returns:
            dict: A nested dictionary containing extracted data and file group organized by column and label.

        Example:
            >>> labels = ['ABL', 'ACC', 'CIT']
            >>> query = ['timestamp', 'mean_HH', 'mean_HV']
            >>> csv_manager = CsvManager("SAR_X_II", subfolder_depth)
            >>> data_dict = csv_manager.extract_data_dict(labels, query)
        """
        data_dict = {column: {label: {'data': [], 'group': []} for label in labels} for column in query}

        return data_dict


    # Transform an irregular 2D list into a regular one.
    @staticmethod
    def transform(nested_list):
        """
        Transform an irregular 2D list into a regular one.

        Args:
            nested_list (list): A list that may contain sublists.

        Returns:
            list: A regular 2D list where each element is either a list or a single item.
        """
        regular_list = []
        for ele in nested_list:
            if type(ele) is list:
                regular_list.append(ele)
            else:
                regular_list.append([ele])
        return regular_list
    

    def filter_patterns(self, labels, patterns):
        """
        Filter files based on specific patterns for each label.

        This method filters files based on a list of patterns for each label. It iterates through the labels and applies the
        corresponding pattern to filter files, returning a dictionary where each label is associated with a list of filtered
        file paths.

        Args:
            labels (list): A list of labels to filter files for.
            patterns (list): A list of patterns to filter files for each label.

        Returns:
            dict: A dictionary where each label is associated with a list of filtered file paths.

        Example:
            >>> csv_manager = CsvManager("SAR_X_II", subfolder_depth)
            >>> labels = ['ABL', 'ACC', 'CIT']
            >>> patterns = ['/X_', '/Y_']
            >>> filtered_files_by_label = csv_manager.filter_patterns(labels, patterns)
        """
        # Initialize an empty list to store files filtered by label
        filtered_files_by_label = {label: [] for label in labels}

        for label in labels:
            # Check if the list is an instance, and flatten if list contains a list
            if isinstance([label, patterns], list):
                regular_2D_list = self.transform([label, patterns])
                filter = functools.reduce(operator.iconcat, regular_2D_list, [])
            filtered_files = self.filter_str(filter)
            
            # Add the list of filtered files to the list corresponding to the label
            filtered_files_by_label[label] = filtered_files
        
        return filtered_files_by_label
    

    def generate_data_dict(self, labels, query, files_dict, convertNP):
        """
        Generate a data dictionary based on the specified labels, columns, and files dictionnary.

        Args:
            labels (list): A list of labels.
            query (list): A list of column names to extract from the CSV files.
            files_dict (dictionnary): A dictionary where each label is associated with a dictionnary of file paths.
            convertNP (bool, optional): Whether to convert data to NumPy arrays. Default is True.

        Returns:
            dict: A nested data dictionary containing extracted data and file group organized by label and column.
        """
        data_dict = {label: {column: {'data': [], 'group': []} for column in query} for label in labels}

        for label, files in files_dict.items():
            for file_path in files:
                extracted_data = self.read_and_extract_data(file_path, query)
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                for column in query:
                    data_dict[label][column]['data'].append(extracted_data[column].to_numpy())
                    data_dict[label][column]['group'].append(file_name)

        if convertNP:
            for label in labels:
                for column in query:
                    data_dict[label][column]['data'] = np.vstack(data_dict[label][column]['data'])

        return data_dict