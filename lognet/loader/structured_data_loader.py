"""
This module is for lazily loading large CSV files.
"""
import os
import linecache
import torch
import numpy as np
import pandas as pd

from torch.utils import data
from lognet.utilities.FileInput import FileInput
from lognet.preprocess.DataPrep import DataPrep





class LazyCSVDataset(data.Dataset):
    """Lazily loads a CSV file.

    Arguments:
        header_file (str): File that contains the header info for CSV files.
        filename (str): CSV file that needs to be loaded
        response_features (str or list): Target variable(s)
    """

    def __init__(self,header_file, root_path ,filename, response_features,model_type):

        self.header_file = header_file
        self.filename = os.path.join(root_path,filename)
        #self.line_count = 0

        #with open(filename, "r") as f:
            #self.line_count = len(f.readlines()) - 1

        line = linecache.getline(self.header_file, 1)
        self.column_names = line.rstrip().split(',')

        if not isinstance(response_features, list):
            response_features = [response_features]

        self.response_feature_columns = []
        for feature in response_features:
            self.response_feature_columns.append(self.column_names.index(feature))
            
        read_data = FileInput(root_path,root_path+filename)
        filelist,data = read_data._readFiles()
        prep_data = DataPrep(model_type)
        data = prep_data._compute_log(data, 'SRESCurve')
        data = prep_data._compute_log(data, 'NeutCurve')
        data = prep_data._compute_cube(data,'DEN')
        data = prep_data._remove_nulls(data)
        self.data_capped = prep_data._cap_feature(data,'SonicCurve',1,100)
        

    def __getitem__(self, index):
        """Returns a row of the CSV file.

        Arguments:
            index (int): Row that is to be returned.

        Returns:
            Returns a tuple of (features, targets)
        """
        #line = linecache.getline(self.filename, index + 2)
        #row = np.array(line.rstrip().split(',')).astype(np.float)
        row = self.data_capped.iloc[index,:].values
        return row[self.response_feature_columns], np.delete(row, self.response_feature_columns)
        
         

    def __len__(self):
        """Number of rows in the CSV file.

        Returns:
            Returns the number of rows in the CSV file.
        """
        return len(self.data_capped)


def create_lazy_loaders(root_path, header_file, file_list, response_features, model_type):
    """
    """
    lazy_loaders = []

    for filename in file_list:
        lazy_loaders.append(LazyCSVDataset(
            header_file, root_path, filename, response_features, model_type))

    return lazy_loaders


class StructuredDataset(data.Dataset):
    """
    Concatenated structed dataset
    """

    def __init__(self, root_path, header_file, file_list, response_features, model_type):
        """
        Constructor
        """
        
        self.lazy_loaders = create_lazy_loaders(
            root_path, header_file, file_list, response_features, model_type)
        self.concatenated_dataset = data.ConcatDataset(self.lazy_loaders)
        self.rec_lengths = [len(self.lazy_loaders[i])
                            for i in range(len(self.lazy_loaders))]
        

    def __getitem__(self, index):
        """
        Gets item
        """
        return self.concatenated_dataset[index]

    def __len__(self):
        """
        Length of the dataset
        """
        return len(self.concatenated_dataset)

    def get_record_lengths(self):
        """
        List of length of each record
        """
        return self.rec_lengths


if __name__ == "__main__":
    pass
