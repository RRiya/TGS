import os
import numpy as np

class LASFileUWIMapper:
    """Mapper for LAS files to UWI

    Arguments:
        las_folder_path (str): Path where the LAS files are stored
    """
    def __init__(self, las_folder_path):
        """
        Constructor
        """

        if os.path.isdir(las_folder_path):
            self.las_folder_path = las_folder_path
        else:
            error_message = 'Path ' + las_folder_path + ' not found.'
            raise FileNotFoundError(error_message)

        self.uwi_list, self.las_file_list = [], []

        for filename in os.listdir(self.las_folder_path):
            if filename.lower().endswith('las'):
                self.uwi_list.append(str(max(filename.split('_'), key=len)))
                self.las_file_list.append(filename)

    def __len__(self):
        """
        Number of unique UWIs.

        Returns:
            Returns the number of unique UWIs
        """
        return len(self.uwi_list)

    def __getitem__(self, uwi):
        """
        Get the LAS file corresponding to the UWI.

        Arguments:
            uwi (str): UWI of the well

        Returns:
            LAS file corresponding to uwi.
        """

        if np.issubdtype(type(uwi), np.integer) == True:
            uwi = str(uwi)

        try:
            file_index = self.uwi_list.index(uwi)
            return self.las_file_list[file_index]
        except:
            return None

