import os
import sys
import pandas as pd
import errno
from tqdm import tqdm

class FileInput():
    """
    Read LAS, Excel and CSV files
    """
    pos_labels = ['WGS84Latitude', 'WGS84Longitude']

    def __init__(self, las_folder_path, csv_file):

        if os.path.isdir(las_folder_path):
            self.las_path = las_folder_path
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), las_folder_path)

        if os.path.isfile(csv_file):
            self.csv_file = csv_file  
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), csv_file)
        self.selected_df = pd.DataFrame()

    def _readFiles(self):
        #print("generating LAS file list")
        lasfiles = [f for f in os.listdir(self.las_path) if f.lower().endswith('.las')]

        try:
            data_df = pd.read_csv(self.csv_file)
        except:
            print(FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.csv_file))
            sys.exit('Metadata file not found.')
        return lasfiles, data_df

    def generateFileList(self, feature_list, target_list):
        assert isinstance(feature_list, list) == True, 'feature_list is not a list'
        assert isinstance(target_list, list) == True, 'target_list is not a list'
        lasfiles, metadata_df = self._readFiles()
        uwi_set = set(metadata_df['uwi'])
        print("UWI set length: " + str(len(uwi_set)))		
        selected_list = []
        info_list = []
        col_labels = FileInput.pos_labels + feature_list + target_list


        for file in tqdm(lasfiles):
            file_uwi = int(max(file.split("_"),key=len))
            if file_uwi in uwi_set:
                file_info = metadata_df.loc[metadata_df['uwi'] == file_uwi][col_labels].values[0]
                selected_list.append(file)
                info_list.append(tuple(file_info))

        self.selected_df = pd.DataFrame.from_records(info_list, index=selected_list, columns=col_labels)
        #print(self.selected_df.head())
        print("Number of files: " + str(len(selected_list)))

    def generateFormationList(self, feature_list):
        lasfiles, metadata_df = self._readFiles()

        uwi_set = set(metadata_df['UWI'])
        print("UWI set length: " + str(len(uwi_set)))       
        counter = 0

        for file in tqdm(lasfiles):
            file_uwi = int(max(file.split("_"),key=len))
            if file_uwi in uwi_set:
                file_info = metadata_df.loc[metadata_df['UWI'] == file_uwi][feature_list]
                file_info['LASFile'] = file
                self.selected_df = pd.concat([self.selected_df, file_info], ignore_index=True)
                counter = counter + 1
        self.selected_df = self.selected_df.set_index('LASFile')
        #print(self.selected_df.head(20))
        print(f"Number of files: {counter}")
        return self.selected_df['Name'].unique(), self.selected_df['FormationName'].unique()

    def getFilePath(self):
        return self.las_path

    def getFileList(self):
        return self.selected_df.index.unique().values

    def getFileInfo(self,filename):
        return self.selected_df.loc[filename].to_dict()

    def getFormationInfo(self,filename):
        if self.selected_df.loc[filename].ndim < 2:
            return pd.DataFrame([self.selected_df.loc[filename]])
        return self.selected_df.loc[filename]

# Testing
if __name__ == '__main__':
    pass 
