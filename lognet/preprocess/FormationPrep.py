import numpy as np
import pandas as pd

FORMATIONS = ['Bell Canyon', 'Mississippian', 'Woodford Shale', 'Rustler',
              'Salado', 'Upper Barnett Shale', 'Fusselman', 'Bone Spring',
              'Brushy Canyon', 'Wolfcamp Shale', 'Simpson', 'Ellenburger',
              'Devonian Ls', 'Dewey Lake', 'Grayburg San Andres',
              'Mississippian Ls Woodford Sh', 'Salado Transil Evaporites',
              'Spraberry', 'Strawn Horseshoe Atol', 'Wolfcamp']

class FormationPrep():
    def __init__(self, basins, formations, column_names, depth_bins):
        self.basin_dict = dict.fromkeys(basins)
        if 'Unknown' not in formations:
            formations = np.append(formations, 'Unknown')
        self.formation_dict = dict(zip(sorted(formations), np.arange(len(formations))))
        self.cols = column_names
        self.depth_bins = depth_bins

    def generateBinSum(self, las_data, info_dict, basin_name):
    
        feature_mnem = [v for v in info_dict.values()]
        mnemonics = ['DEPT'] + feature_mnem

        if not all(column_name in las_data.curvesdict.keys() for column_name in mnemonics):
            return None
    
        features_df = las_data.df()
        features_df.reset_index(level=0, inplace=True)

        for column_name in features_df.columns:
            if column_name not in mnemonics:
                features_df.drop(column_name, axis=1, inplace=True)
        # Maintain same order
        features_df = features_df[mnemonics]
        features_df = features_df[features_df['DEPT']>=0]

        # Rename the mnemonic columns
        inv_map = {v: k for k, v in info_dict.items()}
        features_df.rename(inv_map, axis='columns', inplace=True)
        features_df['bins'] = pd.cut(features_df.DEPT, self.depth_bins, right=False, labels=False)
        fsum_df = features_df.groupby('bins').sum()
        for feature in info_dict.keys():
            nkey = 'n' + feature
            fsum_df[nkey] = features_df[features_df[feature].notnull()].groupby('bins').size()
        fsum_df.fillna(0, inplace=True)
        fsum_df = fsum_df.drop(columns=['DEPT'])
        if self.basin_dict[basin_name] is not None:
            df1 = self.basin_dict[basin_name]
            fsum_df = fsum_df.add(df1, fill_value=0)
        self.basin_dict[basin_name] = fsum_df
        return features_df

    def finalizeBinMean(self):
        for key, val in self.basin_dict.items():
            for curve in self.cols:
                ncurve = 'n' + curve
                val[curve] = val[curve]/val[ncurve]
                val.drop(columns=[ncurve], inplace=True)
            val.fillna(0, inplace=True)
            val.rename(index=str, columns=dict(zip(self.cols,['avg'+ curve for curve in self.cols])), inplace=True)
            val.reset_index(inplace=True)
            val['bins'] = val['bins'].astype('int64')

    def fillMissingValues(self, dataset_df, basin_name):
        merged_df = dataset_df.merge(self.basin_dict[basin_name], left_on=['bins'], right_on=['bins'], how='left')
        # Filling the missing values in the dataset with values from the imputed column
        for curve in self.cols:
            dataset_df[curve] = merged_df[curve].fillna(merged_df['avg'+curve])

    def getNumFormations(self):
        return len(self.formation_dict)

    @staticmethod
    def addLatitudeLongitude(features_df, individual_uwi_df_unsorted):
        features_df['LAT'] = individual_uwi_df_unsorted['WGS84Latitude'].values[0]
        features_df['LON'] = individual_uwi_df_unsorted['WGS84Longitude'].values[0]
    
    # Function for assembling formation depth wise 
    def assembleFormations(self, features_df, individual_uwi_df_unsorted):
        """Description: this function assembles formations depth wise
           Args:
               features_df (dataframe): features dataframe
               individual_uwi_df_unsorted: unsorted individual uwi formation tops
           Modifies features_df by adding formation id, formation names and basin
        """
    
        individual_uwi_df = individual_uwi_df_unsorted.sort_values(['FormationTopMD'])
        # Add a new columns and set them to 'Unknown'
        features_df['FORMATION_ID'] = 0
        features_df['FORMATION_NAME'] = 'Unknown'
        features_df['BASIN'] = individual_uwi_df['Name'].values[0]
   
        for i, row in individual_uwi_df.iterrows():
            top = np.around(row['FormationTopMD'])
            formation_id = row['FormationTopID']
            formation_name = row['FormationName']
            try:
                index, = np.where(features_df['DEPT'] == top)
                features_df['FORMATION_ID'][index[0]:] = formation_id
                features_df['FORMATION_NAME'][index[0]:] = formation_name
            except Exception as err:
                print(f'Error populating {formation_name} for file {individual_uwi_df.index[0]}')
                print(repr(err))
                raise

        features_df['FORMATION_NAME'].replace(self.formation_dict, inplace=True)
        features_df['FORMATION_ID'] = features_df['FORMATION_ID'].astype(int)

    @staticmethod
    def computeLog(dataset_df, column_names):
        for curve in column_names:
            dataset_df['log'+curve] = np.log(dataset_df[curve])