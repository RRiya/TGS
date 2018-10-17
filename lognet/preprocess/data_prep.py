"""
Convert LAS files to CSV files appropriate for deep learning.
"""
import os
import numpy as np
import pandas as pd
import lasio
import errno
import sys
import itertools
from tqdm import tqdm

from .utils import LASFileUWIMapper

from sklearn.preprocessing import RobustScaler 

curve_classes = ['SRESCurve', 'NeutCurve', 'GRCurve', 'SonicCurve']

class LAStoCSV():
    """
    Convert LAS to CSV files
    """
    def __init__(self, las_folder_path, metadata_df, transforms=None):
        """
        Constructor
        """
        if os.path.isdir(las_folder_path):
            self.las_folder_path = las_folder_path
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), las_folder_path)

        self.file_uwi_mapper = LASFileUWIMapper(las_folder_path)
            
        assert isinstance(metadata_df, pd.core.frame.DataFrame) == True,'metadata_df is not a dataframe'
        self.metadata_df = metadata_df

        if transforms:
            assert isinstance(transforms, dict) == True, 'transforms is not a dictionary'
            assert all(curve in curve_classes for curve in transforms.keys()) == True,"transforms.keys do not contain 'Resistivity', 'Neutron', 'Gamma', 'Sonic'"
            self.transforms = transforms
        else:
            self.transforms = None
    
    def _create_features(self, las_filename, features, targets, smoothing_window=31, n=15):
        """
        TODO: Documentation
        """
        try:
            las_file = lasio.read(os.path.join(self.las_folder_path, las_filename))
        except:
            return None

        assert isinstance(features, dict) == True, 'features is not a dictionary'
        assert isinstance(targets, dict) == True, 'targets is not a dictionary'
        assert all(curve in curve_classes for curve in features.keys()) == True,"features.keys do not contain 'Resistivity', 'Neutron', 'Gamma', 'Sonic'"
        assert all(curve in curve_classes for curve in targets.keys()) == True,"targets.keys do not contain 'Resistivity', 'Neutron', 'Gamma', 'Sonic'"
        mnemonics = list(features.values()) + list(targets.values()) + ['DEPT']

        if not all(column_name in las_file.curvesdict.keys() for column_name in mnemonics):
            return None     
        
        features_df = las_file.df()
        features_df.reset_index(level=0, inplace=True)
        
        for column_name in features_df.columns:
            if column_name not in mnemonics:
                features_df.drop(column_name, axis=1, inplace=True)

        features_df.dropna(axis=0, how='any', inplace=True)

        if features_df.empty:
            return None

        # Handle resistivity in a special manner
        if 'SRESCurve' in list(features.keys()):
            features_df[features['SRESCurve']] = np.log(features_df[features['Resistivity']])

        if 'SRESCurve' in list(targets.keys()):
            features_df[targets['SRESCurve']] = np.log(features_df[features['Resistivity']])

        # Transform the features here
        if self.transforms:
            self._apply_transformation(features_df, features, targets)

        for mnem in features.values():
            rolling_window = features_df[mnem].rolling(window=smoothing_window, center=True)
            features_df[mnem] = rolling_window.mean()

        for mnem in targets.values():
            rolling_window = features_df[mnem].rolling(window=smoothing_window, center=True)
            features_df[mnem] = rolling_window.mean()
        
        for curve in features.keys():
            for shift in range(-(n//2), (n//2)+1):
                column_name = curve + "{0:0=3d}".format(shift)
                features_df[column_name] = features_df[features[curve]].shift(shift)

        for curve in targets.keys():
            features_df[curve] = features_df[targets[curve]]
        
        for mnem in features.values():
            if mnem in features_df.columns:
                features_df.drop(mnem, axis=1, inplace=True)

        features_df.dropna(axis=0, how='any', inplace=True)
        
        try:
            assert features_df.isnull().any().any() == False,'features_df contains null values'
            assert features_df.isna().any().any() == False,'features_df contains NAs'
            assert features_df['DEPT'].is_monotonic == True, 'DEPT in features_df is not monotonic'
        except:
            return None
        
        return features_df

    def _create_feature_list(self, features, n=15):

        feature_list = []

        assert isinstance(features, list) == True, 'features is not a list'

        for feature in features:
            for shift in range(-(n//2), (n//2)+1):
                column_name = feature + "{0:0=3d}".format(shift)
                feature_list.append(column_name)

        return feature_list

    def _apply_transformation(self, features_df, features, targets):
        # for key in features.keys():
        #     features_df[features[key]] = self.transforms[key].transform(np.expand_dims(features_df[features[key]], 1))

        # for key in targets.keys():
        #     features_df[targets[key]] = self.transforms[key].transform(np.expand_dims(features_df[targets[key]], 1))
        for key in features.keys():
            robust_scaler = RobustScaler(quantile_range=(10, 90))
            features_df[features[key]] = robust_scaler.fit_transform(np.expand_dims(features_df[features[key]], 1))

        for key in targets.keys():
            robust_scaler = RobustScaler(quantile_range=(10, 90))
            features_df[targets[key]] = robust_scaler.fit_transform(np.expand_dims(features_df[targets[key]], 1))
    
    def to_csv(self, output_folder, feature_curves, target_curves, n=15, smoothing_window=31, debug=400, keep_location=False):

        assert isinstance(feature_curves, list) == True, 'feature_curves not a list'
        assert isinstance(target_curves, list) == True, 'target_curves not a list'
        
        if debug:
            assert isinstance(debug, int) == True, 'debug is not an integer'

        if self.transforms:
            for key in self.transforms:
                if key not in feature_curves + target_curves:
                    error_message = 'Transform for key: ' + key + ' does not exist.'
                    raise ValueError(error_message)

        feature_list = self._create_feature_list(feature_curves, n=n)

        out_file_count = 0
        header_written = False

        for i, row in tqdm(self.metadata_df.iterrows()):

            if debug:
                if i == debug:
                    break

            feature_dict = dict()
            for curve in feature_curves:
                feature_dict[curve] = row[curve]

            target_dict = dict()
            for curve in target_curves:
                target_dict[curve] = row[curve]

            las_file = self.file_uwi_mapper[row['uwi']]

            features_df = self._create_features(las_file, feature_dict, target_dict, n=n, smoothing_window=smoothing_window)
            
            if not isinstance(features_df, pd.core.frame.DataFrame):
                continue

            location_columns = []
            if keep_location:
                location_columns.extend(['Depth', 'Latitude', 'Longitude'])

                features_df['Depth'] = features_df['DEPT']
                features_df['Latitude'] = row['WGS84Latitude']
                features_df['Longitude'] = row['WGS84Longitude']

            reordered_columns = location_columns + feature_list + target_curves

            for column in features_df.columns:
                if column not in reordered_columns:
                    features_df.drop(column, axis=1, inplace=True)

            features_df = features_df[reordered_columns]

            if not header_written:
                header_list = features_df.columns.values.tolist()

                try:
                    import csv
                except ImportError:
                    error_message = 'Please install csv'
                    raise ImportError(error_message)

                with open(os.path.join(output_folder, 'headers.txt'), 'w') as header_file:
                    csv_writer = csv.writer(header_file, dialect='excel')
                    csv_writer.writerow(header_list)

                header_written = True

            if len(features_df) > 0:
                csv_filename = las_file.split('.')[0] + '.csv'
                csv_filename = os.path.join(output_folder, csv_filename)
                print('Writing file ', csv_filename)
                features_df.to_csv(csv_filename, index=False, header=None)
                out_file_count += 1
            
            if out_file_count % 100 == 0:
                print('Number of csv files: ', out_file_count)
