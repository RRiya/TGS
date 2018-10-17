import pandas as pd
import numpy as np
import lasio
from sklearn.preprocessing import RobustScaler

curve_classes = ['SRESCurve', 'NeutCurve', 'GRCurve', 'SonicCurve','DEN']

class DataPrep():
    """
    Performs preprocessing on LogData to generate features for training
    """

    def __init__(self, model_type):
        assert isinstance(model_type, str) == True,"model_type is not a string"
        self.model_type = model_type

    def _smooth(self, data_df, feature_names, window_length):
        for curve in feature_names:
            rolling_window = data_df[curve].rolling(
                window=window_length, center=True)
            data_df[curve] = rolling_window.mean()

    def _shift(self, data_df, feature_names, shift_size):
        new_columns = []
        for curve in feature_names:
            for step in range(-(shift_size//2), (shift_size//2)+1):
                column_name = curve + "{0:0=3d}".format(step)
                new_columns.append(column_name)
                data_df[column_name] = data_df[curve].shift(step)
        return new_columns
    
    def _compute_cube(self, data_df, feature):
        try:
            data_df[feature] = data_df[feature]**3
        except KeyError as e:
            print(f'DataPrep (compute_cube): Key {e.args} not found.')
        except:
            print('DataPrep (compute_cube): Error encountered.')
        return data_df 

    
    def _compute_log(self, data_df, feature):
        try:
            data_df[feature] = np.log(data_df[feature])
        except KeyError as e:
            print(f'DataPrep (compute_log): Key {e.args} not found.')
        except:
            print('DataPrep (compute_log): Error encountered.')
        return data_df 
    
    def _cap_feature(self, data_df, feature, lower_cap, upper_cap):
        data_df = data_df[(data_df[feature]>lower_cap) & (data_df[feature]< upper_cap)]
        return data_df 

    def _apply_transform(self, data_df, feature_names):
        for curve in feature_names:
            robust_scaler = RobustScaler(quantile_range=(10, 90))
            data_df[curve] = robust_scaler.fit_transform(np.expand_dims(data_df[curve], 1))
            
    def _remove_nulls(self,data_df):
        data_df = data_df.replace([np.inf, -np.inf], np.nan)
        data_df.dropna(axis=0, how='any', inplace=True)
        return data_df
    
        

    def _create_features(self, las_data, info_dict, features, targets, keep_location, smoothing_window=31, n=15):
        """
        TODO: Documentation
        """
        assert isinstance(info_dict, dict) == True, "info_dict is not a dictionary"
        assert isinstance(features, list) == True, "features is not a list"
        assert all(curve in curve_classes for curve in features)== True,"features do not contain any of 'Resistivity', 'Neutron', 'Gamma', 'Sonic'"
        assert isinstance(targets, list) == True, "targets is not a list"
        assert all(curve in curve_classes for curve in targets) == True,"target do not contain any of 'Resistivity', 'Neutron', 'Gamma', 'Sonic'"
        feature_mnem = [info_dict[key] for key in features]
        target_mnem = [info_dict[key] for key in targets]
        mnemonics = feature_mnem + target_mnem + ['DEPT']
        # print(mnemonics)

        if not all(column_name in las_data.curvesdict.keys() for column_name in mnemonics):
            return None

        features_df = las_data.df()
        features_df.reset_index(level=0, inplace=True)

        for column_name in features_df.columns:
            if column_name not in mnemonics:
                features_df.drop(column_name, axis=1, inplace=True)

        

        # Rename the mnemonic columns
        inv_map = {v: k for k, v in info_dict.items()}
        features_df.rename(inv_map, axis='columns', inplace=True)                          
               
        feature_list = features
        if self.model_type == 'TorchNN':
            # Smoothing
            self._smooth(features_df, features, smoothing_window)

            # Shifting and renaming columns
            feature_list = self._shift(features_df, features, n)

        location_columns = []
        if keep_location:
            location_columns.extend(['Depth', 'Latitude', 'Longitude'])

            features_df['Depth'] = features_df['DEPT']
            features_df['Latitude'] = info_dict['WGS84Latitude']
            features_df['Longitude'] = info_dict['WGS84Longitude']

        reordered_columns = location_columns + feature_list + targets

        # Drop mnemonic named columns
        for column in features_df.columns:
            if column not in reordered_columns:
                features_df.drop(column, axis=1, inplace=True)

        features_df = features_df[reordered_columns]

        return features_df

    def generateFeatures(self, las_data, info_dict, feature_keys, target_keys, keep_location=False):
        print(f"Generating features for {self.model_type}")
        # Check for normalized sonic values
        # try:
        #    sonicvals = las_data[info_dict['Sonic']]
        # except:
        #    print(info_dict['Sonic'] + ' not found in LAS file.')
        #    return None

        # if np.nanmax(sonicvals) <= 1:
        #    print(f'Skipping file with normalized values <= 1.')
        #    return None
        features_df = self._create_features(
            las_data, info_dict, feature_keys, target_keys, keep_location)

        return features_df
