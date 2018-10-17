import unittest
import pandas as pd
import numpy as np
import lasio

from lognet.preprocess.DataPrep import DataPrep

class TestPreprocess(unittest.TestCase):
    """
    Unit tests for data preprocessing
    """

    def setUp(self):
        self.widget = DataPrep('RandomForest')

    def test_shift(self):
        """
        Test shift function on a dataframe
        """
        print('Testing shift.')
        feature_names = ['lin']
        dvect = np.arange(8)
        data_df = pd.DataFrame(data=dvect, columns=feature_names)
        colnames = self.widget._shift(data_df, feature_names, 2)
        self.assertEqual(len(colnames), 3)
        self.assertEqual(data_df[colnames[0]].dropna().values.tolist(), dvect[1:].tolist())
        self.assertEqual(data_df[colnames[1]].values.tolist(), dvect.tolist())
        self.assertEqual(data_df[colnames[2]].dropna().values.tolist(), dvect[:-1].tolist())

    def test_smooth(self):
        """
        Test smooth function on a dataframe
        """
        print('Testing smooth.')
        feature_names = ['lin']
        dvect = np.arange(8)
        data_df = pd.DataFrame(data=dvect, columns=feature_names)
        self.widget._smooth(data_df, feature_names, 3)
        self.assertEqual(data_df[feature_names[0]].dropna().tolist(), dvect[1:-1].tolist())

    def test_generate_features(self):
        """
        Test feature generation function with las
        """
        print('Testing generate features.')
        las_data = lasio.LASFile()
        depths = np.arange(10, 20, 0.5)
        flen = len(depths)
        c1 = np.random.random(flen)
        c2 = np.random.random(flen)
        lat = 20
        lon = 40
        las_data.add_curve('DEPT', depths, unit='m')
        las_data.add_curve('C1', c1, descr='feature')
        las_data.add_curve('C2', c2, descr='target')
        info_dict = {'feature':'C1', 'target':'C2', 'WGS84Latitude':lat, 'WGS84Longitude':lon}
        feature_keys = ['feature']
        target_keys = ['target']
        keeploc_df = self.widget.generateFeatures(las_data, info_dict, feature_keys, target_keys, True)
        noloc_df = self.widget.generateFeatures(las_data, info_dict, feature_keys, target_keys, False)
        self.assertEqual(keeploc_df.shape[0], flen)
        self.assertEqual(keeploc_df['Latitude'].values.mean(), lat)
        self.assertEqual(keeploc_df['Longitude'].values.mean(), lon)
        self.assertEqual(noloc_df.shape[0], flen)
        self.assertEqual(noloc_df.columns.values.tolist(), feature_keys + target_keys)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

