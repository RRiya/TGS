import unittest
import pandas as pd
import numpy as np
import lasio

from lognet.preprocess.FormationPrep import FormationPrep

class TestFormationPrep(unittest.TestCase):
    """
    Unit tests for formation preprocessing
    """

    def setUp(self):
        basins = ['b1', 'b2']
        formations = ['f1', 'f2', 'f3']
        column_names = ['c1', 'c2', 'c3', 'c4']
        self.depth_bins = np.arange(0, 17, 2)
        self.widget = FormationPrep(basins, formations, column_names, self.depth_bins)
        
    def test_generateBinSum(self):
        """
        Test sum of bins function on a dataframe
        """
        print('Testing bin sum.')
        las_data = lasio.LASFile()
        flen = len(self.depth_bins)
        c1 = np.random.random(flen)
        c2 = np.random.random(flen)

        las_data.add_curve('DEPT', self.depth_bins, unit='m')
        las_data.add_curve('C1', c1, descr='form1')
        las_data.add_curve('C2', c2, descr='form2')
        info_dict = {'form1':'C1', 'form2':'C2'}
        data_df = self.widget.generateBinSum(las_data, info_dict, 'b2')

        self.assertEqual(data_df.columns.values.tolist(), ['DEPT', 'form1', 'form2', 'bins'])
        self.assertEqual(data_df['DEPT'].values.tolist(), self.depth_bins.tolist())
        self.assertEqual(data_df['form1'].values.tolist(), c1.tolist())
        self.assertEqual(data_df['form2'].values.tolist(), c2.tolist())

    def test_fail(self):
        self.assertTrue(False, 'Test failed.')

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

