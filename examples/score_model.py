import os
import sys
import errno
from itertools import accumulate
import numpy as np
import pandas as pd
import lasio
import yaml
from sklearn.model_selection import train_test_split
from lognet.utilities.FileInput import FileInput
from lognet.utilities.FileOutput import FileOutput
from lognet.preprocess.DataPrep import DataPrep
from lognet.models.ModelFactory import ModelFactory
from lognet.loader.structured_data_loader import StructuredDataset



params = yaml.load(open('examples/scoreparams.yml'))
file_input = FileInput(params['las_file_path'], params['metadata_csv'])
file_input.generateFileList(params['feature_curves'], params['target_curves'])
csv_output = FileOutput(params['prepdata_path'])
file_path = file_input.getFilePath()
file_list = file_input.getFileList()


if not params['skip_prep']:
    prep_data = DataPrep(params['model_type'])
    counter = 0
    skip_list = []
    for file in file_list:

        if 'debug' in params and counter == params['debug']:
            break

        try:
            las_file = lasio.read(os.path.join(file_path, file))
        except:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), file)

        file_info = file_input.getFileInfo(file)

        features_df = prep_data.generateFeatures(
            las_file, file_info, params['feature_curves'], params['target_curves'], True)

        if not isinstance(features_df, pd.core.frame.DataFrame):
            continue

        csv_file = file.split('.')[0] + '.csv'
        if len(features_df) > 0:
            csv_output.write_csv(csv_file, features_df)
            counter += 1
    print(f"Number of csv files: {counter}")

# ========================================================
# Scoring

full_file_list = [f for f in os.listdir(params['prepdata_path']) if f.endswith('csv')]
print('Total number of csv files: ', len(full_file_list))

header_file = os.path.join(params['prepdata_path'], 'headers.txt')
if not os.path.isfile(header_file):
    error_message = 'Path: ' + params['prepdata_path'] + ' does not contain headers.txt file'
    raise FileNotFoundError(error_message)

print('Scoring trained model')

test_model = ModelFactory.load(params['model_type'], params['model_path'])
test_dataset = StructuredDataset(params['prepdata_path'], header_file, full_file_list, params['target_curves'])
record_lengths = test_dataset.get_record_lengths()
y_pred = test_model.predict(test_dataset)

# =========================================================
# Output

pred_output = FileOutput(params['output_path'])
counter = 0
rmse = []
mape = []
dept_dict = {}

for length, end in zip(record_lengths, accumulate(record_lengths)):
    file_uwi = int(max(full_file_list[counter].split('_'), key=len))
    output_file = 'Pred_' + str(file_uwi) + '.csv'
    
    
