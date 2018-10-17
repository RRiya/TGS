import os
import sys
import errno
from itertools import accumulate
import numpy as np
import pandas as pd
import lasio
import yaml
import argparse
from sklearn.model_selection import train_test_split

from lognet.utilities.FileInput import FileInput
from lognet.utilities.FileOutput import FileOutput
from lognet.preprocess.DataPrep import DataPrep
from lognet.models.ModelFactory import ModelFactory

parser = argparse.ArgumentParser(description="LogNet model training")
parser.add_argument('params_file', metavar='FILENAME', type=str, help='Parameter file name in yaml format')
args = parser.parse_args()
try:
    params = yaml.load(open(args.params_file))
except:
    print(f'Error loading parameter file: {args.params_file}.')
    sys.exit(1)

if not os.path.exists(params['model_path']):
    os.mkdir(params['model_path'])

full_file_list = [f for f in os.listdir(
    params['prepdata_path']) if f.endswith('csv')]
print('Total number of csv files: ', len(full_file_list))

# Training
training_list, validation_list = train_test_split(full_file_list, test_size=0.25, random_state=424242)

print('Logs in training dataset: ', len(training_list))
print('Logs in validation dataset: ', len(validation_list))

locfeatures = ['DEPT', 'LAT', 'LON'] + params['feature_curves']
training_dataset = pd.concat((pd.read_csv(os.path.join(params['prepdata_path'],f)) for f in training_list), ignore_index=True)
validation_dataset = pd.concat((pd.read_csv(os.path.join(params['prepdata_path'],f)) for f in validation_list), ignore_index=True)
print(f'Training memory usage: {training_dataset.memory_usage()}')
print(f'Validation memory usage: {validation_dataset.memory_usage()}')

# Training and validating
params['model_params'] = params.get('model_params', {})
train_model = ModelFactory.select(params['model_path'], params['model_type'], params['model_params'])
error_metric = train_model.train(training_dataset, validation_dataset, locfeatures, params['target_curves'])

pred_model = ''.join(t for t in params['target_curves']) + '_' + ''.join(f for f in params['feature_curves'])
with open(os.path.join(params['model_path'], 'model_errors.txt'), 'a') as errfile:
    errfile.write(f'{pred_model} rmse {error_metric}\n')