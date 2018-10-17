import itertools
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from lognet.models.ModelFactory import ModelFactory

params = {'metadata_csv': '/ldata/data/PermianIndexFiles/QuadComboDensityPermianWells.csv',
          'prepdata_path': '/ldata/cenyongx/PermianFeatures',
          'model_path': '/ldata/cenyongx/tests/model_output',
          'model_type': 'XGB',
          'model_params': {'n_estimators': 2000, 'learning_rate': 0.03, 'gamma': 0,
                           'subsample': 0.75, 'colsample_bytree': 1.0, 'max_depth': 10}}

flist = ['CUBE_DEN', 'GR', 'LOG_NEUT', 'LOG_RES', 'SON']

full_file_list = [f for f in os.listdir(
    params['prepdata_path']) if f.endswith('csv')]
print('Total number of csv files: ', len(full_file_list))

# Training
training_list, validation_list = train_test_split(full_file_list, test_size=0.25, random_state=424242)

with open(os.path.join(params['model_path'], 'training_list.txt'), 'w') as tfile:
    tfile.write('\n'.join(ff for ff in training_list))
with open(os.path.join(params['model_path'], 'validation_list.txt'), 'w') as vfile:
    vfile.write('\n'.join(ff for ff in validation_list))

print('Logs in training dataset: ', len(training_list))
print('Logs in validation dataset: ', len(validation_list))

training_dataset = pd.concat((pd.read_csv(os.path.join(params['prepdata_path'],f)) for f in training_list), ignore_index=True)
training_dataset = training_dataset.sample(frac=1, random_state=345345)
validation_dataset = pd.concat((pd.read_csv(os.path.join(params['prepdata_path'],f)) for f in validation_list), ignore_index=True)
print(f'Training memory usage: {training_dataset.memory_usage()}')
print(f'Validation memory usage: {validation_dataset.memory_usage()}')

train_model = ModelFactory.select(params['model_path'], params['model_type'], params['model_params'])
for idx in range(len(flist)):
    tgt = flist[idx]
    feat = np.delete(flist, idx)
    for nf in range(1,len(flist)):
        seqs = itertools.combinations(feat, nf)
        for d in seqs:
            print(f'Target: {tgt}, Features: {list(d)}')
            params['feature_curves'] = list(d)
            params['target_curves'] = [tgt]

            locfeatures = ['DEPT', 'LAT', 'LON'] + params['feature_curves']
            error_metric = train_model.train(training_dataset, validation_dataset, locfeatures, params['target_curves'])

            pred_model = ''.join(t for t in params['target_curves']) + 'x' + ''.join(f for f in params['feature_curves'])
            with open(os.path.join(params['model_path'], 'model_errors.txt'), 'a') as errfile:
                errfile.write(f'{pred_model} rmse {error_metric}\n')

            with open(os.path.join('examples/yml', pred_model + '.yml'), 'w') as pfile:
                json.dump(params, pfile, indent=4)