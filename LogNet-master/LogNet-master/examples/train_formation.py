import os
import yaml
import lasio
import linecache
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

from lognet.utilities.FileInput import FileInput
from lognet.utilities.FileOutput import FileOutput
from lognet.preprocess.FormationPrep import FormationPrep
from lognet.loader.structured_data_loader import StructuredDataset
from lognet.models.ModelFactory import ModelFactory

warnings.simplefilter('ignore', category=DeprecationWarning)

# Load parameters
params = yaml.load(open('examples/trainformparams.yml'))

file_input = FileInput(params['las_file_path'], params['metadata_csv'])
basins, formations = file_input.generateFormationList(params['features'])

csv_output = FileOutput(params['prepdata_path'])
train_output = {}
for bname in basins:
    train_output[bname] = FileOutput(os.path.join(params['prepdata_path'], bname))

file_path = file_input.getFilePath()
file_list = file_input.getFileList()

if not os.path.exists(params['model_path']):
    os.mkdir(params['model_path'])

# Compute imputation values
cols = ['SRESCurve', 'NeutCurve', 'SonicCurve', 'GRCurve', 'CaliCurve']
logcols = ['SRESCurve', 'NeutCurve']
independent = ['DEPT', 'SonicCurve', 'GRCurve', 'LAT', 'LON', 'logNeutCurve','logSRESCurve','CaliCurve']
dependent = ['FORMATION_NAME']
dbins = np.arange(0, 28000, 2000)
form_prep = FormationPrep(basins, formations, cols, dbins)
counter = 0
basin_dict = {}

for fname in file_list:

    try:
        las_file = lasio.read(os.path.join(file_path, fname))
    except:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fname)

    file_info = file_input.getFormationInfo(fname)
    info_dict = dict(zip(cols, file_info[cols].values[0]))
    basin_name = file_info['Name'].values[0]
    features_df = form_prep.generateBinSum(las_file, info_dict, basin_name)

    if not isinstance(features_df, pd.core.frame.DataFrame):
        continue

    FormationPrep.addLatitudeLongitude(features_df, file_info)
    form_prep.assembleFormations(features_df, file_info)

    csv_file = fname.split('.')[0] + '.csv'
    if len(features_df) > 0:
        csv_output.write_csv(csv_file, features_df)
        counter += 1

print(f"Number of csv files: {counter}")
form_prep.finalizeBinMean()

# Preprocess data
csv_path = csv_output.getFilePath()
csv_list = [f for f in os.listdir(csv_path) if f.endswith('csv')]
header_file = os.path.join(csv_path, 'headers.txt')
if not os.path.isfile(header_file):
    error_message = 'Path: ' + csv_path + ' does not contain headers.txt file'
    raise FileNotFoundError(error_message)
line = linecache.getline(header_file, 1)
column_names = line.rstrip().split(',')

for fname in csv_list:
    try:
        csv_df = pd.read_csv(os.path.join(csv_path, fname), header=None, names=column_names)
    except:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fname)
    basin_name = csv_df['BASIN'].values[0]
    
    form_prep.fillMissingValues(csv_df, basin_name)
    FormationPrep.computeLog(csv_df, logcols)
    csv_df = csv_df[independent + dependent]
    csv_df = csv_df.replace([np.inf, -np.inf], np.nan)
    csv_df = csv_df.dropna() 
    train_output[basin_name].write_csv(fname, csv_df)

# Train model
bname = params['basin']
basin_path = train_output[bname].getFilePath()
full_file_list = [f for f in os.listdir(basin_path) if f.endswith('csv')]
print('Total number of csv files: ', len(full_file_list))

header_file = os.path.join(basin_path, 'headers.txt')
if not os.path.isfile(header_file):
    error_message = 'Path: ' + basin_path + ' does not contain headers.txt file'
    raise FileNotFoundError(error_message)

# Training
training_list, validation_list = train_test_split(full_file_list, test_size=0.25, random_state=424242)

print('Logs in training dataset: ', len(training_list))
print('Logs in validation dataset: ', len(validation_list))

training_dataset = StructuredDataset(basin_path, header_file, training_list, dependent)
validation_dataset = StructuredDataset(basin_path, header_file, validation_list, dependent)

# Training and validating
train_model = ModelFactory.select(params['model_type'], params['model_path'], form_prep.getNumFormations())
error_metric = train_model.train(training_dataset, validation_dataset)
print(error_metric)