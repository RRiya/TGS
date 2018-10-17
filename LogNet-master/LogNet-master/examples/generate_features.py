import os
import sys
import pandas as pd
import lasio
import yaml
import argparse
from tqdm import tqdm

from lognet.utilities.FileInput import FileInput
from lognet.preprocess.DataPrep import DataPrep

parser = argparse.ArgumentParser(description="LogNet feature generation")
parser.add_argument('params_file', metavar='FILENAME', type=str, help='Parameter file name in yaml format')
args = parser.parse_args()
try:
    params = yaml.load(open(args.params_file))
except:
    print(f'Error loading parameter file: {args.params_file}.')
    sys.exit(1)

file_input = FileInput(params['las_file_path'], params['metadata_csv'])
file_input.generateFileList(params['feature_curves'], params['target_curves'])

csv_path = params['prepdata_path']

file_path = file_input.getFilePath()
file_list = file_input.getFileList()

prep_data = DataPrep(params['model_type'])
counter = 0
skip_list = []
for file in tqdm(file_list):

    if 'debug' in params and counter == params['debug']:
        break
    csv_file = file.split('.')[0] + '.csv'
    if os.path.isfile(os.path.join(csv_path, csv_file)):
        counter += 1
        continue

    try:
        las_file = lasio.read(os.path.join(file_path, file))
    except:
        print(f'There was an error reading file {file}')
        skip_list.append(file)
        continue

    file_info = file_input.getFileInfo(file)

    features_df = prep_data.generateFeatures(las_file, file_info, params['feature_curves'], params['target_curves'], True)

    if not isinstance(features_df, pd.core.frame.DataFrame):
        print(f'Features not generated for file {file}')
        skip_list.append(file)
        continue

    if len(features_df) > 0:
        features_df.to_csv(os.path.join(csv_path, csv_file), index=False)
        counter += 1
    else:
        skip_list.append(file)

with open(os.path.join(csv_path, 'error_files.txt'), 'w') as efile:
    efile.write('\n'.join(fname for fname in skip_list))

print(f"Number of csv files: {counter}")
print(f"Number of error files: {len(skip_list)}")
