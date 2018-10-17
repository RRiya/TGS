import os
import sys
import errno
from itertools import accumulate
import numpy as np
import pandas as pd
import lasio
import yaml
import re
from sklearn.model_selection import train_test_split

from lognet.utilities.FileInput import FileInput
from lognet.utilities.FileOutput import FileOutput
from lognet.preprocess.DataPrep import DataPrep
from lognet.models.ModelFactory import ModelFactory
from lognet.loader.structured_data_loader import StructuredDataset
from lognet.utilities import PlottingMissingCurve
from lognet.utilities import AnalyseResults

params = yaml.load(open('examples/trainparams.yml'))

file_input = FileInput(params['las_file_path'], params['metadata_csv'])
file_input.generateFileList(params['feature_curves'], params['target_curves'])

csv_output = FileOutput(params['prepdata_path'])

file_path = file_input.getFilePath()
file_list = file_input.getFileList()

execute_plots = PlottingMissingCurve.Create_plots(params['feature_curves'],params['target_curves'],params['minmax_dict'])
execute_Analysis = AnalyseResults.Perform_analysis()

if not os.path.exists(params['model_path']):
    os.mkdir(params['model_path'])

if not params['skip_prep']:
    prep_data = DataPrep(params['model_type'])
    counter = 0
    skip_list = []
    error_list = []
    for file in file_list:

        if 'debug' in params and counter == params['debug']:
            break

        try:
            las_file = lasio.read(os.path.join(file_path, file))
            file_info = file_input.getFileInfo(file)
            features_df = prep_data.generateFeatures(
            las_file, file_info, params['feature_curves'], params['target_curves'], True)
            if not isinstance(features_df, pd.core.frame.DataFrame):
                continue
            csv_file = file.split('.')[0] + '.csv'
            print(file,len(features_df))
            if len(features_df) > 0:
                csv_output.write_csv(csv_file, features_df)
                counter += 1
        except:
            error_list.append(file)

    print(f"Number of csv files: {counter}")
    print(f"Error_list_length:{len(error_list)}")
# ========================================================

full_file_list = [f for f in os.listdir(
    params['prepdata_path']) if f.endswith('csv')]
print('Total number of csv files: ', len(full_file_list))

header_file = os.path.join(params['prepdata_path'], 'headers.txt')
if not os.path.isfile(header_file):
    error_message = 'Path: ' + params['prepdata_path'] + ' does not contain headers.txt file'
    raise FileNotFoundError(error_message)

training_list = pd.read_csv('/homedirs/rakskhan/Desktop/LogNet_Repo_v2/training_list.csv',header=None,index_col=0).iloc[:,0].values.tolist()
validation_df = pd.read_excel('/homedirs/vaisbyre/Desktop/Modelling/SON_Modelling/CSV_files/ValidationData_SON.xlsx',encoding = 'latin-1')
validation_list_n = validation_df[0].tolist()
validation_list = []
pattern = r"(\d{14})"
for j in validation_list_n:
    for s in full_file_list:
        if (re.search(pattern,s).group(1) == str(j)):
            validation_list.append(s)
#training_list, validation_list = train_test_split(full_file_list, test_size=0.25, random_state=424242)


print('Logs in training dataset: ', len(training_list))
print('Logs in validation dataset: ', len(validation_list))
training_dataset = StructuredDataset(params['prepdata_path'], header_file, training_list,params['target_curves'],params['model_type'])
validation_dataset = StructuredDataset(params['prepdata_path'], header_file, validation_list, params['target_curves'],params['model_type'])
print(len(training_dataset),len(validation_dataset))


# Training and validating
train_model = ModelFactory.select(params['model_type'], params['model_path'])
error_metric = train_model.train(training_dataset, validation_dataset)
print(error_metric)


print('Validating trained model')

val_model = ModelFactory.load(params['model_type'], params['model_path'])
record_lengths = validation_dataset.get_record_lengths()
y_pred = val_model.predict(validation_dataset)



pred_output = FileOutput(params['output_path'])
counter = 0
rmse = []
mape = []
dept_dict = {}
analysis_output = str(params['analysis_level'])+'_'+str(params['analysis_error']) +'_' + 'Analysis.csv'

for length, end in zip(record_lengths, accumulate(record_lengths)):
    file_uwi = int(max(validation_list[counter].split('_'), key=len))
    output_file = 'Pred_' + str(file_uwi) + '.csv'
    y_test = []
    depth = []
    feature_dict = {feature:[] for feature in params['feature_curves']}                 
   
    for ix in range((end-length), end):
        yt, xt = validation_dataset[ix]
        y_test.append(yt[0])
        depth.append(xt[0])
        ind = 0
        for feature in params['feature_curves']:
            feature_dict[feature].append(xt[ind+3])
            ind = ind+1
          
    rmse.append((output_file, np.sqrt(np.mean((np.array(y_test)-np.array(y_pred[(end-length):end]))**2))))
    mape.append((output_file, np.mean(np.abs((np.array(y_test)-np.array(y_pred[(end-length):end]))/np.array(y_test)))))
    
    #For plotting
    print(len(depth),len(np.unique(depth)))
    dict_df = {'uwi': file_uwi, 'Depth': depth, 'Predicted': y_pred[(end-length):end], 'Actual': y_test}
    dict_df.update(feature_dict)
    dept_dict.update(dict_df)
    pred_df = pd.DataFrame(data= dict_df)
    pred_output.write_csv(output_file, pred_df)
    counter += 1

rmse_df = pd.DataFrame.from_records(rmse, columns=['file', 'rmse'])
mape_df = pd.DataFrame.from_records(mape, columns=['file', 'mape'])
error_df = rmse_df.merge(mape_df,on = 'file', how = 'left')
error_df.to_csv(os.path.join(params['output_path'],'PredictedRMSEMAPE.csv'))
print(len(validation_dataset))
print(np.sum(record_lengths))

dept_df = pd.DataFrame(data= dept_dict)

# Executing the analyse results module to obtain the top performing UWIs based on the error metric selected
analysis_df = execute_Analysis.analysis(dept_df,params['analysis_level'],params['analysis_count'],params['analysis_error'])
pred_output.write_csv(analysis_output, analysis_df)

# Executing PlottingMissingCurve module to generate actual vs predicted plots along with the input curves
for uwi in np.unique(analysis_df['uwi'].tolist()):
    output_plot = 'Plot_' + str(uwi) + '.png'
    plot_df = dept_df[dept_df['uwi']== uwi]
    fig = execute_plots.plots(plot_df)
    pred_output.write_plots(output_plot,fig)
    

