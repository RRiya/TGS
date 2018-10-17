# Importing basic libraries
import pandas as pd
import numpy as np
import os
import math
import sys
import warnings
from lognet.utilities.FileOutput import FileOutput

class Perform_analysis():
    
    """
        Gives a dataframe with the top uwis based on the error received from the YAML input  
    
            Args: 
                preddata (dataframe): dataframe with actual vs predicted at uwi,Depth level
                output_file_path (str): path where the analysis result is to be saved
                level (str): level at which analysis is to be done
                top_count (int): number of top uwis required post analysis
                error (str): criteria for analysis
                
           Returns:
                 Dataframe with the analysis report   
    """     
    
    def __init__(self):
            pass
    
    def error_rmse(self,preddata,level):

            preddata['rmse'] = (preddata['Actual'] - preddata['Predicted'])**2
            rmse=lambda x:math.sqrt(np.mean(x))
            df = preddata.groupby(level, as_index = False).agg({'rmse':rmse})
            return (df)

    def error_mape(self,preddata,level):

            preddata['mape'] = np.abs((preddata['Actual'] - preddata['Predicted'])/preddata['Actual'])
            mape=lambda x:np.mean(x)
            df = preddata.groupby(level, as_index = False).agg({'mape':mape})    
            return (df)

    def analysis(self,preddata,level,top_count,error):
            
            assert(isinstance(preddata, pd.core.frame.DataFrame) == True), "preddata is not a dataframe"
            assert(isinstance(top_count, int) == True),"top_count is not an integer"
            assert(isinstance(level, str) == True),"level is not a string"
            assert(isinstance(error, str) == True),"error is not a string"
            error_functions = {'mape': self.error_mape,'rmse':self.error_rmse}
            error_df= error_functions[str(error.lower())](preddata,level)
            error_df = error_df.sort_values(by=str(error.lower()), ascending=False)
            print(error_df.head(5))
            df_top = error_df.tail(top_count)
            return df_top