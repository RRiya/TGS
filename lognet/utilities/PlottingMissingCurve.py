import pandas as pd
import numpy as np
import os
import math
import sys
import warnings
import matplotlib.pyplot as plt
from lognet.utilities.FileOutput import FileOutput

class Create_plots():
          
    """
        Plots the features and actual v/s prediction plots for the target  
    
            Args: 
                features (list): list of the curves taken as independent
                target (list): list containing the curve to be predicted
                minmax_dict (dict): dict containing the scale for the curves
            
           Returns:
                 Png file for the plots    
    """     
  
    def __init__(self,features,target,minmax_dict):
        
        assert (isinstance(features, list) == True)
        self.features = features
        assert (isinstance(target, list) == True)
        self.target = target
        assert (isinstance(minmax_dict, dict) == True), 'minmax should be a dictionary'
        curve_list = []
        for curve in (self.features+self.target):
            curve_list.append('min_'+str(curve.lower()))
            curve_list.append('max_'+str(curve.lower()))
        assert all(curve in curve_list for curve in minmax_dict) == True , 'minmax_dict do not contain the required curves'
        self.minmax_dict = minmax_dict  
        self.labelsize = 20
        self.fontsize  = 20
        self.weight    = "bold"
       
    def plot_missing_curve_prediction(self,plotdata):

        axoff_fun = np.vectorize(lambda ax:ax.axis('off'))
        figure, ax = plt.subplots(1,len(self.features+self.target), figsize=(21,21))
        axoff_fun(ax)
        depths = plotdata.loc[:,'Depth']
        curves_dict = {}
        for curve in (self.features+['Actual']):
                curves_dict[curve] = plotdata.loc[:,curve]
        predicted_curve = plotdata.loc[:,'Predicted']        
        for axis,curve in enumerate(self.features+['Actual']):
            ax[axis] = figure.add_subplot(1,len(self.features+['Actual']),axis+1)
            ax[axis].plot(curves_dict[curve], depths, 'b--')
            ax[axis].xaxis.tick_top()
            ax[axis].invert_yaxis()
            if(curve == 'Actual'):
                curve= str(self.target[0])
            ax[axis].set_xlim([self.minmax_dict['min'+'_'+str(curve.lower())], self.minmax_dict['max'+'_'+str(curve.lower())]])
            ax[axis].set_ylim([max(depths), min(depths)])
            ax[axis].tick_params(axis='x', labelsize=self.labelsize)
            ax[axis].set_title(str(curve), fontsize=self.fontsize,y= 1.03 ,weight=self.weight)
            ax[axis].minorticks_on()
            ax[axis].grid(which='major', axis='both', linestyle='-', color='black')
            ax[axis].grid(which='minor', axis='both', linestyle='-')
        ax[0].set_ylabel('Depth', fontsize=20,weight="bold") 
        ax[len(self.features+self.target)-1].plot(predicted_curve, depths, 'r--', label='Predicted')
        plt.tight_layout()
        return figure
        
    def plots(self, plotdata):
        
        assert(isinstance(plotdata, pd.core.frame.DataFrame) == True)
        assert(len(np.unique(plotdata['uwi']))==1),'Data should be unique at uwi level for the plots'
        fig = self.plot_missing_curve_prediction(plotdata)
        return fig