"""
This script will make average and standard deviation maps, 
based on all of the predictions for various neural network runs 
found in the root_folder.  

It will save the maps in the root_folder.  It will also save a CSV with the 
average and standard deviations of the predictions. 
"""


import os

import pandas as pd
import numpy as np

from make_map import make_map

# Set the target folder here
root_folder = 'run_A/'


# If using spyder or an IDE which presists variables, you can run this script
# many times on various root folders, and all results will be aggregated
# The following check facilitates this, by not deleting the all_predictions 
# dataframe. It must be deleted manually to reset.
try: 
    len(all_predictions)
except:
    all_predictions = pd.DataFrame()

for path, dirs, files in os.walk(root_folder):
#    print(path)
#    print(dirs)
#    print(files)
    for file in files:
        if file == '3D_data_with_predictions.csv':
            full_path = path + '/' + file
            print(full_path)
            current_file = pd.read_csv(full_path, index_col = 0)
            if len(all_predictions) == 0:
                all_predictions = current_file
            else:
                all_predictions = all_predictions.merge(current_file, 
                                                        how = 'left',
                                                        on = ['X', 'Y'])
del current_file
                
def convert_velocity_to_serp(velocity):
    return 0.119474*(3314 - ((velocity/1000) + 4.9587)/0.0039)

above_predictions = all_predictions.filter(like = 'Above')
above_predictions.columns = ['Above_Velocity' for _ in above_predictions.columns]
below_predictions = all_predictions.filter(like = 'Below')
below_predictions.columns = ['Below_Velocity' for _ in below_predictions.columns]

for column in set(below_predictions.columns):
    serp_predictions = below_predictions[column].apply(convert_velocity_to_serp)

aggr_predictions = all_predictions[['X', 'Y']].copy()

aggr_predictions['Above Velocity - Average'] = above_predictions.mean(axis=1)
aggr_predictions['Above Velocity - Stdev'] = above_predictions.std(axis=1)

aggr_predictions['Below Velocity - Average'] = below_predictions.mean(axis=1)           
aggr_predictions['Below Velocity - Stdev'] = below_predictions.std(axis=1)  

aggr_predictions['Serp Precent - Average'] = serp_predictions.mean(axis=1)
aggr_predictions['Serp Precent - Stdev'] = serp_predictions.std(axis=1)  

aggr_predictions.to_csv(root_folder + "Master_predictions_averaged.csv")

#%%

make_map(np.array(aggr_predictions['X']), 
         np.array(aggr_predictions['Y']), 
         np.array(aggr_predictions['Below Velocity - Average']), 
         point_spacing=25, 
         cmap='gist_heat',
         title='Mantle Velocity m/s - Averaged',
         scale_min=5000,
         scale_max=8500,
         save_location=root_folder + 'Mantle Velocity - Average Map.png')

make_map(np.array(aggr_predictions['X']), 
         np.array(aggr_predictions['Y']), 
         np.array(aggr_predictions['Below Velocity - Stdev']), 
         point_spacing=25, 
         cmap='gist_heat',
         title='Mantle Velocity m/s - Standard Deviation',
         scale_min=None,
         scale_max=None,
         save_location=root_folder + 'Mantle Velocity - Standard Deviation Map.png')

make_map(np.array(aggr_predictions['X']), 
         np.array(aggr_predictions['Y']), 
         np.array(aggr_predictions['Serp Precent - Average']), 
         point_spacing=25, 
         cmap='viridis',
         title='Percent Serpentinite - Averaged',
         scale_min=0,
         scale_max=100,
         save_location=root_folder + 'Percent Serpentinite - Average Map.png')

make_map(np.array(aggr_predictions['X']), 
         np.array(aggr_predictions['Y']), 
         np.array(aggr_predictions['Serp Precent - Stdev']), 
         point_spacing=25, 
         cmap='viridis',
         title='Percent Serpentinite - Standard Deviation',
         scale_min=None,
         scale_max=None,
         save_location=root_folder + 'Percent Serpentinite - Standard Deviation Map.png')

make_map(np.array(aggr_predictions['X']), 
         np.array(aggr_predictions['Y']), 
         np.array(aggr_predictions['Above Velocity - Average']), 
         point_spacing=25, 
         cmap='gnuplot2',
         title='Basement Velocity m/s - Averaged',
         scale_min=5000,
         scale_max=8500,
         save_location=root_folder + 'Basement Velocity - Average Map.png')

make_map(np.array(aggr_predictions['X']), 
         np.array(aggr_predictions['Y']), 
         np.array(aggr_predictions['Above Velocity - Stdev']), 
         point_spacing=25, 
         cmap='gnuplot2',
         title='Basement Velocity m/s - Standard Deviation',
         scale_min=None,
         scale_max=None,
         save_location=root_folder + 'Basement Velocity - Standard Deviation Map.png')
