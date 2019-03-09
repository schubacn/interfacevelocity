#%% Imports
import time
import sys
import os
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from mlp_class import MLP_Regressor_With_Autoencoder
from make_map import make_map
from data_import_utils import import_data_2Y, Scaler


class Logger(object):
    def __init__(self, logfile = './logs/logfile.txt'):
        self.terminal = sys.stdout
        self.logfile = logfile
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()
        self.terminal.flush()
        



#%% Import Data

# import the whole dataset
        
data_folder = "./data/"
IL_420_data_file = data_folder + "IL420_data_for_training.csv"
All_data_file = data_folder + "3D_data_for_prediction.xlsx"

try:
    Master_Combined
except NameError:
    print('Importing Master data')
    Master_Combined = pd.read_excel(All_data_file)
    print('Imported Data')
else:
    print("Master_Combined already exists")

# Make arrays for the inputs and predictions
# The inputs are the 11 columns starting from the 3rd column
X_predict = Master_Combined.iloc[:,2:13].copy()
X_predict = np.array(X_predict)
Y_predict = np.zeros([np.shape(X_predict)[0], 2]) #placeholder

# import just IL420 data, with input and output values included
# This is the training data
print('Importing IL420 data')
X,Y = import_data_2Y(IL_420_data_file)
print('Data Imported')

# Scale Inputs based on the range of values in the whole dataset
Input_Scaler = Scaler('logistic')
Input_Scaler.fit(X_predict)
X_predict = Input_Scaler.transform(X_predict)
X = Input_Scaler.transform(X)

# Scale Outputs based on range of values in the IL420 data
Output_Scaler = Scaler('logistic')
Y = np.array(Y)
Output_Scaler.fit(Y)
Y_Scaled = Output_Scaler.transform(Y)


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, 
                                                    Y_Scaled, 
                                                    test_size = .15)
print(f"Data Scaled")

#%% Set parameters to use for this run
output_folder_root = "./run_A"

# Edit the three lists to test different sets of hyperparameters
# Or, use a single set of hyperparameters and have several repeat training runs
def yield_hyperparameters():
    for l2_beta in [0.02]:
        for hidden_layer_1_size in [9]:
            for hidden_layer_2_size in [12]:
                for _ in range(1):# number of repeats for given hyperparameters
                    yield (hidden_layer_1_size, hidden_layer_2_size, l2_beta)
                
#%% Train
for h1_size, h2_size, l2_beta in yield_hyperparameters():
    start_time = time.time()
    start_time_string = time.strftime("%Y-%m-%d_%H_%M_%S")
    folder_path = output_folder_root + '/' + start_time_string + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    sys.stdout = Logger(folder_path + 'logfile.txt')
#%% Declare MLP and fit
    MLP =  MLP_Regressor_With_Autoencoder(
                         input_size = 11, 
                         hidden_1_size = h1_size, 
                         hidden_2_size = h2_size, 
                         output_size = 2, 
                         hidden_1_activation = tf.nn.relu, 
                         hidden_2_activation = tf.nn.relu, 
                         ae_output_activation = tf.nn.relu,
                         output_activation = tf.nn.relu,
                         tensorboard_path = folder_path + '/tensorboard/',
                         verbose = True, 
                         verbose_debug = False, 
                         weight_init_mean = 0,
                         weight_init_std = 0.25,
                         l2_norm_beta = l2_beta,
                         learning_rate_ae = 0.01,
                         learning_rate_mlp = 0.01,
                         stopping_criteria_ae_1 = 0.000001,
                         stopping_criteria_ae_2 = 0.00000001,
                         stopping_criteria_mlp_1 = 0.000001,
                         stopping_criteria_mlp_2 = 0.00000001,             
                         epochs_ae = 60,
                         epochs_mlp = 1000,
                         batch_size_ae = 1000,
                         batch_size_mlp = 50,
                         display_freq_batch_ae = 200,
                         display_freq_epoch_ae = 5,
                         display_freq_batch_mlp = 100,
                         display_freq_epoch_mlp = 100)
        
    MLP.fit_autoencoder(X_predict.copy(), X)
    
    MLP.fit(X_Train, Y_Train, X_Test, Y_Test)
    
    MLP.plot_training_loss_curves(folder_path)
    
    MLP.save_model(folder_path + 'saved_model/')
    
    MLP.save_hyperparameters(folder_path)
    
    
    # %% Check 
    Ybar_Train = MLP.predict(X_Train)
    Ybar_Test = MLP.predict(X_Test)
    Ybar = MLP.predict(X)
    
    #Calculate quality metrics
    Train_Above_Fig = np.stack((Y_Train[:,1], Ybar_Train[:,1]), axis = 1)
    Train_Above_Fig = Train_Above_Fig[Train_Above_Fig[:,0].argsort()]
    Test_Above_Fig = np.stack((Y_Test[:,1], Ybar_Test[:,1]), axis = 1)
    Test_Above_Fig = Test_Above_Fig[Test_Above_Fig[:,0].argsort()]
    Train_Below_Fig = np.stack((Y_Train[:,0], Ybar_Train[:,0]), axis = 1)
    Train_Below_Fig = Train_Below_Fig[Train_Below_Fig[:,0].argsort()]
    Test_Below_Fig = np.stack((Y_Test[:,0], Ybar_Test[:,0]), axis = 1)
    Test_Below_Fig = Test_Below_Fig[Test_Below_Fig[:,0].argsort()]
        
    plt.figure()
    fig, axes = plt.subplots(2,2)
    fig.set_size_inches(18,6)
    
    ax = axes[0][0]
    ax.plot(Test_Above_Fig[:,0], label = "Real")
    ax.plot(Test_Above_Fig[:,1], label = "Prediction")
    ax.legend()
    ax.set_title("Vp Above Test Set")
    
    ax = axes[1][0]
    ax.plot(Train_Above_Fig[:,0], label = "Real")
    ax.plot(Train_Above_Fig[:,1], label = "Prediction")
    ax.legend()
    ax.set_title("Vp Above Training Set")
    
    ax = axes[0][1]
    ax.plot(Test_Below_Fig[:,0], label = "Real")
    ax.plot(Test_Below_Fig[:,1], label = "Prediction")
    ax.legend()
    ax.set_title("Vp Below Test Set")
    
    ax = axes[1][1]
    ax.plot(Train_Below_Fig[:,0], label = "Real")
    ax.plot(Train_Below_Fig[:,1], label = "Prediction")
    ax.legend()
    ax.set_title("Vp Below Training Set")
    
    plt.savefig(folder_path + 'training-test plots.png')
    
    
    Ybar_Train_Scaled_Up = Output_Scaler.inv_transform(Ybar_Train)
    Ybar_Test_Scaled_Up = Output_Scaler.inv_transform(Ybar_Test)
    Ybar = Output_Scaler.inv_transform(Ybar)
    
    Below_Fig = np.stack((Y[:,0], Ybar[:,0]), axis = 1)
    Above_Fig = np.stack((Y[:,1], Ybar[:,1]), axis = 1)
    
    tick_size = 18
    title_size = 25
    
    plt.figure()
    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(18,6)
    ax = axes[1]
    ax.plot(Below_Fig[:,0], label = "Real")
    ax.plot(Below_Fig[:,1], label = "Prediction")
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title("Vp Below West to East", fontsize=title_size)
    
    
    ax = axes[0]
    ax.plot(Above_Fig[:,0], label = "Real")
    ax.plot(Above_Fig[:,1], label = "Prediction")
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_title("Vp Above West to East", fontsize=title_size)
    
    plt.savefig(folder_path + 'real vs prediction west-to-east.png')
    
    plt.show()
    
    
    Error_Train = Ybar_Train_Scaled_Up - Output_Scaler.inv_transform(Y_Train)
    Error_Test = Ybar_Test_Scaled_Up - Output_Scaler.inv_transform(Y_Test)
    
    Error_Pct_Train = Error_Train / Output_Scaler.inv_transform(Y_Train)
    Error_Pct_Test = Error_Test / Output_Scaler.inv_transform(Y_Test)
    
    Ave_Abs_Error_Train = np.mean(np.absolute(Error_Pct_Train), axis=0)
    Ave_Abs_Error_Test = np.mean(np.absolute(Error_Pct_Test), axis=0)
    print(f"Ave_Abs_Error_Train = {Ave_Abs_Error_Train}")
    print(f"Ave_Abs_Error_Test = {Ave_Abs_Error_Test}")
    
    plt.figure()
    ax = plt.gca()
    ax.hist(100*Error_Pct_Train, label = ["Vp Above","Vp Below"])
    ax.set_title("Histogram of % Error in predictions")
    ax.text(-10, 300, f"Ave Error {100*(np.mean(Error_Pct_Train,axis=0))}")
    plt.legend()
    
    plt.savefig(folder_path + 'histogram of prediction errors.png')
    plt.show()
    
    #%% predict Master data
    
    # Run the model on the master inputs
    Y_predict = MLP.predict(X_predict)
    # The model outputs numbers from 0-1, so they must be scaled up
    Y_predict_Scaled_Up = Output_Scaler.inv_transform(Y_predict)
    
    # Put the data into the dataframe and save it
    Master_Combined.iloc[:,-2:] = Y_predict_Scaled_Up
    
    
    #%% Save Predicted data
    now = time.time()
    print("Predictions made on master data")
    Master_Combined.to_csv(folder_path + '3D_data_with_predictions.csv')
    print(f'Finished saving.  Took {time.time() - now} seconds')
    
    now = time.time()
    Master_Predictions = pd.concat([Master_Combined.iloc[:,0:2], Master_Combined.iloc[:,-2:]], axis=1)
    print('Writing simple predictions sheet')
    Master_Predictions.to_csv(folder_path + '3D_predictions.csv')
    print(f'Finished saving.  Took {time.time() - now} seconds')
    
    now = time.time()
    IL420_Predictions = pd.DataFrame(Ybar)
    IL420_Predictions.columns = Master_Predictions.columns.values[-2:]
    print('Writing IL420 Predictions')
    IL420_Predictions.to_csv(folder_path + 'IL420_predictions.csv')
    print(f'Finished saving.  Took {time.time() - now} seconds')
    
    #%% save a few more things 
    
    with open(folder_path + 'hyperparameters.csv', 'a') as f:
        w = csv.writer(f)
        w.writerow(['Ave_Abs_Error_Train', Ave_Abs_Error_Train])
        w.writerow(['Ave_Abs_Error_Test', Ave_Abs_Error_Test])
        w.writerow(['Ave_Error_Train (bias)', 100*(np.mean(Error_Pct_Train,axis=0))])
        w.writerow(['Ave_Error_Test (bias)', 100*(np.mean(Error_Pct_Test,axis=0))])
    
    #%% Generate maps
    print("Generating maps")
    
    
    
    make_map(np.array(Master_Combined['X']), 
             np.array(Master_Combined['Y']), 
             np.array(Master_Combined['Below Velocity - Mantle (m/s)']), 
             point_spacing=25, 
             cmap='gist_heat',
             title='Mantle Velocity m/s',
             scale_min=5000,
             scale_max=8500,
             save_location=folder_path + 'Map Mantle velocity.png')

    make_map(np.array(Master_Combined['X']), 
             np.array(Master_Combined['Y']), 
             np.array(Master_Combined['Above Velocity - Basement (m/s)']), 
             point_spacing=25, 
             cmap='gnuplot2',
             title='Basement Velocity m/s',
             scale_min=5000,
             scale_max=8500,
             save_location=folder_path + 'Map Basement velocity.png')
    
    #Calculate serpentinite percent from Vp Below
    Serp = [0.119474*(3314 - ((VPBi/1000) + 4.9587)/0.0039) for VPBi in Master_Combined['Below Velocity - Mantle (m/s)']]  
    Serp = [Serpi if Serpi > 0 else 0 for Serpi in Serp]
    
    make_map(np.array(Master_Combined['X']), 
             np.array(Master_Combined['Y']), 
             np.array(Serp), 
             point_spacing=25, 
             cmap='viridis',             
             title='Percent Serpentinization',
             scale_min=0,
             scale_max=100,
             save_location=folder_path + 'Map percent serpentinite.png')
    
    
    # Flush the log, in anticipation of the next run
    sys.stdout.flush()