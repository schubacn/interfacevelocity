'''
Jonathan Schuba
09 March 2018
'''

from collections import Counter 
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

def import_text(filepath,  csv_delimiter = ','):
    with open(filepath, newline='') as f:
        reader = csv.reader(f, delimiter=csv_delimiter)
        X = []
        for row in reader:
            X.append([str(i) for i in row[:]])
        if len(X) == 1:
            X = X[0]
    return X


def import_data(filepath, csv_delimiter = ',', skip_first_line = True):
    '''
    This function reads (filepath) as csv
    The assumption is that the last column contains the classes
    The return is a pair of lists
        X = list of lists of descriptors
        Y = list of corresponding classes
    '''
    with open(filepath, newline='') as f:
        reader = csv.reader(f, delimiter=csv_delimiter)
        first_line = True
        X = []
        Y = []
        for row in reader:
            if skip_first_line and first_line:
                first_line = False
                #column_labels = row
                continue
            X.append([float(i) for i in row[:-1]])
            Y.append(float(row[-1]))
    return X, Y

def import_data_2Y(filepath, csv_delimiter = ',', skip_first_line = True):
    '''
    This function reads (filepath) as csv
    The assumption is that the last column contains the classes
    The return is a pair of lists
        X = list of lists of descriptors
        Y = list of corresponding classes
    '''
    with open(filepath, newline='') as f:
        reader = csv.reader(f, delimiter=csv_delimiter)
        first_line = True
        X = []
        Y = []
        for row in reader:
            if skip_first_line and first_line:
                first_line = False
                #column_labels = row
                continue
            X.append([float(i) for i in row[:-2]])
            Y.append([float(i) for i in row[-2:]])
    return X, Y

def count_labels(X, Y):
    return dict(Counter(Y))

def remove_specified_labels(X, Y, labels_to_delete = []):
    '''
    Returns new lists which have removed the data with the specified labels.
    Inputs: 
        X                   : List of (lists of) descriptors
        Y                   : List of labels, in the same order as X
        labels_to_delete    : List of labels, which are members of Y
    Outputs:
        X_New               : List of (lists of) descriptors
        Y_New               : List of labels corresponding to X_New
    '''
    Data = list(zip(X,Y))
    New_Data = []
    for row in Data:
        if row[-1] in labels_to_delete:
            continue
        else:
            New_Data.append(row)
    X, Y =  zip(*New_Data)       
    return list(X), list(Y)

def extract_specified_labels(X, Y, labels_to_extract = []):
    '''
    Returns new lists which have only the data with the specified labels.
    Inputs: 
        X                   : List of (lists of) descriptors
        Y                   : List of labels, in the same order as X
        labels_to_extract   : List of labels, which are members of Y
    Outputs:
        X_New               : List of (lists of) descriptors
        Y_New               : List of labels corresponding to X_New
    '''
    if type(labels_to_extract) is int or type(labels_to_extract) is float:
        labels_to_extract = list([labels_to_extract])
    Data = list(zip(X,Y))
    New_Data = []
    for row in Data:
        if row[-1] in labels_to_extract:
            New_Data.append(row)
    X, Y =  zip(*New_Data)       
    return list(X), list(Y)

def expand_specified_labels(X, Y, labels_to_expand = {}):
    '''
    Returns new lists which have duplicates of some data
    Inputs: 
        X                   : List of (lists of) descriptors
        Y                   : List of labels, in the same order as X
        labels_to_expand    : Dict of labels, in the form {label:multiplier}
    Outputs:
        X_New               : List of (lists of) descriptors
        Y_New               : List of labels corresponding to X_New
    '''
    Data = list(zip(X,Y))
    orig_len = len(Data)
    for i in range(orig_len):
        row = Data[i]
        if row[-1] in labels_to_expand.keys():
            for _ in range(labels_to_expand[row[-1]]):
                Data.append(row)
    X, Y =  zip(*Data)       
    return list(X), list(Y)

def split_training_and_test_data(X, Y, training_fraction = 0.8):
    '''
    This was written as an exercize. The sklearn version is probably better.
    
    Returns seperate X, Y for training and test sets
    Ensures that all labels are equally represented in the training and test sets
    Note: This function will probably use a lot of memory for large datasets, since it makes copies.
    Inputs: 
        X                   : List of (lists of) descriptors
        Y                   : List of labels, in the same order as X
        training_fraction   : Fraction of data for the training set.  Remainder goes to test set
    Outputs:
        X_Train             : List of (lists of) descriptors
        Y_Train             : List of labels corresponding to X_Train
        X_Test              : List of (lists of) descriptors
        Y_Test              : List of labels corresponding to X_Test
    '''    
    Data = list(zip(X,Y))
    labels = sorted(list(set(Y)))
    label_counts = count_labels(X, Y)
    
    Data_by_label = {}      # Make a dictonary
    for label in labels:
        Data_by_label[label] = []   #For each label as a dict key, make a list
        
    for row in Data:
        Data_by_label[row[-1]].append(row)     #Fill the dict

#   Randomize the data lists for each class
    for label in Data_by_label.keys():
        random.shuffle(Data_by_label[label])
    
#    Make the training and test sets
#    For each class,
#        Grab the first (training_fraction)% for training, and the remainder for test
#        This ensures that the classes are equally represented in the training and test
    Data_Train = []
    Data_Test = []
    for label in Data_by_label.keys():
        TrainSplit = int(training_fraction*label_counts[label])
        for row in Data_by_label[label][:TrainSplit]:
            Data_Train.append(row)
        for row in Data_by_label[label][TrainSplit:]:
            Data_Test.append(row)
            
#    Shuffle and break the data apart again
#        After the last op, the data are grouped by class
#        We will shuffle them, and then break apart the inputs and outputs            
    random.shuffle(Data_Train)
    random.shuffle(Data_Test)
    X_Train, Y_Train = zip(*Data_Train)
    X_Test, Y_Test = zip(*Data_Test)
    
    return list(X_Train), list(Y_Train), list(X_Test), list(Y_Test)

def randomize_data(X, Y):
    Data = list(zip(X,Y))
    random.shuffle(Data)
    X, Y =  zip(*Data)       
    return list(X), list(Y)

def make_batches(n, batch_size):
    """Generator to create slices containing batch_size elements, from 0 to n.

    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n) 
        
    
def compute_MSE_loss(y_true, y_pred):
    """Compute the squared loss for regression.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) values.

    y_pred : array-like or label indicator matrix
        Predicted values, as returned by a regression estimator.

    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    """
    return ((y_true - y_pred) ** 2).mean() / 2


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          verbose = False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if verbose:
            print("Normalized confusion matrix")
    else:
        if verbose:
            print('Confusion matrix, without normalization')
    
    if verbose:
        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
