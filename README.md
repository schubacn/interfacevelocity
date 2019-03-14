# Interface-targeted seismic velocity estimation using machine learning
### C. Nur Schuba, Jonathan P. Schuba, Gary G. Gray, and Richard G. Davy

This is the code and data described in the aforementioned paper.  

The data folder contains the training data from the 2D full-waveform inversion results, from inline-420, plus the inputs used for prediction throughout the 3D seismic volume. A sub-folder contains the averaged results of these predictions, from an ensemble of neural networks trained seperately on the training data.  These are the results described in the paper.

The model can be trained and run using train_main.py.  Additional utility functions are included to make the averaged maps and aggregate results. 

The following software versions were used:

* python 3.6.6
* tensorflow 1.10
* numpy 1.15
* matplotlib 2.2.3
* pandas 0.23.4
* scikit-learn 0.19.1
