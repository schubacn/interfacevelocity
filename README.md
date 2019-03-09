# Interface-targeted seismic velocity estimation using machine learning
### C. Nur Schuba, Jonathan P. Schuba, Gary G. Gray, and Richard G. Davy

This is the code and data described in the aforementioned paper.  

The data folder contains the training data from the 2D full-waveform inversion results, along Inline-420, plus the inputs used for prediction throughout the 3D seismic volume. A sub-folder contains the averaged results of these predictions, from an ensemble of neural networks trained seperately on the training data.

The model can be trained and run using train_main.py.  Additional utility functions are included to make the averaged maps and aggregate results. 