# basic-ann
Basic Artificial Neural Network with Evaluations using Tensorflow in Python.

Uses the sample Bank Data to find out if a customer will exit the bank

build_ann.py : The file that contains the code for the Artificial Neural Network

build_ann_eval.py : This file has the code to evaluate an ANN using mean and 
                    variance. Then we have another function with some added 
                    dropout values to reduce overfitting and inrease variance.

build_ann_tuning.py : This file uses the ANN created to tune the parameters
                      like number of epochs, batch size, etc to improve 
                      performance of the ANN.

Churn_Modelling.csv : The sample data file

 1. The code takes the relevant columns of data from the Churn_Modelling.csv

 2. The data is converted to the features into numeric values.

 3. Then the values are scaled.

 4. The neural network is configured to have 2 hidden layers with 6 cells each and the activation functions to be used in the layer.

 5. The output layer outputs the probability of each of these customers leaving which is then converted to boolean values.

 6. The confusion matrix provides the data to measure the performance of the neural network by recognizing the true positives, false positives, false negatives and true negatives.

The neural network stochastically finds the global minimum of the data.