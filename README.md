# PISA2009_prediction

An SGDregressor prediction model with PISA2009 dataset

# How it works

In this python code, we predict students' grades based on some of their characteristics. To do this, we want to use linear regression. Here we use the SGDregressor 
model in the model_linear module on the scikit library. We set the following values ​​for the stated parameters and find the best value for the learning rate (eta0) 
among the following values ​​using the GridSearchCV and 5fold cross-validation method. 
learning_rate: 'adaptive' 
max_iter : 1000

After finding the best hyperparameters, we fit the model
