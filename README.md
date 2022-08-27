# Sequence prediction in Real-time Systems
This repository contains all the code, data and results for the thesis Sequence prediction in Real-time Systems [link to be added]

## Running
To start run models/models.py. This will display an option to train or load a model. 
Make sure to train a model first. After this you can choose which model you want to run an LSTM, a GRU or a TCN model.
This will run according to all parameters specified in the models.py file.

## Results
All resulting files can be found in the models/results folder. Where each directory is from a certain experiment.
The results/graphs.py is used to generate graphs from these files.

## Data
The data folder contains a data sets we used. [3]
https://data.4tu.nl/articles/dataset/Automotive_Controller_Area_Network_CAN_Bus_Intrusion_Dataset/12696950/2

## Synthetic data generation
To generate data sets a tool by Şerban Vădineanu [2] is used. 
https://github.com/SerbanVadineanu/period_inference#trace-generator

## TCN 
TCN model by:
https://github.com/locuslab/TCN [1]

## Bibliography
[1] Shaojie Bai and J. Zico Kolter and Vladlen Koltun (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. 
https://github.com/locuslab/TCN 

[2] Şerban Vădineanu and Mitra Nasri, Robust and accurate period inference using regression-based techniques. 
https://github.com/SerbanVadineanu/period_inference#trace-generator

[3] Guillaume Dupont, Alexios Lekidis, J. (Jerry) den Hartog, and S. (Sandro) Etalle. Automotive Controller Area Network (CAN) Bus Intrusion Dataset v2,
https://data.4tu.nl/articles/dataset/Automotive_Controller_Area_Network_CAN_Bus_Intrusion_Dataset/12696950/2
