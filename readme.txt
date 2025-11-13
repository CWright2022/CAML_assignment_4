Assignment 4 - IoT Classification

This project classifies IoT Devices using network traffic data. It implements Decision Tree and Random Forest algorithms as part of a two-stage classification system: 

1. Muiltinomial Navie Bayes:
  - classifying ports, domains, and ciphers
2. Custom Tree & Forest: 
  - uses flow features and multinomaial navie bayes outputs to predict device types. 

Setup: 
1. Install packages
- pip -r requirements.txt
2. run main.py
- python3 main.py

Output: 
- model accuracy, training, and prediction time
- classification report
- normalized confusion matrices

Parameters: 
- max_depth --> max tree depth
- min_node --> min samples per leaf
- num_trees --> number of trees
data_fraction --> training fraction per tree

Notes: 
- code includes Gini impurity calculations and node importance analysis 
- random forest uses bootstrap sampling and random feature sub-selection


