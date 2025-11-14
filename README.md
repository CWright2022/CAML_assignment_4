# Assignment 4 – IoT Classification

This project classifies **IoT devices** using **network traffic data**.  
It implements both **Decision Tree** and **Random Forest** algorithms as part of a **two-stage classification system**:

---

## Two-Stage Pipeline

### 1. Multinomial Naive Bayes
- Classifies **ports**, **domains**, and **ciphers** individually.

### 2. Custom Tree & Forest
- Combines **flow features** and **Naive Bayes outputs** to predict **device types**.

---

## Setup

### 1. Install required packages
```bash
pip install -r requirements.txt
```

### 2. Run Main Script

This will train a random forest with the static hyperparameters defined in main.py. This file contains our custom implementations of RandomForest and DecisionTree.

Ensure the IoT data is present in whatever directory is identified with the -r switch.

```bash
python main.py -r ./iot_data     
```

### 3. Run Automatic Tuning Script

This will automatically generate models with different hyperparameters as defined on line 181 of tuning.py. Every combonation of parameters will be generaeted and acccuracy measured. At the end, the best performing model will be selected.

This script also automatically generates matplotlib charts in tuning_output/ showing how the accuracy for each hyperparameter varies.

```bash
python tuning.py -r ./iot_data     
```
