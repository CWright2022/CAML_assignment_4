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
```bash
python main.py -r ./iot_data     
```

### 3. Run Automatic Tuning Script
```bash
python tuning.py -r ./iot_data     
```
