# Chronic_kidney_disease_prediction
This repository includes a project that uses machine learning techniques to predict Chronic Kidney Disease (CKD). Support Vector Machines (SVM), Neural Networks, Logistic Regression, and Decision Trees are among the classification models used in this project to achieve high accuracy in early CKD detection.

## Table of Contents  
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Models Used](#models-used)  
- [Key Features and Findings](#key-features-and-findings)  
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Overview
This study compares the performance of a neural network model (Multi-Layer Perceptron, MLP) to more conventional machine learning models such as Decision Trees, Support Vector Machines (SVM), and Logistic Regression in order to determine how well it can identify Chronic Kidney Disease (CKD) in its early stages.The work aims to enhance detection accuracy while addressing challenges like feature selection and model interpretability.

## Dataset  
The project uses the UCI Chronic Kidney Disease dataset, which includes:  
- **400 instances**  
- **25 attributes**, such as hemoglobin, albumin, packed cell volume, and specific gravity.  
These features are crucial for identifying CKD and evaluating kidney function.

## Models Used  
The following are the machine learning models implemented and evaluated:  
- Logistic Regression  
- Decision Tree  
- Support Vector Machine (SVM)  
- Neural Network(MLP)


## Key Features and Findings  
- **Key Predictors:** The most important characteristics for CKD prediction were determined to be haemoglobin, albumin, packed cell volume, and specific gravity.  
- **Model Performance:**  
  - SVM and Neural Network models achieved highest accuracy (99.17%) and AUC (1.00).  
  - Logistic Regression also performed strongly with an accuracy of 98.33%.  
  - Decision Tree, while slightly less accurate, provided valuable interpretability.  
- **Evaluation Metrics:** To ensure a solid performance evaluation, the models were evaluated using accuracy, precision, recall, F1-score, and AUC.

 ## Results  
| Metric          | Logistic Regression | Decision Tree | SVM           | Neural Network |  
|-----------------|---------------------|---------------|---------------|----------------|  
| Accuracy        | 98.33%              | 96.67%        | 99.17%        | 99.17%         |  
| Precision (CKD) | 1.00                | 1.00          | 1.00          | 1.00           |  
| Recall (CKD)    | 0.97                | 0.95          | 0.99          | 0.99           |  
| F1-Score (CKD)  | 0.99                | 0.97          | 0.99          | 0.99           |  
| AUC             | 1.00                | 0.99          | 1.00          | 1.00           |  

## Usage
1. Clone this repository:  
   ```bash  
   git clone https://github.com/Kottancheril/Chronic_kidney_disease_prediction.git
   ```
2. Open Colab notebook:
   ```bash
   colab notebook chronic_kidney_disease.ipynb
   ```
## Dependencies
The project requires the following Python libraries:
- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
  
Install the dependencies using:
```bash
pip install -r requirements.txt  
```



