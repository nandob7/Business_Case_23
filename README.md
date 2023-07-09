# Business_Case_23

# AI Explainability Research - Shapley and LIME
This repository contains Python code and associated data for demonstrating the use of Kernel Shapley and LIME methods in the field of AI explainability. The main research has been done on the diabetes prediction classification dataset, which can be found [here](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).
The code also includes a demo of these methods on a common small dataset - the Boston Housing dataset.

**Description**

Kernel Shapley and LIME are popular techniques used in AI and Machine Learning to understand and explain the decisions made by complex models. Kernel Shapley provides a unified measure of feature importance by allocating a payoff among features. LIME (Local Interpretable Model-Agnostic Explanations) is a technique that can explain the predictions of any classifier or regressor in a faithful way by approximating it locally with an interpretable model.

Our code utilizes the XGBoost Classifier model, applies them to the dataset, and then uses Kernel Shapley and LIME to assess feature importance while validating using the leave-one-covariate-out method.


**Requirements**

The code requires the following Python libraries:

* sklearn
* lime
* matplotlib
* numpy
* pandas
* shap
* xgboost
* seaborn

You can install the libraries via pip:

```pip install sklearn lime matplotlib numpy pandas shap xgboost seaborn```

**Usage**

To run the code, simply navigate to the root directory of this project in your terminal and type, for example, running the SHAP file:

```python SHAP.py```

The code files in this project perform the following:

1. Demo of methods on the Boston Housing dataset. 
2. Loading the Diabetes Classification dataset.
3. Performing preprocessing and dealing with class imbalance.
4. Splitting the dataset into training and testing sets.
5. Defining, training, and evaluating the XGB Classifier model.
6. Computing and plotting SHAP values.
7. Computing and printing LIME explanations for a random instance in the test set.
