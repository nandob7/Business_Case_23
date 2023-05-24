# Business_Case_23

# AI Explainability Research - Shapley and LIME Demo
This repository contains Python code and associated data for demonstrating the use of Kernel Shapley and LIME methods in the field of AI explainability. The code demonstrates the application of these methods on a small dataset - the Boston Housing dataset.

**Description**

Kernel Shapley and LIME are popular techniques used in AI and Machine Learning to understand and explain the decisions made by complex models. Kernel Shapley provides a unified measure of feature importance by allocating a payoff among features. LIME (Local Interpretable Model-Agnostic Explanations) is a technique that can explain the predictions of any classifier or regressor in a faithful way, by approximating it locally with an interpretable model.

Our code utilizes the RandomForestRegressor and XGBRegressor models, applies them to the Boston Housing dataset, and then uses Kernel Shapley and LIME to assess feature importance.


**Requirements**

The code requires the following Python libraries:

* sklearn
* lime
* matplotlib
* numpy
* pandas
* shap
* xgboost

You can install the libraries via pip:

```pip install sklearn lime matplotlib numpy pandas shap xgboost```

**Usage**

To run the code, simply navigate to the root directory of this project in your terminal and type:

```python main.py```

This script does the following:

1. Loads the Boston Housing dataset.
2. Splits the dataset into training and testing sets.
3. Defines, trains, and evaluates the RandomForestRegressor and XGBRegressor models.
4. Prints the MSE, MAE, and R^2 evaluation metrics.
5. Computes and visualizes the Permutation Feature Importance (PFI) for each model.
6. Computes and plots SHAP values for each model.
7. Computes and prints LIME explanations for a random instance in the test set.
