# Import necessary libraries
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

random_state = 42

# Load boston housing dataset
housing = load_boston()

X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.DataFrame(housing.target, columns=["MEDV"])

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Define the models
models = [
    ("RandomForestRegressor", RandomForestRegressor(n_estimators=100, random_state=random_state)),
    ("XGBRegressor", xgb.XGBRegressor(n_estimators=100, random_state=random_state))
]

for model_name, model in models:
    # Train and evaluate each model
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - MSE: {mse}, MAE: {mae}, R^2: {r2}")

    # Compute and print permutation feature importance
    r = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=random_state)
    x = []
    y = []
    error = []
    for i in r.importances_mean.argsort()[::-1]:
        y.append(housing.feature_names[i])
        x.append(r.importances_mean[i])
        error.append(r.importances_std[i])

    # Plotting the bar graph
    plt.barh(y, x, align='center', alpha=0.5)
    plt.errorbar(x, y, xerr=error, linestyle='None', color='black')

    # Customizing the plot
    plt.gca().invert_yaxis()
    plt.xlabel('Mean feature importance value')
    plt.title(f'{model_name}: PFI + Std Dev error bar')
    plt.show()

    # Compute SHAP values and plot summary
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.plots.bar(shap_values, show=False)

    plt.title(f'SHAP Summary for {model_name}')
    plt.show()

    # Compute LIME explanations for a random instance in the test set
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=housing.feature_names,
                                                       class_names=['MEDV'], mode='regression')
    i = np.random.randint(0, X_test.shape[0])
    exp = explainer.explain_instance(X_test.values[i], model.predict, num_features=5)

    # Print out the explanation in a more readable format
    print('Intercept:', exp.intercept[0])
    for feature in exp.as_list():
        print('Feature:', feature[0])
        print('Weight:', feature[1])
