# Import necessary libraries
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

random_state = 42
sample_size = 1000

# Load the california housing dataset and shuffle and sample down to 1000 instances
housing = fetch_california_housing()
shuffled_housing = shuffle(housing.data, housing.target, random_state=random_state, n_samples=sample_size)

X = pd.DataFrame(shuffled_housing[0], columns=housing.feature_names)
y = pd.DataFrame(shuffled_housing[1], columns=["MEDV"])

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Train a model
model = RandomForestRegressor(n_estimators=100, random_state=random_state)
model.fit(X_train, y_train.values.ravel())

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

# Compute permutation feature importance
r = permutation_importance(model, X_test, y_test,
                           n_repeats=30,
                           random_state=random_state)
for i in r.importances_mean.argsort()[::-1]:
    print(f"{housing.feature_names[i]:<10}: "
          f"{r.importances_mean[i]:.3f}"
          f" +/- {r.importances_std[i]:.3f}")

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
plt.show()

# Compute LIME explanations
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=housing.feature_names,
                                                   class_names=['MEDV'],
                                                   verbose=True,
                                                   mode='regression')
i = np.random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test.values[i], model.predict, num_features=5)

# print out the explanation in a more readable format
print('Intercept:', exp.intercept[0])
for feature in exp.as_list():
    print('Feature:', feature[0])
    print('Weight:', feature[1])
