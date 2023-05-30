import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import time

sns.set_palette('Set2')
pd.options.plotting.backend = 'plotly'

# ignore warnings
import warnings

# Record the starting time
start_time = time.time()

random_state = 42
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes_prediction_dataset.csv')

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
lower_whisker = df['bmi'].where(df['bmi'] >= Q1 - 1.5 * IQR).dropna().min()
upper_whisker = df['bmi'].where(df['bmi'] <= Q3 + 1.5 * IQR).dropna().max()

outliers = df[(df['bmi'] < lower_whisker) | (df['bmi'] > upper_whisker)]
print(outliers['bmi'])

# calculate the IQR
Q1 = np.percentile(df['bmi'], 25)
Q3 = np.percentile(df['bmi'], 75)
IQR = Q3 - Q1

# determine the upper and lower bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# remove the outliers
df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]

# plot the boxplot for BMI
sns.boxplot(df['bmi'])
plt.title('Box plot of BMI (without outliers)')
plt.savefig('boxplot_XGBC.png')
plt.show()

df['gender'] = df['gender'].astype('category')
df['smoking_history'] = df['smoking_history'].astype('category')
df['hypertension'] = df['hypertension'].astype(bool)
df['heart_disease'] = df['heart_disease'].astype(bool)
df['diabetes'] = df['diabetes'].astype(bool)

# drop duplicates
df.drop_duplicates(inplace=True)

# check for duplicates again
print(df.duplicated().any())

X = df.drop('diabetes', axis=1)
y = df.diabetes
X = pd.get_dummies(X, columns=['smoking_history', 'gender'], drop_first=True)
X = X.drop(['gender_Other', 'smoking_history_not current', 'smoking_history_never', 'smoking_history_ever'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

model = XGBClassifier(n_estimators=100, random_state=random_state)
model.fit(X_train, y_train)
cv = 5
weights = [2, 3, 25, 50, 100, 500]

y_train_repo = model.predict(X_train)
y_test_repo = model.predict(X_test)
print(f"the accuracy on train set {accuracy_score(y_train, y_train_repo)}")
print(f"the accuracy on test set {accuracy_score(y_test, y_test_repo)}")
print()
print(classification_report(y_test, y_test_repo))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_repo)).plot()
plt.savefig('confmatrix_XGBC.png')
plt.show()

# Compute SHAP values and plot summary
samples = 1000
X_shap = shap.sample(X_train, samples, random_state=random_state)
mask = shap.maskers.Independent(X_train, max_samples=1000)
explainer = shap.KernelExplainer(model.predict, X_shap, masker=mask)

shap_values = explainer.shap_values(X_shap)

explanation = shap.Explanation(shap_values, data=X_shap,
                               feature_names=X_train.columns)

shap.plots.beeswarm(explanation, show=False)
plt.title(f'SHAP Beeswarm Plot for {samples} instances')
plt.tight_layout()
plt.savefig('beeswarm_SHAP.png')
plt.show()

shap.decision_plot(explainer.expected_value, shap_values,
                   feature_names=list(X_train.columns), xlim=(-0.05, 1), show=False)
plt.title(f'SHAP Decision Plot for {samples} instances')
plt.tight_layout()
plt.savefig('dec_SHAP.png')
plt.show()

shap.plots.bar(explanation, show=False)
plt.title(f'SHAP Bar Plot for {samples} instances')
plt.tight_layout()
plt.savefig('bar_SHAP.png')
plt.show()

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the runtime in seconds
print(f"Runtime: {elapsed_time:.2f}s")
