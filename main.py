import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, PrecisionRecallDisplay
from sklearn.pipeline import Pipeline
sns.set_style('whitegrid')
sns.set_palette('Set2')
pd.options.plotting.backend = 'plotly'

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes_prediction_dataset.csv')

Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
lower_whisker = df['bmi'].where(df['bmi'] >= Q1 - 1.5*IQR).dropna().min()
upper_whisker = df['bmi'].where(df['bmi'] <= Q3 + 1.5*IQR).dropna().max()

outliers = df[(df['bmi'] < lower_whisker) | (df['bmi'] > upper_whisker)]
print(outliers['bmi'])

# calculate the IQR
Q1 = np.percentile(df['bmi'], 25)
Q3 = np.percentile(df['bmi'], 75)
IQR = Q3 - Q1

# determine the upper and lower bounds for outliers
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

# remove the outliers
df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]

# plot the boxplot for BMI
fig = px.box(df, y='bmi')
fig.update_layout(title='Box plot of BMI (without outliers)')
fig.show()

df['gender'] = df['gender'].astype('category')
df['smoking_history'] = df['smoking_history'].astype('category')
df['hypertension'] = df['hypertension'].astype(bool)
df['heart_disease'] = df['heart_disease'].astype(bool)
df['diabetes'] = df['diabetes'].astype(bool)

# drop duplicates
df.drop_duplicates(inplace=True)

# check for duplicates again
print(df.duplicated().any())

X = df.drop('diabetes', axis = 1 )
y = df.diabetes
X = pd.get_dummies(X, columns=['smoking_history', 'gender'], drop_first=True)
X = X.drop(['gender_Other', 'smoking_history_not current', 'smoking_history_never', 'smoking_history_ever'], axis=1)

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_curve, PrecisionRecallDisplay
from sklearn.pipeline import Pipeline
model = XGBClassifier(random_state = 42)
model.fit(X_train, y_train)
cv = 5
weights = [2, 3 , 25, 50, 100]

def report_model(model):
    y_train_repo = model.predict(X_train)
    y_test_repo = model.predict(X_test)
    print(f"the accuracy on train set {accuracy_score(y_train, y_train_repo)}")
    print(f"the accuracy on test set {accuracy_score(y_test, y_test_repo)}")
    print()
    print(classification_report(y_test , y_test_repo))
    ConfusionMatrixDisplay(confusion_matrix(y_test,y_test_repo)).plot()
    plt.show()
report_model(model)
weights = [2, 3 , 25, 50, 100]
param_grid = dict(scale_pos_weight=weights)

grid = GridSearchCV(XGBClassifier(), param_grid = param_grid , cv = cv, scoring = 'recall' )
grid.fit(X_train, y_train)
print(f"best parameters: {grid.best_params_}")
print(f"best scores: {grid.best_score_}")
