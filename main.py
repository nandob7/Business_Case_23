import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
np.random.seed(random_state)
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes_prediction_dataset.csv')

df['gender'] = df['gender'].astype('category')
df['smoking_history'] = df['smoking_history'].astype('category')
df['diabetes'] = df['diabetes'].astype(bool)

# drop duplicates
df.drop_duplicates(inplace=True)

# check for duplicates again
print(df.duplicated().any())

X = df.drop('diabetes', axis=1)
y = df.diabetes
X = pd.get_dummies(X, columns=['smoking_history', 'gender'], drop_first=True)
X = X.drop(['gender_Other', 'smoking_history_not current', 'smoking_history_never',
            'smoking_history_ever'], axis=1)

# Correlation Heatmap
corr = pd.concat([X, y], axis=1).corr()

# Create a mask to hide the upper triangle of the heatmap
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, annot=True, mask=mask, fmt='.2f')
plt.title(f'Correlation Heatmap')
plt.tight_layout()
plt.savefig("heatmap.png")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

class_count_0, class_count_1 = y_train.value_counts()
y_train.value_counts().plot(kind='bar', title='count (target)')
plt.show()

# Get the indices of the instances of each class in y_train
indices_majority = y_train[y_train == 0].index
indices_minority = y_train[y_train == 1].index

# Calculate the desired number of samples in the majority class
desired_majority_count = 4 * len(indices_minority)

# Randomly undersample the majority class to get the desired number of samples
undersampled_indices = np.random.choice(indices_majority, size=desired_majority_count, replace=False)

# Combine the undersampled majority indices and the minority indices
balanced_indices = np.concatenate([undersampled_indices, indices_minority])

# Use the balanced_indices to get the balanced X_train and y_train
X_train = X_train.loc[balanced_indices]
y_train = y_train.loc[balanced_indices]

# Correlation Heatmap
corr = pd.concat([X_train, y_train], axis=1).corr()

# Create a mask to hide the upper triangle of the heatmap
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, annot=True, mask=mask, fmt='.2f')
plt.title(f'Training Correlation Heatmap')
plt.tight_layout()
plt.savefig("heatmap_train.png")
plt.show()

model = XGBClassifier(n_estimators=100, random_state=random_state)
model.fit(X_train, y_train)

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
instance_ids = [2, 39]
samples = 1000 - len(instance_ids)

X_shap = shap.sample(X_test, samples, random_state=random_state)
mask = shap.maskers.Independent(X_train, max_samples=1000)

for i in instance_ids:
    X_shap = X_shap.append(X_test.iloc[i, :])
explainer = shap.KernelExplainer(model.predict, X_shap, masker=mask)

shap_values = explainer.shap_values(X_shap)

explanation = shap.Explanation(shap_values, base_values=np.repeat(explainer.expected_value,len(shap_values)),
                               data=X_shap, feature_names=X_test.columns)

# Beeswarm SHAP Plot
shap.plots.beeswarm(explanation, show=False)
plt.title(f'SHAP Beeswarm Plot for {samples + len(instance_ids)} instances')
plt.tight_layout()
plt.savefig('beeswarm_SHAP.png')
plt.show()

# Decision plot SHAP
shap.decision_plot(explainer.expected_value, shap_values,
                   feature_names=list(X_train.columns), xlim=(-0.05, 1), show=False)
plt.title(f'SHAP Decision Plot for {samples + len(instance_ids)} instances')
plt.tight_layout()
plt.savefig('dec_SHAP.png')
plt.show()

# Mean Bar Plot SHAP
shap.plots.bar(explanation, show=False)
plt.title(f'SHAP Bar Plot for {samples + len(instance_ids)} instances')
plt.tight_layout()
plt.savefig('bar_SHAP.png')
plt.show()

# Single Instance Waterfall plot
it = 0
if len(instance_ids) > 0:
    for i in instance_ids:
        shap.plots.waterfall(explanation[-len(instance_ids) + it], show=False)
        plt.title(f'SHAP Waterfall Plot for instance {i + 1}')
        plt.tight_layout()
        plt.savefig(f'wf_SHAP_{i + 1}.png')
        plt.show()
        it += 1

# Multiple Instance Waterfall
instances = 18
instance_nos = np.random.choice(range(1, samples + len(instance_ids) + 1), instances, replace=False)
for i in instance_nos:
    shap.plots.waterfall(explanation[i - 1], show=False)
    plt.title(f'SHAP Waterfall Plot for instance {i}')
    plt.tight_layout()
    plt.savefig(f'wf_SHAP_{i}.png')
    plt.show()


# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the runtime in seconds
print(f"Runtime: {elapsed_time:.2f}s")
