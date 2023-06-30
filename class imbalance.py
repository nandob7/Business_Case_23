import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import time
from sklearn.metrics import roc_curve, roc_auc_score

sns.set_palette('Paired')
pd.options.plotting.backend = 'plotly'

# ignore warnings
import warnings

# Record the starting time
start_time = time.time()

random_state = 42
np.random.seed(random_state)
warnings.filterwarnings('ignore')

# Reading dataset and data-preprocessing
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


def corr_plot(data, title=""):
    corr = data.corr()

    # Create a mask to hide the upper triangle of the heatmap
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, annot=True, mask=mask, fmt='.2f')
    plt.title(f'Correlation Heatmap {title}')
    plt.tight_layout()
    plt.show()


# Splitting the data for model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# Dealing with class imbalance
class_count_0, class_count_1 = y_train.value_counts()

# Get the indices of the instances of each class in y_train
indices_majority = y_train[y_train == 0].index
indices_minority = y_train[y_train == 1].index

# Training data ratios x:1
ratios = [1, 2, 3, 4, 5, 6, 7, 8]

# Define results list
results = []
rocs = []

for r in ratios:
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()

    # Calculate the desired number of samples in the majority class
    desired_majority_count = r * len(indices_minority)

    # Randomly undersample the majority class to get the desired number of samples
    undersampled_indices = np.random.choice(indices_majority,
                                            size=desired_majority_count,
                                            replace=False)

    # Combine the undersampled majority indices and the minority indices
    balanced_indices = np.concatenate([undersampled_indices, indices_minority])

    # # Use the balanced_indices to get the balanced X_train and y_train
    X_train_copy = X_train_copy.loc[balanced_indices]
    y_train_copy = y_train_copy.loc[balanced_indices]

    # Correlation Heatmaps
    corr_plot(pd.concat([X, y], axis=1), f"Full Data: {r}:1")
    corr_plot(pd.concat([X_train_copy, y_train_copy], axis=1), f"Training Set: {r}:1")

    # Define the cost matrix
    cost_matrix = np.array([[0, 1], [3, 0]])  # Adjust the cost values based on your requirements

    # Model Setup Cost Imbalance
    def custom_cost_objective(y_true, y_pred):
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # Apply the logistic transformation to the predictions
        grad = y_pred - y_true
        hess = y_pred * (1.0 - y_pred)

        # Adjust the gradients based on the cost matrix
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] < 0.5:
                grad[i] *= cost_matrix[1, 0]  # Adjust the gradient for false negatives
            elif y_true[i] == 0 and y_pred[i] >= 0.5:
                grad[i] *= cost_matrix[0, 1]  # Adjust the gradient for false positives

        return grad, hess


    # Define different values of scale_pos_weight to try
    scale_pos_weight_values = [1, 1.5, 2, 2.5, 3, 4, 5, class_count_0 / class_count_1]

    # Train and evaluate XGBoost models with different scale_pos_weight values
    for scale_pos_weight in scale_pos_weight_values:
        # Define the XGBoost classifier with the current scale_pos_weight value
        model = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            # objective=custom_cost_objective,
            scale_pos_weight=scale_pos_weight
        )

        # Fit the model to the training data
        model.fit(X_train_copy, y_train_copy)

        # Predict the labels for the test data
        y_pred = model.predict(X_test)

        # Generate a classification report to evaluate the model's performance
        report = classification_report(y_test, y_pred, output_dict=True)

        # Get the predicted probabilities for the positive class
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute the false positive rate (FPR) and true positive rate (TPR) at different thresholds
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Calculate the ROC AUC score
        roc_auc = roc_auc_score(y_test, y_prob)

        # Plot the ROC curve
        if scale_pos_weight == 1:
            rocs.append([fpr, tpr, f'Ratio {r}:1 (AUC = {roc_auc:.4f})'])

        # Store the results in the list
        results.append({
            'ratio': f'{r}:1',
            'scale_pos_weight': round(scale_pos_weight, 1),
            'accuracy': round(report['accuracy'], 4),
            'f1': round(report['macro avg']['f1-score'], 4),
            'false_recall': round(report['False']['recall'], 4),
            'true_recall': round(report['True']['recall'], 4),
            'roc_auc': round(roc_auc, 4)
        })

# Create a figure and axes object
fig, ax = plt.subplots()

for i in rocs:
    ax.plot(i[0], i[1], label=i[2])

# Show the legend
ax.legend(loc='lower right')

# Set the limits of the x-axis and y-axis
ax.set_xlim(0, 0.2)
ax.set_ylim(0.8, 1)

# Set labels and title
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')

# Show the plot
plt.show()

# Create a pandas DataFrame from the results list
df_results = pd.DataFrame(results)

# Print the results table for latex implementation
for index, row in df_results.iterrows():
    print(
        f'{row.ratio} & {row["scale_pos_weight"]} & {row["accuracy"]} & {row["f1"]} & {row["roc_auc"]} & {row["false_recall"]} & {row["false_recall"]} \\\\')

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Print the runtime in seconds
print(f"Runtime: {elapsed_time:.2f}s")
