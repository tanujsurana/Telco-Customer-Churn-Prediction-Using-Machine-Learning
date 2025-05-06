
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# File path
file_path = 'telco_customer_churn.xlsx'
print(f"Attempting to read file from: {os.path.abspath(file_path)}")

# Load data with error handling
try:
    data = pd.read_excel(file_path)
    print("Data loaded successfully.")
    print(f"Number of rows: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")
    print("\nFirst few rows of the data:")
    print(data.head())
    print("\nColumn names:")
    print(data.columns)
    print("\nData types of columns:")
    print(data.dtypes)
    print("\nSummary statistics:")
    print(data.describe())
    print("\nMissing values:")
    print(data.isnull().sum())
except Exception as e:
    print(f"An error occurred while reading the Excel file: {str(e)}")
    exit(1)

# Handle missing data
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)

# Select relevant features
features = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV']
categorical_features = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 
                        'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 
                        'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 
                        'Contract', 'Paperless Billing', 'Payment Method']

# Convert 'Total Charges' to numeric, replacing any non-numeric values with NaN
data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')

# Fill NaN values in 'Total Charges' with the median
data['Total Charges'].fillna(data['Total Charges'].median(), inplace=True)

# Encode categorical variables
for feature in categorical_features:
    data[feature] = pd.Categorical(data[feature]).codes

# Combine numerical and encoded categorical features
X = data[features + categorical_features]
y = data['Churn Value']

# Feature scaling
X = (X - X.mean()) / X.std()

# Train-Test Split
np.random.seed(42)
mask = np.random.rand(len(X)) < 0.8
X_train, X_test = X[mask], X[~mask]
y_train, y_test = y[mask], y[~mask]

# Logistic Regression implementation
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Model training
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train.values, y_train.values)

# Model evaluation
y_pred = model.predict(X_test.values)
y_pred_proba = model.predict_proba(X_test.values)

# Evaluation metrics
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}"

print("Confusion Matrix:")
print(confusion_matrix(y_test.values, y_pred))

print("\nClassification Report:")
print(classification_report(y_test.values, y_pred))

# ROC Curve
def roc_curve(y_true, y_scores):
    thresholds = np.linspace(0, 1, 100)
    tpr, fpr = [], []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    return fpr, tpr

fpr, tpr = roc_curve(y_test.values, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.savefig('roc_curve.png')
plt.close()

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(model.weights)})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.title('Top 10 Important Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Churn probability distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=50)
plt.title('Distribution of Churn Probabilities')
plt.xlabel('Predicted Probability of Churn')
plt.savefig('churn_probability_distribution.png')
plt.close()

print("All visualizations have been saved as PNG files.")
