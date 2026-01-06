# Decision Tree and Random Forest Implementation as Supervised Learning Models
# Its like a tree structure where each node represents a feature (or attribute),
# each branch represents a decision rule, and each leaf node represents an outcome (or class label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow 
import mlflow.sklearn 

# Load the Iris dataset
iris = load_iris()
X = iris.data  # features, column names are in iris.feature_names
y = iris.target # target labels

df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = y
print(df.shape)
#print(df.head())

# Training and test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(X_train.shape, X_test.shape)


# Decision Tree Classifier
print("\n--- Decision Tree Classifier ---")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, dt_pred))

# Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree")

# Working with Random Forest Classifier
print("\n--- Random Forest Classifier ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, rf_pred))

# Visualize Feature Importances from Random Forest


# Now integrating mlfow for experiment tracking
mlflow.set_experiment('Decision_tree ')

# Logging Decision Tree model
with mlflow.start_run(run_name="Decision_Tree_Model"):
    mlflow.log_param('model_type', 'Decision Tree')
    mlflow.log_param('max_depth', 3)
    mlflow.log_metric('accuracy', dt_accuracy)
    mlflow.sklearn.log_model(dt, "decision_tree_model")

   