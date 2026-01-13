# PNM SST MMM | ILC MST CPP E 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, r2_score
import mlflow
import mlflow.sklearn
import os 
import pickle # To save model we need to import pickle / import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

#------------------------------
# Load Dataset
#------------------------------
df = pd.read_csv('/Users/phanirajendra/Documents/PyCharm-Work/Trainings/EITA/5AM-AWS-AI/Machine - Learning/HR_comma_sep.csv')
print(df.head())


#------------------------------
# Check DataTypes 
#------------------------------
print(df.dtypes)


#------------------------------
# Missing Values Handling 
#------------------------------
print(df.duplicated().sum())  # Check Duplicated Rows
print()
print(df.isnull().sum())  # Check Missing Values


#-------------------------------------------------------
# Performing Label Encoding followed by Standard Scaling
#-------------------------------------------------------
print(df['dept'].head())
print(df['salary'].head())


le_dept = LabelEncoder()
df['dept'] = le_dept.fit_transform(df['dept'])
le_salary = LabelEncoder()
df['salary'] = le_salary.fit_transform(df['salary'])
#------------------------------
print(df['dept'].head())
print(df['salary'].head())
#------------------------------

#------------------------------
# Splitting Dataset
#------------------------------
X = df.drop('left', axis=1)
y = df['left']

print(X.shape, y.shape)
#------------------------------
# Train Test Split
#------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# if we have imblanced dataset we can use stratify=y to maintain the same ratio in train and test datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#------------------------------
# Feature Scaling (Standard Scaling)
#------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # Do not use fit_transform on test data only transform

#------------------------------
#Mlflow Setup
#------------------------------

mlflow.set_experiment("HR_Employee_Attrition_Classification")
mlflow.set_tracking_uri("http://localhost:5000")

#--------------------------------------------------------------
# Creating a helper function to train and evaluate models
#--------------------------------------------------------------

def log_and_evalaute(model, model_name, X_test, y_test, is_classifier=True):
    with mlflow.start_run(run_model = model_name):
        y_pred = model.predict(X_test)

        if is_classifier:
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("Accuracy", acc)
            print(f"{model_name} Accuracy: {acc}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("r2_score", r2_score(y_test, y_pred))
            print(f"{model_name} MSE: {mse}")
        
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_param("Model_type", model_name)


#--------------------------------------------------------------
# Training and Evaluating Different Models
#--------------------------------------------------------------


# Logistic Regression
log_reg =LogisticRegression()
log_reg.fit(X_train, y_train)
log_and_evalaute(log_reg, "Logistic_Regression", X_test, y_test, is_classifier=True)

# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
log_and_evalaute(dt, "Decision_Tree_Classifier", X_test, y_test, is_classifier=True)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
log_and_evalaute(rf, "Random_Forest_Classifier", X_test, y_test, is_classifier=True)

#KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
log_and_evalaute(knn, "KNN_Classifier", X_test, y_test, is_classifier=True) 


# Because the given dataset is for classification problem we are not implementing regression algorithms here.
# To perform regression we need to convert the target variable to continuous numerical values.

#Converting the target variable into float datatype for regression algorithms
y_train_reg = y_train.astype(float)
y_test_reg = y_test.astype(float)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train_reg)
log_and_evalaute(lin_reg, "Linear_Regression", X_test, y_test_reg, is_classifier=False)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train_reg)
log_and_evalaute(ridge, "Ridge_Regression", X_test, y_test_reg, is_classifier=False)        

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train_reg)
log_and_evalaute(lasso, "Lasso_Regression", X_test, y_test_reg, is_classifier=False)        

# ElasticNet Regression
elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5)
elasticnet.fit(X_train, y_train_reg)
log_and_evalaute(elasticnet, "ElasticNet_Regression", X_test, y_test_reg, is_classifier=False)          


# Saving the models using pickle
pickle.dump(log_reg, open("log_reg.pkl", "wb")) 
pickle.dump(dt, open("decision_tree.pkl", "wb"))
pickle.dump(rf, open("random_forest.pkl", "wb"))
pickle.dump(knn, open("knn_classifier.pkl", "wb"))
pickle.dump(lin_reg, open("linear_regression.pkl", "wb"))
pickle.dump(ridge, open("ridge_regression.pkl", "wb"))
pickle.dump(lasso, open("lasso_regression.pkl", "wb"))
pickle.dump(elasticnet, open("elasticnet_regression.pkl", "wb"))
