# PNM SST MMM

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn 
import xgboost as xgb 
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv("/Users/phanirajendra/Documents/PyCharm-Work/Trainings/EITA/5AM-AWS-AI/Machine - Learning/loan_data.csv")
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.dtypes)
print(df.columns)
le = LabelEncoder()
df['property_area'] = le.fit_transform(df['property_area'])
df['education']     = le.fit_transform(df['education'])



X = df.drop('loan_status', axis=1)
y = df['loan_status']   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Integrating mlflow 
mlflow.set_experiment("Loan Application Prediction")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# SEtting mlflow for GB 
with mlflow.start_run(run_name="GB1"):
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3,random_state=42,learning_rate=0.1)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    acc_gb = accuracy_score(y_test, y_pred_gb)

    mlflow.log_params({
        "model": "GradientBoosting",
        "n_estimators" : 100,
        "max_depth": 3,
        "learning_rate":0.01
    })

    mlflow.log_metric("accuracy", acc_gb )
    print(classification_report(y_test, y_pred_gb))


    # XGBOOST

with mlflow.start_run(run_name="XGBoost"):
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3,random_state=42,learning_rate=0.1)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.fit(X_train, y_train)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)

        mlflow.log_params({
            "model": "XGB",
            "n_estimators" : 100,
            "max_depth": 3,
            "learning_rate":0.01
        })

        mlflow.log_metric("accuracy", acx_xgb)
        print(classification_report(y_test, y_pred_xgb))

        mlflow.end_run()