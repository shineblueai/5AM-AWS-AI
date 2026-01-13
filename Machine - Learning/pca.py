# PNM SST MMM  # ILC MST CPP E

# I - Importing libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# 2. Loading dataset 
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df.to_csv("wine.csv", index=False)

# Check datatypes 
#print(df.dtypes)

# Step-4 : Missinig data handling 
print(df.duplicated().sum())
print(df.isnull().sum())

# Step-5 X, y split 
X = df.drop('target', axis=1)   
y = df['target']


#print(df.describe())

# The process of bring data into a range of -1 to 1 or 0 to 1 use StandardScalaer() or MinMaxScalaer()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# Step-6 : Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step-7 : Model Building
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# explained variance ratio
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())
print(pca.components_)
cummulative_variance = np.cumsum(pca.explained_variance_ratio_)



