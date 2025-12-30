# day_8.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset (built-in)
titanic = sns.load_dataset('titanic')

print("Dataset shape:", titanic.shape)
print("\nFirst 5 rows:")
print(titanic.head())

print("\nMissing values:")
print(titanic.isnull().sum())

# Survival rate by class
plt.figure(figsize=(6,4))
sns.barplot(data=titanic, x='class', y='survived')
plt.title('Survival Rate by Class')
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(data=titanic, x='age', hue='survived', kde=True, bins=30)
plt.title('Age Distribution by Survival')
plt.show()

# Correlation heatmap (numeric cols)
numeric_cols = titanic.select_dtypes(include='number')
plt.figure(figsize=(8,6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()