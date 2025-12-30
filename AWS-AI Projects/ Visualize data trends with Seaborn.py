# day_5.py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load tips dataset (built-in)
tips = sns.load_dataset('tips')

# Scatter plot: total_bill vs tip
plt.figure(figsize=(6,4))
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time')
plt.title('Tip vs Total Bill by Time')
plt.show()

# Box plot: tip by day
plt.figure(figsize=(6,4))
sns.boxplot(data=tips, x='day', y='tip')
plt.title('Tip Distribution by Day')
plt.show()

# Histogram of total_bill
plt.figure(figsize=(6,4))
sns.histplot(tips['total_bill'], kde=True)
plt.title('Distribution of Total Bill')
plt.show()