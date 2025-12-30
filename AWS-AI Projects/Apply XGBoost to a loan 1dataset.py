# day_15.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create synthetic customer data
np.random.seed(42)
customers = pd.DataFrame({
    'annual_spending': np.random.randint(1000, 20000, 200),
    'visit_frequency': np.random.randint(1, 50, 200)
})

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customers)

# Fit K-Means (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
customers['cluster'] = kmeans.fit_predict(X_scaled)

# Plot
plt.scatter(customers['annual_spending'], customers['visit_frequency'], c=customers['cluster'], cmap='viridis')
plt.xlabel('Annual Spending')
plt.ylabel('Visit Frequency')
plt.title('Customer Segmentation (K-Means)')
plt.colorbar()
plt.show()