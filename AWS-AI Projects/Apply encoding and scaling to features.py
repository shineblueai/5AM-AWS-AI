# day_7.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Sample data
df = pd.DataFrame({
    'city': ['New York', 'Paris', 'Tokyo', 'Paris'],
    'income': [50000, 70000, 60000, 80000],
    'has_car': ['Yes', 'No', 'Yes', 'No']
})

print("Original data:")
print(df)

# Label encoding for 'city'
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# Label encoding for 'has_car'
df['has_car_encoded'] = le.fit_transform(df['has_car'])

# Standard scaling for 'income'
scaler = StandardScaler()
df['income_scaled'] = scaler.fit_transform(df[['income']])

print("\nAfter encoding and scaling:")
print(df[['city_encoded', 'has_car_encoded', 'income_scaled']])