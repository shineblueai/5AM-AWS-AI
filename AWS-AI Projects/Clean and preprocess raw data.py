# day_6.py
import pandas as pd
import numpy as np

# Create messy data
df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'age': [25, -5, 30, np.nan, 22],
    'salary': ['50k', '60k', '70k', 'error', '55k'],
    'email': ['a@example.com', 'bob@email', 'eve@com', None, 'eve@domain.com']
})

print("Original data:")
print(df)

# 1. Handle missing names: fill with 'Unknown'
df['name'].fillna('Unknown', inplace=True)

# 2. Remove negative age and fill missing age with median
df = df[df['age'] >= 0]
df['age'].fillna(df['age'].median(), inplace=True)

# 3. Clean salary: extract numeric part
df['salary'] = df['salary'].str.extract(r'(\d+)').astype(float) * 1000

# 4. Drop rows with invalid email (no '@')
df = df[df['email'].str.contains('@', na=False)]

print("\nCleaned data:")
print(df)