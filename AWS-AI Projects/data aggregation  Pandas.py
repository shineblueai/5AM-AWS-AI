# day_4.py
import pandas as pd

# Create sample sales data
df = pd.DataFrame({
    'region': ['North', 'South', 'North', 'South', 'East', 'East'],
    'product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'sales': [100, 150, 200, 120, 180, 90],
    'units': [10, 15, 20, 12, 18, 9]
})

# Group by region and compute total sales
region_sales = df.groupby('region')['sales'].sum()
print("Total sales by region:")
print(region_sales)

# Group by product and compute average units
avg_units = df.groupby('product')['units'].mean()
print("\nAverage units by product:")
print(avg_units)

# Pivot table
pivot = df.pivot_table(values='sales', index='region', columns='product', aggfunc='sum', fill_value=0)
print("\nPivot table (sales by region & product):")
print(pivot)