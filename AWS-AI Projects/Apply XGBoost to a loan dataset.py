# day_14.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load loan data
df = pd.read_csv('loan_data.csv')

# Prepare features and target
X = df[['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'credit_length']]
y = (df['loan_status'] == 'Charged Off').astype(int)  # 1 = default

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))