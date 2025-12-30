# day_20.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create synthetic stock prices (100 days)
np.random.seed(42)
prices = np.cumsum(np.random.randn(100)) + 100

# Prepare data: use past 10 days to predict next
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(prices)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(10, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, verbose=1)

# Predict last value
pred = model.predict(X_test[-1].reshape(1,10,1), verbose=0)
print(f"Actual next price: {y_test[-1]:.2f}")
print(f"Predicted: {pred[0][0]:.2f}")