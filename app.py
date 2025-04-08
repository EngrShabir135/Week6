import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit UI
st.title("Stock Price Prediction using ARIMA & LSTM")

# Step 1: Load the dataset
uploaded_file = st.file_uploader("Upload your stock data (Excel file)", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    st.write("### Dataset Preview:", data.head())
    
    # Step 2: Visualizing Stock Price Trends
    st.subheader("Stock Closing Price Over Time")
    fig, ax = plt.subplots()
    ax.plot(data['Close*'], label='Closing Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price')
    ax.set_title('Stock Closing Price Over Time')
    ax.legend()
    st.pyplot(fig)
    
    # Step 3: ARIMA Model for Forecasting
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    st.subheader("ARIMA Forecasting")
    model = ARIMA(data['Close*'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)
    st.write("### ARIMA Forecast:")
    st.write(forecast)
    
    # Step 4: LSTM Model for Forecasting
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close*']])
    
    train_size = int(len(data) * 0.8)
    train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]
    
    # Prepare LSTM data
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 10
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train Model
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
    
    # Predict Future Prices
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Plot Predictions
    st.subheader("LSTM Stock Price Prediction")
    fig2, ax2 = plt.subplots()
    ax2.plot(data.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices')
    ax2.plot(data.index[-len(y_test):], predictions, label='Predicted Prices')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.set_title('Stock Price Prediction using LSTM')
    ax2.legend()
    st.pyplot(fig2)
