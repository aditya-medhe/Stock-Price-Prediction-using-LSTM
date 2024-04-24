import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Define functions
def prepare_data(df, look_back=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back)].flatten())
        y.append(scaled_data[i + look_back, 0])
    return np.array(X), np.array(y), scaler

def create_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_lstm_model(model, X_train, y_train, epochs=100, batch_size=1):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def make_predictions(model, X_test):
    return model.predict(X_test)

def get_next_day_prediction(model, last_data_point, scaler):
    next_day_data = np.array(last_data_point).reshape(1, 1, 1)
    scaled_next_day_prediction = model.predict(next_day_data)
    return scaler.inverse_transform(scaled_next_day_prediction)[0][0]

def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

def plot_training_data(df):
    st.subheader("Training Data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.update_layout(title='Training Data', xaxis_title='Date', yaxis_title='Stock Price')
    st.plotly_chart(fig)

def plot_testing_data(train_data, test_data, predictions, next_day_prediction):
    st.subheader("Testing and Prediction Data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], mode='lines', name='Training Data', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], mode='lines', name='Testing Data', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=test_data.index, y=predictions, mode='lines', name='Predictions', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[test_data.index[-1] + timedelta(days=1)], y=[next_day_prediction], mode='markers', name='Next Day Prediction', marker=dict(color='purple', size=10)))
    fig.update_layout(title='Testing and Prediction Data', xaxis_title='Date', yaxis_title='Stock Price')
    st.plotly_chart(fig)

def plot_next_ten_days_predictions(future_dates, future_predictions):
    st.subheader("Next 10 Days Predictions")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Next 10 Days Predictions', line=dict(color='orange')))
    fig.update_layout(title='Next 10 Days Predictions', xaxis_title='Date', yaxis_title='Stock Price')
    st.plotly_chart(fig)

def main():
    st.title("Stock Price Prediction with LSTM")

    # User inputs
    ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL):")
    start_date = st.date_input("Enter the start date:")
    end_date = st.date_input("Enter the end date:", value=(datetime.now() - timedelta(days=1)))

    if st.button("Predict"):
        if not ticker:
            st.error("Please enter a valid stock ticker symbol.")
            return

        try:
            # Fetch stock data
            df = fetch_stock_data(ticker, start_date, end_date)

            if df.empty:
                st.error("No data available for the specified ticker and date range.")
                return

            # Display data description
            st.subheader("Data Description")
            st.write(df.describe())

            # Plot training data
            plot_training_data(df)

            # Prepare data
            look_back = 1
            X, y, scaler = prepare_data(df['Close'], look_back)

            # Split data into train and test sets
            train_size = int(len(X) * 0.7)
            test_size = len(X) - train_size
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

            # Build and train LSTM model
            model = create_lstm_model(look_back)
            model = train_lstm_model(model, X_train, y_train)

            # Make predictions
            predictions = make_predictions(model, X_test)
            predictions = scaler.inverse_transform(predictions).flatten()

            # Get next day prediction
            next_day_prediction = get_next_day_prediction(model, df['Close'].iloc[-1], scaler)

            # Plot testing and prediction data with next day prediction
            plot_testing_data(df[:train_size], df[train_size:], predictions, next_day_prediction)

            # Extend time series for future predictions
            future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 11)]
            future_predictions = []
            last_data_point = df['Close'].iloc[-1]
            for _ in range(10):
                next_day_data = np.array(last_data_point).reshape(1, 1, 1)
                scaled_next_day_prediction = model.predict(next_day_data)
                next_day_prediction = scaler.inverse_transform(scaled_next_day_prediction)[0][0]
                future_predictions.append(next_day_prediction)
                last_data_point = scaled_next_day_prediction

            # Plot next ten days predictions
            plot_next_ten_days_predictions(future_dates, future_predictions)

            # Show 10-day predictions in a single line
            st.write("Predicted Stock Prices for the Next 10 Days:", ', '.join(map(str, future_predictions)))

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
