import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class modelTrain(Sequential,LSTM,Dense,Dropout, MinMaxScaler) :
    def __init__(self,df):
        self.df = df
    def __len__(self):
        return len(self.df) 
    
    def model(self):
        self.train_size = int(0.8* len(self.df))
        # Train ARIMA on the differenced series
                
        self.df['Close_Diff'] = self.df.iloc[:,0].diff().dropna()

        self.train_data = self.df["Close_Diff"][:self.train_size]
        self.test_data = self.df.iloc[:,[0]][self.train_size:self.train_size+30]
        # last_close = df.iloc[:,0][self.train_size-1]

        # Select closing price & normalize it
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.df["Close_scaled"] = self.scaler.fit_transform(self.df.iloc[:,[0]])

        # Define lookback period (e.g., 60 days)
        timesteps = 60  

        # Prepare training data
        X_train, y_train = [], []
        for i in range(timesteps, len(self.df) - 30):  # Reserve last 30 days for forecasting
            X_train.append(self.df["Close_scaled"].values[i-timesteps:i])
            y_train.append(self.df["Close_scaled"].values[i])

        X_train, y_train = np.array(X_train), np.array(y_train)

        # Build LSTM model
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),  
            Dropout(0.2),  
            LSTM(units=25, return_sequences=True),
            Dropout(0.1),
            LSTM(units=10, return_sequences=False),
            Dense(units=1)
        ])

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="mean_squared_error")

        # Train the model
        self.model.fit(X_train, y_train, epochs=75, batch_size=32)

        # dic = {
        #     'data' :self.df,
        #     'model' : model,
        #     "train_data":self.train_data,
        #     'test_data' : self.test_data
        #     }

        return 
    def prepare_test_data(self):
        self.model()
        timesteps = 60
        # Prepare test data (last `timesteps` days)
        test_df = self.df["Close_scaled"].values[-timesteps:].reshape(1, timesteps, 1)

        x_test =test_df[0,:,]
        y_test =self.df["Close_scaled"][-60:].values

        forecast = []

        for _ in range(180):  # Predict next 60 days
            pred = self.model.predict(test_input[0])  # Predict one day ahead
            forecast.append(pred[0, 0])  # Store prediction

            # Update input by appending the new prediction & removing the first value
            test_input = np.append(test_input[:, 1:, :], pred[0,0].reshape(1, 1, 1), axis=1)

        # Convert predictions back to original scale
        self.forecast_prices = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        return
    def forecast_plot(self, name):
        self.prepare_test_data()
        forecast_dates = pd.date_range(start="2025-02-01", periods=180, freq="D")

        plt.figure(figsize=(12, 6))
        # Plot actual prices (last 100 days before forecast)
        plt.plot(self.df.index[-100:], self.df.iloc[:,0].values[-100:], label="Actual Prices", color="blue")
        # Plot forecasted prices
        plt.plot(forecast_dates, self.forecast_prices, label="LSTM Forecast", color="red", linestyle="dashed")
        plt.title(f"{name} Stock Price Forecast (LSTM)")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        return plt