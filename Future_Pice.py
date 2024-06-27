import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def predict_future_prices(model, scaled_data, scaler, time_step, days_to_predict):
    temp_input = list(scaled_data[-time_step:])
    temp_input = temp_input[:time_step]
    
    future_predictions = []
    for _ in range(days_to_predict):
        if len(temp_input) > time_step:
            temp_input = temp_input[1:]
        input_data = np.array(temp_input).reshape(1, -1, 1)
        
        prediction = model.predict(input_data, verbose=0)
        temp_input.append(prediction[0])
        
        future_predictions.append(prediction[0])
    
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

# Get user input for stock ticker and number of days to predict
ticker = input("Enter the stock ticker symbol: ")
days_to_predict = int(input("Enter the number of days to predict: "))

# Load your historical stock price data
data = pd.read_csv(f'{ticker}_latest_data.csv', index_col='Date', parse_dates=True)

# Filter data to include only the last 6 months
last_6_months = datetime.now() - timedelta(days=6*30)  # Approximately 6 months
data = data[data.index >= last_6_months]

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']].values)

# Load the trained model
model = load_model('stock_lstm_model.h5')

# Predict future prices
future_predictions = predict_future_prices(model, scaled_data, scaler, 100, days_to_predict)

# Prepare dates for the future predictions
last_date = data.index[-1]
future_dates = pd.date_range(last_date, periods=days_to_predict + 1).tolist()[1:]

# Create a DataFrame for plotting
future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Predicted Close'])

# Plot the results with enhancements
plt.figure(figsize=(16,8))
plt.title(f'Future Stock Price Predictions for {ticker}', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price USD ($)', fontsize=16)
plt.grid(True)

# Plot only the last 6 months of historical data
plt.plot(data['Close'], label='Historical Close Price', color='blue')

# Plot the future predictions
plt.plot(future_df['Predicted Close'], label='Predicted Close Price', color='red', linestyle='--')

# Highlight the next few days of predicted prices
highlight_days = min(days_to_predict, 5)
plt.scatter(future_df.index[:highlight_days], future_df['Predicted Close'][:highlight_days], color='orange', s=100, label='Next Few Days Predictions')

# Annotate the next few days of predictions with better readability
for i in range(highlight_days):
    plt.annotate(f'{future_df["Predicted Close"].iloc[i]:.2f}', 
                 (future_df.index[i], future_df["Predicted Close"].iloc[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', 
                 fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

plt.legend(loc='best', fontsize=14)
plt.tight_layout()
plt.show()
