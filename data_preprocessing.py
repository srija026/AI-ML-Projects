import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return data, scaled_data, scaler

# Assume the ticker is stored in a file to maintain consistency
ticker = input("Enter the stock ticker symbol (same as used in data collection): ")
data, scaled_data, scaler = preprocess_data(f'{ticker}_latest_data.csv')
