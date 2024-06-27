import yfinance as yf
import pandas as pd
from datetime import datetime

def download_latest_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print(f"No data available from {start_date}. Attempting to download data from the earliest available date.")
        data = yf.download(ticker, period='max')
        if data.empty:
            raise ValueError(f"No data available for the ticker {ticker}. It might be delisted or invalid.")
        print(f"Data downloaded from the earliest available date.")
    data.to_csv(f'{ticker}_latest_data.csv')
    print(f"Data for {ticker} downloaded successfully.")

# Get user input for stock ticker and start date
ticker = input("Enter the stock ticker symbol: ")
start_date = input("Enter the start date (YYYY-MM-DD): ")

# Update the end date to the current date
end_date = datetime.now().strftime('%Y-%m-%d')

try:
    download_latest_stock_data(ticker, start_date, end_date)
except ValueError as e:
    print(e)
