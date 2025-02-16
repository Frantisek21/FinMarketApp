import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # ChatGPT
from src.base import BaseAnalysis
import pandas as pd
from datetime import datetime, timedelta


class DataFetcher(BaseAnalysis):
    def fetch_aggregates(
            self, ticker,
            start_date,
            end_date,
            timespan="minute",
            multiplier=1,
            limit=50000):
        """
        Fetch aggregate data for a given ticker.
        """
        try:
            aggs = []
            for a in self.client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
                limit=limit
            ):
                aggs.append(a)
            if not aggs:
                print(f"No data found for {ticker}.")
            else:
                print(f"Sample record for {ticker}:", aggs[0])
            return aggs
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return []

    def process_data(self, aggs):
        """
        Convert aggregates into a DataFrame with Timestamp and Close prices.
        """
        if not aggs:
            return pd.DataFrame()

        closing_prices = [agg.close for agg in aggs]
        timestamps = [agg.timestamp for agg in aggs]

        # Convert timestamps to datetime
        timestamps = pd.to_datetime(timestamps, unit='ms')

        # Create a DataFrame
        df = pd.DataFrame({
            "Timestamp": timestamps,
            "Close": closing_prices
        })
        df.set_index("Timestamp", inplace=True)
        return df

    def get_data(self, tickers, days=180):
        """
        Fetch and process data for multiple tickers.
        """
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(
            days=days)).strftime("%Y-%m-%d")

        dataframes = {}
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            aggs = self.fetch_aggregates(ticker, start_date, end_date)
            df = self.process_data(aggs)
            if not df.empty:
                dataframes[ticker] = df
            else:
                print(f"Data for {ticker} is not available.")

        return dataframes
