from moexalgo import Ticker
import pandas as pd
from datetime import datetime, timedelta

def load_stock_data(ticker_with_suffix: str) -> pd.DataFrame | None:
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        ticker_symbol = ticker_with_suffix.replace(".ME", "")
        ticker = Ticker(ticker_symbol)

        df = ticker.candles(start=two_years_ago, end=today, period='1d')
        return df
    except Exception as e:
        print(f"Ошибка загрузки данных по {ticker_with_suffix}: {e}")
        return None