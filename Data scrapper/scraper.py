import pandas as pd
import requests
import time
from datetime import datetime

def fetch_binance_minute_data(symbol='BTCUSDT', start_str='1 Apr, 2021'):
    url = 'https://api.binance.com/api/v3/klines'
    
    def date_to_millis(date_str):
        return int(time.mktime(time.strptime(date_str, '%d %b, %Y'))) * 1000

    start_ts = date_to_millis(start_str)
    end_ts = int(time.time() * 1000)  # Now in milliseconds

    all_data = []
    limit = 1000  # Max per API call
    total_rows = 0

    print(f"Starting scrape from {start_str} to {datetime.fromtimestamp(end_ts / 1000).strftime('%Y-%m-%d')}")

    while start_ts < end_ts:
        params = {
            'symbol': symbol,
            'interval': '1m',
            'startTime': start_ts,
            'limit': limit
        }

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"⚠️ API error {response.status_code}. Retrying in 5s...")
            time.sleep(5)
            continue

        data = response.json()
        if not data:
            print("Reached end of available data.")
            break

        for candle in data:
            ts, open_, high, low, close, volume, *_ = candle
            all_data.append([
                datetime.fromtimestamp(ts / 1000),
                float(open_), float(high), float(low), float(close), float(volume)
            ])

        total_rows += len(data)
        last_time = datetime.fromtimestamp(data[-1][0] / 1000)
        print(f"✅ Retrieved {len(data)} rows | Total: {total_rows} | Latest: {last_time.strftime('%Y-%m-%d %H:%M')}")

        # Move to next window (1 min after last candle)
        start_ts = data[-1][0] + 60_000
        time.sleep(0.25)  # Respect Binance rate limits

    df = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    return df

df_binance = fetch_binance_minute_data(start_str='1 May, 2025')
df_binance.to_csv("btc_last_45_days_minutely.csv", index=False)
print("Full minutely data saved to CSV.")