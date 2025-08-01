import pandas as pd

# Load btcusd_1-min_data.csv (Kaggle file with Unix timestamps)
df_old = pd.read_csv("btcusd_1-min_data.csv")
df_old['Timestamp'] = pd.to_datetime(df_old['Timestamp'], unit='s')

# Load Binance-scraped dataset
df_new = pd.read_csv("btc_scraped_minutely_2021_to_2025.csv")
df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'])

# Standardize column order and structure
df_old = df_old[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
df_new = df_new[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Combine and sort
df_combined = pd.concat([df_old, df_new], ignore_index=True)
df_combined.drop_duplicates(subset='Timestamp', inplace=True)
df_combined.sort_values('Timestamp', inplace=True)
df_combined.reset_index(drop=True, inplace=True)

# Set timestamp as index
df_combined.set_index('Timestamp', inplace=True)

# Create full 1-minute index and reindex
all_minutes = pd.date_range(start=df_combined.index.min(), end=df_combined.index.max(), freq='T')
df_combined = df_combined.reindex(all_minutes)

# Forward-fill missing values
df_combined[['Open', 'High', 'Low', 'Close', 'Volume']] = df_combined[['Open', 'High', 'Low', 'Close', 'Volume']].ffill()

# Final cleanup
df_combined.dropna(inplace=True)

# Save final base version
df_combined.to_csv("btc_base_cleaned_2012_to_2025.csv")
print("✅ Merged and cleaned dataset saved as btc_base_cleaned_2012_to_2025.csv")

print("Old rows:", len(df_old))
print("New rows:", len(df_new))
print("Combined before drop:", len(df_old) + len(df_new))
print("After merge & dedup:", len(df_combined))
print("Dropped rows:", (len(df_old) + len(df_new)) - len(df_combined))