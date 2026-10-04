import os
import fastf1
import pandas as pd

# 1. Create a cache folder so the F1 API works smoothly
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')

print("Downloading F1 telemetry and timing data... This will take a minute or two.")

# 2. Fetch the 2023 Bahrain Grand Prix Race session
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load(telemetry=True, weather=True, messages=True)

# 3. Copy the raw laps data into a table
laps_df = session.laps.copy()

# 4. Convert time columns to strings so Excel doesn't throw an error
for col in laps_df.columns:
    if laps_df[col].dtype == 'timedelta64[ns]':
        laps_df[col] = laps_df[col].astype(str)

# 5. Save the table directly to an Excel file
laps_df.to_excel('F1_Bahrain_2023_Dataset.xlsx', index=False)

print("Success! Your Excel file is ready in the folder.")