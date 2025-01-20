# fetching data with right naming conventions

from entsoe import EntsoeRawClient, EntsoePandasClient
import pandas as pd
import numpy as np
from dotenv import load_dotenv 
import os
import datetime as dt
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

start_time = time.time()


load_dotenv()
ENTSOE_API_KEY="562a20c4-03b0-4ee6-a692-19d534b4393a"
client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

start = pd.Timestamp('20150101', tz='UTC')
change_date = pd.Timestamp('20181001', tz='UTC')
end = pd.Timestamp(dt.datetime.now(), tz='UTC')

print(os.getcwd())
out_dir = '../final-submission/merged_data/data_collection'

country_code_old = 'DE_AT_LU'
country_code_new = 'DE_LU'

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(Exception))
def query_entsoe_data(query_func, country_code, start, end):
    try:
        df = query_func(country_code, start=start, end=end)
        
    except Exception as e:
        print(f"Error querying data: {e}")
        raise
    return df

def merge_data(query_func):
    data_old = query_entsoe_data(query_func, country_code_old, start, change_date)
    
    data_new = query_entsoe_data(query_func, country_code_new, change_date, end)

    if not isinstance(data_old, pd.DataFrame):
        data_old = data_old.to_frame()
    if not isinstance(data_new, pd.DataFrame):
        data_new = data_new.to_frame()
    
    if not data_old.empty and not data_new.empty:
        if len(data_old.columns) != len(data_new.columns):
            same_columns = list(set(data_old.columns) & set(data_new.columns))
            data_old = data_old[same_columns]
            data_new = data_new[same_columns]
        else:
            data_new.columns = data_old.columns
    df_combined = pd.concat([data_old, data_new])
    df_combined.index = df_combined.index.tz_convert('UTC')
    return df_combined

def save_df_with_timestamp(df, filename):
    df_copy = df.copy()
    df_copy.index.name = 'Date'
    df_copy.to_csv(filename)

# Day-ahead prices (EUR/MWh)
day_ahead_prices = merge_data(client.query_day_ahead_prices)
day_ahead_prices = day_ahead_prices.rename(columns={day_ahead_prices.columns[0]: 'day_ahead_prices_EURO'})
save_df_with_timestamp(day_ahead_prices, '../final-submission/merged_data/data_collection/day_ahead_prices.csv')
print('Day-ahead prices done')

# Load forecast (MWh)
load_forecast = merge_data(client.query_load_forecast)
load_forecast = load_forecast.rename(columns={load_forecast.columns[0]: 'E_load_forecast_MWh'})
save_df_with_timestamp(load_forecast, '../final-submission/merged_data/data_collection/load_forecast.csv')
print('Load forecast done')

# Generation forecast (MWh)
generation_forecast = merge_data(client.query_generation_forecast)
generation_forecast = generation_forecast.rename(columns={generation_forecast.columns[0]: 'E_generation_forecast_MWh'})
save_df_with_timestamp(generation_forecast, '../final-submission/merged_data/data_collection/generation_forecast.csv')
print('Generation forecast done')

# Wind and solar forecasts (MWh)
intraday_wind_solar_forecast = merge_data(client.query_intraday_wind_and_solar_forecast)
for col in intraday_wind_solar_forecast.columns:
    if 'Wind' in col:
        intraday_wind_solar_forecast = intraday_wind_solar_forecast.rename(columns={col: 'E_wind_forecast_MWh'})
    elif 'Solar' in col:
        intraday_wind_solar_forecast = intraday_wind_solar_forecast.rename(columns={col: 'E_solar_forecast_MWh'})
save_df_with_timestamp(intraday_wind_solar_forecast, '../final-submission/merged_data/data_collection/intraday_wind_solar_forecast.csv')
print('Intraday wind and solar forecast done')

# Day ahead wind and solar forecast (MWh)
day_ahead_wind_solar_forecast = merge_data(client.query_wind_and_solar_forecast)
for col in day_ahead_wind_solar_forecast.columns:
    if 'Wind' in col:
        day_ahead_wind_solar_forecast = day_ahead_wind_solar_forecast.rename(columns={col: 'E_wind_forecast_MWh'})
    elif 'Solar' in col:
        day_ahead_wind_solar_forecast = day_ahead_wind_solar_forecast.rename(columns={col: 'E_solar_forecast_MWh'})
save_df_with_timestamp(day_ahead_wind_solar_forecast, '../final-submission/merged_data/data_collection/day_ahead_wind_solar_forecast.csv')
print('Day ahead wind and solar forecast done')

# Physical crossborder flows (MWh)
physical_crossborder_flows = merge_data(lambda cc, start, end: client.query_physical_crossborder_allborders(start=start, end=end, country_code=cc, export=True))
for col in physical_crossborder_flows.columns:
    physical_crossborder_flows = physical_crossborder_flows.rename(columns={col: f'E_crossborder_{col}_actual_MWh'})
save_df_with_timestamp(physical_crossborder_flows, '../final-submission/merged_data/data_collection/physical_crossborder_flows.csv')
print('Physical crossborder flows done')

df4 = pd.read_csv('../final-submission/merged_data/data_collection/day_ahead_prices.csv')
df5 = pd.read_csv('../final-submission/merged_data/data_collection/load_forecast.csv')
df6 = pd.read_csv('../final-submission/merged_data/data_collection/generation_forecast.csv')
df7 = pd.read_csv('../final-submission/merged_data/data_collection/intraday_wind_solar_forecast.csv')
df8 = pd.read_csv('../final-submission/merged_data/data_collection/day_ahead_wind_solar_forecast.csv')
df9 = pd.read_csv('../final-submission/merged_data/data_collection/physical_crossborder_flows.csv')


merged_df2 = pd.merge(df5, df4, on='Date', how='outer')
merged_df2 = pd.merge(merged_df2, df6, on='Date', how='outer')
merged_df2 = pd.merge(merged_df2, df7, on='Date', how='outer')
merged_df2 = pd.merge(merged_df2, df8, on='Date', how='outer')
merged_df2 = pd.merge(merged_df2, df9, on='Date', how='outer')

merged_df2.to_csv('../final-submission/merged_data/data_collection/merged_data2.csv', index=False)

df = pd.read_csv('../final-submission/merged_data/data_collection/merged_data2.csv')

df['Date'] = pd.to_datetime(df['Date'])
df_filtered = df[df['Date'].dt.minute == 0]
df_filtered['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_filtered.to_csv('../final-submission/merged_data/data_collection/merged_data3.csv', index=False)

end_time = time.time()
verstrichene_zeit = end_time - start_time
print(f'Ausführungszeit komplett: {verstrichene_zeit} Sekunden')