# fetching data with right naming conventions

from entsoe import EntsoeRawClient, EntsoePandasClient
import pandas as pd
import numpy as np
from dotenv import load_dotenv 
import os
import datetime as dt
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from functools import partial
from entsoe import EntsoePandasClient
import pandas as pd
import os
import datetime as dt
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

start_time_entsoe = time.time()
load_dotenv()
ENTSOE_API_KEY="562a20c4-03b0-4ee6-a692-19d534b4393a"
client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

start = pd.Timestamp('20150101', tz='UTC')
change_date = pd.Timestamp('20181001', tz='UTC')
end = pd.Timestamp(dt.datetime.now(), tz='UTC')

print(os.getcwd())
out_dir = '../final-submission/merged_data/data_collection'
os.makedirs(out_dir, exist_ok=True)

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

#Fetch and Save Data for All Tasks
def start_fetch_save_data(tasks):
    max_workers = min(os.cpu_count(), len(tasks))
    print(f"Starting data fetch for {len(tasks)} tasks with {max_workers} threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(fetch_save_data, task['query_func'], task['save_path'],task.get('rename_columns'), task.get('transform_func')): task for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                future.result()
                print(f"Task completed for {task['save_path'].split('/')[-1]}.")
            except Exception as e:
                print(f"Error in task for {task['save_path'].split('/')[-1]}: {e}")

#Helper function for fetching and saving
def fetch_save_data(query_func, save_path, rename_columns=None, transform_func=None):
    data = merge_data(query_func)

    # Filter: Nur Daten mit vollen Stunden (Minute == 0)
    data['Date'] = data.index
    data = data[data['Date'].dt.minute == 0]

    if rename_columns:
        data = data.rename(columns=rename_columns)
    if transform_func:
        data = transform_func(data)

    # Formatierung der Datetime-Spalte
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    save_df_with_timestamp(data, save_path)
    print(f'{save_path.split("/")[-1]} done')

#Transform solar and wind
def rename_wind_solar_columns(df):
    for col in df.columns:
        if 'Wind' in col:
            df = df.rename(columns={col: 'E_wind_forecast_MWh'})
        elif 'Solar' in col:
            df = df.rename(columns={col: 'E_solar_forecast_MWh'})
    return df

#Tasks for parallel processing
tasks = [
    {
        'query_func': client.query_day_ahead_prices,
        'save_path': f'{out_dir}/day_ahead_prices.csv',
        'rename_columns': {None: 'day_ahead_prices_EURO'}
    },
    {
        'query_func': client.query_load_forecast,
        'save_path': f'{out_dir}/load_forecast.csv',
        'rename_columns': {None: 'E_load_forecast_MWh'}
    },
    {
        'query_func': client.query_generation_forecast,
        'save_path': f'{out_dir}/generation_forecast.csv',
        'rename_columns': {None: 'E_generation_forecast_MWh'}
    },
    {
        'query_func': client.query_intraday_wind_and_solar_forecast,
        'save_path': f'{out_dir}/intraday_wind_solar_forecast.csv',
        'transform_func': rename_wind_solar_columns
    },
    {
        'query_func': client.query_wind_and_solar_forecast,
        'save_path': f'{out_dir}/day_ahead_wind_solar_forecast.csv',
        'transform_func': rename_wind_solar_columns
    },
    {
        'query_func': partial(client.query_physical_crossborder_allborders, export=True),
        'save_path': f'{out_dir}/physical_crossborder_flows.csv',
        'transform_func': lambda df: df.rename(columns={col: f'E_crossborder_{col}_actual_MWh' for col in df.columns})
    }
]

start_fetch_save_data(tasks)

end_time_entsoe = time.time()
verstrichene_zeit_entsoe = end_time_entsoe - start_time_entsoe
print(f'Ausführungszeit nach fetch_save: {verstrichene_zeit_entsoe} Sekunden')

#Combining all datasets
df_list = [pd.read_csv(task['save_path']) for task in tasks]
merged_df = df_list[0]
for df in df_list[1:]:
    merged_df = pd.merge(merged_df, df, on='Date', how='outer')

end_time_entsoe = time.time()
verstrichene_zeit_entsoe = end_time_entsoe - start_time_entsoe
print(f'Ausführungszeit nach Merge vor save als csv: {verstrichene_zeit_entsoe} Sekunden')

merged_df.to_csv(f'{out_dir}/merged_data_multi_2.csv', index=False)

end_time_entsoe = time.time()
verstrichene_zeit_entsoe = end_time_entsoe - start_time_entsoe
print(f'Ausführungszeit nach Merge_to_csv: {verstrichene_zeit_entsoe} Sekunden')

end_time_entsoe = time.time()
verstrichene_zeit_entsoe = end_time_entsoe - start_time_entsoe
print(f'Ausführungszeit komplett: {verstrichene_zeit_entsoe} Sekunden')