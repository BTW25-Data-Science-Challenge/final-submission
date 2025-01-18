import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from functools import partial
from io import StringIO
from pathlib import Path
import csv
import datetime
import io
import json
import os
import re
import sys
import time
import zipfile
import holidays

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from entsoe import EntsoePandasClient, EntsoeRawClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential




##Stockmarket
url_oil = 'https://www.finanzen.net/rohstoffe/oelpreis'
url_gas = 'https://www.finanzen.net/rohstoffe/erdgas-preis-natural-gas'
url_coal = 'https://www.finanzen.net/rohstoffe/kohlepreis'
url_uran = 'https://www.finanzen.net/rohstoffe/uranpreis'



##Entsoe
start_time = time.time()
load_dotenv()
ENTSOE_API_KEY="a5cd0e33-0ad4-4203-b890-b4dfe04a3005"
client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

start = pd.Timestamp('20150101', tz='UTC')
change_date = pd.Timestamp('20181001', tz='UTC')
end = pd.Timestamp(dt.now(), tz='UTC')

print(os.getcwd())
out_dir = '../final-submission/merged_data/data_collection'
os.makedirs(out_dir, exist_ok=True)

country_code_old = 'DE_AT_LU'
country_code_new = 'DE_LU'


##Covid Lockdown Data

FILE_URL = 'https://pada.psycharchives.org/bitstream/9ff033a9-4084-4d0e-87eb-aa963a1324a5'
covid_df = pd.read_csv(FILE_URL, sep=",", header=[0])
print(covid_df.head().iloc[:,:5])

# dict with influence of measure (see readme)
measure_influence = {
    'leavehome': 1,
    'dist': 0,
    'msk': 1,
    'shppng': 2,
    'hcut': 2,
    'ess_shps': 2,
    'zoo': 0,
    'demo': 0,
    'school': 1,
    'church': 0,
    'onefriend': 0,
    'morefriends': 0,
    'plygrnd': 0,
    'daycare': 2,
    'trvl': 1,
    'gastr': 2
}
# dict with state relative population of country
state_percentages = {
    'Baden-Wuerttemberg': 0.133924061,
    'Bayern': 0.158676851,
    'Berlin': 0.044670274,
    'Brandenburg': 0.030491172,
    'Bremen': 0.008169464,
    'Hamburg': 0.022560236,
    'Hessen': 0.075833,
    'Mecklenburg-Vorpommern': 0.019245033,
    'Niedersachsen': 0.096398323,
    'Nordrhein-Westfalen': 0.214840756,
    'Rheinland-Pfalz': 0.049301337,
    'Saarland': 0.011744796,
    'Sachsen': 0.048299274,
    'Sachsen-Anhalt': 0.025752514,
    'Schleswig-Holstein': 0.035026746,
    'Thueringen': 0.025066162
}

## Smard

#-------------translation for Balancing:------------------
balancing_id={
    #automatic frequency, tag=af
    "automatic_frequency":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Volume activated (+) [MWh] Calculated resolutions":"af_E_Volume_Activated_Plus_MWh",
        "Volume activated (-) [MWh] Calculated resolutions":"af_E_Volume_Activated_Minus_MWh",
        "Activation price (+) [€/MWh] Calculated resolutions":"af_Activation_Price_Plus_EUR_MWh",
        "Activation price (-) [€/MWh] Calculated resolutions":"af_Activation_Price_Minus_EUR_MWh",
        "Volume procured (+) [MW] Calculated resolutions":"af_E_Volume_Procured_Plus_MW",
        "Volume procured (-) [MW] Calculated resolutions":"af_E_Volume_Procured_Minus_MW",
        "Procurement price (+) [€/MW] Calculated resolutions":"af_Procurement_Price_Plus_EUR_MW",
        "Procurement price (-) [€/MW] Calculated resolutions":"af_Procurement_Price_Minus_EUR_MW",
    },
    #tag=mf
    "manual_frequency":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Volume activated (+) [MWh] Calculated resolutions":"mf_E_Volume_Activated_Plus_MWh",
        "Volume activated (-) [MWh] Calculated resolutions":"mf_E_Volume_Activated_Minus_MWh",
        "Activation price (+) [€/MWh] Calculated resolutions":"mf_Activation_Price_Plus_EUR_MWh",
        "Activation price (-) [€/MWh] Calculated resolutions":"mf_Activation_Price_Minus_EUR_MWh",
        "Volume procured (+) [MW] Calculated resolutions":"mf_E_Volume_Procured_Plus_MW",
        "Volume procured (-) [MW] Calculated resolutions":"mf_E_Volume_Procured_Minus_MW",
        "Procurement price (+) [€/MW] Calculated resolutions":"mf_Procurement_Price_Plus_EUR_MW",
        "Procurement price (-) [€/MW] Calculated resolutions":"mf_Procurement_Price_Minus_EUR_MW",
    },
     #balancing energy
    "balancing_energy":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Volume (+) [MWh] Calculated resolutions":"E_Volume_Calculated_Plus_MWh",
        "Volume (-) [MWh] Calculated resolutions":"E_Volume_Calculated_Minus_MWh",
        "Price [€/MWh] Calculated resolutions":"Price_Calculated_EUR_MWh",
        "Net income [€] Calculated resolutions":"Net_Income_EUR",
    },
    #costs
    "costs":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Balancing services [€] Calculated resolutions":"Balancing_Services_Calculated_EUR",
        "Network security [€] Calculated resolutions":"Network_Security_Calculated_EUR",
        "Countertrading [€] Calculated resolutions":"Countertrading_Calculated_EUR",
    },
    #frequency_containment_reserve
    "frequency_containment":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Volume procured [MW] Calculated resolutions":"E_Volume_Procured_Calculated_MW",
        "Procurement price [€/MW] Calculated resolutions":"Price_Procument_Calculated_EUR/MW"
    },
    "imported_balancing_services":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Austria [MWh] Calculated resolutions":"import_E_Austria_Calculated_MWh",
    },
    "exported_balancing_services":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Austria [MWh] Calculated resolutions":"export_E_Austria_Calculated_MWh",
    }         
}    

#actual consumption tag=actual
electricity_consumption_id={
    "actual":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Total (grid load) [MWh] Calculated resolutions":"actual_E_Total_Gridload_MWh",
        "Residual load [MWh] Calculated resolutions":"actual_E_Residual_Load_MWh",
        "Hydro pumped storage [MWh] Calculated resolutions":"actual_E_Hydro_Pumped_Storage_MWh",
    },
    #forecasted consumption tag=forecast
    "forecast":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Total (grid load) [MWh] Calculated resolutions":"forecast_E_Total_Gridload_MWh",
        "Residual load [MWh] Calculated resolutions":"forecast_actual_E_Residual_Load_MWh"
    }
}

electricity_generation_id={
    #actual generation
    "actual":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Biomass [MWh] Calculated resolutions":"actual_generation_E_Biomass_MWh",
        "Hydropower [MWh] Calculated resolutions":"actual_generation_E_Hydropower_MWh",
        "Wind offshore [MWh] Calculated resolutions":"actual_generation_E_Windoffshore_MWh",
        "Wind onshore [MWh] Calculated resolutions":"actual_generation_E_Windonshore_MWh",
        "Photovoltaics [MWh] Calculated resolutions":"actual_generation_E_Photovoltaics_MWh",
        "Other renewable [MWh] Calculated resolutions":"actual_generation_E_OtherRenewable_MWh",
        "Nuclear [MWh] Calculated resolutions":"actual_generation_E_Nuclear_MWh",
        "Lignite [MWh] Calculated resolutions":"actual_generation_E_Lignite_MWh",
        "Hard coal [MWh] Calculated resolutions":"actual_generation_E_HardCoal_MWh",
        "Fossil gas [MWh] Calculated resolutions":"actual_generation_E_FossilGas_MWh",
        "Hydro pumped storage [MWh] Calculated resolutions":"actual_generation_E_HydroPumpedStorage_MWh",
        "Other conventional [MWh] Calculated resolutions":"actual_generation_E_OtherConventional_MWh"
    },
    
    #forecastet generation day ahead
    "forecast":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Biomass [MWh] Calculated resolutions":"forecast_generation_E_Biomass_MWh",
        "Hydropower [MWh] Calculated resolutions":"forecast_generation_E_Hydropower_MWh",
        "Wind offshore [MWh] Calculated resolutions":"forecast_generation_E_Windoffshore_MWh",
        "Wind onshore [MWh] Calculated resolutions":"forecast_generation_E_Windonshore_MWh",
        "Photovoltaics [MWh] Calculated resolutions":"forecast_generation_E_Photovoltaics_MWh",
        "Other renewable [MWh] Calculated resolutions":"forecast_generation_E_OtherRenewable_MWh",
        "Nuclear [MWh] Calculated resolutions":"forecast_generation_E_Nuclear_MWh",
        "Lignite [MWh] Calculated resolutions":"forecast_generation_E_Lignite_MWh",
        "Hard coal [MWh] Calculated resolutions":"forecast_generation_E_HardCoal_MWh",
        "Fossil gas [MWh] Calculated resolutions":"forecast_generation_E_FossilGas_MWh",
        "Hydro pumped storage [MWh] Calculated resolutions":"forecast_generation_E_HydroPumpedStorage_MWh",
        "Other [MWh] Calculated resolutions":"forecast_generation_E_Other_MWh",
        "Total [MWh] Original resolutions":"forecast_generation_E_Total_MWh",
        "Photovoltaics and wind [MWh] Calculated resolutions":"forecast_generation_E_PhotovoltaicsAndWind_MWh",
        "Other [MWh] Original resolutions":"forecast_generation_E_Original_MWh"
    },

    #installed generation capacity
    #key=instGenCapacity
    "installed_generation_capacity":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Biomass [MW] Calculated resolutions":"instGenCapacity_E_Biomass_MW",
        "Hydropower [MW] Calculated resolutions":"instGenCapacity_E_Hydropower_MW",
        "Wind offshore [MW] Calculated resolutions":"instGenCapacity_E_Windoffshore_MW",
        "Wind onshore [MW] Calculated resolutions":"instGenCapacity_E_Windonshore_MW",
        "Photovoltaics [MW] Calculated resolutions":"instGenCapacity_E_Photovoltaics_MW",
        "Other renewable [MW] Calculated resolutions":"instGenCapacity_E_OtherRenewable_MW",
        "Nuclear [MW] Calculated resolutions":"instGenCapacity_E_Nuclear_MW",
        "Lignite [MW] Calculated resolutions":"instGenCapacity_E_Lignite_MW",
        "Hard coal [MW] Calculated resolutions":"instGenCapacity_E_HardCoal_MW",
        "Fossil gas [MW] Calculated resolutions":"instGenCapacity_E_FossilGas_MW",
        "Hydro pumped storage [MW] Calculated resolutions":"instGenCapacity_E_HydroPumpedStorage_MW",
        "Other conventional [MW] Calculated resolutions":"instGenCapacity_E_OtherConventional_MW"
    }
}

market_id={
    #key=dayAhead
    "day_ahead_prices":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Germany/Luxembourg [€/MWh] Original resolutions":"dayAhead_Price_GermanyAndLuxembourg_EUR_MWh",
        "∅ DE/LU neighbours [€/MWh] Original resolutions":"dayAhead_Price_GermanyAndLuxembourgAverage_EUR_MWh",
        "Belgium [€/MWh] Original resolutions":"dayAhead_Price_Belgium_EUR_MWh",
        "Denmark 1 [€/MWh] Original resolutions":"dayAhead_Price_Denmark1_EUR_MWh",
        "Denmark 2 [€/MWh] Original resolutions":"dayAhead_Price_Denmark2_EUR_MWh",
        "France [€/MWh] Original resolutions":"dayAhead_Price_France_EUR_MWh",
        "Netherlands [€/MWh] Original resolutions":"dayAhead_Price_Netherlands_EUR_MWh",
        "Norway 2 [€/MWh] Original resolutions":"dayAhead_Price_Norway2_EUR_MWh",
        "Austria [€/MWh] Original resolutions":"dayAhead_Price_Austria_EUR_MWh",
        "Poland [€/MWh] Original resolutions":"dayAhead_Price_Poland_EUR_MWh",
        "Sweden 4 [€/MWh] Original resolutions":"dayAhead_Price_Sweden4_EUR_MWh",
        "Switzerland [€/MWh] Original resolutions":"dayAhead_Price_Switzerland_EUR_MWh",
        "Czech Republic [€/MWh] Original resolutions":"dayAhead_Price_CzechRepublic_EUR_MWh",
        "DE/AT/LU [€/MWh] Original resolutions":"dayAhead_Price_DE/AT/LU_EUR_MWh",
        "Northern Italy [€/MWh] Original resolutions":"dayAhead_Price_NothernItaly_EUR_MWh",
        "Slovenia [€/MWh] Original resolutions":"dayAhead_Price_Slovenia_EUR_MWh",
        "Hungary [€/MWh] Original resolutions":"dayAhead_Price_Hungary_EUR_MWh"
    },
    
    "cross_border_physical":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Net export [MWh] Calculated resolutions":"E_NetExport_crossBorderPhysical_MWh",
        "Netherlands (export) [MWh] Calculated resolutions":"E_NetherlandExport_corssBorderPhysical_MWh",
        "Netherlands (import) [MWh] Calculated resolutions":"E_NetherlandImport_corssBorderPhysical_MW",
        "Switzerland (export) [MWh] Calculated resolutions":"E_SwitzerlandExport_corssBorderPhysical_MWh",
        "Switzerland (import) [MWh] Calculated resolutions":"E_SwitzerlandImport_corssBorderPhysical_MWh",
        "Denmark (export) [MWh] Calculated resolutions":"E_DenmarkExport_corssBorderPhysical_MWh",
        "Denmark (import) [MWh] Calculated resolutions":"E_Denmark_Import_corssBorderPhysical_MWh",
        "Czech Republic (export) [MWh] Calculated resolutions":"E_CzechrepublicExport_corssBorderPhysical_MWh",
        "Czech Republic (import) [MWh] Calculated resolutions":"E_CzechrepublicImport_corssBorderPhysical_MWh",
        "Luxembourg (export) [MWh] Calculated resolutions":"E_LuxembourgExport_corssBorderPhysical_MWh",
        "Luxembourg (import) [MWh] Calculated resolutions":"E_LuxembourgImport_corssBorderPhysical_MWh",
        "Sweden (export) [MWh] Calculated resolutions":"E_SwedenExport_corssBorderPhysical_MWh",
        "Sweden (import) [MWh] Calculated resolutions":"E_SwedenImportv_corssBorderPhysical_MWh",
        "Austria (export) [MWh] Calculated resolutions":"E_AustriaExport_corssBorderPhysical_MWh",
        "Austria (import) [MWh] Calculated resolutions":"E_AustriaImport_corssBorderPhysical_MWh",
        "France (export) [MWh] Calculated resolutions":"E_FranceExport_corssBorderPhysical_MWh",        
        "France (import) [MWh] Calculated resolutions":"E_FranceImport_corssBorderPhysical_MWh",
        "Poland (export) [MWh] Calculated resolutions":"E_PolandExport_corssBorderPhysical_MWh",
        "Poland (import) [MWh] Calculated resolutions":"E_PolandImport_corssBorderPhysical_MWh",
        "Norway (export) [MWh] Calculated resolutions":"E_NorwayExport_corssBorderPhysical_MWh",
        "Norway (import) [MWh] Calculated resolutions":"E_NorwayImport_corssBorderPhysical_MWh",
        "Belgium (export) [MWh] Calculated resolutions":"E_BelgiumExport_corssBorderPhysical_MWh",
        "Belgium (import) [MWh] Calculated resolutions":"E_BelgiumImport_corssBorderPhysical_MWh",
    },
    "scheudled_commercial_exchanges":{
        "Start date":"Start_Date",
        "End date":"End_Date",
        "Net export [MWh] Calculated resolutions":"E_NetExport_MWh",
        "Netherlands (export) [MWh] Calculated resolutions":"E_NetherlandExport_MWh",
        "Netherlands (import) [MWh] Calculated resolutions":"E_NetherlandImport_MW",
        "Switzerland (export) [MWh] Calculated resolutions":"E_SwitzerlandExport_MWh",
        "Switzerland (import) [MWh] Calculated resolutions":"E_SwitzerlandImport_MWh",
        "Denmark (export) [MWh] Calculated resolutions":"E_DenmarkExport_MWh",
        "Denmark (import) [MWh] Calculated resolutions":"E_Denmark_Import_MWh",
        "Czech Republic (export) [MWh] Calculated resolutions":"E_CzechrepublicExport_MWh",
        "Czech Republic (import) [MWh] Calculated resolutions":"E_CzechrepublicImport_MWh",
        "Luxembourg (export) [MWh] Calculated resolutions":"E_LuxembourgExport_MWh",
        "Luxembourg (import) [MWh] Calculated resolutions":"E_LuxembourgImport_MWh",
        "Sweden (export) [MWh] Calculated resolutions":"E_SwedenExport_MWh",
        "Sweden (import) [MWh] Calculated resolutions":"E_SwedenImport_MWh",
        "Austria (export) [MWh] Calculated resolutions":"E_AustriaExport_MWh",
        "Austria (import) [MWh] Calculated resolutions":"E_AustriaImport_MWh",
        "France (export) [MWh] Calculated resolutions":"E_FranceExport_MWh",        
        "France (import) [MWh] Calculated resolutions":"E_FranceImport_MWh",
        "Poland (export) [MWh] Calculated resolutions":"E_PolandExport_MWh",
        "Poland (import) [MWh] Calculated resolutions":"E_PolandImport_MWh",
        "Norway (export) [MWh] Calculated resolutions":"E_NorwayExport_MWh",
        "Norway (import) [MWh] Calculated resolutions":"E_NorwayImport_MWh",
        "Belgium (export) [MWh] Calculated resolutions":"E_BelgiumExport_MWh",
        "Belgium (import) [MWh] Calculated resolutions":"E_BelgiumImport_MWh",
    }
}

##weather
#Define stations
combine_historicforecast_bool =False
station_ids_r = [ "01262", "01975", "02667"]
station_ids_f = [ "10870", "10147", "10513"]
station_place = [ "Muenchen", "Hamburg", "KoelnBonn" ]
#folderstructure
output_folder = "./merged_data/scripts/weather/"
station_folder = "./merged_data/scripts/weather/stations"
computing_folder = "./merged_data/scripts/weather/computing_folder"
stations_combined = "./merged_data/scripts/weather/stations_combined"
data_collection_folder="../final-submission/merged_data/data_collection"
forecas_folder="../final-submission/merged_data/forecast"
#Basis-URL for dwd-data
base_url_review = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"
url_forecast = "https://dwd.api.proxy.bund.dev/v30/stationOverviewExtended"
#collums to remove   
columns_remove_clouds = ["STATIONS_ID","eor", "QN_8","V_N_I"]
columns_remove_pressure = ["STATIONS_ID","eor", "QN_8"]
columns_remove_sun = ["STATIONS_ID","eor", "QN_7"]
columns_remove_temp = ["STATIONS_ID","QN_9", "eor"]
columns_remove_wind = ["STATIONS_ID","eor", "QN_3"]
columns_remove_precipitation = ["STATIONS_ID","eor", "QN_8", "WRTR", "RS_IND"]

columns_remove_forecast = ['isDay','dewPoint2m']
#URL-endings for historical data
data_types = {
    "temperature_historical": "air_temperature/historical/",
    "temperature_recent": "air_temperature/recent/",
    "cloudiness_historical": "cloudiness/historical/",
    "cloudiness_recent": "cloudiness/recent/",
    "pressure_historical": "pressure/historical/",
    "pressure_recent": "pressure/recent/",
    "sun_historical": "sun/historical/",
    "sun_recent": "sun/recent/",
    "wind_historical": "wind/historical/",
    "wind_recent": "wind/recent/",
    "precipitation_recent": "precipitation/recent/",
    "precipitation_historical": "precipitation/historical/",
}
#header for API
headers_weather = {
    "accept": "application/json"
}




##Stockmarket
##Stockmarket

def directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

# gets data from finanzen.net with the given url, the filename and resource have to be put in, it updates an already existing file, to not use selenium
def get_Data(url, filename, resource, before):

    #ellaborate header needed, otherwise finanzen.net will give an access denied error
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.finanzen.net',
        'Connection': 'keep-alive',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }

    session = requests.Session()
    response = session.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    #used website inspection to find the right table from the website
    table = soup.find('table', class_='table table--content-right')

    if table:
        headers = [th.get_text(strip=True) for th in table.find_all('th')]

        #we only need schluss and date, the other columns are irrelevant
        datum_index = headers.index('Datum')
        schlusskurs_index = headers.index('Schlusskurs')
        rows = table.find_all('tr')[1:] 
        extracted_data = []

        for row in rows:
            columns = row.find_all('td')
            if len(columns) > max(datum_index, schlusskurs_index): 
                datum = columns[datum_index].get_text(strip=True)
                schlusskurs = columns[schlusskurs_index].get_text(strip=True)
                schlusskurs = schlusskurs.replace(',', '.')
                extracted_data.append({'Date': datum, resource: schlusskurs})

        df = pd.DataFrame(extracted_data)

    else:
        print("Table not found")

    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')

    old = pd.read_csv(before)
    old['Date'] = pd.to_datetime(old['Date'], format='%Y-%m-%d')  

    df_filtered = df[~df['Date'].isin(old['Date'])]

    if not df_filtered.empty:
        old = pd.concat([old, df_filtered], ignore_index=True)

    old['Date'] = pd.to_datetime(old['Date'], format='%Y-%m-%d')

    # data is in the wrong order, reverses it
    old = old.sort_values(by='Date')

    # Save the updated and sorted data to a new CSV file
    old.to_csv(filename, index=False)
    old.to_csv(before, index=False)

    print("Data saved as", filename)

#the data is missing hour, as it is only daily, fills weekend gaps also
def fill_missing_hours(csv):
    df = pd.read_csv(csv)

    value_Name = df.columns[1]

    # Manually parse the 'date' column using the correct format (DD.MM.YY)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Date'] = df['Date'].dt.normalize()
    df.set_index('Date', inplace=True)

    # start 2015
    full_hourly_range = pd.date_range(start='01.01.2015', end=df.index.max() + pd.Timedelta(days=1), freq='h')[:-1]

    # put prefered null value here
    df_full = df.reindex(full_hourly_range, fill_value=pd.NA)
    df_full.reset_index(inplace=True)
    df_full.rename(columns={'index': 'Date'}, inplace=True)
    df_full[value_Name] = df_full.groupby(df_full['Date'].dt.floor('D'))[value_Name].transform(lambda group: group.ffill().bfill())

    # fills emptys
    df_full.fillna({value_Name:np.nan}, inplace=True)
    df_full.to_csv(csv, index=False)
    print('Missing Hours Filled: ', csv)

##Entsoe

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
    df_copy.index.name = 'timestamp'
    df_copy.to_csv(filename)


##Covid Lockdown Data
def evaluate_date(request_date):
    if request_date in list(covid_df):
        truncated_covid_df = covid_df[['state', 'Measure ', request_date]]
        sum_value = 0
        for index, row in truncated_covid_df.iterrows():
            if row.isnull().values.any(): continue  # if any value in row is missing
            if measure_influence[row['Measure ']] == 0: continue  # if measure has no influence
            sum_value += ((int(row[request_date]) / 5) + 0.6) * state_percentages[row['state']] * measure_influence[
                row['Measure ']]  # see readme documentation
        return sum_value
    else:
        return 0
    
##smard
def main():

    output_path = sys.argv[1]

    dict_ids = [balancing_id["automatic_frequency"],
                balancing_id["balancing_energy"],
                balancing_id["costs"],
                balancing_id["exported_balancing_services"],
                balancing_id["frequency_containment"],
                balancing_id["imported_balancing_services"],
                balancing_id["manual_frequency"],
                electricity_consumption_id["actual"],
                electricity_consumption_id["forecast"],
                electricity_generation_id["actual"],
                electricity_generation_id["forecast"],
                market_id["cross_border_physical"],
                market_id["scheudled_commercial_exchanges"],
                market_id["day_ahead_prices"]    
    ]
    
    final_df = None

    for i in range(3):
        working_df = download(i)
        working_df = new_format(working_df, dict_ids[i])

        #if i > 0:
        working_df=working_df.drop(working_df.columns[1],axis=1)
        #only called once
        if final_df is None:
            final_df = working_df
        else:
            final_df = pd.merge(final_df, working_df, on=working_df.columns[0], how='outer')
            #final_df = pd.merge(final_df, working_df, on=working_df.columns[0], how='inner', copy=True)
    
    final_df=final_df[final_df.duplicated(keep=False) == False]

    final_df.to_csv(output_path, sep=',', index=False)

    #use gzip to compress .csv outputfile to <file_out>.gz
    path_object = Path(output_path)
    output_pathgz = path_object.with_suffix('.gz')
    final_df.to_csv(output_pathgz, sep=',', index=False, compression='gzip')


def download_and_merge_multiple_csv(module_ids):
    steps = ["1420066800000","1600000000000",str(int(datetime.datetime.today().timestamp()))+'000']
    csvfiles = []
    for timestamp_from, timestamp_to in zip(steps,steps[1:]):
        response = requests.post('https://www.smard.de/nip-download-manager/nip/download/market-data',
                                 data='{"request_form":[{"format":"CSV","moduleIds":'+module_ids+',"region":"DE","timestamp_from":'+timestamp_from+',"timestamp_to":'+timestamp_to+',"type":"discrete","language":"en","resolution":"hour"}]}')
        csvfiles.append(response.content.decode('utf-8-sig'))
    csvfile_data = csvfiles[0] + csvfiles[1][csvfiles[1].index('\n'):]
    return csvfile_data


def download(download_id):
    #14 different files
    match download_id:
        # AUTOMATIC FREQUENCY RESTORATION
        case 0:
            csvfile_data = download_and_merge_multiple_csv('[18004368,18004369,18004370,18004351,18004371,18004372,18004373,18004374]')
        # BALANCING ENERGY
        case 1:
            csvfile_data = download_and_merge_multiple_csv('[15004383,15004384,15004382,15004390]')
        # COSTS
        case 2:
            csvfile_data = download_and_merge_multiple_csv('[16004391,16000419,16000418]')
        # EXPORTED BALANCING SERVICES
        case 3:
            csvfile_data = download_and_merge_multiple_csv('[20004385]')
        #FREQUENCY CONTAINMENT RESERVE
        case 4:
            csvfile_data = download_and_merge_multiple_csv('[17004363, 17004367]')
        # IMPORTED BALANCING SERVICES
        case 5:
            csvfile_data = download_and_merge_multiple_csv('[21004386]')
        # MANUAL FREQUENCY RESTORATION RESERVE
        case 6:
            csvfile_data = download_and_merge_multiple_csv('[19004377,19004375,19004376,19004352,19004378,19004379,19004380,19004381]')

        #electricity consumption, actual
        case 7:
            csvfile_data = download_and_merge_multiple_csv('[5000410,5004387,5004359]')
        #forecast consumption
        case 8:
            csvfile_data = download_and_merge_multiple_csv('[6000411,6004362]')
        #electricity generation actual
        case 9:
            csvfile_data = download_and_merge_multiple_csv('[1001224,1004066,1004067,1004068,1001223,1004069,1004071,1004070,1001226,1001228,1001227,1001225]')
        #electricity generation forecast
        case 10:
            csvfile_data = download_and_merge_multiple_csv('[2000122,2005097,2000715,2003791,2000123,2000125]')
        #MARKET
        # CROSSBORDER FLOWS
        case 11:
            csvfile_data = download_and_merge_multiple_csv('[31004963,31004736,31004737,31004740,31004741,31004988,31004990,31004992,31004994,31004738,31004742,31004743,31004744,31004880,31004881,31004882,31004883,31004884,31004885,31004886,31004887,31004888,31004739]')
        # CROSSBORDER SCHEDULED FLOWS
        case 12:
            csvfile_data = download_and_merge_multiple_csv('[22004629,22004722,22004724,22004404,22004409,22004545,22004546,22004548,22004550,22004551,22004552,22004405,22004547,22004403,22004406,22004407,22004408,22004410,22004412,22004549,22004553,22004998,22004712]')
        # DAYAHEAD
        case 13:
            csvfile_data = download_and_merge_multiple_csv('[8004169,8004170,8000251,8005078,8000252,8000253,8000254,8000255,8000256,8000257,8000258,8000259,8000260,8000261,8000262,8004996,8004997]')

    download_df = pd.read_csv(StringIO(csvfile_data), sep=";", header=[0], na_values='-', low_memory=False)
    return download_df


def new_format(df, my_dict):
        
    #use fitting dict to rename table head
    df.rename(columns=my_dict, inplace=True)
    
    #change Datetime_format; replace '-' with np.nan
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    df['End_Date'] = pd.to_datetime(df['End_Date'])
    df.replace("-",np.nan, inplace=True)

    #remove , seperator for thousand
    df.replace(",","", inplace=True, regex=True)
    
    return df
    

def my_merge(fin_df, work_df, i):

    #if i > 0:
        #work_df=work_df.drop(work_df.columns[1],axis=1)
    work_df=work_df.drop(work_df.columns['End_Date'],axis=1)
    #fin_df = pd.merge(fin_df, work_df, on=work_df.columns[0], how='inner', copy=True)
    fin_df = pd.merge(fin_df, work_df, on=work_df.columns[0], how='outer')


##weather

#Definitions of funktions for weather
def combine_historic(station_r, place): 
  #combine data
  try:
    file_r = os.path.join(station_folder, station_r, f"{station_r}_data_combined.csv")
    
    #read data
    df_r = pd.read_csv(file_r)
    combined_df=df_r
    output_file = os.path.join(stations_combined, f"{place}_review.csv")
    combined_df.to_csv(output_file, index=False)

    print(f"Comibe: {station_r} -> {output_file}")

  except FileNotFoundError as e:
    print(f"File not found: {e}")
  except Exception as e:
    print(f"Error while computing{station_r}: {e}")
def combine_all_stations():
  files = [f for f in os.listdir(stations_combined) if f.endswith('.csv')]

  #rename collums to station name
  for file in files:
    file_path = os.path.join(stations_combined, file)
    df = pd.read_csv(file_path)
    #extract filename
    file_name = os.path.splitext(file)[0]
    columname=[df.columns[0]] + [f'{col}_{file_name}' for col in df.columns[1:]]
    df.columns = columname
    print(f'Renamend collums for {file_name}')
    df.to_csv(file_path, index=False)

  #combine dataframes  
  all_data_frames = []
  for file in files:
    file_path = os.path.join(stations_combined, file)  
    
    #load data and add to list
    try:
      df = pd.read_csv(file_path, delimiter=",", parse_dates=["date"], date_format="%Y%m%d%H")
      all_data_frames.append(df)
      print(f"Add data from: {file_path}")
    except Exception as e:
      print(f"Error while loading {file}: {e}")
  
  #if loaded -> combine
  if all_data_frames:
    combined_data = all_data_frames[0]
    for df in all_data_frames[1:]:
      df["date"] = pd.to_datetime(df["date"], errors="coerce")                
      combined_data = pd.merge(combined_data, df, on=[  "date"], how="outer")
    combined_data = combined_data.sort_values(by=[  "date"]).drop_duplicates(subset=[  "date"], keep='last')

  #save
  final_filename = os.path.join(data_collection_folder, f"weather.csv")
  combined_data.to_csv(final_filename, index=False)
  print(f"Combined data saved: {final_filename}")
def start_combine_historic():
    max_workers = min(os.cpu_count(), len(station_ids_r))  
    print(f"Start cmbination of {len(station_ids_r)} stations with {max_workers} threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_station = {executor.submit(combine_historic, station_r, place): (station_r,  place) for station_r,  place in zip(station_ids_r,  station_place) }    
        for future in concurrent.futures.as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                future.result()
                print(f"All data combined for {station_id}.")
            except Exception as e:
                print(f"Error while combination of station {station_id}: {e}")
def combine_forecast():

  files = [f for f in os.listdir(station_folder) if f.endswith('.csv')]

  #rename collumns to stationname
  for file in files:
    file_path = os.path.join(station_folder, file)
    df = pd.read_csv(file_path)
    #extract filename
    file_name = os.path.splitext(file)[0]
    columname=[df.columns[0]] + [f'{col}_{file_name}' for col in df.columns[1:]]
    df.columns = columname
    print(f'renamed collums for {file_name}')
    df.to_csv(file_path, index=False)
 
  all_data_frames = []
  for file in files:
    file_path = os.path.join(station_folder, file)  
    
    #load data and add to list
    try:
      df = pd.read_csv(file_path, delimiter=",", parse_dates=["date"], date_format="%Y%m%d%H")
      all_data_frames.append(df)
      print(f"data added from {file_path}")
    except Exception as e:
      print(f"Error while loading file {file}: {e}")
  
   
  if all_data_frames:
    combined_data = all_data_frames[0]
    for df in all_data_frames[1:]:
      df["date"] = pd.to_datetime(df["date"], errors="coerce")
      combined_data = pd.merge(combined_data, df, on=[  "date"], how="outer")
    combined_data = combined_data.sort_values(by=[  "date"]).drop_duplicates(subset=[  "date"], keep='last')


  final_filename = os.path.join(forecas_folder, f"weather_forecast.csv")
  combined_data.to_csv(final_filename, index=False)
  print(f"Saved combined forecast: {final_filename}")
def create_folder():
  os.makedirs(computing_folder, exist_ok=True)
  os.makedirs(stations_combined, exist_ok=True)
  for station in station_ids_r:
    output_folder_station = os.path.join(computing_folder, station)
    os.makedirs(output_folder_station, exist_ok=True)
    station_folder =os.path.join(output_folder,'stations',station)
    os.makedirs(station_folder, exist_ok=True)

#function to load forecast
def station_folderget_weather_data_for_station_review(station_id):
    output_filepath = os.path.join(computing_folder,station_id)
    print(f"storage location  {output_filepath}, computing_folder {computing_folder}, station_id {station_id}")    
    for data_type, endpoint in data_types.items():
        url = base_url_review + endpoint
        response = requests.get(url)
        response.raise_for_status()

        #lookup zip-file
        for line in response.text.splitlines():
            if station_id in line and "zip" in line:
                filename = re.search(r'href="(.*?)"', line).group(1)
                file_url = url + filename
                
                print(f"Download of: {file_url}")
                file_response = requests.get(file_url)
                file_response.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(file_response.content)) as z:
                    if data_type == "cloudiness_historical" or data_type == "cloudiness_recent":
                        txt_files = [name for name in z.namelist() if re.match(r'produkt_n_stunde_\d{8}_\d{8}_' + station_id + r'\.txt', name)]
                    elif data_type == "pressure_historical" or data_type == "pressure_recent":
                        txt_files = [name for name in z.namelist() if re.match(r'produkt_p0_stunde_\d{8}_\d{8}_' + station_id + r'\.txt', name)]
                    elif data_type == "sun_historical" or data_type == "sun_recent":
                        txt_files = [name for name in z.namelist() if re.match(r'produkt_sd_stunde_\d{8}_\d{8}_' + station_id + r'\.txt', name)]
                    elif data_type == "wind_historical" or data_type == "wind_recent":
                        txt_files = [name for name in z.namelist() if re.match(r'produkt_ff_stunde_\d{8}_\d{8}_' + station_id + r'\.txt', name)]
                    elif data_type == "precipitation_historical" or data_type == "precipitation_recent":
                        txt_files = [name for name in z.namelist() if re.match(r'produkt_rr_stunde_\d{8}_\d{8}_' + station_id + r'\.txt', name)]
                    else:
                        txt_files = [name for name in z.namelist() if re.match(r'produkt_tu_stunde_\d{8}_\d{8}_' + station_id + r'\.txt', name)]
                    
                    if not txt_files:
                        print(f"No TXT file in the expected format for station {station_id} found.")
                        continue  

                    txt_filename = txt_files[0]
                    with z.open(txt_filename) as f:
                        try:
                            df = pd.read_csv(f, sep=";", encoding="utf-8")
                            if df.empty:
                                print(f"Warning: The file {txt_filename} is empty.")
                            else:
                                print("Data loaded for:", txt_filename)
                                if data_type == "temperature_historical":
                                    new_filename = f"temp_{station_id}_hist.txt"
                                elif data_type == "temperature_recent":
                                    new_filename = f"temp_{station_id}_recent.txt"
                                elif data_type == "cloudiness_historical":
                                    new_filename = f"clouds_{station_id}_hist.txt"
                                elif data_type == "cloudiness_recent":
                                    new_filename = f"clouds_{station_id}_recent.txt"
                                elif data_type == "pressure_historical":
                                    new_filename = f"pressure_{station_id}_hist.txt"
                                elif data_type == "pressure_recent":
                                    new_filename = f"pressure_{station_id}_recent.txt"
                                elif data_type == "sun_historical":
                                    new_filename = f"sun_{station_id}_hist.txt"
                                elif data_type == "sun_recent":
                                    new_filename = f"sun_{station_id}_recent.txt"
                                elif data_type == "wind_historical":
                                    new_filename = f"wind_{station_id}_hist.txt"
                                elif data_type == "wind_recent":
                                    new_filename = f"wind_{station_id}_recent.txt"      
                                elif data_type == "precipitation_historical":
                                    new_filename = f"precipitation_{station_id}_hist.txt"
                                elif data_type == "precipitation_recent":
                                    new_filename = f"precipitation_{station_id}_recent.txt"
                                output_filename = os.path.join(output_filepath, new_filename)                                
                                df.to_csv(output_filename, sep=";", encoding="utf-8", index=False)
                                print(f" Saved weather-file as: {output_filepath}")   
                                print(f" Saved file as: {os.path.abspath(output_filepath)}")
                        except Exception as e:
                            print(f"Error while loading file {txt_filename}: {e}")
    cut_historic_bevor_2015(station_id)

def download_weather_data_for_all_stations_review(station_ids):
    max_workers = min(os.cpu_count(), len(station_ids))   
    print(f"Start doanload of {len(station_ids)} stations with {max_workers} threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_station = {executor.submit(station_folderget_weather_data_for_station_review, station_id): station_id for station_id in station_ids}        
        for future in concurrent.futures.as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                future.result()
                print(f"download succeded for{station_id}.")
            except Exception as e:
                print(f"Error while downloading {station_id}: {e}")
def cut_historic_bevor_2015(station_id):
    computing_folder_station = os.path.join(computing_folder, station_id)
    station_files = [f for f in os.listdir(computing_folder_station) if re.match(r'(.+)_hist\.txt', f)]    
    for file in station_files:
        file_path = os.path.join(computing_folder_station, file)
        with open(file_path, 'r') as infile:
            lines = infile.readlines()
        
        filtered_lines = []
        for line in lines[:1]:
            filtered_lines.append(line)
        for line in lines[1:]:
            columns = line.strip().split(';')
            if len(columns) > 1:  
                mess_datum = columns[1]
                year = int(mess_datum[:4])                
                if year >= 2015:
                    filtered_lines.append(line)

        with open(file_path, 'w') as outfile:
            outfile.writelines(filtered_lines)
        print(f"Historically shortened until 2015: {file}")
    remove_columns_review(station_id)

def start_cut_historic_bevor_2015(station_ids):
    max_workers = min(os.cpu_count(), len(station_ids))   
    print(f"start shortening till 2015 for {len(station_ids)} stations with {max_workers} threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_station = {executor.submit(cut_historic_bevor_2015, station_id): station_id for station_id in station_ids}        
        for future in concurrent.futures.as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                future.result()
                print(f"Files shortend to 2015 for {station_id}.")
            except Exception as e:
                print(f"Error while shortening files to 2015 for {station_id}: {e}")

def remove_columns_review(station_id):
    print('Start Remove Columns')
    computing_folder_station =os.path.join(computing_folder, station_id)
    temp_files = [f for f in os.listdir(computing_folder_station) if f.startswith("temp_") and f.endswith(".txt")]
    clouds_files = [f for f in os.listdir(computing_folder_station) if f.startswith("clouds_") and f.endswith(".txt")]
    pressure_files = [f for f in os.listdir(computing_folder_station) if f.startswith("pressure_") and f.endswith(".txt")]
    sun_files = [f for f in os.listdir(computing_folder_station) if f.startswith("sun_") and f.endswith(".txt")]
    wind_files = [f for f in os.listdir(computing_folder_station) if f.startswith("wind_") and f.endswith(".txt")]
    precipitation_files = [f for f in os.listdir(computing_folder_station) if f.startswith("precipitation_") and f.endswith(".txt")]
    
    for file in clouds_files:
        file_path = os.path.join(computing_folder_station, file)        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)
            df = df.drop(columns=[col for col in columns_remove_clouds if col in df.columns])
            df.to_csv(file_path, sep=";", index=False)
            print(f"Colums removed from{file}")
        
        except Exception as e:
            print(f"Error while processing{file}: {e}")
    
    for file in pressure_files:
        file_path = os.path.join(computing_folder_station, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)
            df = df.drop(columns=[col for col in columns_remove_pressure if col in df.columns])
            df.to_csv(file_path, sep=";", index=False)
            print(f"removed collums from {file}")
        
        except Exception as e:
            print(f"Error while prcessing file: {file}: {e}")

    for file in sun_files:
        file_path = os.path.join(computing_folder_station, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True) 
            df = df.drop(columns=[col for col in columns_remove_sun if col in df.columns])
            df.to_csv(file_path, sep=";", index=False)
            print(f"removed collums from {file}")
        
        except Exception as e:
            print(f"Error while prcessing file: {file}: {e}")

    for file in temp_files:
        file_path = os.path.join(computing_folder_station, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)
            df.to_csv(file_path, sep=";", index=False)
            print(f"removed collums from {file}")
        
        except Exception as e:
            print(f"Error while prcessing file: {file}: {e}")

    for file in wind_files:
        file_path = os.path.join(computing_folder_station, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True) 
            df = df.drop(columns=[col for col in columns_remove_wind if col in df.columns]) 
            df.to_csv(file_path, sep=";", index=False)
            print(f"removed collums from {file}")
        
        except Exception as e:
            print(f"Error while prcessing file: {file}: {e}")

    for file in precipitation_files:
        file_path = os.path.join(computing_folder_station, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)
            df = df.drop(columns=[col for col in columns_remove_precipitation if col in df.columns])
            df.to_csv(file_path, sep=";", index=False)
            print(f"removed collums from {file}")
        
        except Exception as e:
            print(f"Error while prcessing file: {file}: {e}")
    combine_historic_recent(station_id)

def start_remove_columns_review(station_ids):
    max_workers = min(os.cpu_count(), len(station_ids))   
    print(f"Start remove collumns {len(station_ids)} stations with {max_workers} threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_station = {executor.submit(remove_columns_review, station_id): station_id for station_id in station_ids}        
        for future in concurrent.futures.as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                future.result()  
                print(f"Collumns deleted for {station_id}.")
            except Exception as e:
                print(f"Error while deletion of collumns {station_id}: {e}")
def combine_historic_recent(station_id):
    computing_folder_station = os.path.join(computing_folder, station_id)
    station_files = [f for f in os.listdir(computing_folder_station) if re.match(r'(.+)_' + station_id + r'_(hist|recent)\.txt', f)]
    file_pairs = {}
    for file in station_files:
        match = re.match(r'(.+)_' + station_id + r'_(hist|recent)\.txt', file)
        if match:
            wettertyp, period = match.groups()
            key = f"{wettertyp}_{station_id}"
            if key not in file_pairs:
                file_pairs[key] = {}
            file_pairs[key][period] = os.path.join(computing_folder_station, file)

    #combine historic an current data
    for key, file_pair in file_pairs.items():
        if 'hist' in file_pair and 'recent' in file_pair:
            hist_df = pd.read_csv(file_pair['hist'], delimiter=";")
            recent_df = pd.read_csv(file_pair['recent'], delimiter=";")
            hist_df["MESS_DATUM"] = pd.to_datetime(hist_df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")
            recent_df["MESS_DATUM"] = pd.to_datetime(recent_df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")

            
            combined_df = pd.concat([hist_df, recent_df]).drop_duplicates(subset=["MESS_DATUM"], keep='last')
            combined_df = combined_df.sort_values(by=["MESS_DATUM"])
            combined_df["MESS_DATUM"] = combined_df["MESS_DATUM"].dt.strftime("%Y%m%d%H")

            combined_filename = os.path.join(computing_folder_station, f"{key}_combined.txt")
            combined_df.to_csv(combined_filename, sep=";", index=False)
            print(f"Combined data saved: {combined_filename}")
        else:
            print(f"Missing file for {key}")
    combine_all_station_data_review(station_id)

def start_combine_historic_recent(station_ids):
    max_workers = min(os.cpu_count(), len(station_ids))   
    print(f"Start combination of historic and current data for {len(station_ids)} stations with {max_workers} threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_station = {executor.submit(combine_historic_recent, station_id): station_id for station_id in station_ids}        
        for future in concurrent.futures.as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                future.result()
                print(f"combined historic and current data for {station_id}.")
            except Exception as e:
                print(f"Error while combining historic and current data {station_id}: {e}")
def combine_all_station_data_review(station_id):
    computing_folder_station = os.path.join(computing_folder, station_id)
    station_folder_station = os.path.join(station_folder, station_id) 
    combined_files = [f for f in os.listdir(computing_folder_station) if f.endswith(f"_{station_id}_combined.txt")]
    all_data_frames = []
    for file in combined_files:
        file_path = os.path.join(computing_folder_station, file)
        try:
            df = pd.read_csv(file_path, delimiter=";", parse_dates=["MESS_DATUM"], date_format="%Y%m%d%H")
            all_data_frames.append(df)
            print(f"data added from {file_path}")
        except Exception as e:
            print(f"Error while loading file {file}: {e}")
    if all_data_frames:
        combined_data = all_data_frames[0]
        for df in all_data_frames[1:]:
            df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")
            combined_data = pd.merge(combined_data, df, on=[  "MESS_DATUM"], how="outer")
        combined_data = combined_data.sort_values(by=[  "MESS_DATUM"]).drop_duplicates(subset=[  "MESS_DATUM"], keep='last')
        combined_data["MESS_DATUM"] = combined_data["MESS_DATUM"].dt.strftime("%Y%m%d%H")
        
        # change header
        header_mapping = {
            "STATIONS_ID": "STATIONS_ID",
            "MESS_DATUM": "date",
            "V_N_I": "Wolken_Interp",
            "V_N": "clouds",
            "P": "stationPressure_hPa",
            "P0": "surfacePressure_hPa",
            "SD_SO": "sunshine_min",
            "TT_TU": "T_temperature_C",
            "RF_TU": "humidity_Percent",
            "F": "wind_speed_ms",
            "D": "wind_direction_degree",
            "R1": "precipitationTotal_mm",
            "RS_IND": "precipitation_indicator"

        }
    
        combined_data.rename(columns=header_mapping, inplace=True)
        final_filename = os.path.join(station_folder_station, f"{station_id}_data_combined.csv")
        combined_data.to_csv(final_filename, sep=",", index=False)
        print(f"All data combined for station {station_id} saved in: {final_filename}")

    else:
        print(f"No combined data for station {station_id} found.")
def start_combine_all_station_data_review(station_ids):
    max_workers = min(os.cpu_count(), len(station_ids))   
    print(f"Start of combination of data for {len(station_ids)} stations with {max_workers} threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_station = {executor.submit(combine_all_station_data_review, station_id): station_id for station_id in station_ids}        
        for future in concurrent.futures.as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                future.result()
                print(f"Files combined for {station_id}.")
            except Exception as e:
                print(f"Error while combination of station {station_id}: {e}")
#Weather forecastfunktions:
def get_weather_data_for_station_forecast(station_id, station_place):
    params = {
        "stationIds": station_id
    }
    #prepare request
    request = requests.Request("GET", url_forecast, headers=headers_weather, params=params)
    prepared_request = request.prepare()
    
    response = requests.Session().send(prepared_request)
    if response.status_code == 200:
        data = response.json()
        
        filename = os.path.join(computing_folder, f"weather_forecast_{station_place}.json")
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Forecast was saved in {filename}")
        with open(filename) as file:
            data = json.load(file)
        
        for station_id, station_data in data.items():
            forecast_data = station_data["forecast1"]
            start_time = forecast_data["start"]
            time_step = forecast_data["timeStep"]

            date = [dt.utcfromtimestamp((start_time + i * time_step) / 1000) for i in range(len(forecast_data["temperature"]))]
            
            variables = {
                "T_temperature_C": forecast_data.get("temperature", []),
                "T_temperature_standarddeviation_C": forecast_data.get("temperatureStd", []),
                "precipitationTotal_mm": forecast_data.get("precipitationTotal", []),
                "sunshine_min": forecast_data.get("sunshine", []),
                "dewPoint2m": forecast_data.get("dewPoint2m", []),
                "surfacePressure_hPa": forecast_data.get("surfacePressure", []),
                "humidity_Percent": forecast_data.get("humidity", []),
                "isDay_bool": forecast_data.get("isDay", [])
            }
            max_length = max(len(date), *(len(values) for values in variables.values()))
            date.extend([None] * (max_length - len(date)))
            for key, values in variables.items():
                variables[key].extend([None] * (max_length - len(values)))
            df = pd.DataFrame({
                "date": date,
                **variables
            })             
            df["T_temperature_C"] = df["T_temperature_C"].apply(lambda x: x / 10 if pd.notnull(x) else x)
            df["T_temperature_standarddeviation_C"] = df["T_temperature_standarddeviation_C"].apply(lambda x: x / 10 if pd.notnull(x) else x)
            df["surfacePressure_hPa"] = df["surfacePressure_hPa"].apply(lambda x: x / 10 if pd.notnull(x) else x)
            df["humidity_Percent"] = df["humidity_Percent"].apply(lambda x: x / 10 if pd.notnull(x) else x)
            df["date"] = df["date"].apply(lambda x: x.strftime("%Y%m%d%H"))
            df.to_csv(os.path.join(station_folder, f"weather_forecast_{station_place}.csv"), index=False)
            print(f"Weather prediction in weather_forecast_{station_place}.csv convertet")
    else:
        print(f"Error while request {response.status_code}")
def download_weatherforecast_data_for_all_stations_forecast(station_ids, station_places):
    for (station_id , station_place) in zip(station_ids, station_places):
        print(f"Start download of Station {station_id}...")
        get_weather_data_for_station_forecast(station_id, station_place)
        print()
def remove_columns_forecast():
    print("Start removing columns")
    forecast_files = [f for f in os.listdir(station_folder) if f.startswith("weather_forecast_")]  
    print(f"File: {forecast_files}...")
    for file in forecast_files:
        print(f"Start removing columns for {file}...")
        file_path = os.path.join(station_folder, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=",", skipinitialspace=True)  
            print(f"Columns in dataframe: {list(df.columns)}") 
            df = df.drop(columns=[col for col in columns_remove_forecast if col in df.columns])
            df.to_csv(file_path, sep=",", index=False)
            print(f"removed collums from {file}")
        
        except Exception as e:
            print(f"Error while prcessing file: {file}: {e}")





start_time = time.time()
##Stockmarket
get_Data(url_oil, '../final-submission/merged_data/data_collection/oilWti.csv', 'Oil WTI', "../final-submission/merged_data/data_collection/oilWtiOld.csv")
get_Data(url_gas, '../final-submission/merged_data/data_collection/naturalGas.csv', 'Natural Gas', "../final-submission/merged_data/data_collection/naturalGasOld.csv")
get_Data(url_coal, '../final-submission/merged_data/data_collection/coal.csv', 'Coal', "../final-submission/merged_data/data_collection/CoalOld.csv")
get_Data(url_uran, '../final-submission/merged_data/data_collection/uran.csv', 'Uran', '../final-submission/merged_data/data_collection/uranOld.csv')

fill_missing_hours('../final-submission/merged_data/data_collection/oilWti.csv')
fill_missing_hours('../final-submission/merged_data/data_collection/naturalGas.csv')
fill_missing_hours('../final-submission/merged_data/data_collection/coal.csv')
fill_missing_hours('../final-submission/merged_data/data_collection/uran.csv')

df1 = pd.read_csv('../final-submission/merged_data/data_collection/oilWti.csv')
df2 = pd.read_csv('../final-submission/merged_data/data_collection/naturalGas.csv')
df3 = pd.read_csv('../final-submission/merged_data/data_collection/coal.csv')
df35 = pd.read_csv('../final-submission/merged_data/data_collection/uran.csv')

merged_df = pd.merge(df1, df2, on='Date', how='outer')
merged_df = pd.merge(merged_df, df3, on='Date', how='outer')
merged_df = pd.merge(merged_df, df35, on='Date', how='outer')

merged_df.to_csv('../final-submission/merged_data/data_collection/merged_data.csv', index=False)

print("CSV files have been merged and saved.")

end_time = time.time()
verstrichene_zeit = end_time - start_time
print(f'Ausführungszeit nach Stockmarket: {verstrichene_zeit} Sekunden')

##entsoe

start_time_entsoe = time.time()
df4 = pd.read_csv('../final-submission/merged_data/data_collection/day_ahead_prices.csv')
df4.drop(df4.columns[2], axis=1, inplace=True)
df5 = pd.read_csv('../final-submission/merged_data/data_collection/load_forecast.csv')
df5.drop(df5.columns[2], axis=1, inplace=True)
df6 = pd.read_csv('../final-submission/merged_data/data_collection/generation_forecast.csv')
df6.drop(df6.columns[2], axis=1, inplace=True)
df7 = pd.read_csv('../final-submission/merged_data/data_collection/intraday_wind_solar_forecast.csv')
df7.drop(df7.columns[4], axis=1, inplace=True)
df8 = pd.read_csv('../final-submission/merged_data/data_collection/day_ahead_wind_solar_forecast.csv')
df8.drop(df8.columns[4], axis=1, inplace=True)
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
end_time = time.time()
verstrichene_zeit = end_time - start_time
print(f'Ausführungszeit nach Entsoe: {verstrichene_zeit} Sekunden')

##Covid Lockdown Data
# generate and populate dataframe with all dates from 2015-1-1 - today
from datetime import date, timedelta

working_dt = date(2015, 1, 1)
end_dt = date(date.today().year, date.today().month, date.today().day)
delta = timedelta(days=1)

data_rows = []

# populate df
while working_dt <= end_dt:
    factor = evaluate_date(working_dt.isoformat())
    date = working_dt.isoformat()
    for hour in range(24):
        timestamp = pd.Timestamp(working_dt.isoformat()) + pd.Timedelta(hours=hour)
        data_rows.append({'Date': timestamp, 'Covid factor': factor})  # Add to rows list
    working_dt += delta

covid_factors_df = pd.DataFrame(data_rows)
print(covid_factors_df.head)

covid_factors_df.to_csv('../final-submission/merged_data/data_collection/covid.csv', index=False)

end_time = time.time()
verstrichene_zeit = end_time - start_time
print(f'Ausführungszeit nach Covidzahlen: {verstrichene_zeit} Sekunden')

##Smard
output_path = '../final-submission/merged_data/data_collection/smard.csv'

dict_ids = [balancing_id["automatic_frequency"],
            balancing_id["balancing_energy"],
            balancing_id["costs"],
            balancing_id["exported_balancing_services"],
            balancing_id["frequency_containment"],
            balancing_id["imported_balancing_services"],
            balancing_id["manual_frequency"],
            electricity_consumption_id["actual"],
            electricity_consumption_id["forecast"],
            electricity_generation_id["actual"],
            electricity_generation_id["forecast"],
            market_id["cross_border_physical"],
            market_id["scheudled_commercial_exchanges"],
            market_id["day_ahead_prices"]    
    ]
    
final_df = None

for i in range(13):
    working_df = download(i)
    working_df = new_format(working_df, dict_ids[i])

    if i > 0:
       working_df=working_df.drop(working_df.columns[1],axis=1)
        #only called once
    if final_df is None:
            final_df = working_df
    else:
        final_df = pd.merge(final_df, working_df, on=working_df.columns[0], how='inner', copy=True)
    
final_df.to_csv(output_path, sep=',', index=False)

#use gzip to compress .csv outputfile to <file_out>.gz
path_object = Path(output_path)
output_pathgz = path_object.with_suffix('.gz')
final_df.to_csv(output_pathgz, sep=',', index=False, compression='gzip')

end_time = time.time()
verstrichene_zeit = end_time - start_time
print(f'Ausführungszeit nach Smard: {verstrichene_zeit} Sekunden')

##weather
start_time_w = time.time()
create_folder()
download_weather_data_for_all_stations_review(station_ids_r)

end_time_w = time.time()
verstrichene_zeit = end_time_w - start_time_w
print(f'Ausführungszeit Wetter: {verstrichene_zeit} Sekunden')
download_weatherforecast_data_for_all_stations_forecast(station_ids_f, station_place)
remove_columns_forecast()

end_time_w = time.time()
verstrichene_zeit = end_time_w - start_time_w
print(f'Ausführungszeit Wetter: {verstrichene_zeit} Sekunden')

start_combine_historic()
enend_time_wd = time.time()
verstrichene_zeit = end_time_w - start_time_w
print(f'Ausführungszeit Wetter: {verstrichene_zeit} Sekunden')
combine_all_stations()
combine_forecast()

end_time_w = time.time()
verstrichene_zeit = end_time_w - start_time_w
print(f'Ausführungszeit Wetter: {verstrichene_zeit} Sekunden')

end_time = time.time()
verstrichene_zeit = end_time - start_time
print(f'Ausführungszeit nach dem Wetter: {verstrichene_zeit} Sekunden')




##Zusammenfassung
df_res = pd.read_csv('../final-submission/merged_data/data_collection/merged_data.csv')
df_ens = pd.read_csv('../final-submission/merged_data/data_collection/merged_data3.csv')
df_smard = pd.read_csv('../final-submission/merged_data/data_collection/smard.csv')
df_smard = df_smard.rename(columns={'Start_Date': 'Date'})
df_smard.to_csv('../final-submission/merged_data/data_collection/smard.csv', index=False)
df_smard = pd.read_csv('../final-submission/merged_data/data_collection/smard.csv')
print(df_smard.head())
df_smard['Date'] = pd.to_datetime(df_smard['Date'])
df_filteredSmard = df_smard[df_smard['Date'].dt.minute == 0]
df_filteredSmard['Date'] = pd.to_datetime(df_filteredSmard['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_filteredSmard.to_csv('../final-submission/merged_data/data_collection/smard.csv', index=False)
df_smard = pd.read_csv('../final-submission/merged_data/data_collection/smard.csv')

df_weather = pd.read_csv('../final-submission/merged_data/data_collection/weather.csv')
df_weather = df_weather.rename(columns={'date': 'Date'})
df_covid = pd.read_csv('../final-submission/merged_data/data_collection/covid.csv')
df_social = pd.read_csv('../final-submission/merged_data/data_collection/major_social_events.csv')
df_carbon = pd.read_csv('../final-submission/merged_data/data_collection/carbon_price_forward_filled.csv')

merge_big = pd.merge(df_ens, df_res, on='Date', how='outer')
merge_big = pd.merge(merge_big, df_smard, on='Date', how='outer')
merge_big = pd.merge(merge_big, df_social, on='Date', how='outer')
merge_big = pd.merge(merge_big, df_carbon, on='Date', how='outer')
merge_big = pd.merge(merge_big, df_weather, on='Date', how='outer')
merge_big = pd.merge(merge_big, df_covid, on='Date', how='outer')

#add weekdays and Holidays
merge_big['Date'] = pd.to_datetime(merge_big['Date'])
merge_big['month'] = merge_big['Date'].dt.month
merge_big['weekday'] = merge_big['Date'].dt.weekday  # 0=Montag, 6=Sonntag
merge_big['week_of_year'] = merge_big['Date'].dt.isocalendar().week
merge_big['is_weekend'] = merge_big['weekday'].isin([5, 6])
german_holidays = holidays.Germany(years=range(merge_big['Date'].dt.year.min(),
                                               merge_big['Date'].dt.year.max() + 1))
merge_big['date'] = merge_big['Date'].dt.date
merge_big['is_holiday'] = merge_big['date'].isin(german_holidays)
merge_big = merge_big.loc[:, ~merge_big.columns.str.endswith('_y')]
merge_big.columns =merge_big.columns.str.replace('_x$', '', regex=True)

merge_big.to_csv('../final-submission/merged_data/allData.csv', index=False)

print("CSV files have been merged and saved.")
end_time = time.time()
verstrichene_zeit = end_time - start_time
print(f'Ausführungszeit komplett: {verstrichene_zeit} Sekunden')