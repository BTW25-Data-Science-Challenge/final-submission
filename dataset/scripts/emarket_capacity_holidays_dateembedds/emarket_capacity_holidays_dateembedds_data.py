from entsoe import EntsoePandasClient
import pandas as pd
import requests
import holidays
from datetime import datetime
import numpy as np
import os
import time
import warnings

warnings.filterwarnings("ignore")

# Entsoe params
country = 'DE_LU'
token = '9a50d1d1-46b1-4376-93ca-4f55da8eca62'
client = EntsoePandasClient(api_key=token)
filename = max(os.listdir(os.path.join(os.getcwd(),'datasets_snapshots')))
path = os.path.join(os.path.join(os.getcwd(),'datasets_snapshots',filename))

class DataUpdater:
    def __init__(self,
                 current_data_path,
                 timestamp_to
                 ):
        self.path = current_data_path
        self.timestamp_from = None
        self.current_data = None
        self.data = None
        self.timestamp_to = timestamp_to.tz_localize(tz='Europe/Brussels')
        self.entsoe_data = None
        self.holidays = None
        self.capacities = None
        self.outages = None
        self.valid_cols = ['electricity_price', 'fcstd_load', 'actual_load', 'cap_nuclear',
                           'cap_lignite', 'cap_hard_coal', 'cap_gas', 'cap_oil',
                           'cap_fossil_other', 'cap_hydro', 'cap_pump_storage', 'cap_biomass',
                           'cap_wind_off', 'cap_wind_on', 'cap_solar', 'cap_batt_power',
                           'cap_batt_capacity', 'cap_coal_gas', 'cap_river', 'cap_re_other',
                           'cap_waste', 'ft_BW', 'ft_BY', 'ft_BE', 'ft_BB', 'ft_HB',
                           'ft_HH', 'ft_HE', 'ft_MV', 'ft_NI', 'ft_NW', 'ft_RP', 'ft_SL', 'ft_SN',
                           'ft_ST', 'ft_SH', 'ft_TH', 'sin_hour_of_day', 'cos_hour_of_day',
                           'sin_day_of_week', 'cos_day_of_week', 'sin_month_of_year',
                           'cos_month_of_year']
        self.date_embedding = None
        self.new_data = None

    def get_current_data(self):
        print(f'reading existing data from: {self.path}...')
        self.current_data = pd.read_csv(self.path, index_col=0)
        self.current_data.index = pd.to_datetime(self.current_data.index, utc=True)
        self.current_data.index = self.current_data.index.tz_convert('Europe/Brussels')
        self.current_data = self.current_data[self.valid_cols]
        self.timestamp_from = self.current_data.index[-1] + pd.Timedelta(hours=1)

    def update_entsoe(self):
        print(f'extracting electricity market data from ENTSO-e for bidding zone {country}...')
        prices = client.query_day_ahead_prices(start=self.timestamp_from, end=self.timestamp_to, country_code=country)
        fcst_load = client.query_load_forecast(start=self.timestamp_from, end=self.timestamp_to,
                                               country_code=country).resample('h').mean()
        load = client.query_load(start=self.timestamp_from, end=self.timestamp_to, country_code=country).resample(
            'h').mean()
        self.entsoe_data = pd.concat((prices, fcst_load, load), axis=1)

    def update_date_embedding(self):
        print(f'generating date embeddings...')
        date_embedding = pd.DataFrame(index=pd.date_range(start=self.timestamp_from, end=self.timestamp_to, freq='h'))
        date_embedding['sin_hour_of_day'] = np.sin(2 * np.pi * date_embedding.index.hour / 24)
        date_embedding['cos_hour_of_day'] = np.cos(2 * np.pi * date_embedding.index.hour / 24)
        date_embedding['sin_day_of_week'] = np.sin(2 * np.pi * date_embedding.index.dayofweek / 7)
        date_embedding['cos_day_of_week'] = np.cos(2 * np.pi * date_embedding.index.dayofweek / 7)
        date_embedding['sin_month_of_year'] = np.sin(2 * np.pi * date_embedding.index.month / 12)
        date_embedding['cos_month_of_year'] = np.cos(2 * np.pi * date_embedding.index.month / 12)
        self.date_embedding = date_embedding

    def update_holidays(self):
        print(f'extracting holidays...')
        states = [
            'BW',  # Baden-Württemberg
            'BY',  # Bavaria
            'BE',  # Berlin
            'BB',  # Brandenburg
            'HB',  # Bremen
            'HH',  # Hamburg
            'HE',  # Hesse
            'MV',  # Mecklenburg-Vorpommern
            'NI',  # Lower Saxony
            'NW',  # North Rhine-Westphalia
            'RP',  # Rhineland-Palatinate
            'SL',  # Saarland
            'SN',  # Saxony
            'ST',  # Saxony-Anhalt
            'SH',  # Schleswig-Holstein
            'TH'  # Thuringia
        ]

        date_range = pd.date_range(start=self.timestamp_from, end=self.timestamp_to, freq='h')
        german_holidays = pd.DataFrame(columns=[f'ft_{s}' for s in states], index=date_range, data='-')

        for state in states:
            for year in range(self.timestamp_from.year, self.timestamp_to.year):
                dict = holidays.Germany(years=year, prov=state)
                for date in [pd.Timestamp(d) for d in dict.keys()]:
                    german_holidays.loc[german_holidays.index.date == date.date(), f'ft_{state}'] = dict[date]
        self.holidays = german_holidays

    def update_capacities(self):
        def query_capacity_echarts(url):
            headers = {'accept': 'application/json'}
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()

            data.pop("deprecated", None)
            dates = data['time']
            prod_types = data['production_types']
            data_dict = {prod['name']: prod['data'] for prod in prod_types}
            df = pd.DataFrame(data_dict, index=dates)

            return df

        url_m = 'https://api.energy-charts.info/installed_power?country=de&time_step=monthly&installation_decommission=false'

        m_cap = query_capacity_echarts(url_m)
        month = str(self.timestamp_from.month)
        month_str = month if len(month) == 2 else f'0{month}'
        try:
            print('extracting capacities [1/2]...')
            m_cap = m_cap.loc[f'{month_str}.{self.timestamp_from.year}':]
        except KeyError:
            m_cap = m_cap.loc[m_cap.index[-1]].to_frame().T
        m_cap.index = pd.to_datetime(m_cap.index, format='%m.%Y')  # + pd.offsets.MonthBegin(1)
        m_cap.index = m_cap.index.tz_localize('Europe/Brussels')
        m_cap = m_cap.reindex(pd.date_range(start=self.timestamp_from, end=self.timestamp_to, freq='h'), method='ffill')
        m_cap = m_cap * 1000  # convert from GW to MW
        m_cap.rename(columns={'Wind offshore': 'Wind Offshore', 'Wind onshore': 'Wind Onshore'}, inplace=True)

        # capacities from entso-e

        columns_entsoe = ['Fossil Brown coal/Lignite', 'Fossil Coal-derived gas',
                          'Fossil Gas', 'Fossil Hard coal', 'Fossil Oil', 'Hydro Pumped Storage',
                          'Hydro Run-of-river and poundage', 'Hydro Water Reservoir',
                          'Nuclear', 'Other', 'Other renewable', 'Waste']

        print('extracting capacities [2/2]...')
        y_cap = client.query_installed_generation_capacity(start=self.timestamp_from, end=self.timestamp_to,
                                                           country_code=country, psr_type=None)
        y_cap = y_cap[[col for col in columns_entsoe if col in y_cap]]  # flexibly include Nuclear for example

        y_cap = y_cap.reindex(pd.date_range(start=self.timestamp_from, end=self.timestamp_to, freq='h'),
                              method='ffill')  # tz='Europe/Brussels')).ffill()
        capacities = pd.concat((m_cap, y_cap), axis=1)

        self.update_outages()
        for col in self.outages:
            capacities[col] = np.maximum((capacities[col] - self.outages[col]), 0)
        if 'Nuclear' not in capacities.columns:
            capacities['Nuclear'] = 0
        self.capacities = capacities

    def update_outages(self):

        max_retries = 3
        delay = 2

        # outages prod units
        print('extracting outages of production units...')
        for attempt in range(1, max_retries + 1):
            try:
                outages_prd = client.query_unavailability_of_production_units(country, start=self.timestamp_from,
                                                                              end=self.timestamp_to, docstatus='A05',
                                                                              periodstartupdate=None,
                                                                              periodendupdate=None)
            except requests.ConnectionError as e:
                print(f'attempt {attempt} failed with ConnectionError: {e}\nretrying...')
                if attempt == max_retries:
                    raise
                time.sleep(delay)

        # outages gen units
        print('extracting outages of generation units...')
        for attempt in range(1, max_retries + 1):
            try:
                outages_gen = client.query_unavailability_of_generation_units(country, start=self.timestamp_from,
                                                                              end=self.timestamp_to, docstatus='A05',
                                                                              periodstartupdate=None,
                                                                              periodendupdate=None)
            except requests.ConnectionError as e:
                print(f'attempt {attempt} failed with ConnectionError: {e}\nretrying...')
                if attempt == max_retries:
                    raise
                time.sleep(delay)

        outages = pd.concat((outages_prd, outages_gen), axis=0)

        outages = outages.reset_index().rename(columns={'index': 'created_doc_time'})
        outages['offset'] = outages['start'] - outages['created_doc_time']
        outages['avail_qty'] = outages['avail_qty'].astype('float64')

        def assign_validity(df):
            for idx, row in df.iterrows():
                if row['created_doc_time'].hour < 12:
                    if row['start'] >= (row['created_doc_time'] + pd.Timedelta(days=1)).normalize() and row[
                        'nominal_power'] > row['avail_qty']:
                        df.at[idx, 'consider'] = 1
                    else:
                        df.at[idx, 'consider'] = 0
                else:
                    if row['start'] >= (row['created_doc_time'] + pd.Timedelta(days=2)).normalize() and row[
                        'nominal_power'] > row['avail_qty']:
                        df.at[idx, 'consider'] = 1
                    else:
                        df.at[idx, 'consider'] = 0

        assign_validity(outages)
        outages = outages[outages['consider'] == 1][
            ['created_doc_time', 'start', 'end', 'avail_qty', 'nominal_power', 'plant_type', 'production_resource_id']]

        outages_ref = pd.DataFrame(index=pd.date_range(start=self.timestamp_from, end=self.timestamp_to, freq='h'),
                                   columns=outages['plant_type'].unique(), data=0)

        for idx in outages_ref.index:
            for col in outages_ref.columns:
                df = outages[(outages['start'] <= idx) & (outages['end'] >= idx) & (outages['plant_type'] == col)]
                if len(df) > 0:
                    df = df.groupby(['created_doc_time', 'start', 'end', 'plant_type', 'production_resource_id']).mean()
                    diff = df['nominal_power'].sum() - df['avail_qty'].sum()
                    outages_ref.at[idx, col] = diff

        self.outages = outages_ref

    def update_data(self):

        column_mapping = {0: 'electricity_price',
                          'Forecasted Load': 'fcstd_load',
                          'Actual Load': 'actual_load',
                          'Biomass': 'cap_biomass',
                          'Wind Offshore': 'cap_wind_off',
                          'Wind Onshore': 'cap_wind_on',
                          'Solar': 'cap_solar',
                          'Battery Storage (Power)': 'cap_batt_power',
                          'Battery Storage (Capacity)': 'cap_batt_capacity',
                          'Fossil Brown coal/Lignite': 'cap_lignite',
                          'Fossil Coal-derived gas': 'cap_coal_gas',
                          'Fossil Gas': 'cap_gas',
                          'Fossil Hard coal': 'cap_hard_coal',
                          'Fossil Oil': 'cap_oil',
                          'Hydro Pumped Storage': 'cap_pump_storage',
                          'Hydro Run-of-river and poundage': 'cap_river',
                          'Hydro Water Reservoir': 'cap_hydro',
                          'Other': 'cap_fossil_other',
                          'Other renewable': 'cap_re_other',
                          'Waste': 'cap_waste',
                          'Nuclear': 'cap_nuclear'}

        self.get_current_data()
        self.update_date_embedding()
        self.update_entsoe()
        self.update_holidays()
        self.update_capacities()
        self.new_data = pd.DataFrame(index=pd.date_range(start=self.timestamp_from, end=self.timestamp_to, freq='h'))
        self.new_data = pd.concat((self.entsoe_data, self.capacities, self.holidays, self.date_embedding), axis=1)
        self.new_data.rename(columns=column_mapping, inplace=True)
        self.new_data.index = self.new_data.index.tz_convert('Europe/Brussels')
        self.data = pd.concat((self.current_data, self.new_data))


updater = DataUpdater(current_data_path=path, timestamp_to=pd.Timestamp(datetime.now()).floor(freq='h'))
updater.update_data()

time_now = pd.Timestamp(datetime.now())
saving_path = os.path.join(os.getcwd(),'datasets_snapshots',f'multivar_dataset_{time_now.strftime('%Y%m%d%H%M%S')}.csv')
updater.data.to_csv(saving_path)
print(f'file successfully saved to {saving_path}')