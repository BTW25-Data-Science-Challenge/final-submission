import pandas as pd
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
#filename = max(os.listdir(os.path.join(os.getcwd(),'datasets_snapshots')))
#path = os.path.join(os.path.join(os.getcwd(),'datasets_snapshots',filename))

class DataUpdater:
    def __init__(self,
                 current_data_path,
                 fosil_prices_path,
                 weather_data_path,
                 timestamp_to,
                 save_dir = None,
                 update_fossil: bool = False,
                 ):
        self.path = current_data_path
        self.fosil_prices_path = fosil_prices_path
        self.weather_data_path = weather_data_path
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
        self.update_fossil = update_fossil
        self.fossil_data = None
        self.weather = None
        self.save_dir = save_dir

    def get_current_data(self):
        print(f'reading existing data from: {self.path}...')
        self.current_data = pd.read_csv(self.path, index_col=0)
        self.current_data.index = pd.to_datetime(self.current_data.index, utc=True)
        self.current_data.index = self.current_data.index.tz_convert('Europe/Brussels')
        #self.current_data = self.current_data[self.valid_cols]
        self.timestamp_from = self.current_data.index[-1] + pd.Timedelta(hours=1)

    def update_entsoe(self):
        print(f'extracting electricity market data from ENTSO-e for bidding zone {country}...')
        prices = client.query_day_ahead_prices(start=self.timestamp_from, end=self.timestamp_to, country_code=country)
        fcst_load = client.query_load_forecast(start=self.timestamp_from, end=self.timestamp_to,
                                               country_code=country).resample('h').mean()
        load = client.query_load(start=self.timestamp_from, end=self.timestamp_to, country_code=country).resample(
            'h').mean()
        #data patch for the last 2 periods for load which are normally non existing
        idx_na = self.current_data.loc[self.current_data['actual_load'].isna()].index
        if len(idx_na) > 0:
            load_patch = client.query_load(start=self.timestamp_from - pd.Timedelta(hours=2), end=self.timestamp_from, country_code=country).resample(
            'h').mean()
            for i in idx_na:
                self.current_data.at[i,'actual_load'] = load_patch.loc[i]

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
        start_of_year = pd.to_datetime(f"{self.timestamp_from.year}-01-01").tz_localize('Europe/Brussels')
        y_cap = client.query_installed_generation_capacity(start=start_of_year, end=self.timestamp_to,
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

    def update_fossil_prices(self, ts_from, ts_to):

        idx = pd.date_range(start=ts_from, end=ts_to, freq='h')
        self.fossil_data = pd.DataFrame(columns=['oil_price', 'gas_price', 'coal_price', 'carbon_price'], index=idx)

        if self.update_fossil:
            #gas
            path_gas = os.path.join(self.fosil_prices_path, 'EEX-ETF-D_gas_price_history.csv')
            gas = pd.read_csv(path_gas, sep = ';', index_col = 0, decimal= ',')
            gas.index = pd.to_datetime(gas.index, dayfirst=True)
            gas = gas.loc[(gas.index.date >= idx[0].date() - pd.Timedelta(days=2)) & (gas.index.date <= idx[-1].date())]
            gas['price'] = gas['Open']
            gas = gas['price'].to_frame().rename(columns={'price': 'gas_price'})
            gas.index = gas.index.tz_localize('Europe/Brussels')
            gas = gas.reindex(idx, method='ffill')

            #oil
            path_oil = os.path.join(self.fosil_prices_path, 'LCOc1_oil_price_history.csv')
            oil = pd.read_csv(path_oil, sep=';', index_col=0, decimal=',')
            oil.index = pd.to_datetime(oil.index, dayfirst=True)
            oil = oil.loc[(oil.index.date >= idx[0].date() - pd.Timedelta(days=2)) & (oil.index.date <= idx[-1].date())]
            oil['price'] = (oil['Open'] + oil['Last']) / 2
            oil = oil['price'].to_frame().rename(columns={'price': 'oil_price'})
            oil.index = oil.index.tz_localize('Europe/Brussels')
            oil = oil.reindex(idx, method='ffill')

            #coal
            path_coal = os.path.join(self.fosil_prices_path, 'TRAPI2_coal_price_history.csv')
            coal = pd.read_csv(path_coal, sep=';', index_col=0, decimal=',')
            coal.index = pd.to_datetime(coal.index, dayfirst=True)
            coal = coal.loc[
                (coal.index.date >= idx[0].date() - pd.Timedelta(days=10)) & (coal.index.date <= idx[-1].date())]
            coal['price'] = coal['Last']
            coal = coal['price'].to_frame().rename(columns={'price': 'coal_price'})
            coal.index = coal.index.tz_localize('Europe/Brussels')
            coal = coal.reindex(idx, method='ffill')

            #emmissions
            path_emmissions = os.path.join(self.fosil_prices_path, 'CFI2Z4_carbon_emission_futures_history.csv')
            emmissions = pd.read_csv(path_emmissions, sep=',', index_col=0, decimal=',', header=None)
            emmissions.index = pd.to_datetime(emmissions.index).tz_localize('Europe/Brussels')
            emmissions = emmissions[1]
            emmissions.index[0]
            emmissions = emmissions.reindex(idx, method='ffill')

            self.fossil_data['oil_price'] = oil
            self.fossil_data['gas_price'] = gas
            self.fossil_data['coal_price'] = coal
            self.fossil_data['carbon_price'] = emmissions

        else:
            self.fossil_data['oil_price'] = self.current_data.loc[ts_from - pd.Timedelta(hours=1), 'oil_price']
            self.fossil_data['gas_price'] = self.current_data.loc[ts_from - pd.Timedelta(hours=1), 'gas_price']
            self.fossil_data['coal_price'] = self.current_data.loc[ts_from - pd.Timedelta(hours=1), 'coal_price']
            self.fossil_data['carbon_price'] = self.current_data.loc[ts_from - pd.Timedelta(hours=1), 'carbon_price']


    def update_weather(self):

        df_weather = pd.DataFrame()
        for file in os.listdir(self.weather_data_path):
            df = pd.read_csv(os.path.join(self.weather_data_path, file), index_col=0)
            df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d%H').tz_localize('UTC')
            df.index = df.index.tz_convert('Europe/Brussels')
            df = df[[i for i in df.columns if i.startswith('sunshine') or i.startswith('wind_speed')]]
            df = df.loc[self.timestamp_from:self.timestamp_to]
            df_weather = pd.concat((df_weather, df), axis=1)

        # replace missing values in sunshine variable
        for col in [col for col in df_weather.columns if col.startswith('sunshine')]:
            # Replace missing values with 0 if nighttime
            idx = df_weather[
                ((df_weather.index.hour < 5) | (df_weather.index.hour > 21)) & (df_weather[col].isna())].index
            df_weather.loc[idx, col] = 0
            # Replace missing values with average if suposedly not nighttime
            idx_ = df_weather[
                (df_weather.index.hour >= 5) & (df_weather.index.hour <= 21) & (df_weather[col].isna())].index
            for i in idx_:
                val = df_weather.loc[((df_weather.index.day == i.day) & (df_weather.index.month == i.month) & (
                            df_weather.index.hour == i.hour)), col].mean()
                df_weather.loc[i, col] = val

        # replace missing values in windspeed variable
        df_weather.replace(-999, np.nan, inplace=True)
        df_weather.interpolate(method='linear', inplace=True)
        df_weather.loc[df_weather[(df_weather < 0).any(axis=1)].index, 'sunshine_LeipzigHalle'] = 0
        self.weather = df_weather

    def adjust_nans(self):
        columns = ['cap_biomass', 'cap_wind_off', 'cap_wind_on', 'cap_solar', 'cap_coal_gas', 'actual_load']
        for col in columns:
            i = self.data[self.data[col].isna()].index[0] - pd.Timedelta(hours=1)
            val = self.data.loc[i, col]
            idxs = self.data[self.data[col].isna()].index
            self.data.loc[idxs, col] = val

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
        self.update_fossil_prices(ts_from = self.timestamp_from, ts_to = self.timestamp_to)
        self.update_weather()
        self.new_data = pd.DataFrame(index=pd.date_range(start=self.timestamp_from, end=self.timestamp_to, freq='h'))
        self.new_data = pd.concat((self.entsoe_data, self.capacities, self.fossil_data, self.weather, self.holidays, self.date_embedding), axis=1)
        self.new_data.rename(columns=column_mapping, inplace=True)
        self.new_data.index = self.new_data.index.tz_convert('Europe/Brussels')
        self.data = pd.concat((self.current_data, self.new_data))
        self.adjust_nans()
        if self.save_dir:
            self.save_data()

    def save_data(self):
        self.data.to_csv(self.save_dir)


if __name__ == '__main__':

    data_path = r'./dataset/multivar_dataset_310125.csv'
    fossil_prices_path = r'./fossil'
    weather_data_path = r'./weather'
    updater = DataUpdater(current_data_path=data_path,
                          fosil_prices_path=fossil_prices_path,
                          weather_data_path=weather_data_path,
                          update_fossil=False,
                          save_dir=r'./dataset/multivar_dataset_010225.csv',
                          timestamp_to=pd.Timestamp(pd.Timestamp('2025-02-02 23:00')))
    #updater.get_current_data()
    #updater.update_fossil_prices(ts_from = updater.timestamp_from, ts_to = updater.timestamp_to)
    updater.update_data()
    #data = updater.data
    #print(data.shape)


'''
time_now = pd.Timestamp(datetime.now())
saving_path = os.path.join(os.getcwd(),'datasets_snapshots',f'multivar_dataset_{time_now.strftime('%Y%m%d%H%M%S')}.csv')
updater.data.to_csv(saving_path)
print(f'file successfully saved to {saving_path}')
'''

