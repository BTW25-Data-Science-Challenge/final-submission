import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import pandas as pd
import os

def directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

# gets data from yahoo finance with the given url, the filename and resource have to be put in
def get_Data(url, filename, resource):
    directory_exists(filename)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')

    # Extract headers
    headers = [header.text.strip() for header in table.find_all('th')]

    # Close column = Value for end of the day, the rest of the columns are not needed
    close_column_index = next(i for i, header in enumerate(headers) if header.lower().startswith('close'))

    # Extract data from the table
    data = []
    for row in table.find_all('tr'):
        columns = row.find_all('td')
        if columns:
            # Get correcct Date format
            date = columns[0].text.strip()
            date = datetime.strptime(date, '%b %d, %Y').strftime('%Y-%m-%d')
            close_value = columns[close_column_index].text.strip()
            data.append([date, close_value])


    # data is in the wrong order, put it from earliest to latest
    data.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))

    # Save the data with headers as a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', resource])
        writer.writerows(data)

    print("Data saved as ",filename)


url_brent = 'https://finance.yahoo.com/quote/BZ%3DF/history/?period1=1420070400&period2=1734903292&guccounter=1'
url_gas = 'https://finance.yahoo.com/quote/NG%3DF/history/?period1=1420070400&period2=1734905333'
url_coal = 'https://finance.yahoo.com/quote/MTFZ24.NYM/history/?period1=1420070400&period2=1734905597'

get_Data(url_brent, '../final-submission/merged_data/data_collection/brent_oil.csv', 'Brent Oil')
get_Data(url_gas, '../final-submission/merged_data/data_collection/naturalGas.csv', 'Natural Gas')
get_Data(url_coal, '../final-submission/merged_data/data_collection/coal.csv', 'Coal')



#the data is missing hour, as it is only daily, fills weekend gaps also
def fill_missing_hours(csv):
    df = pd.read_csv(csv)

    value_Name = df.columns[1]

    # Manually parse the 'date' column using the correct format (DD.MM.YY)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Date'] = df['Date'].dt.normalize()
    df.set_index('Date', inplace=True)

    # start 2015
    full_hourly_range = pd.date_range(start='01.01.2015', end=df.index.max() + pd.Timedelta(days=1), freq='H')[:-1]

    # put prefered null value here
    df_full = df.reindex(full_hourly_range, fill_value=pd.NA)
    df_full.reset_index(inplace=True)
    df_full.rename(columns={'index': 'Date'}, inplace=True)
    df_full[value_Name] = df_full.groupby(df_full['Date'].dt.floor('D'))[value_Name].transform(lambda group: group.ffill().bfill())

    # fills emptys
    df_full[value_Name].fillna('', inplace=True)
    df_full.to_csv(csv, index=False)
    print('Missing Hours Filled: ', csv)


fill_missing_hours('../final-submission/merged_data/data_collection/brent_oil.csv')
fill_missing_hours('../final-submission/merged_data/data_collection/naturalGas.csv')
fill_missing_hours('../final-submission/merged_data/data_collection/coal.csv')

df1 = pd.read_csv('../final-submission/merged_data/data_collection/brent_oil.csv')
df2 = pd.read_csv('../final-submission/merged_data/data_collection/naturalGas.csv')
df3 = pd.read_csv('../final-submission/merged_data/data_collection/coal.csv')

merged_df = pd.merge(df1, df2, on='Date', how='outer')
merged_df = pd.merge(merged_df, df3, on='Date', how='outer')

merged_df.to_csv('../final-submission/merged_data/data_collection/merged_data.csv', index=False)

print("CSV files have been merged and saved.")