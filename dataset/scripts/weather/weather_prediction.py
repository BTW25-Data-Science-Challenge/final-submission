import os
import requests
import json
import pandas as pd
from datetime import datetime

url = "https://dwd.api.proxy.bund.dev/v30/stationOverviewExtended"

headers = {
    "accept": "application/json"
}
computing_folder = "./weather/computing_folder"
output_folder = "./weather/stations"


columns_remove_forecast = ['isDay','dewPoint2m']


def get_weather_data_for_station(station_id, station_place):
    params = {
        "stationIds": station_id
    }
    #Anfrage vorbereiten
    request = requests.Request("GET", url, headers=headers, params=params)
    prepared_request = request.prepare()
    
    response = requests.Session().send(prepared_request)
    #Ausgabeordener checken
    #os.makedirs(computing_folder, exist_ok=True)
    #os.makedirs(output_folder, exist_ok=True)
    if response.status_code == 200:
        data = response.json()
        
        filename = os.path.join(computing_folder, f"weather_forecast_{station_place}.json")
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Die Wettervorhersage wurde in {filename} gespeichert.")
        
        #JSON-Daten laden und verarbeiten
        with open(filename) as file:
            data = json.load(file)
        
        for station_id, station_data in data.items():
            forecast_data = station_data["forecast1"]
            start_time = forecast_data["start"]
            time_step = forecast_data["timeStep"]

            date = [datetime.utcfromtimestamp((start_time + i * time_step) / 1000) for i in range(len(forecast_data["temperature"]))]
            
            variables = {
                "T_temperature_C": forecast_data.get("temperature", []),
                "T_temperature_standarddeviation_C": forecast_data.get("temperatureStd", []),
                "precipitationTotal_mm": forecast_data.get("precipitationTotal", []),
                "sunshine_min": forecast_data.get("sunshine", []),
                "dewPoint2m": forecast_data.get("dewPoint2m", []),
                "surfacePressure_hPa": forecast_data.get("surfacePressure", []),
                "humidity_Percent": forecast_data.get("humidity", []),
                "isDay_bool": forecast_data.get("isDay", []),
                #"icon": forecast_data.get("icon", []),
                #"icon1h": forecast_data.get("icon1h", [])
            }
            
            #Alle Listen auf gleiche Länge bringen
            max_length = max(len(date), *(len(values) for values in variables.values()))
            date.extend([None] * (max_length - len(date)))  # date auf max. Länge auffüllen
            for key, values in variables.items():
                variables[key].extend([None] * (max_length - len(values)))  # Werte-Listen auffüllen
            
            # DataFrame erstellen
            df = pd.DataFrame({
                "date": date,
                **variables
            })

            #DataFrame Temperatur von ZehntelGrad in Grad             
            df["T_temperature_C"] = df["T_temperature_C"].apply(lambda x: x / 10 if pd.notnull(x) else x)
            df["T_temperature_standarddeviation_C"] = df["T_temperature_standarddeviation_C"].apply(lambda x: x / 10 if pd.notnull(x) else x)
            df["surfacePressure_hPa"] = df["surfacePressure_hPa"].apply(lambda x: x / 10 if pd.notnull(x) else x)
            df["humidity_Percent"] = df["humidity_Percent"].apply(lambda x: x / 10 if pd.notnull(x) else x)

            #Date ins richte Foramt konvertieren
            df["date"] = df["date"].apply(lambda x: x.strftime("%Y%m%d%H"))

            df.to_csv(os.path.join(output_folder, f"weather_forecast_{station_place}.csv"), index=False)
            print(f"Die Wettervorhersage wurde in weather_forecast_{station_place}.csv konvertiert")
    else:
        print(f"Fehler bei der Anfrage: {response.status_code}")


def download_weatherforecast_data_for_all_stations(station_ids, station_places):
    for (station_id , station_place) in zip(station_ids, station_places):
        print(f"Starte den Download für Station {station_id}...")
        get_weather_data_for_station(station_id, station_place)
        print()

def remove_columns():
    print("Starte den Spaltenentfernumg")
    forecast_files = [f for f in os.listdir(output_folder) if f.startswith("weather_forecast_")]  
    print(f"File: {forecast_files}...")
    for file in forecast_files:
        print(f"Starte den Spaltenentfernumg für {file}...")
        file_path = os.path.join(output_folder, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=",", skipinitialspace=True)  
            print(f"Spalten im DataFrame: {list(df.columns)}")          
            #Entferne die Spalten, ganz oben definiert
            df = df.drop(columns=[col for col in columns_remove_forecast if col in df.columns])     
            #Speichere modifizierte Datei
            df.to_csv(file_path, sep=",", index=False)
            print(f"Spalten aus {file} entfernt.")
        
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {file}: {e}")

#Wird nur ausgeführt, wenn scrpit direkt ausgeführt
if __name__ == "__main__":
    station_ids_r = ["00722", "01262", "01975", "02667", "02932"]
    station_ids_f = ["10453", "10870", "10147", "10513", "10469"]
    station_place = ["Brocken", "Muenchen", "Hamburg", "KoelnBonn", "LeipzigHalle"]
    #station_ids = ["10453", "10870", "10147", "10513", "10469"]
    download_weatherforecast_data_for_all_stations(station_ids_f, station_place)
    remove_columns()