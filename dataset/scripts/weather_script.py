from weather import weather_prediction
from weather import weather_review
import time

import pandas as pd
import os

import concurrent.futures


station_folder = "./weather/stations"
output_folder = "./weather/"
computing_folder = "./weather/computing_folder"
stations_combined = "./weather/stations_combined"
data_collection_folder="../data_collection"
forecas_folder="../forecast"

def combine_historic(station_r, place): 
  #Kombiniere die Dateien paarweise
  try:
    file_r = os.path.join(station_folder, station_r, f"{station_r}_data_combined.csv")
    
    #Daten einlesen
    df_r = pd.read_csv(file_r)
    combined_df=df_r
    #Ausgabe-Dateiname
    output_file = os.path.join(stations_combined, f"{place}_review.csv")
    combined_df.to_csv(output_file, index=False)

    print(f"Kombiniert: {station_r} -> {output_file}")

  except FileNotFoundError as e:
    print(f"Datei nicht gefunden: {e}")
  except Exception as e:
    print(f"Fehler beim Verarbeiten von {station_r}: {e}")



def combine_all_stations():
  files = [f for f in os.listdir(stations_combined) if f.endswith('.csv')]

  #Umbennenen der Spalten nach Stationsnamen
  for file in files:
    file_path = os.path.join(stations_combined, file)
    df = pd.read_csv(file_path)
    #Extrahiere den Dateinamen
    file_name = os.path.splitext(file)[0]
    columname=[df.columns[0]] + [f'{col}_{file_name}' for col in df.columns[1:]]
    df.columns = columname
    print(f'Spalten umbennant für {file_name}')
    #station_column_filename = os.path.join(stations_combined, file_name)
    df.to_csv(file_path, index=False)

  #Verbinde alle DataFrames nebeneinander  
  all_data_frames = []
  for file in files:
    file_path = os.path.join(stations_combined, file)  
    
    #Lade Daten aus Datei und füge sie zur Liste
    try:
      df = pd.read_csv(file_path, delimiter=",", parse_dates=["date"], date_format="%Y%m%d%H")
      all_data_frames.append(df)
      print(f"Daten hinzugefügt von: {file_path}")
    except Exception as e:
      print(f"Fehler beim Laden der Datei {file}: {e}")
  
  #Wenn geladen wurden -> kombiniere
  if all_data_frames:
    combined_data = all_data_frames[0]
    for df in all_data_frames[1:]:
      #Test MESS_DATUM als Datum
      df["date"] = pd.to_datetime(df["date"], errors="coerce")                
      #Daten zusammenführen
      combined_data = pd.merge(combined_data, df, on=[  "date"], how="outer")

    #Sortieren und doppelte löschen
    combined_data = combined_data.sort_values(by=[  "date"]).drop_duplicates(subset=[  "date"], keep='last')

  #Speichern
  final_filename = os.path.join(data_collection_folder, f"weather.csv")
  combined_data.to_csv(final_filename, index=False)
  print(f"Alle kombinierten Daten gespeichert in: {final_filename}")

def start_combine_historic():
    max_workers = min(os.cpu_count(), len(station_ids_r))  #Maximal so viele Stationen wie vorhanden oder CPU Anzahl
    print(f"Starte die Verknüfung aller Daten für {len(station_ids_r)} Stationen mit {max_workers} Threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #Jede Station wird parallel heruntergeladen
        future_to_station = {executor.submit(combine_historic, station_r, place): (station_r,  place) for station_r,  place in zip(station_ids_r,  station_place) }    
        for future in concurrent.futures.as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                future.result()  #Funktion ausführen und Fehler abfangen
                print(f"Dateien verknüpft aller Daten für Station {station_id}.")
            except Exception as e:
                print(f"Fehler beim Verknüfung aller Daten für Station {station_id}: {e}")

def combine_forecast():

  files = [f for f in os.listdir(station_folder) if f.endswith('.csv')]

  #Umbennenen der Spalten nach Stationsnamen
  for file in files:
    file_path = os.path.join(station_folder, file)
    df = pd.read_csv(file_path)
    #Extrahiere den Dateinamen
    file_name = os.path.splitext(file)[0]
    columname=[df.columns[0]] + [f'{col}_{file_name}' for col in df.columns[1:]]
    df.columns = columname
    print(f'Spalten umbennant für {file_name}')
    #station_column_filename = os.path.join(stations_combined, file_name)
    df.to_csv(file_path, index=False)

  #Verbinde alle DataFrames nebeneinander  
  all_data_frames = []
  for file in files:
    file_path = os.path.join(station_folder, file)  
    
    #Lade Daten aus Datei und füge sie zur Liste
    try:
      df = pd.read_csv(file_path, delimiter=",", parse_dates=["date"], date_format="%Y%m%d%H")
      all_data_frames.append(df)
      print(f"Daten hinzugefügt von: {file_path}")
    except Exception as e:
      print(f"Fehler beim Laden der Datei {file}: {e}")
  
  #Wenn geladen wurden -> kombiniere
  if all_data_frames:
    combined_data = all_data_frames[0]
    for df in all_data_frames[1:]:
      #Test MESS_DATUM als Datum
      df["date"] = pd.to_datetime(df["date"], errors="coerce")                
      #Daten zusammenführen
      combined_data = pd.merge(combined_data, df, on=[  "date"], how="outer")

    #Sortieren und doppelte löschen
    combined_data = combined_data.sort_values(by=[  "date"]).drop_duplicates(subset=[  "date"], keep='last')


  final_filename = os.path.join(forecas_folder, f"weather_forecast.csv")
  combined_data.to_csv(final_filename, index=False)
  print(f"Kombinierter Forecast gespeichert in: {final_filename}")

def create_folder():
  os.makedirs(computing_folder, exist_ok=True)
  os.makedirs(stations_combined, exist_ok=True)
  for station in station_ids_r:
    output_folder_station = os.path.join(computing_folder, station)
    os.makedirs(output_folder_station, exist_ok=True)
    station_folder =os.path.join(output_folder,'stations',station)
    os.makedirs(station_folder, exist_ok=True)


start = time.time()
combine_historicforecast_bool =False
station_ids_r = [ "01262", "01975", "02667"]
station_ids_f = [ "10870", "10147", "10513"]
station_place = [ "Muenchen", "Hamburg", "KoelnBonn" ]

#Erstelle die Ordner
create_folder()

#Starte Rückblick-Download
weather_review.download_weather_data_for_all_stations(station_ids_r)

end = time.time()
verstrichene_zeit = end - start
print(f'Ausführungszeit: {verstrichene_zeit} Sekunden')

#Starte Vorhersagen-Download
weather_prediction.download_weatherforecast_data_for_all_stations(station_ids_f, station_place)
weather_prediction.remove_columns()

end = time.time()
verstrichene_zeit = end - start
print(f'Ausführungszeit: {verstrichene_zeit} Sekunden')

#Kombiniere die historischen und vorhergesagten Daten
start_combine_historic()
end = time.time()
verstrichene_zeit = end - start
print(f'Ausführungszeit: {verstrichene_zeit} Sekunden')
#Alle Stationen kombinieren
combine_all_stations()
combine_forecast()

end = time.time()
verstrichene_zeit = end - start
print(f'Ausführungszeit: {verstrichene_zeit} Sekunden')