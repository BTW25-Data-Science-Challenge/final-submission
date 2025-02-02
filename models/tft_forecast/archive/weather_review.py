import requests
import pandas as pd
import zipfile
import io
import re
import os

import concurrent.futures

#Basis-URL für die DWD Wetterdaten
base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"

#URL Endungen
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



#Speicherort
computing_folder = "./Data/Data/Weather/computing_folder"
output_folder = "./Data/Data/Weather/stations"

# Nicht benötigte Spalten   
columns_remove_clouds = ["STATIONS_ID","eor", "QN_8","V_N_I"]
columns_remove_pressure = ["STATIONS_ID","eor", "QN_8"]
columns_remove_sun = ["STATIONS_ID","eor", "QN_7"]
columns_remove_temp = ["STATIONS_ID","QN_9", "eor"]
columns_remove_wind = ["STATIONS_ID","eor", "QN_3"]
columns_remove_precipitation = ["STATIONS_ID","eor", "QN_8", "WRTR", "RS_IND"]

#Funktion zur Suche und Herunterladen der Wetterdaten pro Station
def get_weather_data_for_station(station_id):
    for data_type, endpoint in data_types.items():
        url = base_url + endpoint
        
        #Esrtellt Liste von Dateien im Verzeichnis
        response = requests.get(url)
        response.raise_for_status()

        #Suche nach passender ZIP-Datei
        for line in response.text.splitlines():
            if station_id in line and "zip" in line:
                filename = re.search(r'href="(.*?)"', line).group(1)
                file_url = url + filename
                
                #Lade ZIP-Datei herunter
                print(f"Lade herunter: {file_url}")
                file_response = requests.get(file_url)
                file_response.raise_for_status()
                
                #Entpacke ZIP-Datei und suche passender TXT-Datei in der ZIP
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
                        print(f"Keine TXT-Datei im erwarteten Format für Station {station_id} gefunden.")
                        continue  

                    #Wenn TXT-Datei gefunden wurde, lade sie in pandas
                    txt_filename = txt_files[0]
                    with z.open(txt_filename) as f:
                        #Test ob ladbar
                        try:
                            df = pd.read_csv(f, sep=";", encoding="utf-8")
                            if df.empty:
                                print(f"Warnung: Die Datei {txt_filename} ist leer.")
                            else:
                                print("Daten geladen für:", txt_filename)

                                #Ausgabeordener checken
                                os.makedirs(computing_folder, exist_ok=True)
                                os.makedirs(output_folder, exist_ok=True)
                                #Dateinamen nach Datenart setzen
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
                                
                                #Speichere TXT-Datei im angegebenen Ordner
                                output_filepath = os.path.join(computing_folder, new_filename)
                                df.to_csv(output_filepath, sep=";", encoding="utf-8", index=False)
                                print(f"Wetterdaten gespeichert unter: {output_filepath}")   
                                print(f"Die Datei wurde erfolgreich gespeichert unter: {os.path.abspath(output_filepath)}")
                        except Exception as e:
                            print(f"Fehler beim Laden der Datei {txt_filename}: {e}")

    return None  #Rückgabe, wenn keine Datei gefunden wird

#Funktion zum Herunterladen der Wetterdaten für alle angegebenen Stationen
def download_weather_data_for_all_stations(station_ids):
    max_workers = min(10, len(station_ids))  #Maximal 10 Threads oder so viele Stationen wie vorhanden
    print(f"Starte den Download für {len(station_ids)} Stationen mit {max_workers} Threads.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #Jede Station wird parallel heruntergeladen
        future_to_station = {executor.submit(get_weather_data_for_station, station_id): station_id for station_id in station_ids}        
        for future in concurrent.futures.as_completed(future_to_station):
            station_id = future_to_station[future]
            try:
                future.result()  #Funktion ausführen und Fehler abfangen
                print(f"Download abgeschlossen für Station {station_id}.")
            except Exception as e:
                print(f"Fehler beim Herunterladen von Daten für Station {station_id}: {e}")

def cut_historic_bevor_2015():
    station_files = [f for f in os.listdir(computing_folder) if re.match(r'(.+)_hist\.txt', f)]    
    for file in station_files:
        file_path = os.path.join(computing_folder, file)
        with open(file_path, 'r') as infile:
            lines = infile.readlines()
        
        #Filtert Zeilen nach 2015 sind
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

        #Schreibe Zeilen in die Datei zurück
        with open(file_path, 'w') as outfile:
            outfile.writelines(filtered_lines)
        print(f"Historisch bis 2015 gekürzt: {file}")


def combine_historic_recent(station_ids):
    for station_id in station_ids:
        #Suche nach Dateien für jeweilige Station
        station_files = [f for f in os.listdir(computing_folder) if re.match(r'(.+)_' + station_id + r'_(hist|recent)\.txt', f)]
        
        #Gruppiere Dateien nach Wettertyp und Station-ID
        file_pairs = {}
        for file in station_files:
            match = re.match(r'(.+)_' + station_id + r'_(hist|recent)\.txt', file)
            if match:
                wettertyp, period = match.groups()  #Wettertyp und Zeitraum
                key = f"{wettertyp}_{station_id}"
                if key not in file_pairs:
                    file_pairs[key] = {}
                file_pairs[key][period] = os.path.join(computing_folder, file)

        #Führe historische und aktuelle Daten zusammen
        for key, file_pair in file_pairs.items():
            if 'hist' in file_pair and 'recent' in file_pair:
                #Einlesen historische, aktuellen Daten
                hist_df = pd.read_csv(file_pair['hist'], delimiter=";")
                recent_df = pd.read_csv(file_pair['recent'], delimiter=";")
                hist_df["MESS_DATUM"] = pd.to_datetime(hist_df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")
                recent_df["MESS_DATUM"] = pd.to_datetime(recent_df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")

                #Kombinieren Daten und entferne Duplikaten
                combined_df = pd.concat([hist_df, recent_df]).drop_duplicates(subset=["MESS_DATUM"], keep='last')
                combined_df = combined_df.sort_values(by=["MESS_DATUM"])
                combined_df["MESS_DATUM"] = combined_df["MESS_DATUM"].dt.strftime("%Y%m%d%H")

                #Speichern unter kombinierten Namen
                combined_filename = os.path.join(computing_folder, f"{key}_combined.txt")
                combined_df.to_csv(combined_filename, sep=";", index=False)
                print(f"Kombinierte Datei gespeichert: {combined_filename}")
            else:
                print(f"Fehlende Datei für {key}: entweder historische oder aktuelle Datei fehlt.")

def combine_all_station_data(station_ids):
    for station_id in station_ids:
        #Suche nach Dateien mit dem Suffix "_combined" 
        combined_files = [f for f in os.listdir(computing_folder) if f.endswith(f"_{station_id}_combined.txt")]
        all_data_frames = []
        for file in combined_files:
            file_path = os.path.join(computing_folder, file)
            
            #Lade Daten aus Datei und füge sie zur Liste
            try:
                df = pd.read_csv(file_path, delimiter=";", parse_dates=["MESS_DATUM"], date_format="%Y%m%d%H")
                all_data_frames.append(df)
                print(f"Daten hinzugefügt von: {file_path}")
            except Exception as e:
                print(f"Fehler beim Laden der Datei {file}: {e}")
        
        #Wenn geladen wurden -> kombiniere
        if all_data_frames:
            combined_data = all_data_frames[0]
            for df in all_data_frames[1:]:
                #Test MESS_DATUM als Datum
                df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")                
                #Daten zusammenführen
                combined_data = pd.merge(combined_data, df, on=[  "MESS_DATUM"], how="outer")

            #Sortieren und doppelte löschen
            combined_data = combined_data.sort_values(by=[  "MESS_DATUM"]).drop_duplicates(subset=[  "MESS_DATUM"], keep='last')
            combined_data["MESS_DATUM"] = combined_data["MESS_DATUM"].dt.strftime("%Y%m%d%H")
            
            # Header ändern
            header_mapping = {
                "STATIONS_ID": "STATIONS_ID",
                "MESS_DATUM": "date",
                "V_N_I": "Wolken_Interp",
                "V_N": "clouds",
                "P": "stationPressure",
                "P0": "surfacePressure",
                "SD_SO": "sunshine",
                "TT_TU": "temperature",
                "RF_TU": "humidity",
                "F": "wind_speed",
                "D": "wind_direction",
                "R1": "precipitationTotal",
                "RS_IND": "precipitation_indicator"

            }
        
            combined_data.rename(columns=header_mapping, inplace=True)

            #Speichern in Datei
            final_filename = os.path.join(output_folder, f"{station_id}_data_combined.csv")
            combined_data.to_csv(final_filename, sep=",", index=False)
            print(f"Alle kombinierten Daten für Station {station_id} gespeichert in: {final_filename}")

        else:
            print(f"Keine kombinierten Dateien für Station {station_id} gefunden.")


def remove_columns():
    temp_files = [f for f in os.listdir(computing_folder) if f.startswith("temp_") and f.endswith(".txt")]
    clouds_files = [f for f in os.listdir(computing_folder) if f.startswith("clouds_") and f.endswith(".txt")]
    pressure_files = [f for f in os.listdir(computing_folder) if f.startswith("pressure_") and f.endswith(".txt")]
    sun_files = [f for f in os.listdir(computing_folder) if f.startswith("sun_") and f.endswith(".txt")]
    wind_files = [f for f in os.listdir(computing_folder) if f.startswith("wind_") and f.endswith(".txt")]
    precipitation_files = [f for f in os.listdir(computing_folder) if f.startswith("precipitation_") and f.endswith(".txt")]
    for file in clouds_files:
        file_path = os.path.join(computing_folder, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)            
            #Entferne die Spalten, ganz oben definiert
            df = df.drop(columns=[col for col in columns_remove_clouds if col in df.columns])            
            #Speichere modifizierte Datei
            df.to_csv(file_path, sep=";", index=False)
            print(f"Spalten aus {file} entfernt.")
        
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {file}: {e}")
    
    for file in pressure_files:
        file_path = os.path.join(computing_folder, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)            
            #Entferne die Spalten, ganz oben definiert
            df = df.drop(columns=[col for col in columns_remove_pressure if col in df.columns])            
            #Speichere modifizierte Datei
            df.to_csv(file_path, sep=";", index=False)
            print(f"Spalten aus {file} entfernt.")
        
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {file}: {e}")

    for file in sun_files:
        file_path = os.path.join(computing_folder, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)            
            #Entferne die Spalten, ganz oben definiert
            df = df.drop(columns=[col for col in columns_remove_sun if col in df.columns])            
            #Speichere modifizierte Datei
            df.to_csv(file_path, sep=";", index=False)
            print(f"Spalten aus {file} entfernt.")
        
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {file}: {e}")

    for file in temp_files:
        file_path = os.path.join(computing_folder, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)            
            #Entferne die Spalten, ganz oben definiert
            df = df.drop(columns=[col for col in columns_remove_temp if col in df.columns])            
            #Speichere modifizierte Datei
            df.to_csv(file_path, sep=";", index=False)
            print(f"Spalten aus {file} entfernt.")
        
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {file}: {e}")

    for file in wind_files:
        file_path = os.path.join(computing_folder, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)            
            #Entferne die Spalten, ganz oben definiert
            df = df.drop(columns=[col for col in columns_remove_wind if col in df.columns])            
            #Speichere modifizierte Datei
            df.to_csv(file_path, sep=";", index=False)
            print(f"Spalten aus {file} entfernt.")
        
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {file}: {e}")

    for file in precipitation_files:
        file_path = os.path.join(computing_folder, file)
        
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)            
            #Entferne die Spalten, ganz oben definiert
            df = df.drop(columns=[col for col in columns_remove_precipitation if col in df.columns])            
            #Speichere modifizierte Datei
            df.to_csv(file_path, sep=";", index=False)
            print(f"Spalten aus {file} entfernt.")
        
        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {file}: {e}")

#Wird nur ausgeführt, wenn scrpit direkt ausgeführt
if __name__ == "__main__":  
    #Liste der Stations-IDs
    station_ids = ["00722", "01262", "01975", "02667", "02932"]
    #station_ids = ["00722"] 

    #Starte Download
    download_weather_data_for_all_stations(station_ids)
    cut_historic_bevor_2015()
    remove_columns()
    combine_historic_recent(station_ids)
    combine_all_station_data(station_ids)

