import sys
import pandas as pd # type: ignore
import numpy as np # type: ignore
from autogluon.timeseries import TimeSeriesPredictor # type: ignore
from autogluon.common import space # type: ignore
from datetime import datetime, timedelta
import os
from matplotlib import pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import time
import tensorflow as tf # type: ignore
import csv
from dateutil.relativedelta import relativedelta
from sklearn.metrics import *

def dataset_Setup(dataset):
    #CSV-Datei laden
    file_path = "/data/horse/ws/fewa833b-time-series-forecast/AutoGluon/Felix/allData.csv"
    #file_path = "../final-submission/merged_data/allData.csv"
    if dataset=='all':        
        loaded_dataframe = "AllData.csv"
        data = pd.read_csv(file_path)
    elif dataset=='less':       
        columns_to_load = [
            "Date", "Forecasted Load", "day_ahead_prices_EURO", "Actual Aggregated", "Oil WTI", 
            "Natural Gas", "Coal", "Uran", "actual_E_Total_Gridload_MWh", "actual_E_Residual_Load_MWh", 
            "actual_generation_E_Biomass_MWh", "actual_generation_E_Hydropower_MWh", "actual_generation_E_Windoffshore_MWh", 
            "actual_generation_E_Windonshore_MWh", "actual_generation_E_Photovoltaics_MWh", "actual_generation_E_OtherRenewable_MWh", 
            "actual_generation_E_Nuclear_MWh", "actual_generation_E_Lignite_MWh", "actual_generation_E_HardCoal_MWh", 
            "actual_generation_E_FossilGas_MWh", "actual_generation_E_HydroPumpedStorage_MWh", "actual_generation_E_OtherConventional_MWh", 
            "carbon_price_EURO", "wind_speed_ms_Hamburg_review", "T_temperature_C_Muenchen_review", 
            "sunshine_min_Muenchen_review", "month", "weekday", "week_of_year", "is_weekend", "is_holiday"
        ]
        loaded_dataframe = "lessData.csv"        
        data = pd.read_csv(file_path, usecols=columns_to_load)
        print(data.head())
    elif dataset=='dayhead':
        loaded_dataframe = "Dayahead only"
        df = pd.read_csv(file_path)
        data = df[["Date", "day_ahead_prices_EURO"]]
    else:
        print(f"Error: No valid 'dataset': {dataset}")
        sys.exit(1)
    data['Date'] = pd.to_datetime(data['Date'])
    data.rename(columns={'Date': 'timestamp'}, inplace=True)
    return loaded_dataframe, file_path, data

def correlation_calculation(data, output_folder_autogluon):
    os.makedirs(output_folder_autogluon, exist_ok=True)
    data_copy = data.copy()
    data_copy = data_copy.select_dtypes(include=[np.number])
    correlation_matrix = data_copy.corr()
    print(correlation_matrix)
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Extra Features")
    plt.savefig(f"{output_folder_autogluon}/correlation.png")

    corr_matrix = correlation_matrix 
    threshold = 0.95
    high_corr_pairs = []
    for col in corr_matrix.columns: # Loop through the correlation matrix to find pairs with correlation above the threshold
        high_corr = corr_matrix.index[corr_matrix[col] > threshold].tolist()
        high_corr = [x for x in high_corr if x != col]
        for pair in high_corr:
            high_corr_pairs.append((col, pair, corr_matrix[col][pair]))

    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Column 1', 'Column 2', 'Correlation'])

    high_corr_df.to_csv(f"{output_folder_autogluon}/high_correlation_pairs.csv", index=False)
    print("High correlation pairs saved to 'high_correlation_pairs.csv'")

    missing_percentage = data.isnull().mean() * 100
    print("Missing values percentage per column:")
    print(missing_percentage)

    missing_df = pd.DataFrame(missing_percentage)
    missing_df.to_csv(f"{output_folder_autogluon}/missing.csv", index=True)

    high_missing_cols = missing_percentage[missing_percentage > 50].index #more than 50% are missing
    print("Columns with more than 50% missing values:", high_missing_cols) 

def set_known_covariates(data):    
    known_covariates_columns = ['item_id','month','weekday','week_of_year','is_weekend','is_holiday','superbowl_bool','oktoberfest_bool','berlinale_bool','precipitationTotal_mm_KoelnBonn_review', 'sunshine_min_KoelnBonn_review','stationPressure_hPa_KoelnBonn_review', 'surfacePressure_hPa_KoelnBonn_review','T_temperature_C_KoelnBonn_review', 'humidity_Percent_KoelnBonn_review','wind_speed_ms_KoelnBonn_review', 'wind_direction_degree_KoelnBonn_review', 'clouds_KoelnBonn_review','T_temperature_C_Hamburg_review', 'humidity_Percent_Hamburg_review', 'stationPressure_hPa_Hamburg_review','surfacePressure_hPa_Hamburg_review', 'wind_speed_ms_Hamburg_review', 'wind_direction_degree_Hamburg_review','clouds_Hamburg_review', 'precipitationTotal_mm_Hamburg_review', 'sunshine_min_Hamburg_review', 'precipitationTotal_mm_Muenchen_review', 'sunshine_min_Muenchen_review', 'stationPressure_hPa_Muenchen_review','surfacePressure_hPa_Muenchen_review', 'T_temperature_C_Muenchen_review', 'humidity_Percent_Muenchen_review','clouds_Muenchen_review', 'wind_speed_ms_Muenchen_review', 'wind_direction_degree_Muenchen_review', 'Covid factor']

    existing_columns = []
    for col in known_covariates_columns:
        if col in data.columns:
            existing_columns.append(col)  
        else:
            print(f"Warning: Column '{col}' is missing and will be ignored.")

    known_covariates = data[['timestamp'] + existing_columns]
    return known_covariates

def train_autogluon(set_preset,set_time_limit,dataset,provide_known_covariables,output_folder_autogluon,loaded_dataframe, file_path, data):
    print(f"Zeitlimit:{set_time_limit}")
    print(f"Dataset: {dataset}")
    print(f"Preset: {set_preset}")
    #Zeitstempel beim Start des Skripts erstellen
    start_time_script = time.time()
    script_start_time = datetime.now().strftime('%Y.%m.%d-%H.%M')
    add_path=f"{script_start_time}_{set_preset}_{dataset}"
    output_folder_train_ag=os.path.join(output_folder_autogluon, add_path)
    #output_folder = f"/data/horse/ws/fewa833b-time-series-forecast/AutoGluon/Felix/AG-Preset/{script_start_time}_{set_preset}_{dataset}"
    os.makedirs(output_folder_train_ag, exist_ok=True)  #Ordner erstellen, falls er nicht existiert
    
        
    data['item_id'] = 'item_1'    
    required_columns = ['timestamp', 'day_ahead_prices_EURO', 'item_id']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Spalte '{col}' fehlt in der Datei {file_path}.")
        
    if provide_known_covariables:
        known_covariates=set_known_covariates(data)  

    
    train_data =data[(data['timestamp'] > '2015-01-04') & (data['timestamp'] < '2023-12-01')]
    val_data = data[(data['timestamp'] >= '2023-12-01') & (data['timestamp'] < '2024-06-01')]
    test_data = data[(data['timestamp'] >= '2024-06-01') & (data['timestamp'] < '2024-12-01')]

    print(f"Trainingsdaten bis 01.12.2023: {train_data.shape[0]} Zeilen")
    print(f"Validierungsdaten bis 01.12.2023: {val_data.shape[0]} Zeilen")
    print(f"Testdaten ab 01.06.2024: {test_data.shape[0]} Zeilen")

    
    if train_data.empty or val_data.empty or test_data.empty:
        raise ValueError("Eine der Datenaufteilungen (Train, Validation, Test) ist leer.")

    
    train_data.to_csv(f"{output_folder_train_ag}/train_data.csv", index=False)
    val_data.to_csv(f"{output_folder_train_ag}/val_data.csv", index=False)
    test_data.to_csv(f"{output_folder_train_ag}/test_data.csv", index=False)

    autogluon_path = os.path.join(output_folder_train_ag, "AutogluonModels")
    predictor = TimeSeriesPredictor(
        target="day_ahead_prices_EURO",  
        prediction_length=24,
        path=autogluon_path,                   
        eval_metric="MAE"             
    )
    if set_preset=='best_quality' or set_preset=='fast_training' or set_preset=='high_quality' or set_preset=='medium_quality':
        predictor.fit(train_data=train_data, 
                    tuning_data=val_data,            
                    presets=set_preset,
                    time_limit=set_time_limit
                    )
    elif set_preset=='hp1':
        predictor.fit(train_data=train_data, 
                        tuning_data=val_data,
                        time_limit=set_time_limit,            
                        hyperparameters = {
                            'ADIDA':{},
                            'AutoARIMA':{}, 
                            'AutoCES':{}, 
                            'AutoETS':{}, 
                            'Croston':{}, 
                            'IMAPA':{}, 
                            'NPTS':{},                  
                            'Chronos': {
                                'num_epochs': 2000,        
                                'batch_size': 64,          
                            },
                            'DeepAR': {
                                'num_lstm_layers': 4,      
                                'num_lstm_units': 256,     
                                'dropout_rate': 0.3,       
                                'learning_rate': 1e-4,     
                                'num_epochs': 2000,        
                                'batch_size': 64,          
                            },
                            'DirectTabular': {
                                'num_layers': 10,           
                                'hidden_size': 512,        
                                'dropout_rate': 0.3,       
                            },
                            'DLinear': {
                                'num_layers': 10,          
                                'hidden_size': 512,       
                                'dropout_rate': 0.3,       
                            },
                            'PatchTST': {
                                'num_layers': 10,           
                                'hidden_size': 512,        
                                'dropout_rate': 0.3,       
                            },
                            'RecursiveTabular': {
                                'num_layers': 10,           
                                'hidden_size': 512,        
                                'dropout_rate': 0.3,       
                            },
                            'SimpleFeedForward': {
                                'num_layers': 10,           
                                'hidden_size': 512,        
                                'dropout_rate': 0.3,       
                            },
                            'TemporalFusionTransformer': {
                                'num_encoder_layers': 6,  
                                'num_decoder_layers': 6,
                                'attention_heads': 16,
                                'hidden_size': 512,        
                                'dropout_rate': 0.3,       
                            },
                            'TiDE': {
                                'num_layers': 10,           
                                'hidden_size': 512,        
                                'dropout_rate': 0.3,       
                            },
                            'WaveNet': {
                                'num_layers': 8,           
                                'hidden_size': 512,        
                                'dropout_rate': 0.3,       
                            }
                        }
            )
    
    else: 
        print('No preset found')
    
    #Check Feature Importance
    try:
        importance = predictor.feature_importance(data=val_data)
        importance_df = pd.DataFrame(importance)
        importance_df.to_csv(f"{output_folder_train_ag}/feature_importance.csv", index=True)
        print("Feature Importance saved.")
    except ValueError as e:
        print("Feature importance could not be computed:", e)
    
    start_time = test_data["timestamp"].min()
    end_time = test_data["timestamp"].max()
    print(f"Starttime: {start_time}, endtime: {end_time}")

    predictions = []
    current_time = start_time

    while current_time <= end_time:
        current_day_data = data[            
            (data["timestamp"] < current_time)
        ]
        if not current_day_data.empty:
            if provide_known_covariables:
                predicted_values = predictor.predict(current_day_data,known_covariates=known_covariates)
            else:
                predicted_values = predictor.predict(current_day_data)
            print(f"Predicted Values: {predicted_values}")            
            if "mean" in predicted_values:
                mean_predicted_values = predicted_values["mean"].values
            else:
                raise ValueError("'mean' column not found in predictions")
            next_24_hours = [current_time + timedelta(hours=i) for i in range(24)]
            print(f"Next 24 timestamps: {next_24_hours}")            
            if len(mean_predicted_values) != len(next_24_hours):
                raise ValueError("Mismatch between predicted values and generated timestamps")        
            predictions.extend([
                {"timestamp": ts, "predicted": pred}
                for ts, pred in zip(next_24_hours, mean_predicted_values)
            ])        
        current_time += timedelta(days=1)

    if predictions:
        final_predictions = pd.DataFrame(predictions)
        final_predictions["timestamp"] = pd.to_datetime(final_predictions["timestamp"])  # Sicherstellen, dass der Timestamp korrekt ist
        final_predictions.to_csv(f"{output_folder_train_ag}/results_data.csv", index=False)
    else:
        print("Keine Vorhersagen konnten gemacht werden.")
        final_predictions = pd.DataFrame(columns=["timestamp", "predicted"])

    test_data_comparison = test_data[['timestamp', 'day_ahead_prices_EURO']].rename(columns={'day_ahead_prices_EURO': 'actual_price'})
    test_data_comparison['timestamp'] = pd.to_datetime(test_data_comparison['timestamp'])

    comparison = pd.merge(test_data_comparison, final_predictions, on='timestamp', how='outer')
    comparison.to_csv(f"{output_folder_train_ag}/comparison.csv", index=False)


    y_true = comparison["actual_price"]
    y_pred = comparison["predicted"]
    mae_sklearn= mae_sklearn = mean_absolute_error(y_true, y_pred)
    rmse_sklearn = root_mean_squared_error(y_true, y_pred)    

    print(f"Alle Ergebnisse wurden im Ordner '{output_folder_train_ag}' gespeichert.")
    duration_skript=  time.time() - start_time_script
    print(f"Dauer: {duration_skript}")
    filename_mape = f"MAE{mae_sklearn:.2f}.txt"
    file_path_mape = os.path.join(output_folder_train_ag, filename_mape)
    with open(file_path_mape, "w") as file:
        file.write(f"MAE SKlearn: {mae_sklearn}\n")
        file.write(f"RMSE SKlearn: {rmse_sklearn}\n")
        file.write(f"Modelpreset: {set_preset}\n")
        file.write(f"Dataset: {dataset}\n")
        file.write(f"Duration: {duration_skript}")
        file.write(f"Knowncoavariables: {provide_known_covariables}")
    print(f"Ergebnisse wurden in die Datei '{file_path_mape}' gespeichert.")

    data = {
        "Dataset": dataset,
        "Modelpreset": set_preset,
        "Knowncoavariables": provide_known_covariables,
        "MAE SKlearn": mae_sklearn,
        "RMSE SKlearn": rmse_sklearn,
        "Duration": duration_skript,        
        "Date":script_start_time,
    }

    #write to csv
    file_path_csv =os.path.join(output_folder_autogluon, "preset-comparison.csv")
    file_exists = os.path.isfile(file_path_csv)
    with open(file_path_csv, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            #set header if new file
            writer.writeheader()
        #write data
        writer.writerow(data)
    print(f"Ergebnisse wurden in die Datei '{file_path_csv}' gespeichert.")

#only if sheduled directly
if __name__ == "__main__": 
    preset_list=["high_quality","best_quality","fast_training","medium_quality","hp1"]   
    set_time_limit=100
    output_folder_autogluon = f"/data/horse/ws/fewa833b-time-series-forecast/AutoGluon/Felix/AllInOne"

    #Start baseline    
    dataset='dayhead'
    loaded_dataframe, file_path, data=dataset_Setup(dataset)
    correlation_output_folder=os.path.join(output_folder_autogluon,"Correlation", "Baseline")
    correlation_calculation(data, correlation_output_folder)      
    provide_known_covariables=None
    set_preset="fast_training"
    train_autogluon(set_preset,set_time_limit,dataset,provide_known_covariables,output_folder_autogluon,loaded_dataframe, file_path, data)

    #start all-data
    dataset='all'
    correlation_output_folder=os.path.join(output_folder_autogluon,"Correlation", "Less-Data")
    correlation_calculation(data, correlation_output_folder) 
    for set_preset in preset_list:
        provide_known_covariables=None
        train_autogluon(set_preset,set_time_limit,dataset,provide_known_covariables,output_folder_autogluon,loaded_dataframe, file_path, data)

        provide_known_covariables=True
        train_autogluon(set_preset,set_time_limit,dataset,provide_known_covariables,output_folder_autogluon,loaded_dataframe, file_path, data)


    #start less-data
    dataset="less"
    correlation_output_folder=os.path.join(output_folder_autogluon,"Correlation", "Less-Data")
    correlation_calculation(data, correlation_output_folder)  
    for set_preset in preset_list:
        provide_known_covariables=None
        train_autogluon(set_preset,set_time_limit,dataset,provide_known_covariables,output_folder_autogluon,loaded_dataframe, file_path, data)

        provide_known_covariables=True
        train_autogluon(set_preset,set_time_limit,dataset,provide_known_covariables,output_folder_autogluon,loaded_dataframe, file_path, data)


    
    