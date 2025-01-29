import pandas as pd
import numpy as np
import os
import inspect
from sklearn.preprocessing import FunctionTransformer
import optuna
from plotly.io import show
import torch
import joblib

from models.LSTM_based.dataset_preparation import train_test_val_split, sin_transformer, cos_transformer
from models.LSTM_based.encoder_decoder_LSTM import EncoderDecoderAttentionLSTM
from models.LSTM_based.multivariate_LSTM import MultivariateBiLSTM
from benchmarking.benchmarker import BenchmarkMaker

torch.manual_seed(0)


def optimize_hyperparameters():
    # load dataset
    datasets_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data')
    df = pd.read_csv(datasets_path + '\\allData_cleaned.csv', index_col=0)
    prev_day_prices = df[:-24]['day_ahead_prices_EURO'].values
    df = df[24:]
    df['prev_range_prices'] = prev_day_prices
    df['hour_sin'] = sin_transformer(24).fit_transform(df['hour'].values)
    df['hour_cos'] = cos_transformer(24).fit_transform(df["hour"].values)

    df['day_of_week_sin'] = sin_transformer(7).fit_transform(df['weekday'].values)
    df['day_of_week_cos'] = cos_transformer(7).fit_transform(df["weekday"].values)

    df['month_sin'] = sin_transformer(12).fit_transform(df['month'].values)
    df['month_cos'] = cos_transformer(12).fit_transform(df["month"].values)
    # prepare dataset for LSTM
    # features = ['month', 'hour', 'day_of_week']
    # features = list(df.drop(['Date', 'day_ahead_prices_EURO_x'], axis=1).columns)
    features = ['Forecasted Load', 'E_solar_forecast_MWh', 'E_wind_forecast_MWh', 'E_wind_forecast_MWh.1',
                'E_crossborder_DK_2_actual_MWh', 'E_crossborder_SE_4_actual_MWh', 'E_crossborder_DK_1_actual_MWh',
                'E_crossborder_FR_actual_MWh', 'E_crossborder_CH_actual_MWh', 'E_crossborder_NL_actual_MWh',
                'E_crossborder_sum_actual_MWh', 'E_crossborder_CZ_actual_MWh', 'E_crossborder_PL_actual_MWh',
                'Oil WTI',
                'Natural Gas', 'Price_Calculated_EUR_MWh', 'actual_E_Total_Gridload_MWh',
                'actual_E_Residual_Load_MWh',
                'actual_E_Hydro_Pumped_Storage_MWh', 'forecast_E_Total_Gridload_MWh',
                'forecast_actual_E_Residual_Load_MWh', 'actual_generation_E_Biomass_MWh',
                'actual_generation_E_Hydropower_MWh', 'actual_generation_E_Windoffshore_MWh',
                'actual_generation_E_Windonshore_MWh', 'actual_generation_E_Photovoltaics_MWh',
                'actual_generation_E_OtherRenewable_MWh', 'actual_generation_E_Lignite_MWh',
                'actual_generation_E_HardCoal_MWh', 'actual_generation_E_FossilGas_MWh',
                'actual_generation_E_HydroPumpedStorage_MWh', 'actual_generation_E_OtherConventional_MWh',
                'forecast_generation_E_Total_MWh', 'forecast_generation_E_PhotovoltaicsAndWind_MWh',
                'forecast_generation_E_Windoffshore_MWh', 'forecast_generation_E_Windonshore_MWh',
                'forecast_generation_E_Photovoltaics_MWh', 'forecast_generation_E_Original_MWh',
                'E_NetherlandExport_corssBorderPhysical_MWh', 'E_NetherlandImport_corssBorderPhysical_MW',
                'E_DenmarkExport_corssBorderPhysical_MWh', 'E_Denmark_Import_corssBorderPhysical_MWh',
                'E_CzechrepublicExport_corssBorderPhysical_MWh', 'E_CzechrepublicImport_corssBorderPhysical_MWh',
                'E_SwedenExport_corssBorderPhysical_MWh', 'E_SwedenImportv_corssBorderPhysical_MWh',
                'E_AustriaExport_corssBorderPhysical_MWh', 'E_AustriaImport_corssBorderPhysical_MWh',
                'E_FranceExport_corssBorderPhysical_MWh', 'E_FranceImport_corssBorderPhysical_MWh',
                'E_PolandExport_corssBorderPhysical_MWh', 'E_PolandImport_corssBorderPhysical_MWh',
                'E_NetherlandExport_MWh', 'E_NetherlandImport_MW', 'E_SwitzerlandExport_MWh',
                'E_SwitzerlandImport_MWh',
                'E_DenmarkExport_MWh', 'E_Denmark_Import_MWh', 'E_CzechrepublicExport_MWh',
                'E_CzechrepublicImport_MWh',
                'E_LuxembourgExport_MWh', 'E_LuxembourgImport_MWh', 'E_SwedenExport_MWh', 'E_SwedenImport_MWh',
                'E_AustriaExport_MWh', 'E_AustriaImport_MWh', 'E_FranceExport_MWh', 'E_FranceImport_MWh',
                'E_PolandExport_MWh', 'E_PolandImport_MWh', 'superbowl_bool', 'oktoberfest_bool', 'berlinale_bool',
                'carnival_bool', 'carbon_price_EURO', 'wind_speed_ms_KoelnBonn_review',
                'wind_direction_degree_KoelnBonn_review', 'precipitationTotal_mm_KoelnBonn_review',
                'QN_9_KoelnBonn_review', 'T_temperature_C_KoelnBonn_review', 'humidity_Percent_KoelnBonn_review',
                'stationPressure_hPa_KoelnBonn_review', 'surfacePressure_hPa_KoelnBonn_review',
                'sunshine_min_Muenchen_review', 'wind_speed_ms_Muenchen_review',
                'wind_direction_degree_Muenchen_review', 'precipitationTotal_mm_Muenchen_review',
                'clouds_Muenchen_review', 'T_temperature_C_Muenchen_review', 'humidity_Percent_Muenchen_review',
                'stationPressure_hPa_Muenchen_review', 'surfacePressure_hPa_Muenchen_review', 'Covid factor',
                'month',
                'weekday', 'week_of_year', 'is_weekend', 'is_holiday', 'hour', 'prev_range_prices',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
    target = 'day_ahead_prices_EURO'

    # split into train, test, val
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(df, target_column=target)

    def objective(trial):
        # set hyperparameter search space
        hidden_size = trial.suggest_int("hidden_size", 16, 256, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 6)

        # define model
        model = EncoderDecoderAttentionLSTM(target_length=24, features=features, target=target,
                                            hidden_size=hidden_size, num_layers=num_layers)

        # train model
        training_history = model.train(X_train=X_train, y_train=y_train,
                                       X_val=X_val, y_val=y_val,
                                       X_test=X_test, y_test=y_test,
                                       n_epochs=500, batch_size=1024, learning_rate=0.001)
        min_test_loss = training_history['test loss'].values.min()

        return min_test_loss

    save_study_cb = SaveStudyCallback()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25, callbacks=[save_study_cb])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print('Best Trial: ')
    best_trial = study.best_trial
    for k, v in best_trial.params.items():
        print('     {}: {}'.format(k, v))

    best_params = best_trial.params

    fig = optuna.visualization.plot_parallel_coordinate(study, params=["learning_rate", "batch_size"])
    show(fig)
    return best_params


class SaveStudyCallback:
    def __init__(self):
        pass

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        i = trial.number
        joblib.dump(study, f"optuna_trials\\study_{i}.pkl")


def run_encoder_decoder_attention_LSTM():
    # load dataset
    datasets_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data')
    df = pd.read_csv(datasets_path + '\\allData_cleaned.csv', index_col=0)
    prev_day_prices = df[:-24]['day_ahead_prices_EURO'].values
    df = df[24:]
    df['prev_range_prices'] = prev_day_prices
    df['hour_sin'] = sin_transformer(24).fit_transform(df['hour'].values)
    df['hour_cos'] = cos_transformer(24).fit_transform(df["hour"].values)

    df['day_of_week_sin'] = sin_transformer(7).fit_transform(df['weekday'].values)
    df['day_of_week_cos'] = cos_transformer(7).fit_transform(df["weekday"].values)

    df['month_sin'] = sin_transformer(12).fit_transform(df['month'].values)
    df['month_cos'] = cos_transformer(12).fit_transform(df["month"].values)

    df['renew_total'] = df['E_solar_forecast_MWh'].values + df['E_wind_forecast_MWh'].values + df['E_wind_forecast_MWh.1'].values

    df['delta_renew_load'] = df['Forecasted Load'].values - df['renew_total'].values
    # prepare dataset for LSTM
    features = ['Forecasted Load', 'E_solar_forecast_MWh', 'E_wind_forecast_MWh', 'E_wind_forecast_MWh.1',
                'E_crossborder_DK_2_actual_MWh', 'E_crossborder_SE_4_actual_MWh', 'E_crossborder_DK_1_actual_MWh',
                'E_crossborder_FR_actual_MWh', 'E_crossborder_CH_actual_MWh', 'E_crossborder_NL_actual_MWh',
                'E_crossborder_sum_actual_MWh', 'E_crossborder_CZ_actual_MWh', 'E_crossborder_PL_actual_MWh', 'Oil WTI',
                'Natural Gas', 'Price_Calculated_EUR_MWh', 'actual_E_Total_Gridload_MWh', 'actual_E_Residual_Load_MWh',
                'actual_E_Hydro_Pumped_Storage_MWh', 'forecast_E_Total_Gridload_MWh',
                'forecast_actual_E_Residual_Load_MWh', 'actual_generation_E_Biomass_MWh',
                'actual_generation_E_Hydropower_MWh', 'actual_generation_E_Windoffshore_MWh',
                'actual_generation_E_Windonshore_MWh', 'actual_generation_E_Photovoltaics_MWh',
                'actual_generation_E_OtherRenewable_MWh', 'actual_generation_E_Lignite_MWh',
                'actual_generation_E_HardCoal_MWh', 'actual_generation_E_FossilGas_MWh',
                'actual_generation_E_HydroPumpedStorage_MWh', 'actual_generation_E_OtherConventional_MWh',
                'forecast_generation_E_Total_MWh', 'forecast_generation_E_PhotovoltaicsAndWind_MWh',
                'forecast_generation_E_Windoffshore_MWh', 'forecast_generation_E_Windonshore_MWh',
                'forecast_generation_E_Photovoltaics_MWh', 'forecast_generation_E_Original_MWh',
                'E_NetherlandExport_corssBorderPhysical_MWh', 'E_NetherlandImport_corssBorderPhysical_MW',
                'E_DenmarkExport_corssBorderPhysical_MWh', 'E_Denmark_Import_corssBorderPhysical_MWh',
                'E_CzechrepublicExport_corssBorderPhysical_MWh', 'E_CzechrepublicImport_corssBorderPhysical_MWh',
                'E_SwedenExport_corssBorderPhysical_MWh', 'E_SwedenImportv_corssBorderPhysical_MWh',
                'E_AustriaExport_corssBorderPhysical_MWh', 'E_AustriaImport_corssBorderPhysical_MWh',
                'E_FranceExport_corssBorderPhysical_MWh', 'E_FranceImport_corssBorderPhysical_MWh',
                'E_PolandExport_corssBorderPhysical_MWh', 'E_PolandImport_corssBorderPhysical_MWh',
                'E_NetherlandExport_MWh', 'E_NetherlandImport_MW', 'E_SwitzerlandExport_MWh', 'E_SwitzerlandImport_MWh',
                'E_DenmarkExport_MWh', 'E_Denmark_Import_MWh', 'E_CzechrepublicExport_MWh', 'E_CzechrepublicImport_MWh',
                'E_LuxembourgExport_MWh', 'E_LuxembourgImport_MWh', 'E_SwedenExport_MWh', 'E_SwedenImport_MWh',
                'E_AustriaExport_MWh', 'E_AustriaImport_MWh', 'E_FranceExport_MWh', 'E_FranceImport_MWh',
                'E_PolandExport_MWh', 'E_PolandImport_MWh', 'superbowl_bool', 'oktoberfest_bool', 'berlinale_bool',
                'carnival_bool', 'carbon_price_EURO', 'wind_speed_ms_KoelnBonn_review',
                'wind_direction_degree_KoelnBonn_review', 'precipitationTotal_mm_KoelnBonn_review',
                'QN_9_KoelnBonn_review', 'T_temperature_C_KoelnBonn_review', 'humidity_Percent_KoelnBonn_review',
                'stationPressure_hPa_KoelnBonn_review', 'surfacePressure_hPa_KoelnBonn_review',
                'sunshine_min_Muenchen_review', 'wind_speed_ms_Muenchen_review',
                'wind_direction_degree_Muenchen_review', 'precipitationTotal_mm_Muenchen_review',
                'clouds_Muenchen_review', 'T_temperature_C_Muenchen_review', 'humidity_Percent_Muenchen_review',
                'stationPressure_hPa_Muenchen_review', 'surfacePressure_hPa_Muenchen_review', 'Covid factor', 'month',
                'weekday', 'week_of_year', 'is_weekend', 'is_holiday', 'hour', 'prev_range_prices',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
    features2 = ['renew_total', 'delta_renew_load', 'Forecasted Load', 'E_solar_forecast_MWh', 'E_wind_forecast_MWh', 'E_wind_forecast_MWh.1',
                 'is_weekend', 'is_holiday',
                 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
                 'prev_range_prices']
    target = 'day_ahead_prices_EURO'

    # split into train, test, val
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(df, target_column=target)

    # df_pred_test = pd.read_csv('results_prediction.csv')
    # y_compare = y_test.copy(deep=True)[:df_pred_test.shape[0]]
    # y_compare['pred'] = df_pred_test['day_ahead_price_predicted'].values
    # import matplotlib.pyplot as plt
    # y_compare.plot()
    # plt.show(block=True)

    # define model
    model = EncoderDecoderAttentionLSTM(target_length=24, features=features, target=target,
                                        hidden_size=256, num_layers=6, use_attention=True)
    # 112, 6
    # train model
    training_history = model.train(X_train=X_train, y_train=y_train,
                                   X_val=X_val, y_val=y_val,
                                   X_test=X_test, y_test=y_test,
                                   n_epochs=1000, batch_size=2048, learning_rate=0.001)

    # store the model
    # model.custom_save(model.model, filename='BiEncDecAttLSTM.pth')

    # load the model
    # m = model.custom_load(filename='BiEncDecAttLSTM.pth')

    # create feature and target scalers in training data
    model.create_scalers(X_train, y_train)

    # predict on model
    prediction_train_result = model.predict(X=X_train, exp_dir=None).set_index('timestamp')
    prediction_test_result = model.predict(X=X_test, exp_dir=None).set_index('timestamp')
    prediction_val_result = model.predict(X=X_val, exp_dir=None).set_index('timestamp')

    BenchMaker = BenchmarkMaker(export_dir='result')
    BenchMaker.load_dataframes(predictions={'EncDecLSTM': prediction_test_result}, prices=y_test)
    BenchMaker.calc_errors()

    BenchMaker.plot_rmse_per_hour()
    BenchMaker.plot_mae_per_day()
    BenchMaker.plot_rmse_per_day()
    BenchMaker.plot_rmse_per_day()
    BenchMaker.plot_mae_per_day()
    BenchMaker.plot_compare_mae()
    BenchMaker.plot_compare_mape()
    BenchMaker.plot_compare_rmse()
    BenchMaker.plot_compare_predictions_daily()
    BenchMaker.plot_compare_predictions_hourly()
    pass


def run_multivariate_LSTM():
    # load dataset
    datasets_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data')
    df = pd.read_csv(datasets_path + '\\allData_cleaned.csv', index_col=0)
    prev_day_prices = df[:-24]['day_ahead_prices_EURO'].values
    df = df[24:]
    df['prev_range_prices'] = prev_day_prices
    df['hour_sin'] = sin_transformer(24).fit_transform(df['hour'].values)
    df['hour_cos'] = cos_transformer(24).fit_transform(df["hour"].values)

    df['day_of_week_sin'] = sin_transformer(7).fit_transform(df['weekday'].values)
    df['day_of_week_cos'] = cos_transformer(7).fit_transform(df["weekday"].values)

    df['month_sin'] = sin_transformer(12).fit_transform(df['month'].values)
    df['month_cos'] = cos_transformer(12).fit_transform(df["month"].values)
    # prepare dataset for LSTM
    features2 = ['Forecasted Load', 'E_solar_forecast_MWh', 'E_wind_forecast_MWh', 'E_wind_forecast_MWh.1',
                'E_crossborder_DK_2_actual_MWh', 'E_crossborder_SE_4_actual_MWh', 'E_crossborder_DK_1_actual_MWh',
                'E_crossborder_FR_actual_MWh', 'E_crossborder_CH_actual_MWh', 'E_crossborder_NL_actual_MWh',
                'E_crossborder_sum_actual_MWh', 'E_crossborder_CZ_actual_MWh', 'E_crossborder_PL_actual_MWh', 'Oil WTI',
                'Natural Gas', 'Price_Calculated_EUR_MWh', 'actual_E_Total_Gridload_MWh', 'actual_E_Residual_Load_MWh',
                'actual_E_Hydro_Pumped_Storage_MWh', 'forecast_E_Total_Gridload_MWh',
                'forecast_actual_E_Residual_Load_MWh', 'actual_generation_E_Biomass_MWh',
                'actual_generation_E_Hydropower_MWh', 'actual_generation_E_Windoffshore_MWh',
                'actual_generation_E_Windonshore_MWh', 'actual_generation_E_Photovoltaics_MWh',
                'actual_generation_E_OtherRenewable_MWh', 'actual_generation_E_Lignite_MWh',
                'actual_generation_E_HardCoal_MWh', 'actual_generation_E_FossilGas_MWh',
                'actual_generation_E_HydroPumpedStorage_MWh', 'actual_generation_E_OtherConventional_MWh',
                'forecast_generation_E_Total_MWh', 'forecast_generation_E_PhotovoltaicsAndWind_MWh',
                'forecast_generation_E_Windoffshore_MWh', 'forecast_generation_E_Windonshore_MWh',
                'forecast_generation_E_Photovoltaics_MWh', 'forecast_generation_E_Original_MWh',
                'E_NetherlandExport_corssBorderPhysical_MWh', 'E_NetherlandImport_corssBorderPhysical_MW',
                'E_DenmarkExport_corssBorderPhysical_MWh', 'E_Denmark_Import_corssBorderPhysical_MWh',
                'E_CzechrepublicExport_corssBorderPhysical_MWh', 'E_CzechrepublicImport_corssBorderPhysical_MWh',
                'E_SwedenExport_corssBorderPhysical_MWh', 'E_SwedenImportv_corssBorderPhysical_MWh',
                'E_AustriaExport_corssBorderPhysical_MWh', 'E_AustriaImport_corssBorderPhysical_MWh',
                'E_FranceExport_corssBorderPhysical_MWh', 'E_FranceImport_corssBorderPhysical_MWh',
                'E_PolandExport_corssBorderPhysical_MWh', 'E_PolandImport_corssBorderPhysical_MWh',
                'E_NetherlandExport_MWh', 'E_NetherlandImport_MW', 'E_SwitzerlandExport_MWh', 'E_SwitzerlandImport_MWh',
                'E_DenmarkExport_MWh', 'E_Denmark_Import_MWh', 'E_CzechrepublicExport_MWh', 'E_CzechrepublicImport_MWh',
                'E_LuxembourgExport_MWh', 'E_LuxembourgImport_MWh', 'E_SwedenExport_MWh', 'E_SwedenImport_MWh',
                'E_AustriaExport_MWh', 'E_AustriaImport_MWh', 'E_FranceExport_MWh', 'E_FranceImport_MWh',
                'E_PolandExport_MWh', 'E_PolandImport_MWh', 'superbowl_bool', 'oktoberfest_bool', 'berlinale_bool',
                'carnival_bool', 'carbon_price_EURO', 'wind_speed_ms_KoelnBonn_review',
                'wind_direction_degree_KoelnBonn_review', 'precipitationTotal_mm_KoelnBonn_review',
                'QN_9_KoelnBonn_review', 'T_temperature_C_KoelnBonn_review', 'humidity_Percent_KoelnBonn_review',
                'stationPressure_hPa_KoelnBonn_review', 'surfacePressure_hPa_KoelnBonn_review',
                'sunshine_min_Muenchen_review', 'wind_speed_ms_Muenchen_review',
                'wind_direction_degree_Muenchen_review', 'precipitationTotal_mm_Muenchen_review',
                'clouds_Muenchen_review', 'T_temperature_C_Muenchen_review', 'humidity_Percent_Muenchen_review',
                'stationPressure_hPa_Muenchen_review', 'surfacePressure_hPa_Muenchen_review', 'Covid factor', 'month',
                'weekday', 'week_of_year', 'is_weekend', 'is_holiday', 'hour', 'prev_range_prices',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
    features1 = ['Forecasted Load', 'E_solar_forecast_MWh', 'E_wind_forecast_MWh', 'E_wind_forecast_MWh.1',
                'is_weekend', 'is_holiday', 'prev_range_prices',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
    target = 'day_ahead_prices_EURO'

    # split into train, test, val
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(df, target_column=target)

    # df_pred_test = pd.read_csv('results_prediction.csv')
    # y_compare = y_test.copy(deep=True)[:df_pred_test.shape[0]]
    # y_compare['pred'] = df_pred_test['day_ahead_price_predicted'].values
    # import matplotlib.pyplot as plt
    # y_compare.plot()
    # plt.show(block=True)

    # define model
    model1 = MultivariateBiLSTM(features=features2, target=target)

    # train model
    # training_history = model1.train(X_train=X_train, y_train=y_train,
    #                                 X_val=X_val, y_val=y_val,
    #                                 X_test=X_test, y_test=y_test,
    #                                 n_epochs=200, batch_size=1024, learning_rate=0.001)

    model1.create_scalers(X_train, y_train)
    model1.custom_load('MultivarLSTM.pth')
    prediction1 = model1.run_prediction(X_val).set_index('timestamp')

    model2 = EncoderDecoderAttentionLSTM(target_length=24, features=features2, target=target,
                                         hidden_size=256, num_layers=6, use_attention=True)
    model2.custom_load(filename='BiEncDecAttLSTM.pth')
    model2.create_scalers(X_train, y_train)
    prediction2 = model2.run_prediction(X_val).set_index('timestamp')

    model3 = EncoderDecoderAttentionLSTM(target_length=24, features=features2, target=target,
                                         hidden_size=256, num_layers=6, use_attention=False)
    model3.custom_load(filename='BiEncDecLSTM.pth')
    model3.create_scalers(X_train, y_train)
    prediction3 = model3.run_prediction(X_val).set_index('timestamp')

    BenchMaker = BenchmarkMaker(export_dir='result')
    BenchMaker.load_dataframes(predictions={'MultivarLSTM': prediction1,
                                            'EncDecAttLSTM': prediction2,
                                            'EncDecLSTM': prediction3}, prices=y_val)
    BenchMaker.calc_errors()

    BenchMaker.plot_rmse_per_hour()
    BenchMaker.plot_mae_per_day()
    BenchMaker.plot_rmse_per_day()
    BenchMaker.plot_rmse_per_day()
    BenchMaker.plot_mae_per_day()
    BenchMaker.plot_compare_mae()
    BenchMaker.plot_compare_mape()
    BenchMaker.plot_compare_rmse()
    BenchMaker.plot_compare_predictions_daily()
    BenchMaker.plot_compare_predictions_hourly()

    pass


if __name__ == '__main__':
    run_multivariate_LSTM()
    # run_encoder_decoder_attention_LSTM()
    pass
