import pandas as pd
import numpy as np
import os
import inspect
from sklearn.preprocessing import FunctionTransformer

from models.LSTM_based.dataset_preparation import train_test_val_split
from models.LSTM_based.encoder_decoder_LSTM import EncoderDecoderAttentionLSTM


def sin_transformer(period):
	return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def run_encoder_decoder_attention_LSTM():
    # load dataset
    datasets_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data')
    df = pd.read_csv(datasets_path + '\\allData_cleaned.csv', index_col=0)
    prev_day_prices = df[:-24]['day_ahead_prices_EURO_x'].values
    df = df[24:]
    df['prev_range_prices'] = prev_day_prices
    df['hour_sin'] = sin_transformer(24).fit_transform(df['hour'].values*24)
    df['hour_cos'] = cos_transformer(24).fit_transform(df["hour"].values*24)

    df['day_of_week_sin'] = sin_transformer(7).fit_transform(df['day_of_week'].values*7)
    df['day_of_week_cos'] = cos_transformer(7).fit_transform(df["day_of_week"].values*7)

    df['month_sin'] = sin_transformer(12).fit_transform(df['month'].values*12)
    df['month_cos'] = cos_transformer(12).fit_transform(df["month"].values*12)
    # prepare dataset for LSTM
    # features = ['month', 'hour', 'day_of_week']
    # features = list(df.drop(['Date', 'day_ahead_prices_EURO_x'], axis=1).columns)
    features = ['E_load_forecast_MWh_x', 'E_solar_forecast_MWh_x_x', 'E_wind_forecast_MWh_x_x',
                'E_wind_forecast_MWh.1_x_x', 'E_solar_forecast_MWh_y_x', 'E_wind_forecast_MWh_y_x',
                'E_wind_forecast_MWh.1_y_x', 'E_crossborder_DK_2_actual_MWh_x', 'E_crossborder_SE_4_actual_MWh_x',
                'E_crossborder_DK_1_actual_MWh_x', 'E_crossborder_FR_actual_MWh_x', 'E_crossborder_CH_actual_MWh_x',
                'E_crossborder_NL_actual_MWh_x', 'E_crossborder_sum_actual_MWh_x', 'E_crossborder_CZ_actual_MWh_x',
                'E_crossborder_PL_actual_MWh_x', 'Brent Oil', 'Natural Gas', 'carbon_price_EURO', 'hour', 'month',
                'day_of_week', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',]
    #             'prev_range_prices']
    # features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']
    # features = ['hour', 'month', 'day_of_week']
    target = 'day_ahead_prices_EURO_x'

    # split into train, test, val
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(df, target_column=target)

    # define model
    model = EncoderDecoderAttentionLSTM(target_length=24, features=features, target=target)

    # train model
    training_history = model.train(X_train=X_train, y_train=y_train,
                                   X_val=X_val, y_val=y_val,
                                   X_test=X_test, y_test=y_test,
                                   n_epochs=500, batch_size=1024, learning_rate=0.001)

    # predict on model
    prediction_result = model.predict(X=X_test, exp_dir=None)

    pass


if __name__ == '__main__':
    run_encoder_decoder_attention_LSTM()
    pass
