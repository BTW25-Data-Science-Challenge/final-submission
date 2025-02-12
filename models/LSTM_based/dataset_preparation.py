from xgboost import XGBRegressor
import os
import inspect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import time
from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))


def fill_missing_with_xgboost(df, target_column):
    """
    Fills nan values in the target_column of the DataFrame using an XGBoost regressor trained
    with the other columns as features.

    :param df: input dataframe
    :param target_column: column which contains nan-values, that needs to be replaces
    :return: dataframe without nan-values in target column
    """
    # separate the target column
    target = df[target_column]

    # identify rows with and without NaN in the target column
    missing_mask = target.isna()
    complete_mask = ~missing_mask

    # split the data into training (non-nan) and prediction (nan)
    X_train = df.loc[complete_mask].drop(columns=[target_column])
    y_train = target[complete_mask]
    X_predict = df.loc[missing_mask].drop(columns=[target_column])

    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train, y_train, test_size=0.05, random_state=42
    )
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # train the XGBoost model
    model = XGBRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        max_depth=10,
        random_state=42,
        verbosity=0,
        device=device
    )
    model.fit(X_train_split, y_train_split,
              eval_set=[(X_valid_split, y_valid_split)],
              verbose=False)

    # predict the missing values
    predictions = model.predict(X_predict)

    # fill the missing values in the original DataFrame
    df.loc[missing_mask, target_column] = predictions

    return df


def has_long_nan_streak(series, threshold):
    """Detect long continuous nan-value sequences in a series

    :param series: series of floats, may contain nan
    :param threshold: maximum continuous nan values in sequence in a row
    :return: True if less that threshold nan-values in a row at any point in the series, otherwise False
    """
    max_nan_streak = (series.isna().astype(int)
                      .groupby(series.notna().astype(int).cumsum())
                      .transform('sum').max())
    return max_nan_streak > threshold


def run_preprocessing(df, nan_streak_threshold, start_ts, end_ts):
    time_n = time.time()
    # df = df.drop(['date', 'End_Date'], axis=1)
    # df['is_holiday'] = df['is_holiday'].values.astype(int)
    # df['is_weekend'] = df['is_weekend'].values.astype(int)

    unique_df = df.loc[:, ~df.T.duplicated()]

    # unique_df['Date'] = df.index.values
    unique_df['Date'] = pd.DatetimeIndex(unique_df.index.values)
    unique_df = unique_df.set_index('Date')
    start_date = pd.Timestamp('2018-10-01 00:00:00')
    end_date = pd.Timestamp('2025-02-11 23:00:00')
    unique_df = unique_df[start_date:end_date]

    unique_df = unique_df.reset_index()
    timestamps = unique_df['Date']
    unique_df = unique_df.drop(['Date'], axis=1)

    threshold = 168     # 24h * 7 = 7days

    columns_to_keep = ~unique_df.apply(has_long_nan_streak, threshold=threshold, axis=0)
    no_nan_streaks_df = unique_df.loc[:, columns_to_keep]

    features = list(no_nan_streaks_df.columns)
    new_df = no_nan_streaks_df.copy(deep=True)
    # new_df = unique_df.copy(deep=True).drop(['Network security of the TSOs [€] Calculated resolutions'], axis=1)
    features = list(new_df.columns)
    for i, c in enumerate(features):
        if new_df[c].isna().sum() > 0:
            new_df = fill_missing_with_xgboost(new_df, target_column=c)
        print(f'feature {i+1}: {c} done after: {int((time.time() - time_n)/60)}')
        # plot_filled_values(df=new_df, target_column=c, missing_mask=no_nan_streaks_df[c].isna())

    new_df['Date'] = timestamps
    new_df['hour'] = new_df['Date'].dt.hour

    datasets_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data\\lstm_small_subset')
    new_df.to_csv(datasets_path + '\\small_subset_lstm_cleaned.csv')

    print(f'preprocessing done in: {int((time.time() - time_n)/60)}')

    return new_df


def train_test_val_split(df, target_column):
    """Split dataset on the in project determined timeranges for train, test, validation.

    :param df: full input dataframe
    :param target_column: column to predict/forecast later on
    :returns: dataframes for X_train, y_train, X_test, y_test, X_val, y_val
    """
    df['Date'] = pd.DatetimeIndex(df['Date'].values)
    df = df.set_index('Date')
    train_df = df[:pd.Timestamp('2023-11-30 23:00:00')]
    val_df = df[pd.Timestamp('2023-12-01 00:00:00'):pd.Timestamp('2024-05-31 23:00:00')]
    test_df = df[pd.Timestamp('2024-06-01 00:00:00'):pd.Timestamp('2024-11-30 23:00:00')]

    X_train = train_df.drop(target_column, axis=1)
    X_val = val_df.drop(target_column, axis=1)
    X_test = test_df.drop(target_column, axis=1)
    y_train = train_df[[target_column]]
    y_val = val_df[[target_column]]
    y_test = test_df[[target_column]]

    return X_train, y_train, X_test, y_test, X_val, y_val


def plot_filled_values(df, target_column, missing_mask):
    """Plots the DataFrame, marking filled (previously nan) values.

    :param df: dataframe with filled nan-values in target column
    :param target_column: column contained nan-values in the original dataframe
    :param missing_mask: mask positions indication nan-values in target column from original dataframe
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(df.shape[0])

    # plot all values in the column
    plt.plot(x, df[target_column], label='Filled Values', color='blue')

    # highlight the originally missing nan-values
    plt.scatter(
        x[missing_mask],
        df.loc[missing_mask, target_column],
        color='red',
        label='Originally nan',
        zorder=5,
        marker='x',
        s=100
    )

    plt.title(f"Filled Missing Values in '{target_column}'")
    plt.xlabel('Index')
    plt.ylabel(target_column)
    plt.legend()
    plt.show(block=True)


def combine_df(dir):
    dfs = []
    for name in os.listdir(dir):
        filename = dir + f'\\{name}'
        dfs.append(pd.read_csv(filename).set_index('timestamp').drop(['Unnamed: 0'], axis=1))
    result_df = dfs[0]
    for df in dfs[1:]:
        result_df = result_df.join(df, how='outer')
    result_df = result_df.dropna(subset=['hour'])
    return result_df


def renew_features(df):
    df['renew_total'] = df['Solar'].values + df['Wind Offshore'].values + df['Wind Onshore'].values

    df['delta_renew_load'] = df['Forecasted Load'].values - df['renew_total'].values

    return df


def add_time_features(df):
    prev_day_prices = df[:-24]['day_ahead_prices'].values
    df = df[24:]
    df['prev_range_prices'] = prev_day_prices
    df['hour_sin'] = sin_transformer(24).fit_transform(df['hour'].values)
    df['hour_cos'] = cos_transformer(24).fit_transform(df["hour"].values)

    df['day_of_week_sin'] = sin_transformer(7).fit_transform(df['day_of_week'].values)
    df['day_of_week_cos'] = cos_transformer(7).fit_transform(df["day_of_week"].values)

    df['month_sin'] = sin_transformer(12).fit_transform(df['month'].values)
    df['month_cos'] = cos_transformer(12).fit_transform(df["month"].values)

    return df


if __name__ == '__main__':
    datasets_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data\\lstm_small_subset')
    datasets_load_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data')
    # full_df = combine_df(dir=datasets_path + '\\sources')
    # full_df.to_csv(datasets_path + '\\small_subset_lstm.csv')

    df = pd.read_csv(datasets_path + '\\small_subset_lstm_cleaned.csv', index_col=0)
    df_new = add_time_features(df)
    df_new = renew_features(df_new).set_index('Date')
    # j = train_test_val_split(df, target_column='day_ahead_prices_EURO_x')
    # run_preprocessing(df, nan_streak_threshold=168, start_ts=None, end_ts=None)
    pass
