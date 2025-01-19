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


def fill_missing_with_xgboost(df, target_column):
    """
    Fills NaN values in the target_column of the DataFrame using an XGBoost regressor trained
    with the other columns as features.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - target_column (str): The column name that contains NaN values to be predicted.

    Returns:
    - pd.DataFrame: A DataFrame with NaN values in the target_column filled.
    """
    # Check if the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in the DataFrame.")

    # Separate the target column
    target = df[target_column]

    # Identify rows with and without NaN in the target column
    missing_mask = target.isna()
    complete_mask = ~missing_mask

    # Ensure there are missing values to predict
    if missing_mask.sum() == 0:
        raise ValueError(f"No missing values found in column '{target_column}'.")

    # Split the data into training (non-NaN) and prediction (NaN)
    X_train = df.loc[complete_mask].drop(columns=[target_column])
    y_train = target[complete_mask]
    X_predict = df.loc[missing_mask].drop(columns=[target_column])

    # Ensure there is enough data to train
    if X_train.shape[0] < 10:  # Arbitrary threshold to ensure some training data exists
        raise ValueError("Not enough data to train the model.")

    # Train/test split for model validation
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train, y_train, test_size=0.05, random_state=42
    )
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Train the XGBoost model
    model = XGBRegressor(
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=10,
        random_state=42,
        verbosity=0,
        device=device
    )
    model.fit(X_train_split, y_train_split,
              eval_set=[(X_valid_split, y_valid_split)],
              verbose=False)

    # Predict the missing values
    predictions = model.predict(X_predict)

    # Fill the missing values in the original DataFrame
    df.loc[missing_mask, target_column] = predictions

    return df


def has_long_nan_streak(series, threshold):
    max_nan_streak = (series.isna().astype(int)
                      .groupby(series.notna().astype(int).cumsum())
                      .transform('sum').max())
    return max_nan_streak > threshold


def run_preprocessing(df):
    time_n = time.time()
    unique_df = df.loc[:, ~df.T.duplicated()]

    unique_df['Date'] = pd.DatetimeIndex(unique_df['Date'].values)
    unique_df = unique_df.set_index('Date')
    start_date = pd.Timestamp('2015-01-05 00:00:00')
    end_date = pd.Timestamp('2024-11-30 00:00:00')
    unique_df = unique_df[start_date:end_date]

    unique_df = unique_df.reset_index()
    timestamps = unique_df['Date']
    unique_df = unique_df.drop(['Date'], axis=1)

    threshold = 168     # 24h * 7 = 7days

    columns_to_keep = ~unique_df.apply(has_long_nan_streak, threshold=threshold, axis=0)
    no_nan_streaks_df = unique_df.loc[:, columns_to_keep]

    features = list(no_nan_streaks_df.columns)
    new_df = no_nan_streaks_df.copy(deep=True)
    for i, c in enumerate(features):
        if no_nan_streaks_df[c].isna().sum() > 0:
            new_df = fill_missing_with_xgboost(new_df, target_column=c)
        print('----------------')
        print(f'feature {i+1}: {c} done after: {int((time.time() - time_n)/60)}')
        # plot_filled_values(df=new_df, target_column=c, missing_mask=no_nan_streaks_df[c].isna())

    new_df['hour'] = new_df['Date'].apply(lambda x: x.hour / 24)
    new_df['month'] = new_df['Date'].apply(lambda x: x.month / 12)
    new_df['day_of_week'] = new_df['Date'].apply(lambda x: x.day_of_week / 7)

    datasets_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data')
    new_df.to_csv(datasets_path + '\\allData_cleaned.csv')

    print(f'preprocessing done in: {int((time.time() - time_n)/60)}')
    new_df['Date'] = timestamps
    print('done')

    pass


def train_test_val_split(df, target_column):
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
    """
    Plots the DataFrame, marking filled values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with filled values.
    - target_column (str): The target column that was filled.
    - missing_mask (pd.Series): A boolean mask indicating where NaN values were originally.
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(df.shape[0])

    # Plot all values in the column
    plt.plot(x, df[target_column], label='Filled Values', color='blue')

    # Highlight the originally missing values
    plt.scatter(
        x[missing_mask],
        df.loc[missing_mask, target_column],
        color='red',
        label='Originally NaN',
        zorder=5,
        marker='x',
        s=100
    )

    plt.title(f"Filled Missing Values in '{target_column}'")
    plt.xlabel('Index')
    plt.ylabel(target_column)
    plt.legend()
    plt.show(block=True)


if __name__ == '__main__':
    datasets_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).replace(
        '\\models\\LSTM_based', '\\data')
    df = pd.read_csv(datasets_path + '\\allData_cleaned.csv', index_col=0)
    j = train_test_val_split(df, target_column='day_ahead_prices_EURO_x')
    # run_preprocessing(df)
    pass