import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import os
import random


class BenchmarkMaker:
    def __init__(self, config):
        self.gt_name = config['ground_truth_file_name']
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_file_names = os.listdir(curr_dir + '\\' + config['input_dir'])
        self.export_dir = curr_dir + '\\' + config['export_dir']  # output for plots and benchmark .csv files

        # load input data
        data = self.__load_data(dir=curr_dir + '\\' + config['input_dir'], file_names=self.input_file_names,
                                tz=config['tz'])
        self.predictions_data = {n: data[n] for n in data.keys() if n != self.gt_name.replace('.csv', '')}   # prediction results Dataframes
        self.ground_truth_data = data[self.gt_name.replace('.csv', '')]   # gt df

        pred_data = self.predictions_data['biLSTM_prediction']
        r = np.random.randint(low=-1, high=10, size=pred_data.shape[0])
        pred_data['day_ahead_prices_predicted'] = pred_data['day_ahead_prices_predicted'].values + r

        # calculate errors for each model prediction
        self.__calc_errors()

        # combine dataframes
        self.full_df = None
        self.__create_full_df()

        pass

    def __load_data(self, dir, file_names, tz):
        data = {n.replace('.csv', ''): pd.read_csv(dir + '\\' + n, index_col=0) for n in file_names}
        # adjust timestamp
        for k in data.keys():
            ts_start = pd.Timestamp(data[k]['timestamp'].values[0], tz=tz)
            ts_end = pd.Timestamp(data[k]['timestamp'].values[-1], tz=tz)
            timestamp_range = pd.date_range(start=ts_start, end=ts_end, freq='1H')
            data[k]['timestamp'] = timestamp_range
        return data

    def __create_full_df(self):

        pass

    def __calc_errors(self):
        gt_values = self.ground_truth_data['day_ahead_prices']
        for pred_model in self.predictions_data.keys():
            pred_values = self.predictions_data[pred_model]['day_ahead_prices_predicted']
            self.predictions_data[pred_model]['RMSE'] = self.calc_rmse(gt_values, pred_values)
            self.predictions_data[pred_model]['MAE'] = self.calc_mae(gt_values, pred_values)
            self.predictions_data[pred_model]['SE'] = self.calc_squared_error(gt_values, pred_values)
            self.predictions_data[pred_model]['AE'] = self.calc_absolute_error(gt_values, pred_values)
        pass

    def calc_squared_error(self, actual_values, predicted_values):
        se = (actual_values - predicted_values)**2
        return se

    def calc_absolute_error(self, actual_values, predicted_values):
        ae = abs(actual_values - predicted_values)
        return ae

    def calc_rmse(self, actual_values, predicted_values):
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        return rmse

    def calc_mae(self, actual_values, predicted_values):
        mae = mean_absolute_error(actual_values, predicted_values)
        return mae

    def plot_rmse_per_hour(self):
        for k in self.predictions_data.keys():
            self.predictions_data[k] = self.predictions_data[k][:1000]
            plt.plot(self.predictions_data[k]['timestamp'], np.sqrt(self.predictions_data[k]['SE']), label=str(k))
        plt.xticks(rotation=45)
        plt.title('RMSE per Hour')
        plt.xlabel('Timestamp')
        plt.ylabel('RMSE')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'hourly_rmse.png')
        plt.show(block=True)

    def plot_mae_per_hour(self):
        for k in self.predictions_data.keys():
            plt.plot(self.predictions_data[k]['timestamp'], self.predictions_data[k]['AE'], label=str(k))
        plt.xticks(rotation=45)
        plt.title('Absolute Error per Hour')
        plt.xlabel('Timestamp')
        plt.ylabel('Absolute Error')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'hourly_mae.png')
        plt.show(block=True)

    def plot_rmse_per_day(self):
        for k in self.predictions_data.keys():
            days_df = self.predictions_data[k].groupby(self.predictions_data[k].timestamp.dt.date).mean().reset_index(names='date')
            plt.scatter(days_df['timestamp'], np.sqrt(days_df['SE']))
        plt.xticks(rotation=45)
        plt.title('RMSE per Day')
        plt.xlabel('Timestamp')
        plt.ylabel('RMSE')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'daily_rmse.png')
        plt.show(block=True)

    def plot_mae_per_day(self):
        for k in self.predictions_data.keys():
            days_df = self.predictions_data[k].groupby(self.predictions_data[k].timestamp.dt.date).mean().reset_index(names='date')
            plt.scatter(days_df['timestamp'], np.sqrt(days_df['AE']))
        plt.xticks(rotation=45)
        plt.title('MAE per Day')
        plt.xlabel('Timestamp')
        plt.ylabel('MAE')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'daily_mae.png')
        plt.show(block=True)

    def plot_compare_predictions(self):
        pass


