import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import numpy as np
import os


class BenchmarkMaker:
    def __init__(self, config):
        self.gt_name = config['ground_truth_file_name']
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_file_names = os.listdir(curr_dir + '\\' + config['input_dir'])
        self.export_dir = curr_dir + '\\' + config['export_dir']  # output for plots and benchmark .csv files

        # load input data
        data = self.__load_data(dir=curr_dir + '\\' + config['input_dir'], file_names=self.input_file_names)
        self.predictions_data = {n: data[n] for n in data.keys() if n != self.gt_name}   # prediction results Dataframes
        self.ground_truth_data = data[self.gt_name.replace('.csv', '')]   # gt df

        # calculate errors for each model prediction
        self.__calc_errors()

        # combine dataframes
        self.full_df = None
        self.__create_full_df()

        pass

    def __load_data(self, dir, file_names):
        data = {n.replace('.csv', ''): pd.read_csv(dir + '\\' + n, index_col=0) for n in file_names}
        return data

    def __create_full_df(self):
        pass

    def __calc_errors(self):
        gt_values = self.ground_truth_data['day_ahead_prices']
        for pred_model in self.predictions_data.keys():
            pred_values = self.predictions_data[pred_model]['day_ahead_prices_predicted']
            self.predictions_data[pred_model]['RMSE'] = self.calc_rmse(gt_values, pred_values)
            self.predictions_data[pred_model]['MAE'] = self.calc_mae(gt_values, pred_values)
        pass

    def calc_rmse(self, actual_values, predicted_values):
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        return rmse

    def calc_mae(self, actual_values, predicted_values):
        mae = mean_absolute_error(actual_values, predicted_values)
        return mae

    def plot_compare_rmse(self):
        pass

    def plot_compare_mae(self):
        pass

    def plot_compare_f1(self):
        pass

    def plot_compare_predictions(self):
        pass


