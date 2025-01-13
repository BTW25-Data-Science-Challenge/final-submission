import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os


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

        # pred_data = self.ground_truth_data.copy(deep=True)
        # r = np.random.randint(low=-10, high=10, size=pred_data.shape[0])
        # pred_data['day_ahead_prices_predicted'] = pred_data['day_ahead_prices'].values + r
        # pred_data = pred_data.drop('day_ahead_prices', axis=1)
        # pred_data.to_csv(curr_dir + '\\' + config['input_dir'] + '\\' + 'biLSTM_prediction.csv')
        #
        # pred_data = self.ground_truth_data.copy(deep=True)
        # r = np.random.randint(low=-10, high=10, size=pred_data.shape[0])
        # pred_data['day_ahead_prices_predicted'] = pred_data['day_ahead_prices'].values + r
        # pred_data = pred_data.drop('day_ahead_prices', axis=1)
        # pred_data.to_csv(curr_dir + '\\' + config['input_dir'] + '\\' + 'XGBoost_prediction.csv')

        # calculate errors for each model prediction
        self.__calc_errors()

        # combine dataframes
        self.full_df = None
        self.__create_full_df()

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
        gt_values = self.ground_truth_data['day_ahead_prices'].values
        for pred_model in self.predictions_data.keys():
            pred_values = self.predictions_data[pred_model]['day_ahead_prices_predicted'].values
            self.predictions_data[pred_model]['RMSE'] = self.calc_rmse(gt_values, pred_values)
            self.predictions_data[pred_model]['MAE'] = self.calc_mae(gt_values, pred_values)
            self.predictions_data[pred_model]['MAPE'] = self.calc_mape(gt_values, pred_values)
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

    def calc_mape(self, actual_values, predicted_values):
        scaler = StandardScaler()
        y_true_scaled = scaler.fit_transform(actual_values.reshape(-1, 1)).flatten()
        y_pred_scaled = scaler.transform(predicted_values.reshape(-1, 1)).flatten()
        mape = mean_absolute_percentage_error(y_true_scaled, y_pred_scaled)
        return mape

    def plot_rmse_per_hour(self):
        for k in self.predictions_data.keys():
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
            plt.plot(days_df['timestamp'], np.sqrt(days_df['SE']), label=str(k))
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
            plt.plot(days_df['timestamp'], np.sqrt(days_df['AE']), label=str(k))
        plt.xticks(rotation=45)
        plt.title('MAE per Day')
        plt.xlabel('Timestamp')
        plt.ylabel('MAE')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'daily_mae.png')
        plt.show(block=True)

    def plot_compare_mae(self):
        gt_values = self.ground_truth_data['day_ahead_prices'].values

        mae_values = []
        names = []
        for i, pred_model in enumerate(self.predictions_data.keys()):
            pred_values = self.predictions_data[pred_model]['day_ahead_prices_predicted'].values
            mae = self.calc_mae(gt_values, pred_values)
            mae_values.append(mae)
            names.append(str(pred_model))
        x = np.arange(len(self.predictions_data.keys()))

        ax = plt.subplot(111)
        my_cmap = plt.get_cmap("jet")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        bar = ax.bar(x, mae_values, width=0.4, align='center', label='MAE', color=my_cmap(rescale(mae_values)))

        def digit_label(rects):
            for rect in rects:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, f'{int(h)}',
                        ha='center', va='bottom', color=rect.get_facecolor())
        digit_label(bar)
        plt.title('MAE per Model')
        ax.set_ylabel('MAE')
        ax.set_xlabel('Model')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim([0, max(mae_values) + 2])
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'compare_mae.png')
        plt.show(block=True)

    def plot_compare_rmse(self):
        gt_values = self.ground_truth_data['day_ahead_prices'].values

        rmse_values = []
        names = []
        for i, pred_model in enumerate(self.predictions_data.keys()):
            pred_values = self.predictions_data[pred_model]['day_ahead_prices_predicted'].values
            rmse = self.calc_rmse(gt_values, pred_values)
            rmse_values.append(rmse)
            names.append(str(pred_model).replace('_prediction', ''))
        x = np.arange(len(self.predictions_data.keys()))

        ax = plt.subplot(111)
        my_cmap = plt.get_cmap("jet")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        bar = ax.bar(x, rmse_values, width=0.4, align='center', label='RMSE', color=my_cmap(rescale(rmse_values)))

        def digit_label(rects):
            for rect in rects:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, f'{int(h)}',
                        ha='center', va='bottom', color=rect.get_facecolor())
        digit_label(bar)
        plt.title('RMSE per Model')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Model')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim([0, max(rmse_values) + 2])
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'compare_rmse.png')
        plt.show(block=True)

    def plot_compare_mape(self):
        gt_values = self.ground_truth_data['day_ahead_prices'].values

        mape_values = []
        names = []
        for i, pred_model in enumerate(self.predictions_data.keys()):
            pred_values = self.predictions_data[pred_model]['day_ahead_prices_predicted'].values
            mape = self.calc_mape(gt_values, pred_values)
            mape_values.append(mape*100)
            names.append(str(pred_model).replace('_prediction', ''))
        x = np.arange(len(self.predictions_data.keys()))

        ax = plt.subplot(111)
        my_cmap = plt.get_cmap("jet")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        bar = ax.bar(x, mape_values, width=0.4, align='center', label='MAPE', color=my_cmap(rescale(mape_values)))

        def digit_label(rects):
            for rect in rects:
                h = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * h, f'{int(h)}',
                        ha='center', va='bottom', color=rect.get_facecolor())
        digit_label(bar)
        plt.title('MAPE per Model')
        ax.set_ylabel('MAPE in %')
        ax.set_xlabel('Model')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim([0, 100])
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'compare_mape.png')
        plt.show(block=True)

