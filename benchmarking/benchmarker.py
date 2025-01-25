import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import os


class BenchmarkMaker:
    def __init__(self, export_dir):
        self.export_dir = export_dir  # output for plots and benchmark .csv files

        if not os.path.exists(self.export_dir):
            # create the directory in case it does not already exist
            os.makedirs(self.export_dir)

        # load input data
        self.data = None
        self.model_names = []

    def load_from_files(self, dir, file_names, tz):
        data = {n.replace('.csv', ''): pd.read_csv(dir + '\\' + n, index_col=0) for n in file_names}
        # adjust timestamp
        for k in data.keys():
            ts_start = pd.Timestamp(data[k]['timestamp'].values[0], tz=tz)
            ts_end = pd.Timestamp(data[k]['timestamp'].values[-1], tz=tz)
            timestamp_range = pd.date_range(start=ts_start, end=ts_end, freq='1H')
            data[k]['timestamp'] = timestamp_range
        return data

    def load_dataframes(self, predictions: dict, prices: pd.DataFrame):
        """loading datasets of predictions from different models

        :param predictions: dict form: {model_name1: df1, model_name2: df2, ...}
        :param prices: ground truth prices dataframe
        """
        for k in predictions.keys():
            predictions[k] = predictions[k].set_axis([str(k)], axis='columns')
            self.model_names.append(str(k))
        prices = prices.set_axis(['day_ahead_prices'], axis='columns')
        self.align_dataframes(list(predictions.values()) + [prices])

    def align_dataframes(self, dataframes):
        """Align multiple DataFrames with timestamp indices by merging them on their index.
        """
        result_df = dataframes[0]
        for df in dataframes[1:]:
            result_df = result_df.join(df, how='outer')
        self.data = result_df

    def calc_errors(self):
        no_nan = self.data.copy(deep=True).dropna(how='any')
        gt_values = no_nan['day_ahead_prices'].values
        for pred_model in self.model_names:
            pred_values = no_nan[pred_model].values
            self.data[str(pred_model) + '_RMSE'] = self.calc_rmse(gt_values, pred_values)
            self.data[str(pred_model) + '_MAE'] = self.calc_mae(gt_values, pred_values)
            self.data[str(pred_model) + '_MAPE'] = self.calc_mape(gt_values, pred_values)
            self.data[str(pred_model) + '_SE'] = self.calc_squared_error(self.data['day_ahead_prices'].values,
                                                                         self.data[pred_model].values)
            self.data[str(pred_model) + '_AE'] = self.calc_absolute_error(self.data['day_ahead_prices'].values,
                                                                          self.data[pred_model].values)
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

    def calc_mape(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        """calculate mean average percentage error.
        close to 0: good
        close to 1: bad

        :param actual_values: correct underlying values
        :param predicted_values: forecasted values
        :return: mape"""
        mape = mean_absolute_percentage_error(actual_values, predicted_values)
        return mape

    def plot_rmse_per_hour(self):
        for model in self.model_names:
            plt.plot(self.data.index.values, np.sqrt(self.data[model + '_SE'].values), label=model)
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
        for model in self.model_names:
            plt.plot(self.data.index.values, self.data[model + '_AE'].values, label=model)
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
        copy_df = self.data.reset_index(names='timestamp')
        for model in self.model_names:
            days_df = copy_df.groupby(copy_df.timestamp.dt.date).mean().reset_index(names='date')
            plt.plot(days_df['timestamp'], np.sqrt(days_df[model + '_SE']), label=model)
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
        copy_df = self.data.reset_index(names='timestamp')
        for model in self.model_names:
            days_df = copy_df.groupby(copy_df.timestamp.dt.date).mean().reset_index(names='date')
            plt.plot(days_df['timestamp'], np.sqrt(days_df[model + '_AE']), label=model)
        plt.xticks(rotation=45)
        plt.title('MAE per Day')
        plt.xlabel('Timestamp')
        plt.ylabel('MAE')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'daily_mae.png')
        plt.show(block=True)

    def plot_compare_predictions_hourly(self):
        gt_values = self.data['day_ahead_prices'].values
        timestamps = self.data.index.values
        plt.plot(timestamps, gt_values, label='Actual Values')

        for model in self.model_names:
            pred_values = self.data[model].values
            plt.plot(timestamps, pred_values, label=model)

        plt.xticks(rotation=45)
        plt.title('Model Predictions per Hour')
        plt.ylabel('Day Ahead Price in €')
        plt.xlabel('Timestamp')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'hourly_compare_predictions.png')
        plt.show(block=True)

    def plot_compare_predictions_daily(self):
        copy_df = self.data.reset_index(names='timestamp')
        days_df = copy_df.groupby(copy_df.timestamp.dt.date).mean().reset_index(
            names='date')
        timestamps = days_df['timestamp'].values

        gt_values = days_df['day_ahead_prices'].values
        plt.plot(timestamps, gt_values, label='Actual Values')

        for model in self.model_names:
            pred_values = days_df[model].values
            plt.plot(timestamps, pred_values, label=model)

        plt.xticks(rotation=45)
        plt.title('Model Predictions per Day')
        plt.ylabel('Day Ahead Price in €')
        plt.xlabel('Timestamp')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'daily_compare_predictions.png')
        plt.show(block=True)

    def plot_compare_mae(self):
        no_nan = self.data.copy(deep=True).dropna(how='any')
        gt_values = no_nan['day_ahead_prices'].values

        mae_values = []
        for model in self.model_names:
            pred_values = no_nan[model].values
            mae = self.calc_mae(gt_values, pred_values)
            mae_values.append(mae)
        x = np.arange(len(self.model_names))

        ax = plt.subplot(111)
        my_cmap = plt.get_cmap("jet")
        colors = my_cmap(np.linspace(0, 1, len(self.model_names)))
        bar = ax.bar(x, mae_values, width=0.4, align='center', label='MAE', color=colors)

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
        ax.set_xticklabels(self.model_names)
        ax.set_ylim([0, max(mae_values) + 5])
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'compare_mae.png')
        plt.show(block=True)

    def plot_compare_rmse(self):
        no_nan = self.data.copy(deep=True).dropna(how='any')
        gt_values = no_nan['day_ahead_prices'].values

        rmse_values = []
        for model in self.model_names:
            pred_values = no_nan[model].values
            rmse = self.calc_rmse(gt_values, pred_values)
            rmse_values.append(rmse)
        x = np.arange(len(self.model_names))

        ax = plt.subplot(111)
        my_cmap = plt.get_cmap("jet")
        colors = my_cmap(np.linspace(0, 1, len(self.model_names)))
        bar = ax.bar(x, rmse_values, width=0.4, align='center', label='RMSE', color=colors)

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
        ax.set_xticklabels(self.model_names)
        ax.set_ylim([0, max(rmse_values) + 5])
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'compare_rmse.png')
        plt.show(block=True)

    def plot_compare_mape(self):
        no_nan = self.data.copy(deep=True).dropna(how='any')
        gt_values = no_nan['day_ahead_prices'].values

        mape_values = []
        for model in self.model_names:
            pred_values = no_nan[model].values
            mape = self.calc_mae(gt_values, pred_values)
            mape_values.append(mape / abs(no_nan[model].max() - no_nan[model].min()) * 100)
        x = np.arange(len(self.model_names))

        ax = plt.subplot(111)
        my_cmap = plt.get_cmap("jet")
        colors = my_cmap(np.linspace(0, 1, len(self.model_names)))
        bar = ax.bar(x, mape_values, width=0.4, align='center', label='MAPE', color=colors)

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
        ax.set_xticklabels(self.model_names)
        ax.set_ylim([0, 100])
        plt.tight_layout()
        if self.export_dir is not None:
            plt.savefig(self.export_dir + '\\' + 'compare_mape.png')
        plt.show(block=True)

    def plot_single_model(self, model_name: str = ''):
        x = self.data.index.values
        y1 = self.data[model_name].values
        y2 = self.data[model_name + '_AE'].values
        y3 = self.data[model_name + '_SE'].values

        fig, ax1 = plt.subplots()

        # plot prediction and absolute error with € scale
        ax1.plot(x, y1, 'b', label='y1 (sin(x)')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('y1', color='b')
        ax1.tick_params('y', colors='b')

        # plot RMSE with additional scale
        ax2 = ax1.twinx()

        ax2.plot(x, y2, 'g', label='y2 (exp(-x))')
        ax2.set_ylabel('y2', color='g')
        ax2.tick_params('y', colors='g')

        ax3 = ax1.twinx()

        ax3.plot(x, y3, 'r', label='y3 (100*cos(x))')
        ax3.spines['right'].set_position(('outward', 60))
        ax3.set_ylabel('y3', color='r')
        ax3.tick_params('y', colors='r')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines = lines1 + lines2 + lines3
        labels = labels1 + labels2 + labels3
        plt.legend(lines, labels, loc='upper right')

        plt.title('Multiple Y-axis Scales')
        plt.show()

