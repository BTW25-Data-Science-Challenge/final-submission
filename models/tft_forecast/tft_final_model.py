#import libs
import numpy as np
import pandas as pd
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet, NaNLabelEncoder
import torch
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

#metrics calculation
def calc_metric(y_pred,y_actuals,metric, allow_zeros=True):
	epsilon = 1e-10
	if metric == 'mape':
		if not allow_zeros:
			idx_zeros = np.where((y_actuals <= -2) | (y_actuals >= 2 ))
			y_actuals = y_actuals[idx_zeros]
			y_pred = y_pred[idx_zeros]
		res = np.mean(np.abs((y_pred - y_actuals) / (y_actuals + epsilon))) * 100
	elif metric == 'mae':
		res = np.mean(np.abs(y_pred - y_actuals))
	elif metric == 'rmse':
		res = np.sqrt(np.mean(np.power(y_pred - y_actuals, 2)))
	elif metric == 'smape':
		if not allow_zeros:
			idx_zeros = np.where((y_actuals <= -2) | (y_actuals >= 2 ))
			y_actuals = y_actuals[idx_zeros]
			y_pred = y_pred[idx_zeros]
		res = np.mean(np.abs(y_pred - y_actuals) / ((np.abs(y_pred) + np.abs(y_actuals))/2))*100
	return res

model_path_gpu =r"./trained_models/GPU_model_epoch=33-step=3672.ckpt"
model_path_cpu = r"./trained_models/CPU_model_epoch=14-step=6510.ckpt"

if torch.cuda.is_available():
    model_path = model_path_gpu
else:
    model_path = model_path_cpu

data_path = r"./dataset/multivar_dataset_020225.csv"

class TftForecaster:
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.data = None
        self.data_index = None
        self.data_predict = None
        self.y_pred = None
        self.y_actuals = None


    def get_current_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.data_path, index_col=0)
        data.index = pd.to_datetime(data.index, utc=True)
        data.index =data.index.tz_convert('Europe/Brussels')

        data_index = pd.DataFrame(index = range(1, len(data) + 1), data = data.index)
        data_index.rename(columns = {0: 'timestamp'}, inplace = True)
        data.reset_index(drop=True, inplace=True)
        data['country'] = 'DE_LU'
        data['time_idx'] = range(1, len(data) + 1)
        self.data = data
        self.data_index = data_index

    def get_data_predict(self, predict_date):
        #TODO: error management for a non-valid date predict_date

        '''
        :param date: date to forecast in YYYY-MM-DD HH:MM format.
        :return: encoder decoder forecasting dataset
        '''

        max_prediction_length = 24
        max_encoder_length = 24 * 7
        date = pd.Timestamp(predict_date, tz='Europe/Brussels')
        last_encoder = date - pd.Timedelta(hours = 1)
        first_encoder = last_encoder - pd.Timedelta(hours = max_encoder_length)
        last_decoder = last_encoder + pd.Timedelta(hours = max_prediction_length)
        last_encoder_idx = self.data_index[self.data_index['timestamp'] == last_encoder].index[0]
        first_encoder_idx = self.data_index[self.data_index['timestamp'] == first_encoder].index[0]
        last_decoder_idx = self.data_index[self.data_index['timestamp'] == last_decoder].index[0]
        encoder_data = self.data[lambda x: (x.time_idx > first_encoder_idx) & (x.time_idx <= last_encoder_idx)]
        decoder_data = self.data[lambda x: (x.time_idx > last_encoder_idx) & (x.time_idx <= last_decoder_idx)]

        self.data_predict = pd.concat([encoder_data, decoder_data], ignore_index=True)


    def custom_load(self):
        '''
        :return: load TFT model from a trained .ckpt file
        '''
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path)

    def predict(self):
        #TODO: bool param to indicate if day-ahead prices are available
        raw_prediction = self.model.predict(self.data_predict, return_y=True)
        self.y_pred = raw_prediction.output.cpu().numpy()
        self.y_actuals = raw_prediction.y[0].cpu().numpy()

    def plot(self, date):
        text_mape = f"MAPE = {calc_metric(self.y_pred,self.y_actuals, metric='mape'):.2f} %"
        text_smape = f"SMAPE = {calc_metric(self.y_pred,self.y_actuals, metric='smape'):.2f} %"
        text_mae = f"MAE = {calc_metric(self.y_pred,self.y_actuals, metric='mae'):.2f} EUR/MWh"
        text_rmse = f"RMSE = {calc_metric(self.y_pred,self.y_actuals, metric='rmse'):.2f} EUR/MWh"
        plt.figure(figsize=(8, 5))
        plt.plot(self.y_pred[0], label = 'Forecast')
        plt.plot(self.y_actuals[0], label = 'Actuals')

        plt.title(f'forecast for {date.date()}')
        plt.xlabel('hour index')
        plt.ylabel('electricity price [EUR/MWh]')
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.legend()

        plt.text(0.75, 0.15, text_mape, transform=plt.gca().transAxes,
                    verticalalignment='bottom')
        plt.text(0.75, 0.1, text_mae, transform=plt.gca().transAxes,
                    verticalalignment='bottom')
        plt.text(0.75, 0.05, text_rmse, transform=plt.gca().transAxes,
                    verticalalignment='bottom')
        plt.text(0.75, 0.0, text_smape, transform=plt.gca().transAxes,
                    verticalalignment='bottom')

        plt.show()

date = pd.to_datetime('2024-02-02 00:00')

if __name__ == "__main__":
    tft = TftForecaster(model_path, data_path)
    tft.custom_load()
    tft.get_current_data()
    tft.get_data_predict(predict_date=date)
    tft.predict()
    tft.plot(date=date)
    print(f'Predicted day-ahead prices: {tft.y_pred}')




