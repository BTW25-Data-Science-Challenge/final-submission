import pandas as pd
import pickle
from base_model import BaseModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class ETSModel(BaseModel):
    def _BaseModel__create_model(self):
        self.model = None

    def train(self, X_train, y_train, X_val = None, y_val = None, X_test = None, y_test = None):
        #time_series = X_train
        if y_train.isnull().any():
            print("Warning: Missing values in y_train that are being automatically filled.")
            y_train = y_train.ffill()
        model = ExponentialSmoothing(y_train, seasonal_periods=24, trend="add", seasonal="add")
        self.model = model.fit()
        print("ETS model filled with values")
        return None
    
    def _BaseModel__run_prediction(self, X):
        if self.model is None:
            raise ValueError("The model is not trained. Call `train` before prediction.")
        forecast = self.model.forecast(len(X))
        #print(forecast)
        #print(X)
        prediction_results = pd.DataFrame({
            'timestamp': X['Date'],  # Use the 'Date' column in X for timestamps
            'value': forecast.values
        })
        return prediction_results

    def _BaseModel__custom_save(self, model, filename):
         with open(f"{filename}.pkl", "wb") as file:
            pickle.dump(model, file)
    
    def _BaseModel__custom_load(self, filename):
        with open(f"{filename}.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        return loaded_model
