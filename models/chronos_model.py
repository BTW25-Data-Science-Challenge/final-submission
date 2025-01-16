from base_model import BaseModel
import torch
import numpy as np
import pandas as pd

import requests
import tempfile

#install pip install "chronos-forecasting[training] @ git+https://github.com/amazon-science/chronos-forecasting.git" before using class

from chronos import ChronosPipeline

class ChronosModel(BaseModel):
    def _BaseModel__create_model(self):
        self.model = None

    def train(self, X_train = None, y_train = None, X_val = None, y_val = None, X_test = None, y_test = None):
        return None
    
    def _BaseModel__run_prediction(self, X):

        #define which column to be forecasted and forecast legth
        target_column = "day_ahead_prices_EURO"
        prediction_length = 24

        
        context = torch.tensor(X[target_column].values)[-512:]  # Limit context to last 512 samples
        forecast = self.model.predict(context, prediction_length)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

        context_dates = X.index[-512:]
        last_date = context_dates[-1]
        forecast_index = pd.date_range(last_date + pd.Timedelta(hours=1), periods=prediction_length, freq="H")

        prediction_results = pd.DataFrame({
            "timestamp": forecast_index,
            "forecasted_values": median
        })
        
        return prediction_results

    
    def _BaseModel__custom_load(self, filename):
        """
        :param filename: URL to uploadad model on github
        """
         
        #ToDO Upload Model to GiHub and anapt URL
        model_url = filename
    
        # Download the model file
        response = requests.get(model_url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(response.content)
                model_path = tmp_file.name
        else:
            raise ValueError(f"Failed to download model checkpoint. HTTP Status: {response.status_code}")


        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=("cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=torch.bfloat16,
)
        return pipeline