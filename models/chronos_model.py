from base_model import BaseModel
import torch


class ChronosModel(BaseModel):
    def __init__(self, model_name: str, model_type: str):
        """Call the BaseModel constructor with the required arguments."""
        super().__init__(model_name, model_type)

    def _BaseModel__create_model(self):
        self.model = None

    
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

    
    def _BaseModel__custom_load(self, model_dir):
        # Directory to clone
        
        # Load the model pipeline
        pipeline = ChronosPipeline.from_pretrained(
            model_dir,
            device_map=("cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=torch.bfloat16,
        )
        self.model = pipeline  # Set the model to the loaded pipeline

        return pipeline
    
    
                
    def _BaseModel__custom_save(self, model = None, filename = None):
        return

    def train(self, X_train = None, y_train = None, X_val = None, y_val = None, X_test = None, y_test = None):
        return None
    