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

    
    def _BaseModel__custom_load(self, filename):
        # Directory to clone
        github_repo_url = filename
        local_dir = "./final-submission"

        # Clone the repository if it doesn't already exist
        if not os.path.exists(local_dir):
            os.system(f"git clone {github_repo_url} {local_dir}")

        # Path to the specific directory containing the checkpoint
        checkpoint_dir = os.path.join(local_dir, "models/models/chronos-tiny-2015-1000/checkpoint-final")
        
        # Load the model pipeline
        pipeline = ChronosPipeline.from_pretrained(
            checkpoint_dir,
            device_map=("cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=torch.bfloat16,
        )

        return pipeline
                
    def _BaseModel__custom_save(self, model = None, filename = None):
        return

    def train(self, X_train = None, y_train = None, X_val = None, y_val = None, X_test = None, y_test = None):
        return None