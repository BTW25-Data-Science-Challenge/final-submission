import pandas as pd
import pickle
from base_model import BaseModel
from sklearn.linear_model import LinearRegression

class LinearRegression(BaseModel):

    def _BaseModel__create_model(self):
        self.model = None


    def train(self, X_train, y_train, X_val = None, y_val = None, X_test = None, y_test = None):
        #drop column 'Start Date' -> sklearn cant use DateTime values
        X_train = X_train.drop['Start Date']
        y_train = y_train.drop['Start Date']
        
        #fill nan vaules ->: skleanr cant use NaN values
        X_train = X_train.apply(lambda col: col.fillna(0), axis=0)
        y_train = y_train.apply(lambda col: col.fillna(0), axis=0)

        model = LinearRegression()
        self.model = model.fit(X_train, y_train)
        return None
        

    def _BaseModel__run_prediction(self, X):
        #copy of col 'Start Date' for mapping after training
        X_date_cpy = X['Start Date']
        #preprocessing to avoid sklearn specific errors
        X = X.drop['Start Date']
        X = X.apply(lambda col: col.fillna(0), axis=0)

        target = "day_ahead_prices_EURO"
        prediction_values = self.model.predict(X)
        predicted_results_df = pd.DataFrame({'timestamp': X_date_cpy['Date'],
                                                'value':prediction_values})
        return predicted_results_df


    def _BaseModel__custom_save(self, model, filename):
         with open(f"{filename}.pkl", "wb") as file:
            pickle.dump(model, file)
    
    def _BaseModel__custom_load(self, filename):
        with open(f"{filename}.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        return loaded_model

        
        
