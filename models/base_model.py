import abc


class BaseModel(abc.ABC):

    def __init__(self):
        self.model_name = ''
        self.model = None

    def name_model(self, model_name: str):
        self.model_name = model_name

    def create_model(self, model_name: str):
        # create your own model under self.model
        pass

    def train(self, X, y):
        # the train function should take the training data
        history = self.model.train(X, y)
        return history

    def predict(self, X, store=None):
        # the prediction should output a pandas dataframe timestamp|value
        prediction = self.model.predict(X)
        if store:
            prediction.to_csv('base_model_prediction_output.csv')
        return prediction

    def save_model(self, file):
        # use your own dataformat to save your model
        self.model.save(file)

