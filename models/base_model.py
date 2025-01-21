"""
Base model structure.
Every Forecasting model should inherit from this.

Note: override all abstract methods and keep the final methods unchanged
"""
import pandas as pd
from abc import ABC, abstractmethod
from typing import final


class BaseModel(ABC):

    def __init__(self, model_name: str, model_type: str):
        """Init and create model.

        :param model_name: Name of your model
        :param model_type: Type of your model e.g. LSTM
        """
        self.model_name = model_name
        self.model_type = model_type
        self.model = None  # this is a placeholder for your model
        self.create_model()

    @abstractmethod
    def create_model(self):
        """Define your own model under self.model.
        """
        ...

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame = None, y_val: pd.DataFrame = None,
              X_test: pd.DataFrame = None, y_test: pd.DataFrame = None) -> pd.DataFrame | None:
        """train the model on the training data.
        test and validation data can be used only for evaluation (if available).

        :param X_train: training features dataset
        :param y_train: training target values
        :param X_val: validation features' dataset
        :param y_val: validation target values
        :param X_test: testing features' dataset
        :param y_test: testing target values
        :return: training history (losses while training, if available else None) [epoch | train_loss | test_loss]
        """
        # call the training loop/function of your model
        # and return a history (if available, otherwise None)
        ...

    @abstractmethod
    def run_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """run prediction on your defined model

        :param X: features dataset
        :return: prediction output, [timestamp | value]
        """
        ...

    @final
    def predict(self, X: pd.DataFrame, exp_dir: str = None) -> pd.DataFrame:
        """call this to run prediction

        :param X: features dataset
        :param exp_dir: dir to store prediction result
        :return: prediction output, [timestamp | value]
        """
        # run your custom prediction
        prediction_results = self.run_prediction(X)

        # store if dir is provided
        if exp_dir is not None:
            prediction_results.to_csv(f'{exp_dir}\\{self.model_type}_{self.model_name}_prediction.csv')
        return prediction_results

    @abstractmethod
    def custom_save(self, model: object, filename: str):
        """Use your own dataformat to save your model here

        :param filename: filename or path
        """
        ...

    @abstractmethod
    def custom_load(self, filename: str) -> object:
        """Use your own dataformat to load your model here

        :param filename: filename or path
        :return: your loaded model
        """
        # return model
        ...

    @final
    def save(self, exp_dir: str):
        """call this to save self.model.

        :param exp_dir: dir name or path to dir
        """
        self.custom_save(model=self.model, filename=f'{exp_dir}\\{self.model_type}_{self.model_name}')

    @final
    def load(self, exp_dir: str):
        """call this to load a retrained model

        :param exp_dir: dir name or path to dir
        """
        self.model = self.__custom_load(filename=f'{exp_dir}\\{self.model_type}_{self.model_name}')
