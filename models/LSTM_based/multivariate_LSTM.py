import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.base_model import BaseModel


class MultivarLSTM(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size, output_size):
        super().__init__()
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # number of input features
        self.output_size = output_size  # output sequence length
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2,
                            bidirectional=True)
        self.linear = nn.Linear(hidden_size * 24 * 2, output_size)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        device = x.device
        # hidden state init
        h_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device))
        # cell state init
        c_0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=device))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # (input, hidden, and internal state)
        # reduce dimension to the required output sequence length
        predictions = self.linear(output.reshape(output.size(0), output.size(1) * output.size(2)))
        pred_out = self.activation(predictions)

        return pred_out


class MultivariateBiLSTM(BaseModel):
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.input_length = 24
        self.hidden_layer_size = 256
        self.num_layers = 12
        self.output_length = 24
        # Check For GPU -> If available send model and data to it
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        super(MultivariateBiLSTM, self).__init__(model_name='MultivarLSTM',
                                                 model_type='MultivariateBidirectionalLSTM')

    def create_model(self):
        self.model = MultivarLSTM(num_layers=self.num_layers, hidden_size=self.hidden_layer_size,
                                  input_size=len(self.features), output_size=self.output_length)
        self.model = self.model.to(self.device)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame = None, y_val: pd.DataFrame = None,
              X_test: pd.DataFrame = None, y_test: pd.DataFrame = None,
              n_epochs=500, batch_size=1024, learning_rate=0.001) -> pd.DataFrame | None:
        self.target_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()

        # select features and target columns
        if len(self.features) > 0:
            # use only selected features (of self.features not defined: use all columns as features)
            X_train = X_train[self.features]
            X_test = X_test[self.features]
            X_val = X_val[self.features]
        y_train = y_train[self.target].values
        y_test = y_test[self.target].values
        y_val = y_val[self.target].values

        # fit scalar on training data
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train.reshape(-1, 1))

        X_train = self.feature_scaler.transform(X_train)
        y_train = self.target_scaler.transform(y_train.reshape(-1, 1))
        X_test = self.feature_scaler.transform(X_test)
        y_test = self.target_scaler.transform(y_test.reshape(-1, 1))
        X_val = self.feature_scaler.transform(X_val)
        y_val = self.target_scaler.transform(y_val.reshape(-1, 1))

        # convert dataset to tensors suitable for training the model
        X_train_tensors = self.__prepare_feature_dataset(X_train)
        y_train_tensors = self.__prepare_target_dataset(y_train)
        X_test_tensors = self.__prepare_feature_dataset(X_test)
        y_test_tensors = self.__prepare_target_dataset(y_test)
        X_val_tensors = self.__prepare_feature_dataset(X_val)
        y_val_tensors = self.__prepare_target_dataset(y_val)

        history = self.__training_loop(n_epochs=n_epochs,
                                       X_train=X_train_tensors,
                                       y_train=y_train_tensors,
                                       X_test=X_test_tensors,
                                       y_test=y_test_tensors,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate)
        pass

    def __training_loop(self, n_epochs, X_train, y_train, X_test, y_test, batch_size, learning_rate):
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        loss_fn = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        train_history = {'epoch': [], 'train loss': [], 'test loss': []}

        for epoch in range(n_epochs):
            train_losses = []
            self.model.train()
            # for seq, labels in tqdm(train_loader):
            for seq, labels in train_loader:
                outputs = self.model(seq)
                optimizer.zero_grad()

                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.to('cpu').item())
            # test loss
            test_losses = []
            self.model.eval()
            # for seq, labels in tqdm(test_loader):
            for seq, labels in test_loader:
                test_preds = self.model(seq)
                test_loss = loss_fn(test_preds, labels)
                test_losses.append(test_loss.to('cpu').item())
            if epoch % 1 == 0:
                print(
                    f'Epoch {epoch + 1}:\t|\tTrain_Loss: {round(sum(train_losses) / len(train_losses), 5)}\t|\t'
                    f'Test_Loss: {round(sum(test_losses) / len(test_losses), 5)}')
                train_history['epoch'].append(epoch + 1)
                train_history['train loss'].append((sum(train_losses) / len(train_losses)))
                train_history['test loss'].append((sum(test_losses) / len(test_losses)))
        train_history = pd.DataFrame(train_history).set_index('epoch')

        return train_history

    def run_prediction(self, X: pd.DataFrame, batch_size=1024) -> pd.DataFrame:
        """run prediction on your defined model.

        :param X: features dataset
        :return: prediction output, [timestamp | value]
        """
        X = X.reset_index(names='timestamp')
        timestamps = X['timestamp']
        X = X.drop(['timestamp'], axis=1)

        # scale features using the training scaler
        X = self.feature_scaler.transform(X[self.features])
        # prepare dataset
        X_pred_tensors = self.__prepare_feature_dataset(X)
        dataset = TensorDataset(X_pred_tensors)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        pred_list = []
        # predict each batch
        for seq in data_loader:
            X_batch = seq[0]
            predictions = self.model.forward(X_batch)
            pred_list.append(predictions.to('cpu').detach().numpy())

        pred_np = np.concatenate(pred_list)
        pred_steps = pred_np[::self.output_length]
        pred_shaped = np.reshape(pred_steps, pred_steps.shape[0] * pred_steps.shape[1]).reshape(-1, 1)
        # rescale using training scaler and reshape into one continuous sequence
        pred_sequence = self.target_scaler.inverse_transform(pred_shaped).reshape(-1)

        df_result = pd.DataFrame({'timestamp': timestamps[:pred_sequence.shape[0]],
                                  'day_ahead_price_predicted': pred_sequence})
        return df_result

    def __prepare_feature_dataset(self, X):
        X_seq = self.__split_feature_sequences(features_seq=X)

        X_tensor = Variable(torch.Tensor(X_seq))
        X_tensor_format = torch.reshape(X_tensor, (X_tensor.shape[0], self.input_length, X_tensor.shape[2]))
        X_tensor_format = X_tensor_format.to(self.device)

        return X_tensor_format

    def __prepare_target_dataset(self, y):
        y_seq = self.__split_target_sequences(y)

        y_tensor = Variable(torch.Tensor(y_seq))
        y_tensor = y_tensor.to(self.device)

        return y_tensor

    def __split_feature_sequences(self, features_seq):
        X = []  # instantiate X and y
        for i in range(len(features_seq)):
            # find the end of the sequence
            end_ix = i + self.input_length
            # check if we are beyond the dataset
            if end_ix > len(features_seq):
                break
            # gather input and output of the pattern
            seq_x = features_seq[i:end_ix]
            X.append(seq_x)
        return np.array(X)

    def __split_target_sequences(self, target_seq):
        y = []  # instantiate y
        for i in range(len(target_seq)):
            # find the end of the sequence
            end_ix = i + self.input_length
            # check if we are beyond the dataset
            if end_ix > len(target_seq):
                break
            # gather input and output of the pattern
            seq_y = target_seq[i:end_ix, -1]
            y.append(seq_y)
        return np.array(y)

    def create_scalers(self, X_train, y_train):
        self.target_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()

        # select features and target columns
        if len(self.features) > 0:
            # use only selected features (of self.features not defined: use all columns as features)
            X_train = X_train[self.features]
        y_train = y_train[self.target].values

        # fit scalar on training data
        self.feature_scaler.fit(X_train)
        self.target_scaler.fit(y_train.reshape(-1, 1))

    def custom_load(self, filename: str) -> object:
        """Use your own dataformat to load your model here

        :param filename: filename or path
        :return: your loaded model
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        return self.model

    def custom_save(self, model: object, filename: str):
        """Use your own dataformat to save your model here

        :param filename: filename or path
        """
        torch.save(self.model.state_dict(), filename)
        pass
