import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import copy

# from preprocessing.outlier_detection import hampel_filter
from models.base_model import BaseModel

torch.manual_seed(0)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        # self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """forward calculating the attention context and weights
        :param hidden: (batch_size, 1, hidden_dim)
        :param encoder_outputs: (batch_size, seq_len, hidden_dim)
        :return: attention context and weights
        """
        # Repeat decoder hidden state across sequence length
        hidden = hidden.repeat(1, encoder_outputs.size(1), 1)  # (batch_size, seq_len, hidden_dim)

        # concatenate hidden and encoder outputs
        combined = torch.cat((hidden, encoder_outputs), dim=2)  # (batch_size, seq_len, hidden_dim * 2)

        # calculate attention scores
        energy = torch.tanh(self.attn(combined))  # (batch_size, seq_len, hidden_dim)
        scores = self.v(energy).squeeze(2)  # (batch_size, seq_len)

        # apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len)

        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim*num_dir)

        return context, attn_weights


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidir=False):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidir)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)    # if bidir
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)    # not bidir
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # output: (batch_size, seq_len, num_layer*hidden_dim)
        # hidden: (num_layers*2, batch_size, hidden_dim)        -> *2 for num_layers if bidirectional=True
        # cell: (num_layers*2, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(x, (h0, c0))
        return output, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim=1, hidden_dim=16, num_layers=1, num_heads=4, bidir=False, use_attention=True):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_attention = use_attention
        self.lstm = nn.LSTM(10 + (2 * hidden_dim), hidden_dim, self.num_layers, batch_first=True,
                            bidirectional=bidir)
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.attention = Attention(hidden_dim)
        # self.attention = MultiHeadAttention(hidden_dim, num_heads)

    def forward(self, x, hidden, cell, encoder_out):
        """Calculate decoder output using attention from encoder input.

        :param x: decoder input, shape: (batch_size, 1, encoder_out_dim)
        :param hidden: encoder hidden state, shape: (num_layers*num_directions, batch_size, hidden_dim)
        :param cell: encoder cell state, shape: (num_layers*num_directions, batch_size, hidden_dim)
        :param encoder_out: encoder output, shape: (batch_size, seq_len, hidden_dim*num_directions)
        :return: decoder output, shape: (batch_size, 1, seq_len)
        """
        if self.use_attention:
            # Compute attention
            # context: (batch_size, 1, hidden_dim*num_directions)
            # att_weights: (batch_size, seq_len)
            context, att_weights = self.attention(hidden[-1].unsqueeze(1), encoder_out)
        else:
            # use equal attention weights
            seq_len = encoder_out.shape[1]
            batch_size = x.shape[0]
            value = 1 / seq_len
            att_weights = torch.full((batch_size, seq_len), value, device=x.device)
            context = torch.bmm(att_weights.unsqueeze(1), encoder_out)

        # Concatenate context vector and decoder input
        # x: (batch_size, 1, hidden_dim*num_directions + enc_output_dim)
        x = x.repeat(1, 1, 10)
        x = torch.cat((x, context), dim=2)

        # output: (batch_size, seq_len, num_layer*hidden_dim)
        # hidden: (num_layers*2, batch_size, hidden_dim)        -> *2 for num_layers if bidirectional=True
        # cell: (num_layers*2, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        prediction = self.fc(output)
        return prediction, hidden, cell, att_weights


class EncDecLSTM(nn.Module):
    def __init__(self, encoder, decoder, target_length):
        super(EncDecLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, dec_init=None, teacher_forcing_ratio=0.5, target_values=None):
        """Forward pass the input sequences, containing feature space through the
        Encoder-Attention-Decoder model and output the predicted sequence

        :param x: input sequences tensor
        :param dec_init: encoder input price value
        :return: prediction sequence
        """
        batch_size, sequence_length, input_dim = x.size()
        encoder_output, hidden, cell = self.encoder(x)
        # encoder_output, hidden, cell, feature_attn_weights = self.encoder(x)
        # Todo: use last encoder input price value as init for first decoder value
        # decoder_input = x[:, -1:, :]
        if dec_init is not None:
            decoder_input = dec_init
        else:
            decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)

        out = torch.zeros(batch_size, self.target_length, 1).to(x.device)
        # Todo: compare to zero init hidden and cell variables

        # step by step decoding
        # Todo: compare autoregressive decoding with seq2seq decoding (replace loop, torch.reshape..,
        #  linear to seq length in decoder for fc)
        # out, hidden, cell, attn_weights = self.decoder(decoder_input, hidden, cell, encoder_output)
        for t in range(self.target_length):
            decoder_output, hidden, cell, attn_weights = self.decoder(decoder_input, hidden, cell, encoder_output)
            out[:, t, :] = decoder_output.squeeze(1)

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            decoder_input = target_values[:, t].unsqueeze(1).unsqueeze(1) if teacher_force and target_values is not None else decoder_output
            # ecoder_input = decoder_output

        out = torch.reshape(out, (out.size(0), out.size(1)))
        # out = out.squeeze(1)
        out = self.activation(out)
        return out


class EncoderDecoderAttentionLSTM(BaseModel):
    def __init__(self, target_length, features, target, hidden_size=64, num_layers=3, use_attention=True):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target = target
        self.features = features
        self.target_length = target_length
        self.use_attention = use_attention
        # Check For GPU -> If available send model and data to it
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        super(EncoderDecoderAttentionLSTM, self).__init__(model_name='EncDecAttLSTM',
                                                          model_type='EncoderDecoderAttentionLSTM')

    def create_model(self):
        """Define your own model under self.model.
        """
        num_heads = 1
        input_size = len(self.features)
        self.input_length = 24
        # case seq2seq decoder: use output_size = self.output_length
        # case autoregressive decoder: use output_size = 1
        output_size = 24

        Enc = Encoder(input_dim=input_size, hidden_dim=self.hidden_size, num_layers=self.num_layers, bidir=True)
        # Enc = EncoderWithFeatureAttention(input_dim=input_size, hidden_dim=hidden_size, num_layers=num_layers)
        Dec = Decoder(output_dim=output_size, hidden_dim=self.hidden_size, num_layers=self.num_layers,
                      num_heads=num_heads, bidir=True, use_attention=self.use_attention)
        self.model = EncDecLSTM(encoder=Enc, decoder=Dec, target_length=self.target_length)

        self.model = self.model.to(self.device)

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame = None,
              y_val: pd.DataFrame = None, X_test: pd.DataFrame = None,
              y_test: pd.DataFrame = None, n_epochs: int = 500, batch_size: int = 1024,
              learning_rate: float = 0.001) -> pd.DataFrame | None:
        """train the model on the training data.
        test and validation data can be used only for evaluation (if available).

        :param X_train: training features dataset
        :param y_train: training target values
        :param X_val: validation features' dataset
        :param y_val: validation target values
        :param X_test: testing features' dataset
        :param y_test: testing target values
        :param n_epochs: number of training iterations
        :param batch_size: size of each processed chunk of data in trainings loop
        :param learning_rate: learning rate
        :return: training history (losses while training, if available else None) [epoch | train_loss | test_loss]
        """

        self.target_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()  #

        start = 22560
        end = 37248

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
        # y_scaled = y.reshape(-1, 1)

        # convert dataset to tensors suitable for training the model
        X_train_tensors = self.__prepare_feature_dataset(X_train)
        X_train_tensors = torch.cat((X_train_tensors[:start], X_train_tensors[end:]))
        y_train_tensors = self.__prepare_target_dataset(y_train)
        y_train_tensors = torch.cat((y_train_tensors[:start], y_train_tensors[end:]))
        X_test_tensors = self.__prepare_feature_dataset(X_test)
        y_test_tensors = self.__prepare_target_dataset(y_test)
        X_val_tensors = self.__prepare_feature_dataset(X_val)
        y_val_tensors = self.__prepare_target_dataset(y_val)

        loss_fn = torch.nn.L1Loss()  # mean-squared error for regression
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        history = self.__training_loop(n_epochs=n_epochs,
                                       optimiser=optimizer,
                                       loss_fn=loss_fn,
                                       X_train=X_train_tensors,
                                       y_train=y_train_tensors,
                                       X_test=X_test_tensors,
                                       y_test=y_test_tensors,
                                       batch_size=batch_size)
        return history

    def __training_loop(self, n_epochs, optimiser, loss_fn, X_train, y_train,
                        X_test, y_test, batch_size):
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_history = {'epoch': [], 'train loss': [], 'test loss': []}
        # timestep_weights = np.asarray(generate_linear_weights(len(train_loader)))
        print(datetime.datetime.now())
        tr = 0.5
        tr_reduce = tr / n_epochs

        min_loss = 1000
        model_state = self.model.state_dict()
        for epoch in range(n_epochs):
            # tr -= tr_reduce
            train_losses = []
            self.model.train()
            timestep = 0
            # for seq, labels in tqdm(train_loader):
            for seq, labels in train_loader:
                # current_weight = timestep_weights[timestep]
                decoder_input = seq[:, -1:, -1:]
                outputs = self.model.forward(seq, decoder_input, teacher_forcing_ratio=tr, target_values=labels)
                optimiser.zero_grad()

                loss = loss_fn(outputs, labels)
                # loss = loss * current_weight
                loss.backward()
                optimiser.step()
                train_losses.append(loss.to('cpu').item())
                timestep += 1
            # test loss
            test_losses = []
            self.model.eval()
            # for seq, labels in tqdm(test_loader):
            for seq, labels in test_loader:
                decoder_input = seq[:, -1:, -1:]
                test_preds = self.model(seq, decoder_input)

                test_loss = loss_fn(test_preds, labels)
                test_losses.append(test_loss.to('cpu').item())
            if epoch % 1 == 0:
                print(
                    f'Epoch {epoch + 1}:\t|\tTrain_Loss: {round(sum(train_losses) / len(train_losses), 5)}\t|\t'
                    f'Test_Loss: {round(sum(test_losses) / len(test_losses), 5)}')
                train_history['epoch'].append(epoch + 1)
                train_history['train loss'].append((sum(train_losses) / len(train_losses)))
                train_history['test loss'].append((sum(test_losses) / len(test_losses)))
            if min_loss > (sum(test_losses) / len(test_losses)):
                model_state = copy.deepcopy(self.model.state_dict())
                # torch.save(model_state, f'BiEncDecAttLSTM_small_autoreg_val.pth')
                min_loss = (sum(test_losses) / len(test_losses))

        # import matplotlib.pyplot as plt
        train_history = pd.DataFrame(train_history).set_index('epoch')
        # plt.show(block=True)
        # torch.save(model_state, f'BiEncDecAttLSTM.pth')

        return train_history

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

    def run_prediction(self, X: pd.DataFrame, batch_size=1024) -> pd.DataFrame:
        """run prediction on your defined model.

        :param X: features dataset
        :return: prediction output, [timestamp | value]
        """
        self.model.eval()
        pred_length = X.shape[0]
        X = X.reset_index(names='timestamp')
        timestamps_x = X['timestamp']
        X = X.drop(['timestamp'], axis=1)
        start_timestamp = timestamps_x[0]

        # scale features using the training scaler
        X = self.feature_scaler.transform(X[self.features])
        X_pred_tensors = self.__prepare_feature_dataset(X)
        dataset = TensorDataset(X_pred_tensors)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        pred_list = []
        for seq in data_loader:
            X_batch = seq[0]
            decoder_input = X_batch[:, -1:, -1:]
            predictions = self.model.forward(X_batch, decoder_input)
            pred_list.append(predictions.to('cpu').detach().numpy())
        # predictions = self.model.forward(X_pred_tensors, X_pred_tensors[:, -1:, -1:])
        sequences = np.concatenate(pred_list)
        # pred_np = predictions.to('cpu').detach().numpy()

        total_length = len(sequences) + self.target_length - 1  # Total number of positions covered
        sum_values = np.zeros(total_length)
        count_values = np.zeros(total_length)
        min_values = np.full(total_length, np.inf)  # initialize with infinity
        max_values = np.full(total_length, -np.inf)  # initialize with negative infinity

        for i, seq in enumerate(sequences):
            for j in range(self.target_length):
                index = i + j  # global index in the expanded array
                value = seq[j]

                sum_values[index] += value
                count_values[index] += 1
                min_values[index] = min(min_values[index], value)
                max_values[index] = max(max_values[index], value)

        mean_values = sum_values / count_values  # Compute the mean

        end_timestamp = start_timestamp + pd.Timedelta(hours=total_length - 1)
        timestamps = pd.date_range(start_timestamp, end_timestamp, freq='1H')

        # pred_steps = pred_np[::self.target_length]
        # pred_shaped = np.reshape(pred_steps, pred_steps.shape[0] * pred_steps.shape[1]).reshape(-1, 1)
        # # rescale using training scaler and reshape into one continuous sequence
        # pred_sequence = self.target_scaler.inverse_transform(pred_shaped).reshape(-1)
        pred_sequence = self.target_scaler.inverse_transform(mean_values.reshape(-1, 1)).reshape(-1)
        mins = self.target_scaler.inverse_transform(min_values.reshape(-1, 1)).reshape(-1)
        maxs = self.target_scaler.inverse_transform(max_values.reshape(-1, 1)).reshape(-1)

        df_result = pd.DataFrame({'timestamp': timestamps[:pred_sequence.shape[0]],
                                  'day_ahead_price_predicted': pred_sequence,
                                  'pred_min': mins,
                                  'pred_max': maxs})
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
        X = []  # instantiate X
        for i in range(len(features_seq)):
            # find the end of the input, output sequence
            end_ix = i + self.input_length
            if end_ix > len(features_seq):
                break
            # gather input and output of the pattern
            seq_x = features_seq[i:end_ix]
            X.append(seq_x)
        return np.asarray(X)

    def __split_target_sequences(self, target_seq):
        y = []  # instantiate y
        for i in range(len(target_seq)):
            # find the end of the input, output sequence
            end_ix = i + self.input_length
            if end_ix > len(target_seq):
                break
            # gather input and output of the pattern
            seq_y = target_seq[i:end_ix, -1]
            y.append(seq_y)
        return np.asarray(y)

    def custom_save(self, model: object, filename: str):
        """Use your own dataformat to save your model here

        :param filename: filename or path
        """
        torch.save(self.model.state_dict(), filename)
        pass

    def custom_load(self, filename: str) -> object:
        """Use your own dataformat to load your model here

        :param filename: filename or path
        :return: your loaded model
        """
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        return self.model
