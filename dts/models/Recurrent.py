import numpy as np
from keras.layers import LSTMCell, GRUCell, RNN, SimpleRNNCell, Dense, Input, Flatten, Concatenate
from keras import Model
from keras import backend as K
from keras.metrics import get
from tqdm import tqdm


class RecurrentNN(object):
    """
    A wrapper around the RNN, LSTM and GRU classes that allows to build model
    and performs predictions using two different multi-step forecasting strategies:
    Multiple Input Multiple Output (MIMO) and Recursive
    """

    def __init__(self, layers, cell_type, cell_params):
        """
        Build the rnn with the given number of layers.
        :param layers: list
            list of integers. The i-th element of the list is the number of hidden neurons for the i-th layer.
        :param cell_type: 'gru', 'rnn', 'lstm'
        :param cell_params: dict
            A dictionary containing all the paramters for the RNN cell.
            see keras.layers.LSTMCell, keras.layers.GRUCell or keras.layers.SimpleRNNCell for more details.
        """
        # init params
        self.model = None
        self.horizon = None
        self.layers = layers
        self.cell_params = cell_params
        if cell_type == 'lstm':
            self.cell = LSTMCell
        elif cell_type == 'gru':
            self.cell = GRUCell
        elif cell_type == 'rnn':
            self.cell = SimpleRNNCell
        else:
            raise NotImplementedError('{0} is not a valid cell type.'.format(cell_type))
        # Build deep rnn
        self.rnn = self._build_rnn()

    def _build_rnn(self):
        cells = []
        for _ in range(self.layers):
            cells.append(self.cell(**self.cell_params))
        deep_rnn = RNN(cells, return_sequences=False, return_state=False)
        return deep_rnn

    def build_model(self, input_shape, horizon):
        pass

    def predict(self, inputs):
        pass

    def evaluate(self, inputs):
        pass

    def _eval(self, y, y_hat):
        results = []
        for m in self.model.metrics:
            if isinstance(m, str):
                results.append(K.eval(K.mean(get(m)(y, y_hat))))
            else:
                results.append(K.eval(K.mean(m(y, y_hat))))
        return results


class RecurrentNN_MIMO(RecurrentNN):
    """
    Recurrent Neural network using MIMO forecasting startegy.
    """

    def build_model(self, input_shape, horizon, exogenous_shape=None):
        """
        Create a Model that takes as inputs:
            - 3D Tensor of shape (batch_size, window_size, n_features)
            - (optional) 3D Tensor of shape (batch_size, window_size, n_features-1)
        and outputs:
            - 2D tensor of shape (batch_size, horizon)

        :param input_shape:
            (window_size, n_features)
        :param horizon: int
            The forecasting horizon
        :param conditions_shape:
            (horizon, n_features)
        :return: a keras Model
        """
        self.horizon = horizon
        if len(input_shape) < 2:
            input_shape = (input_shape[0], 1)
        inputs = Input(shape=input_shape, dtype='float32', name='input')
        # [batch_size, hidden_state_length]
        out_rnn = self.rnn(inputs)

        if exogenous_shape is not None:
            # Include exogenous in the prediction
            exogenous = Input(exogenous_shape, dtype='float32', name='exogenous')  # [batch_size, horizon, n_features]
            out_rnn = Dense(horizon, activation='relu')(out_rnn)
            ex = Flatten()(exogenous)                                              # [batch_size, horizon * n_features]
            ex = Dense(horizon, activation='relu')(ex)
            out_rnn = Concatenate()([out_rnn, ex])                                 # [batch_size, 2*horizon]

        # [batch_size, horizon]
        outputs = Dense(horizon, activation=None)(out_rnn)

        if exogenous_shape is not None:
            self.model = Model(inputs=[inputs, exogenous], outputs=outputs)
        else:
            self.model = Model(inputs=[inputs], outputs=[outputs])
        self.model.summary()
        return self.model

    def predict(self, inputs):
        """
        :param inputs: np.array
            (batch_size, window_size, n_features)
        :return: np.array
            (batch_size, horizon)
        """
        return self.model.predict(inputs)

    def evaluate(self, inputs, fn_inverse=None, fn_plot=None):
        try:
            X, y_exog, y = inputs
            y_hat = self.model.predict([X, y_exog])
        except:
            X, y = inputs
            y_hat = self.model.predict(X)
        y_hat = np.asarray(y_hat, dtype=y.dtype)

        if fn_inverse is not None:
            y_hat = fn_inverse(y_hat)
            y = fn_inverse(y)

        if fn_plot is not None:
            fn_plot([y, y_hat])

        return self._eval(y, y_hat)


class RecurrentNN_Rec(RecurrentNN):
    """
    Recurrent Neural network using Recursive forecasting startegy.
    The model's training and predictions phase are different.
    """

    def __init__(self, *args, **kwargs):
        self.return_sequence = False
        super().__init__(*args, **kwargs)

    def build_model(self, input_shape, horizon):
        """
        Create a Model that takes as inputs:
            - 3D Tensor of shape (batch_size, window_size, n_features)
        and outputs:
            - 2D tensor of shape (batch_size, 1)

        :param input_shape:
            (window_size, n_features)
        :param horizon: int
            The forecasting horizon
        :return: a keras Model
        """
        self.horizon = horizon
        if len(input_shape) < 2:
            input_shape = (input_shape[0], 1)
        inputs = Input(shape=input_shape, dtype='float32')
        out_rnn = self.rnn(inputs)                    # [batch_size, hidden_state_length]
        outputs = Dense(1, activation=None)(out_rnn)  # [batch_size, 1]

        self.model = Model(inputs=[inputs], outputs=[outputs])
        self.model.summary()
        return self.model

    def predict(self, inputs, exogenous=None):
        """
        Perform recursive prediction by feeding the network input at time t+1 with the prediction at
        time t. This is repeted 'horizon' number of time.

        :param input: np.array
            (batch_size, window_size, n_features), n_features is supposed to be 1 (univariate time-series)
        :param exogenous: np.array
            exogenous feature for the loads to be predicted
            (batch_size, horizon, n_exog_features)
        :return: np.array
            (batch_size, horizon)
        """
        input_seq = inputs                                         # (batch_size, n_timestamps, n_features)
        output_seq = np.zeros((input_seq.shape[0], self.horizon))  # (batch_size, horizon)
        for i in tqdm(range(self.horizon)):
            if self.return_sequence:
                output = self.model.predict(input_seq)             # [batch_size, input_timesteps]
                output = output[:,-1:]
            else:
                output = self.model.predict(input_seq)             # [batch_size, 1]
            input_seq[:, :-1, :] = input_seq[:, 1:, :]
            input_seq[:, -1:, 0] = output
            if exogenous is not None:
                input_seq[:, -1, 1:] = exogenous[:, i, :]
            # input_seq = np.concatenate([input_seq[:, 1:, :], np.expand_dims(output,axis=-1)], axis=1)
            output_seq[:, i] = output[:,0]
        return output_seq

    def evaluate(self, inputs, fn_inverse=None, fn_plot=None):
        try:
            X, y = inputs
        except:
            X, y, _ = inputs
        try:
            X, exogenous = X
        except:
            exogenous = None
        y_hat = self.predict(X, exogenous)

        if fn_inverse is not None:
            y_hat = fn_inverse(y_hat)
            y = fn_inverse(y)

        if fn_plot is not None:
            fn_plot([y, y_hat])

        return self._eval(y, y_hat)
