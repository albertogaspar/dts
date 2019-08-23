import keras.backend as K
from keras.metrics import get
from keras import Model
from keras.layers import Dense, Dropout, Input, Flatten, Add, BatchNormalization, Concatenate
from keras import regularizers, initializers, constraints, activations
import numpy as np


class FFNN:
    def __init__(self,
                 layers,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 dropout=0.0,
                 recursive_forecast=False,
                 ):
        """
        Base FeedForward Neural Network.

        :param layers: list of integers. The i-th elem. of the list is the number of units of the i-th layer.
        :param dropout: An integer or tuple/list of a single integer, specifying the length
            of the 1D convolution window.
        :param recursive_forecast: an integer or tuple/list of a single integer, specifying the dilation rate
            to use for dilated convolution.
            Usually dilation rate increases exponentially with the depth of the network.
        :param for all the other parameters see keras.layers.Dense
        """
        self.layers = layers
        self.dropout = dropout
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.recursive_forecast = recursive_forecast
        self.model = None
        self.horizon = None

    def model_inputs(self, input_shape, conditions_shape=None):
        """
        :param input_shape: np.array
            (window_size, n_features)
        :param conditions_shape: np.array
            (horizon, n_features)
        :return: a tuple containing:
            - a list containing all the Input Layers needed by the model
            - the tensor that has to be feeded to the subsequent layers of the archotecture
        """
        inputs = Input(shape=input_shape, name='input')
        if conditions_shape is not None:
            conditions = Input(shape=conditions_shape, name='exogenous')
            # pass through different filters in order for them to have = no. channels
            out = Concatenate(axis=1)(
                [Dense(units=128, activation='sigmoid')(inputs),
                 Dense(units=128, activation='tanh')(conditions)]
            )  # concatenate over temporal axis
            return [inputs, conditions], out
        return inputs, inputs

    def build_model(self, input_shape, horizon, conditions_shape=None):
        """
        Create a Model that takes as inputs:
            - 3D tensor of shape tesor (batch_size, window_size, n_features)
            - 3D Tensor of shape (batch_size, window_size, n_features)
        and outputs:
            - 2D tensor of shape (batch_size, 1) or (batch_size, horizon), depending on the value of
            recursive_forecast.
        :param input_shape: np.array
            (window_size, n_features)
        :param horizon: int
            the forecasting horizon
        :param conditions_shape: np.array
            (horizon, n_features)
        :return: a keras Model
        """
        pass

    def predict(self, inputs):
        if self.recursive_forecast:
            return self._predict_rec(inputs)
        else:
            return self.model.predict(inputs)

    def _predict_rec(self, inputs):
        """
        Perform prediction when the model's recursive_forecast flag is set to True.
        Perform recursive prediction by feeding the network input at time t+1 with the prediction at
        time t. This is repeted 'horizon' number of time.
        If exogenous features are available they are integrated into the fore casting process.

        :param inputs:
            np.array with shape: `(batch, window_size, n_features)`
            or list of tensor having shape:
               - np.array with shape: `(batch, window_size, n_features)`
               - np.array with shape: `(batch, horizon, n_features)`
        :return: np.array
            (batch, horizon)
        """
        try:
            inputs, conditions = inputs
        except ValueError:
            conditions = None

        outputs = np.zeros((inputs.shape[0], self.horizon))  # (batch_size, pred_steps)
        for i in range(self.horizon):
            if conditions is not None:
                next_exog = conditions[:, i:i + 1, :] # exog at time i
                out = self.model.predict([inputs, next_exog]) # output at time i
                inputs = np.concatenate([inputs[:, 1:, :],
                                         np.concatenate([np.expand_dims(out,-1), next_exog], -1)],
                                        1) # shift input and concat exog
            else:
                out = self.model.predict(inputs)  # [batch, 1]
                inputs = np.concatenate([inputs[:, 1:, :],
                                         np.expand_dims(out, -1)],
                                        1)
            outputs[:, i] = out[:, 0]
        return outputs

    def evaluate(self, inputs, fn_inverse=None, fn_plot=None):
        try:
            X, y = inputs
            inputs = X
        except:
            X, conditions, y = inputs
            inputs = [X, conditions]

        y_hat = self.predict(inputs)

        if fn_inverse is not None:
            y_hat = fn_inverse(y_hat)
            y = fn_inverse(y)

        if fn_plot is not None:
            fn_plot([y, y_hat])

        results = []
        for m in self.model.metrics:
            try:
                if isinstance(m, str):
                    results.append(K.eval(K.mean(get(m)(y, y_hat))))
                else:
                    results.append(K.eval(K.mean(m(y, y_hat))))
            except:
                print(m)
                continue
        return results


class SimpleNet(FFNN):

    def build_model(self, input_shape, horizon, conditions_shape=None):
        self.horizon = horizon
        model_inputs, inputs = self.model_inputs(input_shape, conditions_shape)
        out = Flatten()(inputs)
        for units in self.layers:
            out = Dense(units=units, kernel_regularizer=self.kernel_regularizer, activation=self.activation,
                        kernel_initializer=self.kernel_initializer, kernel_constraint=self.kernel_constraint,
                        use_bias=self.use_bias, bias_regularizer=self.bias_regularizer,
                        bias_initializer=self.bias_initializer, bias_constraint=self.bias_constraint)(out)
            out = Dropout(self.dropout)(out)
        if self.recursive_forecast:
            out = Dense(units=1, activation='linear')(out)
        else:
            out = Dense(units=self.horizon, activation='linear')(out)
        self.model = Model(model_inputs, out)
        self.model.summary()
        return self.model


class ResNet(FFNN):

    def _residual_block(self, units, inputs):
        out = Dense(units=units, kernel_regularizer=self.kernel_regularizer, activation=self.activation,
                        kernel_initializer=self.kernel_initializer, kernel_constraint=self.kernel_constraint,
                        use_bias=self.use_bias, bias_regularizer=self.bias_regularizer,
                        bias_initializer=self.bias_initializer, bias_constraint=self.bias_constraint)(inputs)
        out = Dropout(self.dropout)(out)
        out = Dense(units=units, kernel_regularizer=self.kernel_regularizer, activation=self.activation,
                        kernel_initializer=self.kernel_initializer, kernel_constraint=self.kernel_constraint,
                        use_bias=self.use_bias, bias_regularizer=self.bias_regularizer,
                        bias_initializer=self.bias_initializer, bias_constraint=self.bias_constraint)(out)
        out = BatchNormalization(trainable=True)(out)

        if K.int_shape(inputs)[-1] != K.int_shape(out)[-1]:
            inputs = Dense(units=units, kernel_regularizer=self.kernel_regularizer, activation=self.activation,
                        kernel_initializer=self.kernel_initializer, kernel_constraint=self.kernel_constraint,
                        use_bias=self.use_bias, bias_regularizer=self.bias_regularizer,
                        bias_initializer=self.bias_initializer, bias_constraint=self.bias_constraint)(inputs)
        out = Add()([inputs, out])
        return out

    def build_model(self, input_shape, horizon, conditions_shape=None):
        self.horizon = horizon
        model_inputs, inputs = self.model_inputs(input_shape, conditions_shape)
        out = Flatten()(inputs)
        for units in self.layers:
            out = self._residual_block(units, out)
        if self.recursive_forecast:
            out = Dense(units=1, activation='linear')(out)
        else:
            out = Dense(units=self.horizon, activation='linear')(out)
        self.model = Model(model_inputs, out)
        self.model.summary()
        return self.model

