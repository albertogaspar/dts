from keras.layers import Concatenate, Conv1D, Add, Multiply, Input, Dense, Lambda, Dropout, Activation, \
    SpatialDropout1D, Layer
from keras.regularizers import l2
import numpy as np
from keras import Model
import keras.backend as K
from keras.metrics import get


def tcn_residual_block(inputs,
                       filters=1,
                       kernel_size=2,
                       dilation_rate=None,
                       kernel_initializer='glorot_normal',
                       bias_initializer='glorot_normal',
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       use_bias=False,
                       dropout_rate=0.0
                       ):
    """
    TCN Residual Block.
    TCN uses zero-padding to maintain `steps` value of the ouput equal to the one in the input.
    See [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling]
    (https://arxiv.org/abs/1803.01271).
    A Residual Block is obtained by stacking togeather (2x) the following:
        - 1D Dilated Convolution
        - WeightNorm (here absent)
        - ReLu
        - Spatial Dropout
    And adding the input after trasnforming it with a 1x1 Conv
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Usually dilation rate increases exponentially with the depth of the network.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
    # Input shape
        3D tensor with shape: `(batch, steps, n_features)`
    # Output shape
        3D tensor with shape: `(batch, steps, filters)`
    """
    outputs = Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     use_bias=use_bias,
                     bias_initializer=bias_initializer,
                     bias_regularizer=bias_regularizer,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     padding='causal',
                     dilation_rate=dilation_rate,
                     activation='relu')(inputs)
    outputs = SpatialDropout1D(dropout_rate, trainable=True)(outputs)
    # skip connection
    skip_out = Conv1D(filters=filters, kernel_size=1, activation='linear')(inputs)
    residual_out = Add()([outputs, inputs])
    return skip_out, residual_out


def wavenet_residual_block(inputs,
                           filters=1,
                           kernel_size=2,
                           dilation_rate=None,
                           kernel_initializer='glorot_normal',
                           bias_initializer='glorot_normal',
                           kernel_regularizer=None,
                           bias_regularizer=None,
                           use_bias=False,
                           dropout_rate=0.0
                           ):
    """
    Wavenet Residual Block.
    See [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499).
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Usually dilation rate increases exponentially with the depth of the network.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
    # Input shape
        3D tensor with shape: `(batch, steps, n_features)`
    # Output shape
        3D tensor with shape: `(batch, steps, filters)`
    """
    if kernel_regularizer is None:
        kernel_regularizer = l2(0.001)
    bias_regularizer = bias_regularizer
    if bias_regularizer is None:
        bias_regularizer = l2(0.001)

    outputs = Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     use_bias=use_bias,
                     bias_initializer=bias_initializer,
                     bias_regularizer=bias_regularizer,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     padding='causal',
                     dilation_rate=dilation_rate,
                     activation='relu')(inputs)
    outputs = SpatialDropout1D(dropout_rate, trainable=True)(outputs)
    sig_out = Activation('sigmoid')(outputs)
    tanh_out = Activation('tanh')(outputs)
    outputs = Multiply()([sig_out, tanh_out])
    if K.int_shape(outputs) != K.int_shape(inputs):
        inputs = Conv1D(filters=1, kernel_size=1, activation='linear')(inputs)
    residual_out = Add()([outputs, inputs])
    return residual_out


def simple_residual_block(inputs,
                          filters=1,
                          kernel_size=2,
                          dilation_rate=None,
                          kernel_initializer='glorot_normal',
                          bias_initializer='glorot_normal',
                          kernel_regularizer=None,
                          bias_regularizer=None,
                          use_bias=False,
                          dropout_rate=0.0):
    """
    Simple Residual Block for building Temporal Convolutional Neural Networks.
    See [Conditional Time Series Forecasting with Convolutional Neural Networks]
    (https://arxiv.org/abs/1703.04691)
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Usually dilation rate increases exponentially with the depth of the network.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
    # Input shape
        3D tensor with shape: `(batch, steps, n_features)`
    # Output shape
        3D tensor with shape: `(batch, steps, filters)`
    """
    if kernel_regularizer is None:
        kernel_regularizer = l2(0.001)
    bias_regularizer = bias_regularizer
    if bias_regularizer is None:
        bias_regularizer = l2(0.001)

    outputs = Conv1D(filters=filters,
                     kernel_size=kernel_size,
                     use_bias=use_bias,
                     bias_initializer=bias_initializer,
                     bias_regularizer=bias_regularizer,
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer,
                     padding='causal',
                     dilation_rate=dilation_rate,
                     activation='relu')(inputs)
    # outputs = self.dropout(outputs)
    if K.int_shape(outputs) != K.int_shape(inputs):
        inputs = Conv1D(filters=1, kernel_size=1, activation='linear')(inputs)
    residual_out = Add()([outputs, inputs])
    return residual_out


def conditional_block(inputs,
                      filters=1,
                      kernel_size=2,
                      dilation_rate=None,
                      kernel_initializer='glorot_normal',
                      bias_initializer='glorot_normal',
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      use_bias=False,
                      dropout_rate=0.0):
    """
    Residual Block for building Conditioned Temporal Convolutional Neural Networks.
    See [Conditional Time Series Forecasting with Convolutional Neural Networks]
    (https://arxiv.org/abs/1703.04691)
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer,
            specifying the length of the 1D convolution window.
        dilation_rate: an integer or tuple/list of a single integer, specifying
            the dilation rate to use for dilated convolution.
            Usually dilation rate increases exponentially with the depth of the network.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
    # Input shape
        3D tensor with shape: `(batch, steps, n_features_1)`
        3D tensor with shape: `(batch, steps, n_features_2)`
    # Output shape
        3D tensor with shape: `(batch, steps, filters)`
    """
    # input shape = (batch_size, n_samples, 1), output shape = (batch_size, n_samples, out_channels)
    try:
        if len(inputs) == 1:
            conditions = None
        else:
            inputs, conditions = inputs
    except TypeError:
        conditions = None
    inputs_conv = Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         use_bias=use_bias,
                         bias_initializer=bias_initializer,
                         bias_regularizer=bias_regularizer,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         padding='causal',
                         dilation_rate=1,
                         activation='relu')(inputs)
    # if filters > 1 then the parametrized skip connecion uses 1x1 conv
    inputs_skip_conn = Conv1D(filters=filters, kernel_size=1, activation='linear')(inputs)
    inputs = Add()([inputs_conv, inputs_skip_conn])
    if conditions is not None:
        conditions_conv = Conv1D(filters=filters,
                                 kernel_size=kernel_size,
                                 use_bias=use_bias,
                                 bias_initializer=bias_initializer,
                                 bias_regularizer=bias_regularizer,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer,
                                 padding='causal',
                                 dilation_rate=1,
                                 activation='relu')(conditions)
        # if filters > 1 then the parametrized skip connecion uses 1x1 conv
        conditions_skip_conn = Conv1D(filters=filters, kernel_size=1, activation='linear')(conditions)
        conditions = Add()([conditions_conv, conditions_skip_conn])
        inputs = Concatenate(axis=1)([inputs, conditions])  # concatenate over temporal axis
    return inputs


class TemporalConvNet():

    def __init__(self,
                 layers=None,
                 filters=1,
                 kernel_size=2,
                 dilation_rate=None,
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_normal',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 use_bias=False,
                 dropout_rate=0.0,
                 return_sequence=False):
        self.layers = layers
        self.dilation_rate = dilation_rate
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.return_sequence = return_sequence

    def call(self, inputs):
        pass

    def __call__(self, inputs, *args, **kwargs):
        return self.call(inputs)


class Wavenet(TemporalConvNet):
    """
    Wavenet Network for time series forecasting.
    See [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499).
    """

    def call(self, inputs):
        inputs = Conv1D(filters=self.filters,
                            kernel_size=self.kernel_size,
                            use_bias=self.use_bias,
                            bias_initializer=self.bias_initializer,
                            bias_regularizer=self.bias_regularizer,
                            kernel_initializer=self.kernel_initializer,
                            kernel_regularizer=self.kernel_regularizer,
                            padding='causal',
                            dilation_rate=self.dilation_rate,
                            activation='linear')(inputs)
        skip_outs = []
        for i in range(1, self.layers):
            # output shape = (batch_size, n_samples, out_channels)
            skip_out, res_out = wavenet_residual_block(inputs,
                                                       filters=self.filters,
                                                       kernel_size=self.kernel_size,
                                                       use_bias=self.use_bias,
                                                       bias_initializer=self.bias_initializer,
                                                       bias_regularizer=self.bias_regularizer,
                                                       kernel_initializer=self.kernel_initializer,
                                                       kernel_regularizer=self.kernel_regularizer,
                                                       dilation_rate=2 ** i)
            skip_outs.append(skip_out)
            inputs = res_out
        output = Add()(skip_outs)
        output = Activation('relu')(output)
        output = Conv1D(filters=1,
                            kernel_size=1,
                            activation='relu',
                            kernel_regularizer=self.kernel_regularizer)(output)
        output = Conv1D(filters=1,
                            kernel_size=1,
                            activation='linear',
                            kernel_regularizer=self.kernel_regularizer)(output)
        if self.return_sequence:
            output = Lambda(lambda x: x[:, :, 0])(output)
        else:
            output = Lambda(lambda x: x[:, -1:, 0])(output)
        return output


class ConditionalTCN(TemporalConvNet):
    """
    Wavenet variant used to condition the forecast on exogenous variables.
    See [Conditional Time Series Forecasting with Convolutional Neural Networks]
    (https://arxiv.org/abs/1703.04691)
    """

    def call(self, inputs):
        original_inputs = inputs
        inputs = conditional_block(original_inputs,
                                   filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   use_bias=self.use_bias,
                                   bias_initializer=self.bias_initializer,
                                   bias_regularizer=self.bias_regularizer,
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self.kernel_regularizer,
                                   dilation_rate=1)
        for i in range(1, self.layers):
            # output shape = (batch_size, n_samples, out_channels)
            outputs = simple_residual_block(inputs,
                                            filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            use_bias=self.use_bias,
                                            bias_initializer=self.bias_initializer,
                                            bias_regularizer=self.bias_regularizer,
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            dilation_rate=2 ** i)
            inputs = Add()([inputs, outputs])
        output = Conv1D(filters=1,
                        kernel_size=1,
                        activation='linear',
                        kernel_regularizer=self.kernel_regularizer)(inputs)
        if self.return_sequence:
            output = Lambda(lambda x: x[:, :, 0])(output)
        #            output = Dense(output_shape[0])(output)
        else:
            output = Lambda(lambda x: x[:, -1:, 0])(output)
        return output

    def __call__(self, inputs, *args, **kwargs):
        return self.call(inputs)


class TCN(TemporalConvNet):
    """
    Temporal CNN from
    'An Emprical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modelling'
    Returned model to be compiled with AdamWithWeightnorm (see https://github.com/openai/weightnorm/tree/master/keras)
    """

    def call(self, inputs):
        for i in range(self.layers):
            inputs = tcn_residual_block(inputs,
                                        filters=self.filters,
                                        kernel_size=self.kernel_size,
                                        use_bias=self.use_bias,
                                        bias_initializer=self.bias_initializer,
                                        bias_regularizer=self.bias_regularizer,
                                        kernel_initializer=self.kernel_initializer,
                                        kernel_regularizer=self.kernel_regularizer,
                                        dilation_rate=2 ** i,
                                        )

        if self.return_sequence:
            output = Lambda(lambda x: x[:, :, 0])(inputs)
        else:
            output = Lambda(lambda x: x[:, -1:, 0])(inputs)
        return output


class TCNModel:
    """
    Utility class to create a Keras Model using a predefined TCN Architecture.
    Available architectures are:
        - Wavenet
        - TCN
        - ConditionalTCN
    """

    def __init__(self,
                 layers=None,
                 filters=1,
                 kernel_size=2,
                 dilation_rate=None,
                 kernel_initializer='glorot_normal',
                 bias_initializer='glorot_normal',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 use_bias=False,
                 dropout_rate=0.0,
                 return_sequence=True,
                 tcn_type='conditional_tcn'):
        """
        :param layers: int
            Number of layers for the network
        :param filters: int
            the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        :param kernel_size: int or tuple
            the length of the 1D convolution window
        :param dilation_rate: int
            the dilation rate to use for dilated convolution.
            Usually dilation rate increases exponentially with the depth of the network.
        :param return_sequence: bool
            if True the network output is (batch_size, horizon) ~MIMO training
            else,  the network output is (batch_size, 1) ~Rec training
        :param tcn_type: str
            The TCN architecture that has to be used
        """
        self.layers = layers
        self.dilation_rate = dilation_rate
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.return_sequence = return_sequence
        self.built = False
        self.tcn_type = tcn_type
        self.model = None
        self.out_shape = None
        params = dict(
            layers=layers,
            dilation_rate=dilation_rate,
            filters=filters,
            kernel_size=kernel_size,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            return_sequence=return_sequence,
        )
        if tcn_type == 'conditional_tcn':
            self.tcn = ConditionalTCN(**params)
        elif tcn_type == 'wavenet':
            self.tcn = Wavenet(**params)
        else:
            self.tcn = TCN(**params)

    def build_model(self, input_shape, horizon, conditions_shape=None, use_final_dense=False):
        """
        Create a Model that takes as inputs:
            - 3D Tensor of shape (batch_size, window_size, n_features)
            - (optional) 3D Tensor of shape (batch_size, horizon, n_features-1)
        and outputs:
            - 2D tensor of shape (batch_size, horizon)

        :param input_shape:
            (window_size, n_features)`
        :param horizon: int
            The forecasting horizon
        :param conditions_shape:
            (window_size, n_features)
        :param use_final_dense:
            if True transfrom output using a Dense Net
        """
        # inputs
        self.horizon = horizon
        inputs = Input(shape=input_shape)

        # define model
        if self.tcn_type == 'conditional_tcn':
            if conditions_shape is None:
                output = self.tcn(inputs)
            else:
                conditions = Input(shape=conditions_shape)
                output = self.tcn([inputs, conditions])
        else:
            output = self.tcn(inputs)

        # outputs
        if self.return_sequence and use_final_dense:
            output = Dense(horizon)(output)
        elif self.return_sequence:
            output = Lambda(lambda x: x[:, -horizon:])(output)
        else:
            output = Lambda(lambda x: x[:, -1:])(output)

        # build model
        if conditions_shape is None:
            model = Model(inputs=[inputs], outputs=[output])
        else:
            model = Model(inputs=[inputs, conditions], outputs=[output])
        self.model = model
        model.summary()
        return model

    def predict(self, inputs):
        if self.return_sequence:
            return self.model.predict(inputs)
        else:
            raise NotImplementedError
            # return self._predict_rec(inputs)

    def _predict_rec(self, inputs):
        """
        Recursive Forecast. To be used with return_sequence = False and forecasting horizon > 1.

        :param inputs: list
            Input array to be used for prediction
            (batch_size, window_size, n_features)

            (optional) Exogenous features to be included for prediction pruposes
            (batch_size, horizon, n_features - 1)
        :return:
        """
        try:
            inputs, conditions = inputs
        except ValueError:
            conditions = None

        outputs = np.zeros((inputs.shape[0], self.horizon))  # (batch_size, pred_steps)
        for i in range(self.horizon):
            if conditions is not None:
                next_exog = conditions[:, i:i + 1, :]          # exog at time i, (batch_size, 1, n_features - 1)
                if self.tcn_type == 'conditional_tcn':
                    out = self.model.predict([inputs, next_exog])  # output at time i, (batch_size, 1)
                else:
                    out = self.model.predict(inputs)
                inputs = np.concatenate([inputs[:, 1:, :],
                                         np.concatenate([np.expand_dims(out,-1), next_exog], -1)],
                                        1)                     # (batch_size, window_size, n_features)
            else:
                out = self.model.predict(inputs)  # prediction at i, (batch, 1)
                inputs = np.concatenate([inputs[:, 1:, :],
                                         np.expand_dims(out, -1)],
                                        1)        # (batch_size, window_size, 1)
            outputs[:, i] = out[:, 0]
        return outputs

    def evaluate(self, data, fn_inverse=None, fn_plot=None):
        try:
            X, y = data
            y_hat = self.predict(X)
        except:
            X, X_ex, y = data
            y_hat = self.predict([X, X_ex])

        if fn_inverse is not None:
            y = fn_inverse(y)
            y_hat = fn_inverse(y_hat)

        y = np.float32(y)
        y_hat = np.float32(y_hat)

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