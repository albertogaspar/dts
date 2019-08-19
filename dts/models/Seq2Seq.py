from keras.layers import Concatenate, LSTMCell, GRU, GRUCell, RNN, Dense, Input, Lambda, TimeDistributed, Dropout
from keras.optimizers import Adam
from keras.initializers import Zeros, glorot_normal
from keras.losses import mean_squared_error
from keras import Model
import keras.backend as K
import numpy as np
from itertools import chain
import keras
from keras.regularizers import l2
from keras.metrics import get


class Seq2SeqBase:
    """
    Base class for a RNN-based Sequnece to Sequence model (for time-series prediction)
    """

    def __init__(self, encoder_layers,
                 decoder_layers,
                 output_sequence_length,
                 dropout=0.0,
                 l2=0.01,
                 cell_type='lstm'):
        """
        :param encoder_layers: list
            encoder (RNN) architecture: [n_hidden_units_1st_layer, n_hidden_units_2nd_layer, ...]
        :param decoder_layers:
        decoder (RNN) architecture: [n_hidden_units_1st_layer, n_hidden_units_2nd_layer, ...]
        :param output_sequence_length: int
            number of timestep to be predicted.
        :param cell_type: str
            gru or lstm.
        """
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.output_sequence_length = output_sequence_length
        self.dropout = dropout
        self.l2 = l2
        if cell_type == 'lstm':
            self.cell = LSTMCell
        elif cell_type == 'gru':
            self.cell = GRUCell
        else:
            raise ValueError('{0} is not a valid cell type. Choose between gru and lstm.'.format(cell_type))

    def _build_encoder(self):
        """
        Build the encoder multilayer RNN (stacked RNN). Return it as a keras.Model
        :param encoder_inputs: keras.layers.Input
            shape=(batch_size, input_sequence_length, 1)
        :return: keras.Model
            model inputs=encoder_inputs,
            outputs=encoder_states, i.e. (last) hidden and cell state for each layer [h_i, c_i, h_i-1, c_i-1, ...]
        """
        # Create a list of RNN Cells, these get stacked one after the other in the RNN,
        # implementing an efficient stacked RNN
        encoder_cells = []
        for n_hidden_neurons in self.encoder_layers:
            encoder_cells.append(self.cell(units=n_hidden_neurons,
                                           dropout=self.dropout,
                                           kernel_regularizer=l2(self.l2),
                                           recurrent_regularizer=l2(self.l2)))

        self.encoder = RNN(encoder_cells, return_state=True, name='encoder')

    def _build_decoder(self):
        decoder_cells = []
        for n_hidden_neurons in self.decoder_layers:
            decoder_cells.append(self.cell(units=n_hidden_neurons,
                                           dropout=self.dropout,
                                           kernel_regularizer=l2(self.l2),
                                           recurrent_regularizer=l2(self.l2)
                                           ))
        # return output for EACH timestamp
        self.decoder = RNN(decoder_cells, return_sequences=True, return_state=True, name='decoder')

    def _get_decoder_initial_states(self):
        """
        Return decoder states as Input layers
        """
        decoder_states_inputs = []
        for units in self.encoder_layers:
            decoder_state_input_h = Input(shape=(units,))
            input_states = [decoder_state_input_h]
            if self.cell == LSTMCell:
                decoder_state_input_c = Input(shape=(units,))
                input_states = [decoder_state_input_h, decoder_state_input_c]
            decoder_states_inputs.extend(input_states)
        if keras.__version__ < '2.2':
            return list(reversed(decoder_states_inputs))
        else:
            return decoder_states_inputs

    def _format_encoder_states(self, encoder_states, use_first=True):
        """
        Format the encoder states in such a way that only the last state from the first layer of the encoder
        is used to init the first layer of the decoder.
        If the cell type used is LSTM then both c and h are kept.
        :param encoder_states: Keras.tensor
            (last) hidden state of the decoder
        :param use_first: bool
            if True use only the last hidden state from first layer of the encoder, while the other are init to zero.
            if False use last hidden state for all layers
        :return:
            masked encoder states
        """
        if use_first:
            # Keras version 2.1.4 has encoder states reversed w.r.t later versions
            if keras.__version__ < '2.2':
                if self.cell == 'lstm':
                    encoder_states = [Lambda(lambda x: K.zeros_like(x))(s) for s in encoder_states[:-2]] + [
                        encoder_states[-2]]
                else:
                    encoder_states = [Lambda(lambda x: K.zeros_like(x))(s) for s in encoder_states[:-1]] + [
                        encoder_states[-1]]
            else:
                if self.cell == 'lstm':
                    encoder_states = encoder_states[:2] + [Lambda(lambda x: K.zeros_like(x))(s) for s in
                                                                encoder_states[2:]]
                else:
                    encoder_states = encoder_states[:1] + [Lambda(lambda x: K.zeros_like(x))(s) for s in
                                                                encoder_states[1:]]
        return encoder_states


class Seq2SeqTF(Seq2SeqBase):
    """
    Sequence 2 Sequence model with RNNs encoder-decoder.
    Training process uses Teacher Forcing.
    """
    def __init__(self, *args, **kwargs):
        self.decoder_pred = None
        self.model = None
        super().__init__(*args, **kwargs)

    def build(self, encoder_inputs, decoder_inputs):
        """
        Build a Sequence to Sequence model to be trained via teacher forcing.
        :param encoder_inputs:
        :param decoder_inputs:
        """
        encoder_inputs = Input(shape=encoder_inputs, name='encoder_inputs')
        decoder_inputs = Input(shape=decoder_inputs, name='decoder_inputs')

        self._build_encoder()
        self._build_decoder()
        self.decoder_dense = Dense(1)

        self.encoder_states = self.encoder(encoder_inputs)[1:]
        self.encoder_model = Model(inputs=encoder_inputs, outputs=self.encoder_states)

        encoder_states = self._format_encoder_states(self.encoder_states, use_first=False)

        decoder_outputs = self.decoder(decoder_inputs, initial_state=encoder_states)[0]

        # FC layer after decoder to produce a single real value for each timestamp (univariate time-series prediction)
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Full encoder-decoder model
        self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs, name='train_model')
        self.model.summary()
        return self.model

    def build_prediction_model(self, decoder_inputs):
        """
        A modified version of the decoder is used for prediction.
        Inputs = Predicted target inputs and encoded state vectors,
        Outputs = Predicted target outputs and decoder state vectors.
        We need to hang onto these state vectors to run the next step of the inference loop.
        :param decoder_inputs:
        :return:
        """
        decoder_inputs = Input(shape=decoder_inputs)
        decoder_states_inputs = self._get_decoder_initial_states()

        decoder_outputs = self.decoder(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = decoder_outputs[1:]
        decoder_outputs = decoder_outputs[0]
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Decoder model to be used during inference
        self.decoder_pred = Model([decoder_inputs] + decoder_states_inputs,
                                  [decoder_outputs] + decoder_states,
                                  name='pred_model')

    def predict(self, encoder_inputs, pred_steps, decoder_input_exog=None):
        """
        Multi step Inference (1 at a time)
        :param encoder_inputs: numpy.array
            Encoder input: shape(n_samples, input_sequnece_length, n_features)
        :param pred_steps: int
            number of steps to be predicted in the future
        :param decoder_input_exog: numpy.array
            Decoder_input (if exogenous variables are given) shape(n_samples, output_sequnece_length, n_features-1).
            Important: REMOVE the target variable from this array of values.
        :return: numpy.array
            shape(n_samples, output_sequence_length, 1)
        """
        # predictions, shape (batch_size, pred_steps, 1)
        predictions = np.zeros((encoder_inputs.shape[0], pred_steps, 1))

        # produce embeddings with encoder
        states_value = self.encoder_model.predict(encoder_inputs)  # [h,c](lstm) or [h](gru) each of dim (batch_size, n_hidden)

        # populate the decoder input with the last encoder input
        decoder_input = np.zeros((encoder_inputs.shape[0], 1, encoder_inputs.shape[-1]))  # decoder input for a single timestep
        decoder_input[:, 0, 0] = encoder_inputs[:, -1, 0]

        for i in range(pred_steps):
            if decoder_input_exog is not None:
                # add exogenous variables if any
                decoder_input[:, 0, 1:] = decoder_input_exog[:, i, :]

            if isinstance(states_value, list):
                outputs = self.decoder_pred.predict([decoder_input] + states_value)
            else:
                outputs = self.decoder_pred.predict([decoder_input, states_value])

            # prediction at timestep i
            output = outputs[0]  # output (batch_size, 1, 1)
            predictions[:, i, 0] = output[:, 0, 0]

            # Update the decoder input with the predicted value (of length 1).
            decoder_input = np.zeros((encoder_inputs.shape[0], 1, encoder_inputs.shape[-1]))
            decoder_input[:, 0, 0] = output[:, 0, 0]

            # Update states
            states_value = outputs[1:] # h, c (both [batch_size, n_hidden]) or just h

        return predictions

    def evaluate(self, data, fn_inverse=None, horizon=1, fn_plot=None):
        """
        Evaluate model
        :return:
        """
        encoder_input_data, decoder_input_exog, y = data

        y_hat = self.predict(encoder_inputs=encoder_input_data,
                             pred_steps=horizon,
                             decoder_input_exog=decoder_input_exog)

        if fn_inverse is not None:
            y = fn_inverse(y)
            y_hat = fn_inverse(y_hat)

        y = np.float32(y)
        y_hat = np.float32(y_hat)

        if fn_plot is not None:
            fn_plot([y,y_hat])

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


class Seq2SeqTFPretr(Seq2SeqBase):
    """
    Sequence 2 Sequence model with RNNs encoder-decoder.
    Training process uses Teacher Forcing, Pretrain encoder then freeze it and train decoder.
    """
    def _build_encoder(self):
        encoder_cells = []
        for n_hidden_neurons in self.encoder_layers:
            encoder_cells.append(self.cell(units=n_hidden_neurons,
                                           dropout=self.dropout,
                                           kernel_regularizer=l2(self.l2),
                                           recurrent_regularizer=l2(self.l2)
                                           ))
        self.encoder = RNN(encoder_cells, return_state=True, return_sequences=True, name='encoder')

    def pretrain_encoder(self,  encoder_inputs):
        encoder_inputs = Input(shape=encoder_inputs, name='encoder_inputs')
        self._build_encoder()
        encoder_outputs = Dense(1)(self.encoder(encoder_inputs)[0])
        return Model(encoder_inputs, encoder_outputs)

    def build(self, encoder_inputs, decoder_inputs):
        """
        Build a Sequence to Sequence model to be trained via teacher forcing.
        :param encoder_inputs:
        :param decoder_inputs:
        """
        encoder_inputs = Input(shape=encoder_inputs, name='encoder_inputs')
        decoder_inputs = Input(shape=decoder_inputs, name='decoder_inputs')

        self._build_decoder()
        self.decoder_dense = Dense(1)

        self.encoder_states = self.encoder(encoder_inputs)[1:]
        self.encoder.trainable = False
        self.encoder_model = Model(inputs=encoder_inputs, outputs=self.encoder_states)

        encoder_states = self._format_encoder_states(self.encoder_states, use_first=False)
        decoder_outputs = self.decoder(decoder_inputs, initial_state=encoder_states)[0]

        # FC layer after decoder to produce a single real value for each timestamp (univariate time-series prediction)
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Full encoder-decoder model
        self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs, name='train_model')
        self.model.summary()
        return self.model

    def build_prediction_model(self, decoder_inputs):
        """
        A modified version of the decoder is used for prediction.
        Inputs = Predicted target inputs and encoded state vectors,
        Outputs = Predicted target outputs and decoder state vectors.
        We need to hang onto these state vectors to run the next step of the inference loop.
        :param decoder_inputs:
        :return:
        """
        decoder_inputs = Input(shape=decoder_inputs)
        decoder_states_inputs = self._get_decoder_initial_states()

        decoder_outputs = self.decoder(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = decoder_outputs[1:]
        decoder_outputs = decoder_outputs[0]
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Decoder model to be used during inference
        self.decoder_pred = Model([decoder_inputs] + decoder_states_inputs,
                                  [decoder_outputs] + decoder_states,
                                  name='pred_model')

    def predict(self, encoder_inputs, pred_steps, decoder_input_exog=None):
        """
        Multi step Inference (1 at a time)
        :param encoder_inputs: numpy.array
            Encoder input: shape(n_samples, input_sequnece_length, n_features)
        :param pred_steps: int
            number of steps to be predicted in the future
        :param decoder_input_exog: numpy.array
            Decoder_input (if exogenous variables are given) shape(n_samples, output_sequnece_length, n_features-1).
            Important: REMOVE the target variable from this array of values.
        :return: numpy.array
            shape(n_samples, output_sequence_length, 1)
        """
        # predictions, shape (batch_size, pred_steps, 1)
        predictions = np.zeros((encoder_inputs.shape[0], pred_steps, 1))

        # produce embeddings with encoder
        states_value = self.encoder_model.predict(encoder_inputs)  # [h,c](lstm) or [h](gru) each of dim (batch_size, n_hidden)

        # populate the decoder input with the last encoder input
        decoder_input = np.zeros((encoder_inputs.shape[0], 1, encoder_inputs.shape[-1]))  # decoder input for a single timestep
        decoder_input[:, 0, 0] = encoder_inputs[:, -1, 0]

        for i in range(pred_steps):
            if decoder_input_exog is not None:
                # add exogenous variables if any
                decoder_input[:, 0, 1:] = decoder_input_exog[:, i, :]

            if isinstance(states_value, list):
                outputs = self.decoder_pred.predict([decoder_input] + states_value)
            else:
                outputs = self.decoder_pred.predict([decoder_input, states_value])

            # prediction at timestep i
            output = outputs[0]  # output (batch_size, 1, 1)
            predictions[:, i, 0] = output[:, 0, 0]

            # Update the decoder input with the predicted value (of length 1).
            decoder_input = np.zeros((encoder_inputs.shape[0], 1, encoder_inputs.shape[-1]))
            decoder_input[:, 0, 0] = output[:, 0, 0]

            # Update states
            states_value = outputs[1:] # h, c (both [batch_size, n_hidden]) or just h

        return predictions


class Seq2SeqStatic(Seq2SeqBase):
    """
    Sequence 2 Sequence model with RNNs encoder-decoder.
    Training process without Teacher Forcing. Even during training self-generfated samples are used.
    """

    def build(self, encoder_inputs, decoder_inputs, decoder_inputs_exog=None):
        """
        Build a Sequence to Sequence model to be trained with a static loop
        (predictions are recursively fed into the decoder's input until a sequence of length 'output_sequence_length'
        is formed).
        :param encoder_inputs: tuple or list
            [batch_size, input_sequence_length, n_features]
        :param decoder_inputs: tuple or list
            [batch_size, 1, 1]
        :param decoder_inputs_exog: tuple or list
            [batch_size, output_sequence_length, n_features - 1]
        """
        encoder_inputs = Input(shape=encoder_inputs, name='encoder_inputs')
        decoder_inputs = Input(shape=decoder_inputs, name='decoder_inputs')

        if decoder_inputs_exog is not None:
            decoder_inputs_exog = Input(shape=decoder_inputs_exog, name='decoder_exog')
            decoder_inputs_with_ex = Lambda(lambda x: x[:, :1, 1:])(decoder_inputs_exog)
            decoder_inputs_with_ex = Concatenate(axis=-1, name='decoder_inputs_plus_exog')([decoder_inputs, decoder_inputs_with_ex])

        self._build_encoder()
        self._build_decoder()
        self.decoder_dense = Dense(1)

        self.encoder_states = self.encoder(encoder_inputs)[1:]
        encoder_states = self._format_encoder_states(self.encoder_states)

        # decoder inputs should have shape [batch_size, 1, 1] not [batch_size, output_sequence_length, 1]
        # beacuse we want to reinject the last output into the inputs.
        decoder_outputs = self.build_static_loop(encoder_states, decoder_inputs)

        # Full encoder-decoder model
        if decoder_inputs_exog is None:
            self.model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        else:
            self.model = Model(inputs=[encoder_inputs, decoder_inputs, decoder_inputs_exog], outputs=decoder_outputs)
        self.model.summary()
        return self.model

    def build_static_loop(self, init_states, decoder_inputs, decoder_inputs_exog=None):
        """
        :param init_states:
        :param decoder_inputs:
        :param decoder_inputs_exog:
        :return:
        """
        inputs = decoder_inputs  # [batch,1,1]
        all_outputs = []
        for i in range(self.output_sequence_length):
            if decoder_inputs_exog is not None:
                exog_var = Lambda(lambda x: x[:, i:i + 1, 1:])(decoder_inputs_exog)  # [batch,1,features]
                inputs = Concatenate(axis=-1)([inputs, exog_var])
            decoder_outputs = self.decoder(inputs, initial_state=init_states)
            init_states = decoder_outputs[1:] # state update
            decoder_outputs = decoder_outputs[0]
            decoder_outputs = self.decoder_dense(decoder_outputs)  # (batch, 1, 1)
            all_outputs.append(decoder_outputs)
            inputs = decoder_outputs # input update

        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
        return decoder_outputs

    def evaluate(self, data, fn_inverse=None, fn_plot=None):
        """
        Evaluate model
        :return:
        """
        try:
            encoder_inputs, decoder_inputs, decoder_inputs_exog, y = data
            y_hat = self.model.predict([encoder_inputs, decoder_inputs, decoder_inputs_exog])
        except:
            encoder_inputs, decoder_inputs, y = data
            y_hat = self.model.predict([encoder_inputs, decoder_inputs])


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



