from tqdm import tqdm
import numpy as np


def simple_split(X, train_len=None, test_len=None, valid_len=None):
    """
    Split the data in train-test-validation using the given dimensions for each set.
    :param X: numpy.array or pandas.DataFrame
        Univariate data of shape (n_samples, n_features)
    :param train_len: int
        Length in number of data points (measurements) for training.
        If None then allow_muliple_split cannot be True.
    :param test_len: int
        Length in number of data points (measurements) for testing
    :param valid_len: int
        Length in number of data points (measurements) for validation
    :return: list
        train: numpy.array, shape=(train_len, n_features)
        validation: numpy.array, shape=(valid_len, n_features)
        test: numpy.array, shape=(test_len, n_features)
    """
    if test_len is None:
        raise ValueError('test_len cannot be None.')
    if train_len is None:
        train_len = X.shape[0] - test_len
        valid_len = 0
    if valid_len is None:
        valid_len = X.shape[0] - train_len - test_len
    return X[:train_len], \
           X[train_len:train_len+valid_len], \
           X[train_len+valid_len:]


def multiple_splits(X, train_len=None, test_len=None, valid_len=None):
        """
        Split the data in train-test-validation using a window approach.
        Each window has length train_len+test_len+valid_len and the total number of windows is
        M = n_samples // train_len + valid_len + test_len
        This technique creates sets that are not contiguous.
        :param X: numpy.array or pandas.DataFrame
            Univariate data of shape (n_samples, n_features)
        :param train_len: int
            Length in number of data points (measurements) for training.
            If None then allow_muliple_split cannot be True.
        :param test_len: int
            Length in number of data points (measurements) for testing
        :param valid_len: int
            Length in number of data points (measurements) for validation
        :return: list
            train: numpy.array, shape=(M, train_len, n_features)
            validation: numpy.array, shape=(M, valid_len, n_features)
            test: numpy.array, shape=(M, test_len, n_features)
        """
        if train_len is None:
            raise ValueError('train_len cannot be None.')
        if test_len is None:
            raise ValueError('test_len cannot be None.')
        if train_len is None:
            train_len = X.shape[0] - test_len
            valid_len = 0
        if valid_len is None:
            valid_len = X.shape[0] - train_len - test_len

        groups = X.shape[0] // (train_len + valid_len + test_len)
        print('train_len={0}, \n test_len = {1}, \n valid_len = {2}, '
              '\n Number of groups = {3}.'
              '\n Wasted samples: {4}.'
              .format(train_len, test_len, valid_len, groups, X.shape[0] - (train_len+test_len+valid_len)*groups))
        train = []
        test = []
        validation = []
        for i in range(groups):
            train_start = i * (train_len + valid_len + test_len)
            test_start = train_start + train_len + valid_len
            valid_start = train_start + train_len
            test.append(X[test_start:test_start+test_len])
            validation.append(X[valid_start:test_start])
            train.append(X[train_start:valid_start])
        return np.asarray(train), np.asarray(validation), np.asarray(test)


def get_rnn_inputs(data, window_size, horizon,
                   multivariate_output=False, shuffle=False, other_horizon=None):
    """
    Prepare data for feeding a RNN model.
    :param X: numpy.array
        shape (n_samples, n_features) or (M, n_samples, n_features)
    :return: list
        Return two numpy.arrays: the input and the target for the model.
        the inputs has shape (n_samples, input_sequence_len, n_features)
        the target has shape (n_samples, output_sequence_len)
    """
    if data.ndim == 2:
        data = np.expand_dims(data, 0)
    inputs = []
    targets = []
    for X in tqdm(data):  # for each array of shape (n_samples, n_features)
        n_used_samples = X.shape[0] - horizon - window_size + 1
        for i in range(n_used_samples):
            inputs.append(X[i: i + window_size])
            # TARGET FEATURE SHOULD BE THE FIRST
            if multivariate_output:
                if other_horizon is None:
                    targets.append(
                        X[i + window_size: i + window_size + horizon])
                else:
                    targets.append(
                        X[i + 1: i + window_size + 1])
            else:
                if other_horizon is None:
                    targets.append(
                        X[i + window_size: i + window_size + horizon, 0])
                else:
                    targets.append(
                        X[i + 1: i + window_size + 1, 0])
    encoder_input_data = np.asarray(inputs)  # (n_samples, n_features, sequence_len)
    decoder_target_data = np.asarray(targets)  # (n_samples, horizon) or (n_samples, n_features, horizon) if multivariate_output
    idxs = np.arange(encoder_input_data.shape[0])
    if shuffle:
        np.random.shuffle(idxs)
    return encoder_input_data[idxs], decoder_target_data[idxs]


def get_seq2seq_inputs(data, window_size, horizon, noise_model=False, shuffle=False):
    """
    Prepare data for feeding a Sequnece2Sequence model.
    :param data: numpy.array
        shape (n_samples, n_features) or (M, n_samples, n_features)
    :param noise_model: bool
        If True add gaussian random noise to the decoder input data (for training when using TF mode).
    :return: list
        Return three numpy.arrays: the encoder input, the decoder input and the target for the model.
        the encoder inputs has shape (n_samples, input_sequence_len, n_features)
        the decoder inputs has shape (n_samples, input_sequence_len, n_features) and corresponds to the target lagged by 1. (teacher forcing)
        the target has shape (n_samples, output_sequence_len, 1)
    """
    encoder_input_data, decoder_target_data = \
        get_rnn_inputs(data, window_size=window_size, horizon=horizon, multivariate_output=True, shuffle=False)
    # lagged target series for teacher forcing
    # i.e, during training, the true series values (lagged by one time step) are fed as inputs to the decoder.
    # Intuitively, we are trying to teach the NN how to condition on previous time steps to predict the next.
    # At prediction time, the true values in this process will be replaced by predicted values for each previous time step.
    decoder_input_data = np.zeros((decoder_target_data.shape[0], decoder_target_data.shape[1], encoder_input_data.shape[2]))
    decoder_input_data[:, 1:, :] = decoder_target_data[:, :-1, :]  # target = input shifted by one
    decoder_input_data[:, 0, :] = encoder_input_data[:, -1, :]
    # Set decoder target data to univariate output
    decoder_target_data = decoder_target_data[:,:,:1]
    # add noise in the decoder load values input
    if noise_model:
        decoder_input_data[:,:,0] = \
            decoder_input_data[:,:,0] + \
            np.random.normal(loc=0, scale=0.05, size=decoder_input_data[:,:,0].shape)
    idxs = np.arange(encoder_input_data.shape[0])
    if shuffle:
        np.random.shuffle(idxs)

    return encoder_input_data[idxs], decoder_input_data[idxs], decoder_target_data[idxs]