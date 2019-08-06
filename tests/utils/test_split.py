import numpy as np
from dts.utils.split import *
import unittest


class TestSplit(unittest.TestCase):

    def __init__(self,*args,**kwargs):
        self.X = np.random.uniform(0., 10., size=(10000,10))
        super().__init__(*args,**kwargs)

    def test_simple_split(self):
        test_len = 4000
        res = simple_split(self.X, train_len=None, valid_len=None, test_len=test_len)
        self.assertTupleEqual(res[0].shape, (self.X.shape[0] - test_len, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (0, self.X.shape[1]))
        self.assertTupleEqual(res[2].shape, (test_len, self.X.shape[1]))

        train_len = 6000
        test_len = 1000
        res = simple_split(self.X, train_len=train_len, valid_len=None, test_len=test_len)
        self.assertTupleEqual(res[0].shape, (train_len, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (3000, self.X.shape[1]))
        self.assertTupleEqual(res[2].shape, (self.X.shape[0] - train_len - 3000, self.X.shape[1]))

        train_len = 3000
        valid_len = 500
        test_len = 200
        res = simple_split(self.X, train_len=train_len, valid_len=valid_len, test_len=test_len)
        self.assertTupleEqual(res[0].shape, (train_len, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (valid_len, self.X.shape[1]))
        self.assertTupleEqual(res[2].shape, (self.X.shape[0] - train_len - valid_len, self.X.shape[1]))

    def test_multiple_split(self):
        test_len = 4000
        self.assertRaises(ValueError, multiple_splits ,self.X, None, None, test_len)

        train_len = 6000
        test_len = 1000
        res = multiple_splits(self.X, train_len=train_len, valid_len=None, test_len=test_len)
        self.assertTupleEqual(res[0].shape, (1, train_len, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (1, 3000, self.X.shape[1]))
        self.assertTupleEqual(res[2].shape, (1, test_len, self.X.shape[1]))

        train_len = 1000
        valid_len = 500
        test_len = 200
        groups = self.X.shape[0] // (train_len + valid_len + test_len)
        res = multiple_splits(self.X, train_len=train_len, valid_len=valid_len, test_len=test_len)
        self.assertTupleEqual(res[0].shape, (groups, train_len, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (groups, valid_len, self.X.shape[1]))
        self.assertTupleEqual(res[2].shape, (groups, test_len, self.X.shape[1]))

    def test_rnn_inputs(self):
        window_size = 100
        horizon = 10

        train_len = 3000
        valid_len = 0
        test_len = 200
        X = simple_split(self.X, train_len=train_len, valid_len=valid_len, test_len=test_len)[0]
        res = get_rnn_inputs(X, window_size, horizon, multivariate_output=False, shuffle=False)

        n_samples = X.shape[0] - horizon - window_size + 1
        self.assertTupleEqual(res[0].shape, (n_samples, window_size, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (n_samples, horizon))

        train_len = 1000
        valid_len = 0
        test_len = 200
        groups = self.X.shape[0] // (train_len + valid_len + test_len)
        X = multiple_splits(self.X, train_len=train_len, valid_len=valid_len, test_len=test_len)[0]
        res = get_rnn_inputs(X, window_size, horizon, multivariate_output=False, shuffle=False)

        n_samples = groups*(train_len - horizon - window_size + 1)
        self.assertTupleEqual(res[0].shape, (n_samples, window_size, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (n_samples, horizon))

        train_len = 3000
        valid_len = 0
        test_len = 200
        X = simple_split(self.X, train_len=train_len, valid_len=valid_len, test_len=test_len)[0]
        res = get_rnn_inputs(X, window_size, horizon, multivariate_output=True, shuffle=False)

        n_samples = X.shape[0] - horizon - window_size + 1
        self.assertTupleEqual(res[0].shape, (n_samples, window_size, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (n_samples, horizon, self.X.shape[1]))

        train_len = 1000
        valid_len = 0
        test_len = 200
        groups = self.X.shape[0] // (train_len + valid_len + test_len)
        X = multiple_splits(self.X, train_len=train_len, valid_len=valid_len, test_len=test_len)[0]
        res = get_rnn_inputs(X, window_size, horizon, multivariate_output=True, shuffle=False)

        n_samples = groups*(train_len - horizon - window_size + 1)
        self.assertTupleEqual(res[0].shape, (n_samples, window_size, self.X.shape[1]))
        self.assertTupleEqual(res[1].shape, (n_samples, horizon, self.X.shape[1]))

    def test_seq2seq_inputs(self):
        window_size = 100
        horizon = 10

        train_len = 3000
        valid_len = 0
        test_len = 200
        X = simple_split(self.X, train_len=train_len, valid_len=valid_len, test_len=test_len)[0]
        enc_in, dec_in, dec_tar = get_seq2seq_inputs(X, window_size, horizon, shuffle=False)

        n_samples = X.shape[0] - horizon - window_size + 1
        self.assertTupleEqual(enc_in.shape, (n_samples, window_size, self.X.shape[1]))
        self.assertTupleEqual(dec_in.shape, (n_samples, horizon, self.X.shape[1]))
        self.assertTupleEqual(dec_tar.shape, (n_samples, horizon, 1))

        train_len = 1000
        valid_len = 0
        test_len = 200
        groups = self.X.shape[0] // (train_len + valid_len + test_len)
        X = multiple_splits(self.X, train_len=train_len, valid_len=valid_len, test_len=test_len)[0]
        enc_in, dec_in, dec_tar = get_seq2seq_inputs(X, window_size, horizon, shuffle=False)

        n_samples = groups * (train_len - horizon - window_size + 1)
        self.assertTupleEqual(enc_in.shape, (n_samples, window_size, self.X.shape[1]))
        self.assertTupleEqual(dec_in.shape, (n_samples, horizon, self.X.shape[1]))
        self.assertTupleEqual(dec_tar.shape, (n_samples, horizon, 1))


if __name__ == "__main__":
    unittest.main()