import keras.backend as K
from keras.losses import mean_absolute_percentage_error, mse, mae
import numpy as np


def r2(y_true, y_pred):
    """
    R2 score
    $$ 1 - \frac{MSE}{TSS}, \quad TSS= \sum\limits_{i}^{N} (y_i - \bar{y})^2) $$
    $$ \bar{y} = \frac{1}{N} sum\limits_{i}^{N} y_i $$
    """
    return 1. - K.mean(K.sum(K.square(y_true - y_pred))) / K.sum(K.square(y_true - K.mean(K.identity(y_true))))


def nrmse_a(y_true, y_pred):
    """
    Root relative squared error: divide MSE by some variation of y_pred
    RRSE = sqrt( MSE / MESS) with  ESS = sum((y_true - mean(y_true))**2), MESS = 1/N*ESS
    """
    return K.sqrt(K.mean(K.square(y_true - y_pred)) / K.mean(K.square(y_pred - K.mean(K.identity(y_true)))))


def nrmse_b(y_true, y_pred):
    " If this value is larger than 1, you 'd obtain a better model by simply generating a random time series " \
    "of the same mean and standard deviation as Y."
    return K.sqrt(K.mean(K.sum(K.square(y_true - y_pred)))) / K.std(K.identity(y_true))


def nrmse_c(y_true, y_pred):
    return K.sqrt(K.mean(K.sum(K.square(y_true - y_pred))) / K.abs(K.identity(y_true)))


def nrmsd(y_true, y_pred):
    """
    Normalized root mean squared deviation
    NRMSD = sqrt( MSE / (y_true.max - y_true.min)
    """
    return K.sqrt(K.mean(K.square(y_true - y_pred)))/(
           K.maximum(K.epsilon(),
                     K.max(y_true, axis=None, keepdims=True) - K.min(y_true, axis=None, keepdims=True)))


def smape(y_true, y_pred):
        """
        Symmetric mean absolute percentage error
        """
        return 100*K.mean(K.abs(y_true -y_pred)/K.maximum((K.abs(y_true) + K.abs(y_pred)/2),
                                                          K.epsilon()))


def acf_loss(y_true, y_pred):
    """
    Loss based on the autocorrelation of residuals (reduce sum). n_lags=10 (fixed)
    """
    n_lags=5
    lags = range(1,2)
    residuals = (y_true - y_pred)
    # acf = []
    # for k in lags:
    #     mean = K.mean(residuals, axis=1, keepdims=True)
    #     autocorrelation_at_lag_k = K.square(K.sum((residuals[:,:-k] - mean) * (residuals[:,k:] - mean), axis=1) / \
    #                                         K.sum(K.square(residuals - mean), axis=1))
    #     acf.append(autocorrelation_at_lag_k)
    # acf = K.transpose(K.tf.convert_to_tensor(acf))
    mean = K.mean(residuals, axis=1, keepdims=True)
    autocorrelation_at_lag_k = K.square(K.sum((residuals[:, :-1] - mean) * (residuals[:, 1:] - mean), axis=1) / \
                                        K.sum(K.square(residuals - mean), axis=1))
    return K.mean(autocorrelation_at_lag_k)


def write(op, y_true, y_pred):
    y_test = y_true.astype(np.float32)
    pred = y_pred.astype(np.float32)
    op('MAE {}\n'.format(K.eval(K.mean(mae(y_test, pred)))))
    op('MSE {}\n'.format(K.eval((K.mean(mse(y_test, pred))))))
    op('RMSE {}\n'.format(K.eval(K.sqrt(K.mean(mse(y_test, pred))))))
    op('NRMSE_a {}\n'.format(K.eval(K.sqrt(K.mean(nrmse_a(y_test, pred))))))
    op('NRMSE_b {}\n'.format(K.eval(K.sqrt(K.mean(nrmse_b(y_test, pred))))))
    op('MAPE {}\n'.format(K.eval(K.mean(mean_absolute_percentage_error(y_test, pred)))))
    op('NRMSD {}\n'.format(K.eval(K.mean(nrmsd(y_test, pred)))))
    op('SMAPE {}\n'.format(K.eval(K.mean(smape(y_test, pred)))))
    op('R2 {}\n'.format(K.eval(K.mean(r2(y_test, pred)))))