from keras.callbacks import EarlyStopping, LambdaCallback
from keras.regularizers import l2
from keras.optimizers import Adam

from dts import config
from dts.datasets import uci_single_households
from dts.datasets import gefcom2014
from dts import logger
from dts.utils.plot import plot
from dts.utils import metrics
from dts.utils import run_grid_search, run_single_experiment
from dts.utils import DTSExperiment, log_metrics
from dts.utils.split import *
from dts.models.Recurrent import *
from argparse import ArgumentParser
import yaml
import time
import os


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Training FLAG. Use SACRED args if True')
    parser.add_argument("--dataset", type=str, default='gefcom', help='Dataset to be used: uci, gefcom')
    parser.add_argument('--exogenous', action='store_true', help='Add exogenous features to the data')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell", type=str, default='rnn', help='RNN cell to be used: rnn (Elmann), lstm, gru')
    parser.add_argument('--MIMO', action='store_true',
                        help='Multi step forecast strategy. If True use MIMO else Recursive')
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-3, help='L2 regularization')
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--input_sequence_length", type=int, default=128, help='window size')
    parser.add_argument("--output_sequence_length", type=int, default=10, help='forecasting horizon')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--units", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()
    return args


args = get_args()


def ex_config():
    train = False
    dataset = 'uci'
    exogenous = False
    epochs = 2
    batch_size = 1024
    input_sequence_length = 96 * 4
    output_sequence_length = 96
    dropout = 0.0
    layers = 1
    units = 30
    learning_rate = 1e-3
    cell = 'lstm'
    l2 = 0.001
    MIMO = True
    load = False
    detrend = False


# implementing the "f_main" API
def main(ex, _run, f_log_metrics):
    """
    Updates the main experiment function arguments, calls it and save the
    experiment results and artifacts.
    """
    # Override argparse arguments with sacred arguments
    cmd_args = args  # argparse command line arguments
    vars(cmd_args).update(_run.config)

    # call main script
    val_loss, test_loss, model_name = rnn_main(f_log_metrics=f_log_metrics)

    # save the result metrics to db
    _run.info['model_metrics'] = dict(val_loss=val_loss, test_loss=test_loss)
    # save an artifact (keras model) to db
    ex.add_artifact(model_name)

    return test_loss


def rnn_main(f_log_metrics):
    ################################
    # Load Experiment's paramaters #
    ################################
    params = vars(args)
    print(params)

    ################################
    #         Load Dataset         #
    ################################
    dataset_name = params['dataset']
    if dataset_name == 'gefcom':
        dataset = gefcom2014
    else:
        dataset = uci_single_households

    data = dataset.load_data(fill_nan='median',
                             preprocessing=True,
                             split_type='default',
                             is_train=params['train'],
                             detrend=params['detrend'],
                             exogenous_vars=params['exogenous'],
                             use_prebuilt=True)
    scaler, train, test, trend = data['scaler'], data['train'], data['test'], data['trend']
    if not params['detrend']:
        trend = None

    if params['MIMO']:
        X_train, y_train = get_rnn_inputs(train,
                                          window_size=params['input_sequence_length'],
                                          horizon=params['output_sequence_length'],
                                          shuffle=True)
    else:
        X_train, y_train = get_rnn_inputs(train,
                                          window_size=params['input_sequence_length'],
                                          horizon=1,
                                          shuffle=True)

    if params['exogenous']:
        exog_var_train = y_train[:, :, 1:]
        y_train = y_train[:, :, 0]
        exogenous_shape = (exog_var_train.shape[1], exog_var_train.shape[-1])

        X_test, y_test = get_rnn_inputs(test,
                                        window_size=params['input_sequence_length'],
                                        horizon=params['output_sequence_length'],
                                        shuffle=False,
                                        multivariate_output=True)
        exog_var_test = y_test[:, :, 1:]  # [n_samples, 1, n_features]
        y_test = y_test[:, :, 0]  # [n_samples, 1, 1]
    else:
        X_test, y_test = get_rnn_inputs(test,
                                        window_size=params['input_sequence_length'],
                                        horizon=params['output_sequence_length'],
                                        shuffle=False)
        exog_var_train = None
        exog_var_test = None
        exogenous_shape = None

    # IMPORTANT: Remember to pass the trend values through the same ops as the inputs values
    if params['detrend']:
        X_trend_test, y_trend_test = get_rnn_inputs(trend[1],
                                                    window_size=params['input_sequence_length'],
                                                    horizon=params['output_sequence_length'],
                                                    shuffle=False)
        trend = y_trend_test

    ################################
    #     Build & Train Model      #
    ################################
    cell_params = dict(units=params['units'],
                       activation='tanh',
                       dropout=params['dropout'],
                       kernel_regularizer=l2(params['l2']),
                       recurrent_regularizer=l2(params['l2']),
                       kernel_initializer='lecun_uniform',
                       recurrent_initializer='lecun_uniform')

    if params['MIMO']:
        rnn = RecurrentNN_MIMO(cell_type=params['cell'],
                               layers=params['layers'],
                               cell_params=cell_params)
        if params['exogenous']:
            model = rnn.build_model(input_shape=(params['input_sequence_length'], X_train.shape[-1]),
                                    horizon=params['output_sequence_length'],
                                    exogenous_shape=exogenous_shape)
        else:
            model = rnn.build_model(input_shape=(params['input_sequence_length'], X_train.shape[-1]),
                                    horizon=params['output_sequence_length'])
    else:
        rnn = RecurrentNN_Rec(cell_type=params['cell'],
                              layers=params['layers'],
                              cell_params=cell_params)
        model = rnn.build_model(input_shape=(params['input_sequence_length'], X_train.shape[-1]),
                                horizon=params['output_sequence_length'])

    model.compile(optimizer=Adam(params['learning_rate']), loss='mse', metrics=metrics)
    callbacks = [EarlyStopping(patience=100, monitor='val_loss'),
                 # LambdaCallback(on_epoch_end=lambda _, logs: f_log_metrics(logs=logs))
                 ]
    if params['load']:
        logger.info("Loading model's weights from disk using {}".format(params['load']))
        model.load_weights(params['load'])

    history = model.fit(X_train, y_train,
                        validation_split=0.1,
                        batch_size = params['batch_size'],
                        # steps_per_epoch= train.shape[0] // params['batch_size'],
                        epochs=params['epochs'],
                        callbacks=callbacks,
                        verbose=2)

    ################################
    #          Save weights        #
    ################################
    model_filepath = os.path.join(config['weights'],'{}_{}_{}'.format(
        params['cell'], params['dataset'], time.time()))
    model.save_weights(model_filepath)
    logger.info("Model's weights saved at {}".format(model_filepath))

    #################################
    # Evaluate on Validation & Test #
    #################################
    fn_inverse_val = lambda x: dataset.inverse_transform(x, scaler=scaler, trend=None)
    fn_inverse_test = lambda x: dataset.inverse_transform(x, scaler=scaler, trend=trend)
    fn_plot = lambda x: plot(x, dataset.SAMPLES_PER_DAY, save_at=None)

    if params['exogenous']:
        val_scores = rnn.evaluate(history.validation_data, fn_inverse=fn_inverse_val)
        test_scores = rnn.evaluate([[X_test, exog_var_test], y_test], fn_inverse=fn_inverse_test, fn_plot=fn_plot)
    else:
        val_scores = rnn.evaluate(history.validation_data, fn_inverse=fn_inverse_val)
        test_scores = rnn.evaluate([X_test, y_test], fn_inverse=fn_inverse_test, fn_plot=fn_plot)

    metrics_names = [m.__name__ if not isinstance(m, str) else m for m in model.metrics]
    return dict(zip(metrics_names, val_scores)), \
           dict(zip(metrics_names, test_scores)), \
           model_filepath


if __name__ == '__main__':
    grid_search = True
    if grid_search:
        run_grid_search(
            experimentclass=DTSExperiment,
            parameters=yaml.load(open(os.path.join(config['config'], 'recurrent.yaml'))),
            db_name=config['db'],
            ex_name='rnn',
            f_main=main,
            f_config=ex_config,
            f_metrics=log_metrics,
            cmd_args=vars(args),
            observer_type='mongodb')
    else:
        run_single_experiment(
            experimentclass=DTSExperiment,
            db_name=config['db'],
            ex_name='rnn',
            f_main=main,
            f_config=ex_config,
            f_capture=log_metrics,
            cmd_args=vars(args),
            observer_type='mongodb')