"""
Main script for training and evaluating a FFNN or a (cutsom) ResNet trained either in MIMO or Recurrent fashion
for multi-step forecasting task.
You can chose bewteen:
    - Running a simple experiment
    - Running multiple experiments trying out diffrent combinations of hyperparamters (grid-search)
"""
import warnings
warnings.filterwarnings(action='ignore')

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
from dts.utils import DTSExperiment, log_metrics, get_args
from dts.utils.split import *
from dts.models.FFNN import *
from dts.utils.decorators import f_main
import time
import os


args = get_args()

@f_main(args=args)
def main(_run):
    ################################
    # Load Experiment's paramaters #
    ################################
    params = vars(args)
    logger.info(params)

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
                             split_type='simple',
                             is_train=params['train'],
                             detrend=params['detrend'],
                             exogenous_vars=params['exogenous'],
                             use_prebuilt=True)
    scaler, train, test, trend = data['scaler'], data['train'], data['test'], data['trend']
    if not params['detrend']:
        trend = None

    if params['recursive_forecast']:
        horizon = 1
    else:
        horizon = params['output_sequence_length']
    X_train, y_train = get_rnn_inputs(train,
                                      window_size=params['input_sequence_length'],
                                      horizon=horizon,
                                      shuffle=True,
                                      multivariate_output=params['exogenous'])

    ################################
    #     Build & Train Model      #
    ################################
    if params['ffnn_type'] == 'simple':
        ffnn = SimpleNet
    else:
        ffnn = ResNet
    ffnn = ffnn(layers=params['layers'],
                kernel_initializer='glorot_normal',
                kernel_regularizer=l2(params['l2_reg']),
                bias_regularizer=l2(params['l2_reg']),
                use_bias=False,
                recursive_forecast=params['recursive_forecast'])

    if params['exogenous']:
        exog_var_train = y_train[:, :, 1:]  # [n_samples, 1, n_features]
        y_train = y_train[:, :, 0]  # [n_samples, 1]
        conditions_shape = (exog_var_train.shape[1], exog_var_train.shape[-1])

        X_test, y_test = get_rnn_inputs(test,
                                        window_size=params['input_sequence_length'],
                                        horizon=params['output_sequence_length'],
                                        shuffle=False,
                                        multivariate_output=True)
        exog_var_test = y_test[:, :, 1:]  # [n_samples, 1, n_features]
        y_test = y_test[:, :, 0]  # [n_samples, 1]
    else:
        X_test, y_test = get_rnn_inputs(test,
                                        window_size=params['input_sequence_length'],
                                        horizon=params['output_sequence_length'],
                                        shuffle=False)
        exog_var_train = None
        exog_var_test = None
        conditions_shape = None

    # IMPORTANT: Remember to pass the trend values through the same ops as the inputs values
    if params['detrend']:
        X_trend_test, y_trend_test = get_rnn_inputs(trend[1],
                                                    window_size=params['input_sequence_length'],
                                                    horizon=params['output_sequence_length'],
                                                    shuffle=False)
        trend = y_trend_test

    model = ffnn.build_model(input_shape=(X_train.shape[1], X_train.shape[-1]),
                             horizon=params['output_sequence_length'],
                             conditions_shape=conditions_shape)

    if params['load']:
        logger.info("Loading model's weights from disk using {}".format(params['load']))
        model.load_weights(params['load'])

    optimizer = Adam(params['learning_rate'])
    model.compile(optimizer=optimizer, loss=['mse'], metrics=metrics)
    callbacks = [EarlyStopping(patience=50, monitor='val_loss')]

    if params['exogenous']:
        history = model.fit([X_train, exog_var_train], y_train,
                            validation_split=0.1,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            callbacks=callbacks,
                            verbose=2)
    else:
        history = model.fit(X_train, y_train,
                            validation_split=0.1,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            callbacks=callbacks,
                            verbose=2)

    ################################
    #          Save weights        #
    ################################
    model_filepath = os.path.join(
        config['weights'],'{}_{}_{}'
            .format(params['ffnn_type'], params['dataset'], time.time()))
    model.save_weights(model_filepath)
    logger.info("Model's weights saved at {}".format(model_filepath))

    #################################
    # Evaluate on Validation & Test #
    #################################
    fn_inverse_val = lambda x: dataset.inverse_transform(x, scaler=scaler, trend=None)
    fn_inverse_test = lambda x: dataset.inverse_transform(x, scaler=scaler, trend=trend)
    fn_plot = lambda x: plot(x, dataset.SAMPLES_PER_DAY, save_at=None)

    if params['recursive_forecast']:
        val_scores = []
        txt = "When FFNN is trained in Recursive mode training and inference are different. Specifically, training is "\
              "a 1 step forecasting problem and inference is multi step forecasting problem. Thus, "\
              "validation results will not be provided as they are not comparable with test results"
        logger.warn(txt)
        _run.info['extra'] = txt
    else:
        # has to add this filter because of unexpected behaviour of history.validation_data when using resent.
        validation_data = list(filter(lambda x: isinstance(x, np.ndarray), history.validation_data))
        val_scores = ffnn.evaluate(validation_data[:-1], fn_inverse=fn_inverse_val)

    if params['exogenous']:
        test_scores = ffnn.evaluate([[X_test, exog_var_test], y_test], fn_inverse=fn_inverse_test, fn_plot=fn_plot)
    else:
        test_scores = ffnn.evaluate([X_test, y_test], fn_inverse=fn_inverse_test, fn_plot=fn_plot)

    metrics_names = [m.__name__ if not isinstance(m, str) else m for m in model.metrics]
    return dict(zip(metrics_names, val_scores)), \
           dict(zip(metrics_names, test_scores)), \
           model_filepath


if __name__ == '__main__':
    if args.grid_search:
        run_grid_search(
            experimentclass=DTSExperiment,
            db_name=config['db'],
            ex_name='ffnn_grid_search',
            f_main=main,
            f_metrics=log_metrics,
            f_config=args.add_config,
            observer_type=args.observer)
    else:
        run_single_experiment(
            experimentclass=DTSExperiment,
            db_name=config['db'],
            ex_name='ffnn',
            f_main=main,
            f_config=args.add_config,
            f_metrics=log_metrics,
            observer_type=args.observer)