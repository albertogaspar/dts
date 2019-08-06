"""
Evaluate performances on different datasets of some baselines.
TODO: ARMA, ARIMA, EXP SMOOTHING
DONE: TREND eval.
"""
from dts import config
from dts.datasets import uci_single_households
from dts.datasets import gefcom2014
from dts import logger
from dts.utils.plot import plot
from dts.utils import metrics
from dts.utils import run_grid_search, run_single_experiment
from dts.utils import DTSExperiment, log_metrics
from dts.utils.split import *
from dts.models.Seq2Seq import *
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
    parser.add_argument("--input_sequence_length", type=int, default=128, help='window size')
    parser.add_argument("--output_sequence_length", type=int, default=10, help='forecasting horizon')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()
    return args


args = get_args()


def ex_config():
    train = False
    dataset = 'gefcom'
    exogenous = False
    batch_size = 1024
    input_sequence_length = 24 * 4
    output_sequence_length = 24


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
    _, test_loss, _ = static_main(f_log_metrics=f_log_metrics)

    # save the result metrics to db
    _run.info['model_metrics'] = dict(test_loss=test_loss)
    # save an artifact (keras model) to db
    # ex.add_artifact(model_name)

    return test_loss


def trend_eval(y, y_hat, fn_inverse, fn_plot):
    if fn_inverse is not None:
        y = fn_inverse(y)

    y = np.float32(y)
    y_hat = np.float32(y_hat)

    if fn_plot is not None:
        fn_plot([y, y_hat])

    results = []
    for m in metrics:
        try:
            if isinstance(m, str):
                results.append(K.eval(K.mean(get(m)(y, y_hat))))
            else:
                results.append(K.eval(K.mean(m(y, y_hat))))
        except:
            print(m)
            continue
    return results


def static_main(f_log_metrics):
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
                             split_type='default',
                             is_train=params['train'],
                             detrend=True,
                             use_prebuilt=True)
    scaler, train, test, trend = data['scaler'], data['train'], data['test'], data['trend']

    #################################
    # Evaluate on Validation & Test #
    #################################
    X_test, y_test = get_rnn_inputs(test,
                                    window_size=params['input_sequence_length'],
                                    horizon=params['output_sequence_length'],
                                    shuffle=False,
                                    multivariate_output=False)
    # IMPORTANT: Remember to pass the trend values through the same ops as the inputs values
    _, y_trend_test = get_rnn_inputs(trend[1],
                                     window_size=params['input_sequence_length'],
                                     horizon=params['output_sequence_length'],
                                     shuffle=False)
    trend = y_trend_test

    # Define lambdas to be used during eval
    fn_inverse_val = lambda x: dataset.inverse_transform(x, scaler=scaler, trend=None)
    fn_inverse_test = lambda x: dataset.inverse_transform(x, scaler=scaler, trend=trend)
    fn_plot = lambda x: plot(x, dataset.SAMPLES_PER_DAY, save_at=None)

    test_scores = trend_eval(y_test, trend, fn_inverse=fn_inverse_test, fn_plot=fn_plot)

    metrics_names = [m.__name__ if not isinstance(m, str) else m for m in metrics]
    return None, \
           dict(zip(metrics_names, test_scores)), \
           None


if __name__ == '__main__':
    grid_search = False
    if grid_search:
        run_grid_search(
            experimentclass=DTSExperiment,
            parameters=yaml.load(open(os.path.join(config['config'], 'static.yaml'))),
            db_name=config['db'],
            ex_name='static+grid_search',
            f_main=main,
            f_config=ex_config,
            f_metrics=log_metrics,
            cmd_args=vars(args),
            observer_type='mongodb')
    else:
        run_single_experiment(
            experimentclass=DTSExperiment,
            db_name=config['db'],
            ex_name='static',
            f_main=main,
            f_config=ex_config,
            f_metrics=log_metrics,
            cmd_args=vars(args),
            observer_type='mongodb')