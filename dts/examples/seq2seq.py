from keras.callbacks import EarlyStopping, LambdaCallback

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
    parser.add_argument("--cell", type=str, default='rnn', help='RNN cell to be used: rnn (Elmann), lstm, gru')
    parser.add_argument('--teacher_forcing', action='store_true',
                        help='Multi step forecast strategy. If True use teacher forcing else use self-generated mode')
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-3, help='L2 regularization')
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--input_sequence_length", type=int, default=128, help='window size')
    parser.add_argument("--output_sequence_length", type=int, default=10, help='forecasting horizon')
    parser.add_argument("--batch_size", type=int, default=32)
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
    units = [30]
    learning_rate = 1e-3
    cell = 'lstm'
    l2 = 0.005
    teacher_forcing = True
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
    val_loss, test_loss, model_name = seq2seq_main(f_log_metrics=f_log_metrics)

    # save the result metrics to db
    _run.info['model_metrics'] = dict(val_loss=val_loss, test_loss=test_loss)
    # save an artifact (keras model) to db
    ex.add_artifact(model_name)

    return test_loss


def seq2seq_main(f_log_metrics):
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
                             detrend=params['detrend'],
                             use_prebuilt=True)
    scaler, train, test, trend = data['scaler'], data['train'], data['test'], data['trend']
    if not params['detrend']:
        trend = None

    encoder_input_data, decoder_input_data, decoder_target_data = \
        get_seq2seq_inputs(train,
                           window_size=params['input_sequence_length'],
                           horizon=params['output_sequence_length'],
                           shuffle=True)

    ################################
    #     Build & Train Model      #
    ################################

    if params['teacher_forcing']:
        encoder_inputs_shape = (params['input_sequence_length'], encoder_input_data.shape[-1])
        decoder_inputs_shape = (params['output_sequence_length'], decoder_input_data.shape[-1])

        seq2seq = Seq2SeqTF(encoder_layers=params['units'],
                            decoder_layers=params['units'],
                            output_sequence_length=params['output_sequence_length'],
                            cell_type=params['cell'],
                            l2=params['l2'])
        model = seq2seq.build(encoder_inputs=encoder_inputs_shape,
                              decoder_inputs=decoder_inputs_shape)

        all_inputs = [encoder_input_data, decoder_input_data]
    else:
        encoder_inputs_shape = (params['input_sequence_length'], encoder_input_data.shape[-1])
        decoder_inputs_shape = (1, 1)
        if params['exogenous']:
            decoder_inputs_exog_shape = (params['output_sequence_length'], decoder_input_data.shape[-1] - 1)
            exog_input_data = decoder_input_data[:, :, 1:]  # [batch_size, output_sequence_length, n_features -1]
            decoder_input_data = decoder_input_data[:, :1, :1]  # [batch_size, 1, 1]
            all_inputs = [encoder_input_data, decoder_input_data, exog_input_data]
        else:
            decoder_inputs_exog_shape = None
            decoder_input_data = decoder_input_data[:, :1, :1]
            all_inputs = [encoder_input_data, decoder_input_data]

        seq2seq = Seq2SeqStatic(encoder_layers=params['units'],
                                decoder_layers=params['units'],
                                output_sequence_length=params['output_sequence_length'],
                                cell_type=params['cell'])
        model = seq2seq.build(encoder_inputs=encoder_inputs_shape,
                              decoder_inputs=decoder_inputs_shape,
                              decoder_inputs_exog=decoder_inputs_exog_shape)




    if params['load']:
        logger.info("Loading model's weights from disk using {}".format(params['load']))
        model.load_weights(params['load'])

    callbacks = [EarlyStopping(patience=50, monitor='val_loss')]
    model.compile(optimizer=Adam(lr=params['learning_rate']), loss='mse', metrics=metrics)
    history = model.fit(all_inputs, decoder_target_data,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_split=0.1,
                        callbacks=callbacks,
                        verbose=2)

    ################################
    #          Save weights        #
    ################################
    model_filepath = os.path.join(
        config['weights'],'seq2seq_{}_{}_{}'
            .format(params['cell'], params['dataset'], time.time()))
    model.save_weights(model_filepath)
    logger.info("Model's weights saved at {}".format(model_filepath))

    #################################
    # Evaluate on Validation & Test #
    #################################
    encoder_input_data, decoder_input_data, decoder_target_data = \
        get_seq2seq_inputs(test,
                           window_size=params['input_sequence_length'],
                           horizon=params['output_sequence_length'],
                           shuffle=False)
    # IMPORTANT: Remember to pass the trend values through the same ops as the inputs values
    if params['detrend']:
        _, _, decoder_target_trend = get_seq2seq_inputs(
            trend[1],
            window_size=params['input_sequence_length'],
            horizon=params['output_sequence_length'],
            shuffle=False)
        trend = decoder_target_trend

    # Define lambdas to be used during eval
    fn_inverse_val = lambda x: dataset.inverse_transform(x, scaler=scaler, trend=None)
    fn_inverse_test = lambda x: dataset.inverse_transform(x, scaler=scaler, trend=trend)
    fn_plot = lambda x: plot(x, dataset.SAMPLES_PER_DAY, save_at=None)

    if params['teacher_forcing']:
        # decoder_target_data = np.squeeze(decoder_target_data)
        seq2seq.build_prediction_model((1, decoder_input_data.shape[-1]))
        if params['exogenous']:
            val_scores = seq2seq.evaluate(history.validation_data[:-1],
                                          fn_inverse=fn_inverse_val,
                                          horizon=params['output_sequence_length'])
            test_scores = seq2seq.evaluate([encoder_input_data,
                                            decoder_input_data[:, :, 1:],
                                            decoder_target_data],
                                           fn_inverse=fn_inverse_test,
                                           horizon=params['output_sequence_length'],
                                           fn_plot=fn_plot)
        else:
            val_scores = seq2seq.evaluate([history.validation_data[0],
                                           None,
                                           history.validation_data[2]],
                                          fn_inverse=fn_inverse_val,
                                          horizon=params['output_sequence_length'])
            test_scores = seq2seq.evaluate([encoder_input_data,
                                            None,
                                            decoder_target_data],
                                           fn_inverse=fn_inverse_test,
                                           horizon=params['output_sequence_length'],
                                           fn_plot=fn_plot)
    else:
        val_scores = seq2seq.evaluate(history.validation_data[:-1], fn_inverse=fn_inverse_val)
        if params['exogenous']:
            test_scores = seq2seq.evaluate([encoder_input_data,
                                            decoder_input_data[:, :1, :1],
                                            decoder_input_data[:, :, 1:],
                                            decoder_target_data],
                                           fn_inverse=fn_inverse_test,
                                           fn_plot=fn_plot)
        else:
            test_scores = seq2seq.evaluate([encoder_input_data,
                                            decoder_input_data[:, :1, :1],
                                            decoder_target_data],
                                           fn_inverse=fn_inverse_test,
                                           fn_plot=fn_plot)



    metrics_names = [m.__name__ if not isinstance(m, str) else m for m in model.metrics]
    return dict(zip(metrics_names, val_scores)), \
           dict(zip(metrics_names, test_scores)), \
           model_filepath


if __name__ == '__main__':
    grid_search = True
    if grid_search:
        run_grid_search(
            experimentclass=DTSExperiment,
            parameters=yaml.load(open(os.path.join(config['config'], 'seq2seq.yaml'))),
            db_name=config['db'],
            ex_name='seq2seq_grid_search',
            f_main=main,
            f_config=ex_config,
            f_metrics=log_metrics,
            cmd_args=vars(args),
            observer_type='mongodb')
    else:
        run_single_experiment(
            experimentclass=DTSExperiment,
            db_name=config['db'],
            ex_name='seq2seq',
            f_main=main,
            f_config=ex_config,
            f_metrics=log_metrics,
            cmd_args=vars(args),
            observer_type='mongodb')