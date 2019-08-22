"""
Main script for training and evaluating a seq2seq on a multi-step forecasting task.
You can chose bewteen:
    - Running a simple experiment
    - Running multiple experiments trying out diffrent combinations of hyperparamters (grid-search)
"""

from keras.callbacks import EarlyStopping
from dts import config
from dts.datasets import uci_single_households
from dts.datasets import gefcom2014
from dts import logger
from dts.utils.plot import plot
from dts.utils.decorators import f_main
from dts.utils import metrics
from dts.utils import run_grid_search, run_single_experiment
from dts.utils import DTSExperiment, log_metrics, get_args
from dts.utils.split import *
from dts.models.Seq2Seq import *
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




    if params['load'] is not None:
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
    if args.grid_search:
        run_grid_search(
            experimentclass=DTSExperiment,
            #parameters=yaml.load(open(os.path.join(config['config'], 'seq2seq_gs.yaml'))),
            f_config=args.add_config,
            db_name=config['db'],
            ex_name='seq2seq_grid_search',
            f_main=main,
            f_metrics=log_metrics,
            observer_type=args.observer)
    else:
        run_single_experiment(
            experimentclass=DTSExperiment,
            db_name=config['db'],
            ex_name='seq2seq',
            f_main=main,
            f_config=args.add_config,
            f_metrics=log_metrics,
            observer_type=args.observer)