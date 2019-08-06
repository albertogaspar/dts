import os
from datetime import datetime
import numpy as np
from sklearn.externals import joblib
from dts import config


def save_data(data, split_type=None, exogenous_vars=False, is_train=False, dataset_name=None):
    """
    Save all relevant information as a numpy array
    :param data: dict having as keys: 'train', 'test', 'scaler', 'trend'.
                 Use load_data method of one of the datasets in dts.dataset to generate it.
    :param split_type: 'simple', 'multi', 'default'
    :param exogenous_vars: True if exogenous vars have been used
    :param is_train: True if training
    :param dataset_name: 'uci', 'gefcom2014'
    """
    dirname = '{}_{}'.format(dataset_name, split_type)
    path = os.path.join(config['data'], '{}/{}'.format(dataset_name, dirname))
    if not os.path.exists(path):
        os.mkdir(path)
    data_filename, scaler_filename = build_filenames(data=data, is_train=is_train,
                                                     exogenous_vars=exogenous_vars,
                                                     dataset_name=dataset_name)
    try:
        data['trend'] = [data['trend'][0], data['trend'][1]]
    except:
        data['trend'] = [None, None]
    np.savez_compressed(os.path.join(path, data_filename),
             train=data['train'],
             test=data['test'],
             trend_train=data['trend'][0],
             trend_test=data['trend'][1])
    joblib.dump(data['scaler'],
                os.path.join(path, scaler_filename))


def load_prebuilt_data(split_type=None, exogenous_vars=False, detrend=False, is_train=False, dataset_name=None):
    """
    Load one of the prebuilt dataset from disk.
    :param split_type: 'simple', 'multi', 'default'
    :param exogenous_vars:
    :param is_train:
    :return: a dict having the following (key, value) pairs:
        - train = training dataset, np.array of shape()
        - test = test dataset, np.array of shape()
        - scaler = the scaler used to preprocess the data
        - trend  = None or the values that has to be added back after prediction if pdetrending has been used.
    """
    dirname = '{}_{}'.format(dataset_name, split_type)
    path = os.path.join(config['data'], '{}/{}'.format(dataset_name, dirname))
    if is_train:
        t = 'train'
    else:
        t = 'test'

    data_files = sorted(list(filter(
        lambda x: x.startswith('{}_data_{}'.format(dataset_name, t)), os.listdir(path))))
    scaler_files = sorted(list(filter(
        lambda x: x.startswith('{}_scaler_{}'.format(dataset_name, t)), os.listdir(path))))
    if exogenous_vars:
        data_files = sorted(list(filter(
            lambda x: x.startswith('{}_data_{}'.format(dataset_name, t)) and
                      'exog' in x, data_files)))
        scaler_files = sorted(list(filter(
            lambda x: x.startswith('{}_scaler_{}'.format(dataset_name, t)) and
                      'exog' in x, scaler_files)))
    else:
        data_files = sorted(list(filter(
            lambda x: x.startswith('{}_data_{}'.format(dataset_name, t)) and
                      'exog' not in x, data_files)))
        scaler_files = sorted(list(filter(
            lambda x: x.startswith('{}_scaler_{}'.format(dataset_name, t)) and
                      'exog' not in x, scaler_files)))
    if detrend:
        data_files = sorted(list(filter(
            lambda x: x.startswith('{}_data_{}'.format(dataset_name, t)) and
                      'detrend' in x, data_files)))
        scaler_files = sorted(list(filter(
            lambda x: x.startswith('{}_scaler_{}'.format(dataset_name, t)) and
                      'detrend' in x, scaler_files)))
    else:
        data_files = sorted(list(filter(
            lambda x: x.startswith('{}_data_{}'.format(dataset_name, t)) and
                      'detrend' not in x, data_files)))
        scaler_files = sorted(list(filter(
            lambda x: x.startswith('{}_scaler_{}'.format(dataset_name, t)) and
                      'detrend' not in x, scaler_files)))
    data_file = data_files[-1]
    scaler_file = scaler_files[-1]
    del scaler_files
    del data_files

    data = np.load(os.path.join(path, data_file))
    return dict(
        scaler=joblib.load(os.path.join(path, scaler_file)),
        train=data['train'],
        test=data['test'],
        trend=[data['trend_train'],data['trend_test']])


def build_filenames(data, is_train=False, exogenous_vars=False, dataset_name=None):
    time = datetime.today()
    if is_train:
        t = 'train'
    else:
        t = 'test'
    data_filename = '{}_data_{}_{}'.format(dataset_name, t, time)
    scaler_filename = '{}_scaler_{}_{}'.format(dataset_name, t, time)
    if exogenous_vars:
        data_filename += '_exog'
        scaler_filename += '_exog'
    if data['trend'] is not None and data['trend'] != [None,None]:
        data_filename += '_detrend'
        scaler_filename += '_detrend'
    return data_filename, scaler_filename