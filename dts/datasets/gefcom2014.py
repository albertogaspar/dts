import pandas as pd
from dts import config, logger
from dts.datasets.utils import *
from dts.utils.split import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import requests, zipfile, io
import warnings

NAME = 'gefcom'
SAMPLES_PER_DAY = 24
FREQ = 'H'
TARGET = 'LOAD'
DATETIME = 'datetime'


def download():
    """
    Download data and extract them in the data directory
    :return:
    """
    url = 'https://www.dropbox.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip?dl=1'
    logger.info('Donwloading .zip file from {}'.format(url))
    r = requests.get(url)
    logger.info('File downloaded.')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path=config['data'])

    new_loc = os.path.join(config['data'], 'GEFCom2014')
    os.rename(os.path.join(config['data'], 'GEFCom2014 Data'),
              new_loc)

    z = zipfile.ZipFile(os.path.join(new_loc, 'GEFCom2014-L_V2.zip'))
    z.extractall(path=os.path.join(new_loc))

    logger.info('File extracted and available at {}'.format(new_loc))
    process_csv()
    logger.info('The original CSV has been parsed and is now available at {}'.format(
        os.path.join(config['data'], 'UCI_household_power_consumption_synth.csv')))


def process_csv():
    """
    Parse the datetime field, Sort the values accordingly and save the new dataframe to disk
    """
    df = load_raw_dataset()
    df.to_csv(os.path.join(config['data'], 'GEFCom2014/Load/gefcom2014.csv'), index=False)


def load_raw_dataset():
    """
    Load the dataset as is.
    :return: pandas.DataFrame: sorted dataframe with parsed datetime
    """
    data_dir = os.path.join(config['data'], 'GEFCom2014/Load/Task 1/')
    df = pd.read_csv(os.path.join(data_dir, 'L1-train.csv'))
    for i in range(2, 16):
        data_dir = os.path.join(config['data'], 'GEFCom2014/Load/Task {}/'.format(i))
        tmp = pd.read_csv(os.path.join(data_dir, 'L{}-train.csv'.format(i)))
        df = pd.concat([df, tmp], axis=0)
    df[DATETIME] = pd.date_range('01-01-2001', '12-01-2011', freq=FREQ)[1:]
    df = df[~pd.isnull(df.LOAD)].reset_index(drop=True)
    return df


def load_dataset():
    """
    Load an already cleaned version of the dataset
    """
    df = pd.read_csv(os.path.join(config['data'], 'GEFCom2014/Load/gefcom2014.csv'))
    df[DATETIME] = df[DATETIME].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return df


def load_data(fill_nan=None,
              preprocessing=True,
              detrend=False,
              exogenous_vars=False,
              train_len=365 * 3 * SAMPLES_PER_DAY,
              test_len=365 * SAMPLES_PER_DAY,
              valid_len=0,
              split_type='simple',
              is_train=False,
              use_prebuilt = True):
    """
    Create a split of the data according to the given dimensions for each set.
    :param fill_nan: string that identifies how NaN values should be filled. Options are:
        -bfill: fill NaN value at index i with value at index i-1
        -ffill: fill NaN value at index i with value at index i+1
        -mean: fill NaN value at index i  with the mean value over all dataset at the same hour,minute
        -median: fill NaN value at index i  with the median value over all dataset at the same hour,minute
        -drop: drop all rows with missing values
    :param preprocessing: if True, standardize features using standard scaler
    :param detrend: if True, use train weekly statistics to detrend the time series.
        (WORKS ONLY FOR split_type=simple or split_type=default when is_train=False)
    :param exogenous_vars: if True, add exogenous features to the input data
        (temperatures + date/time feature + holiday feature)
    :param train_len: length of the train dataset
    :param test_len: length of the test set
    :param valid_len: length of the validation set
    :param split_type: 'simple', 'multi' or 'default'.
        - 'simple': See dts.utils.split.simple_split
        - 'multi':  See dts.utils.split.multiple_split
        - 'default': Uses 'simple' split for train-test, then divides training using the 'multi' approach.
    :param use_prebuilt: if True, load already splitted data files from disk
    :return: a dict having the following (key, value) pairs:
        - train = training dataset, np.array of shape()
        - test = test dataset, np.array of shape()
        - scaler = the scaler used to preprocess the data
        - trend  = None or the values that has to be added back after prediction if pdetrending has been used.
    """
    dataset = dict(
        train=None,
        test=None,
        scaler=None,
        trend=[None,None],
    )
    if valid_len == 0:
        valid_len = int(0.1*train_len)

    if split_type == 'simple':
        train_test_split = lambda x: simple_split(x, train_len=None, valid_len=0, test_len=test_len)
        train_valid_split = lambda x: simple_split(
            train_test_split(x)[0],
            train_len=len(train_test_split(x)[0]) - valid_len,
            valid_len=0,
            test_len=valid_len)
    elif split_type == 'multi':
        train_test_split = lambda x: multiple_splits(x, train_len=train_len + valid_len, valid_len=0, test_len=test_len)
        train_valid_split = lambda x: [x[0][:, :train_len, :], None, x[0][:, train_len:, :]]
    elif split_type == 'default':
        train_test_split =  lambda x: simple_split(x, train_len=None, valid_len=0, test_len=int(0.1*df.shape[0]))
        train_valid_split = lambda x: multiple_splits(
            train_test_split(x)[0],
            train_len=5 * 31 * SAMPLES_PER_DAY,
            valid_len=0,
            test_len=31 * SAMPLES_PER_DAY)
    else:
        raise ValueError('{} is not a valid split type.'.format(split_type))

    if not use_prebuilt:
        logger.info('Fetching and preprocessing data. This will take a while...')
        try:
            df = load_dataset()
        except:
            logger.info('The dataset seems to be unavailable on your disk. '
                        'It will be downloaded'
                        'This will take some time...')
            download()
            df = load_dataset()

        if detrend:
            if split_type == 'default' and not is_train:
                df, trend_values = apply_detrend(df, df.shape[0] - 365*SAMPLES_PER_DAY)
                trend_values = train_test_split(np.expand_dims(trend_values,-1))[::2]
            elif split_type == 'simple' and is_train:
                df, trend_values = apply_detrend(df, train_len)
                trend_values = train_valid_split(np.expand_dims(trend_values,-1))[::2]
            elif split_type == 'simple':
                df, trend_values = apply_detrend(df, train_len+valid_len)
                trend_values = train_test_split(np.expand_dims(trend_values,-1))[::2]
            else:
                raise ValueError('Detrend cannot be applied with this type of split.')
            dataset['trend'] = trend_values


        X = df[TARGET].values[:-1] # load values
        X = np.expand_dims(X, axis=-1)
        if preprocessing:
            # init scaler using only information for training
            scaler, _ = transform(X[:train_len])
            # actual preprocess
            _, X = transform(X, scaler)
        else:
            scaler = None
        if exogenous_vars:
            # init scaler using only temperature information for training
            X_temp, X_ex = add_exogenous_variables(df, one_hot=True)
            scaler_temp, _ = transform(X_temp[:train_len], scaler_type='minmax')
            _, X_temp = transform(X_temp, scaler_temp)
            X = np.concatenate([X, X_temp, X_ex], axis=1)  # Load @ t-1, Datetime @ t, Temp @ t

        if is_train:
            data = train_valid_split(X)
        else:
            data = train_test_split(X)

        dataset['scaler'] = scaler
        dataset['train'] = np.array(data[0], dtype=np.float32)
        dataset['test'] = np.array(data[2], dtype=np.float32)
        return dataset

    else:
        logger.info('Fetching preprocessed data from disk...')
        try:
            return load_prebuilt_data(split_type=split_type, exogenous_vars=exogenous_vars, detrend=detrend,
                                      is_train=is_train, dataset_name=NAME)
        except FileNotFoundError as e:
            logger.warn('An already preprocessed version of the data do not exists on disk. '
                        'The train/test data will be created now.')
            return load_data(fill_nan, preprocessing, detrend, exogenous_vars, train_len, test_len,
                             valid_len, split_type, is_train, use_prebuilt=False)


def add_exogenous_variables(df, one_hot=True):
    """
    Augument the dataframe with exogenous features (date/time feature + holiday feature).
    The feature's values can be kept as they are or they can be one hot encoded
    :param df: the dataframe
    :param one_hot: if True, one hot encode all the features.
    :return: two matrix of exogenous features,
     the first for temperatures only the second one contains all the other variables.
    """
    X_temp = df[['w{}'.format(i) for i in range(1, 26)]].values[1:]  # temperature values
    df['year'] = df.datetime.map(lambda x: x.year)
    df['month'] = df.datetime.map(lambda x: x.month)
    df['day'] = df.datetime.map(lambda x: x.day)
    df['hour'] = df.datetime.map(lambda x: x.hour)
    df['holiday'] = [0] * len(df)
    df = _add_holidays(df)
    if one_hot:
        ex_feat = pd.get_dummies(df, columns=['year', 'month', 'day', 'hour', 'holiday'])
        return X_temp, ex_feat.values[:, -4 - (ex_feat.shape[1] - df.shape[1]):][1:]
    else:
        return X_temp, df.values


def _add_holidays(df):
    """
    Add a binary variable to the dataset that takes value: 1 if the day is a holiday, 0 otherwise.
    Main holidays for the New England area are considered.
    :param df: the datafrme
    :return: the agumented dtaframe
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idx = []
        idx.extend(df[df.day == 1][df.month == 1].index.tolist())  # new year's eve
        idx.extend(df[df.day == 4][df.month == 7].index.tolist())  # independence day
        idx.extend(df[df.day == 11][df.month == 11].index.tolist())  # veternas day
        idx.extend(df[df.day == 25][df.month == 12].index.tolist())  # christams
        df.loc[idx, 'holiday'] = 1
        return df


def transform(X, scaler=None, scaler_type=None):
    """
    Apply standard scaling to the input variables
    :param X: the data
    :param scaler: the scaler to use, None if StandardScaler has to be used
    :return:
        scaler used
        X transformed using scaler
    """
    if scaler is None:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        scaler.fit(X)
    return scaler, scaler.transform(X)


def inverse_transform(X, scaler, trend=None):
    """
    :param X: the data
    :param scaler: the scaler that have been used for transforming X
    :param trend: the trebd values that has been removed from X. None if no detrending has been used.
        It has to be the same dim. as X.
    :return:
        X with trend adds back and de-standardized
    """
    X = X.astype(np.float32)
    X = scaler.inverse_transform(X)
    try:
        X += trend
    except TypeError as e:
        logger.warn(str(e))
    except Exception as e:
        logger.warn('General error (not a TypeError) while adding back time series trend. \n {}'.format(str(e)))
    return X


def apply_detrend(df, train_len):
    """
    Perform detrending on a time series by subtrating from each value of the dataset
    the average value computed over the training dataset for each hour/weekday
    :param df: the dataset
    :param test_len: test length,
    :return:
        - the detrended datasets
        - the trend values that has to be added back after computing the prediction
    """
    # Compute mean values for each hour of every day of the week (STATS ARE COMPUTED USING ONLY TRAIN SET)
    dt_idx = pd.DatetimeIndex(df[DATETIME])
    df_copy = df.set_index(dt_idx, drop=False)
    df_train_mean = \
        df_copy.iloc[:train_len].groupby(
            [df_copy.iloc[:train_len].index.hour])[TARGET].mean()
    # Remove mean values from dataset
    df_copy['trend'] = None
    for h in df_train_mean.index:
            mu = df_train_mean[h]
            idxs = df_copy.loc[(df_copy.index.hour == h)].index
            df_copy.loc[idxs, TARGET] = df_copy.loc[idxs, TARGET].apply(lambda x: x - mu)
            df_copy.loc[idxs, 'trend'] = mu
    df[TARGET] = df_copy[TARGET].values
    return df, np.float32(df_copy['trend'].values[:-1])





