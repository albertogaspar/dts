import keras.backend as K
import numpy as np
import pandas as pd
from datetime import time, datetime
import tensorflow as tf
from argparse import ArgumentParser


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def get_df_time_slice(df, hour, minute):
    t = time(hour, minute, 0)
    mask = df.date.apply(lambda x: x.to_pydatetime().time()) == t
    return df[mask]


def shuffle_x_y(X, y):
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    return X[idxs], y[idxs]


def split_on_date(df, split_date='2007/1/1'):
    df = df.sort_values('date').reset_index()
    split_pt = min(df[df['date'] == datetime.strptime(split_date, '%Y/%m/%d').date()].index)
    return df.iloc[:split_pt], df.iloc[split_pt:]


def set_datetime_index(df, datetime_col='datetime'):
    if not isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex):
        try:
            dt_idx = pd.DatetimeIndex(df[datetime_col])
            df = df.set_index(dt_idx, drop=False)
            return df
        except ValueError:
            raise ValueError('{0} is not the correct datetime column name or the column values '
                             'are not formatted correctly (use datetime)'.format(datetime_col))
    else:
        return df


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--add_config', type=str, default=None,
                        help="full path to the yaml file containing the experiment's (hyper)parameters.")
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument("--observer", type=str, default='mongodb', help="mongodb or file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", type=str, default=None, help="full path to the model's weights")
    args = parser.parse_args()
    return args