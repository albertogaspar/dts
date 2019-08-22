import keras.backend as K
from keras.losses import mean_absolute_percentage_error
import numpy as np
import pandas as pd
from datetime import date, time, datetime
import re
import tensorflow as tf
from tqdm import tqdm
from itertools import product
from argparse import ArgumentParser


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.

def print_monthly_mape(df, y_true, y_pred):
    """
    Print Monthly MAPE (i.e for each month in test set provides the performance)
    :param df: pd.DataFrame
    :param y_true: np.array
    :param y_pred: np.array
    :return: dict
    """
    months = df['month'].values
    mape_by_month = dict()
    for i in list(set(months)):
        mask = months == i
        mask = np.reshape(mask,(-1,1))
        row_idxs = np.arange(len(y_true))[:, np.newaxis][mask]
        mape_by_month['{0}) {1}'.format(i, date(1900, i, 1).strftime('%B'))] = \
            K.eval(K.mean(mean_absolute_percentage_error(y_true[row_idxs], y_pred[row_idxs])))
    return mape_by_month


def get_df_time_slice(df, hour, minute):
    """
    Return a dataframe continaing only the samples timestamped with the given time of the day
    :param df: pd.DataFrame
        Original dataframe
    :param hour: int
    :param minut: int
    :return:
    """
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


def transform_data(series_array):
    series_array = np.log1p(np.nan_to_num(series_array))  # filling NaN with 0
    series_mean = series_array.mean()
    series_array = series_array - series_mean
    return series_array, series_mean


def inverse_transform_data(series_array, series_mean):
    series_array = series_array + series_mean
    series_array = np.exp(series_array) - 1.
    return series_array


def autocorrelation(x, lags=1, plot=False, alpha=0.05):
    from statsmodels.tsa.stattools import acf
    from statsmodels.graphics.tsaplots import plot_acf
    ac = acf(x, nlags=lags)
    if plot:
        return ac, plot_acf(x, lags=lags, alpha=alpha, use_vlines=True, markersize=0.)
    return ac


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

def print_test_results(filepath):
    ll = open(filepath).readlines()
    lines = ' '.join(ll)
    rmse = ('RMSE', [float(x) for x in re.findall('(?<=RMSE) [0-9]*.[0-9]*', lines)])
    mae = ('MAE', [float(x) for x in re.findall('(?<=MAE) [0-9]*.[0-9]*', lines)])
    nrmse = ('NRMSD', [1e2*float(x) for x in re.findall('(?<=NRMSD) [0-9]*.[0-9]*', lines)])
    r2 = ('R2', [float(x) for x in re.findall('(?<=R2) [-]?[0-9]*.[0-9]*', lines)[::2]])
    for var in [rmse, mae, nrmse, r2]:
        print('{}: {} +- {}'.format(var[0], np.mean(var[1]), np.std(var[1])))


def get_all_models_results_one_dataset(filepath):
    import os
    names = []
    results = []
    for filename in list(filter(lambda x: 'best' in x, os.listdir(filepath))):
        ll = open(os.path.join(filepath,filename)).readlines()
        lines = ' '.join(ll)
        rmse = [float(x) for x in re.findall('(?<=RMSE) [0-9]*.[0-9]*', lines)]
        mae = [float(x) for x in re.findall('(?<=MAE) [0-9]*.[0-9]*', lines)]
        nrmse = [1e2*float(x) for x in re.findall('(?<=NRMSD) [0-9]*.[0-9]*', lines)]
        r2 = [float(x) for x in re.findall('(?<=R2) [-]?[0-9]*.[0-9]*', lines)[::2]]
        names.append(filename)
        results.append([np.mean(x) for x in [rmse, mae,nrmse,r2]])
    return names, results


def get_model_results_all_datasets(filepaths):
    names = []
    results = []
    for filename in filepaths:
        ll = open(filename).readlines()
        lines = ' '.join(ll)
        rmse = [float(x) for x in re.findall('(?<=RMSE) [0-9]*.[0-9]*', lines)]
        mae = [float(x) for x in re.findall('(?<=MAE) [0-9]*.[0-9]*', lines)]
        nrmse = [1e2*float(x) for x in re.findall('(?<=NRMSD) [0-9]*.[0-9]*', lines)]
        r2 = [float(x) for x in re.findall('(?<=R2) [-]?[0-9]*.[0-9]*', lines)[::2]]
        names.append(filename)
        results.append([np.mean(x) for x in [rmse, mae,nrmse,r2]])
    return names, results


def normality_test(filepath, test_type='ks'):
    """
    2 sampled Kolomogorov Smirnov test for normality on each metric
    """
    from scipy.stats import ks_2samp, shapiro
    ll = open(filepath).readlines()
    lines = ' '.join(ll)
    rmse = ('RMSE', [float(x) for x in re.findall('(?<=RMSE) [0-9]*.[0-9]*', lines)])
    mae = ('MAE', [float(x) for x in re.findall('(?<=MAE) [0-9]*.[0-9]*', lines)])
    nrmse = ('NRMSD', [1e2*float(x) for x in re.findall('(?<=NRMSD) [0-9]*.[0-9]*', lines)])
    r2 = ('R2', [float(x) for x in re.findall('(?<=R2) [-]?[0-9]*.[0-9]*', lines)[::2]])
    normal = []
    for var in [rmse, mae, nrmse, r2]:
        if test_type == 'ks':
            pvalue = ks_2samp((np.array(var[1])-np.mean(var[1]))/np.std(var[1]),
                              np.random.normal(0,1, size=len(var[1]))).pvalue
            # pvalue = ks_2samp(np.array(var[1]),
            #                   np.random.normal(0, 1, size=len(var[1]))).pvalue
        else:
            pvalue = shapiro(var[1])[1]
        if pvalue < 0.05:
            normal.append(False)
            txt = 'the data differs significantly from that which is normally distributed'
        else:
            normal.append(True)
            txt = 'the data do not differs significantly from that which is normally distributed'
        print('{} Normality test: p-value={} --- > {}'.format(var[0], pvalue, txt))
    return np.array(normal)

def unpaired_ttest(fp1, fp2):
    from scipy.stats import ttest_ind, bartlett, mannwhitneyu
    from itertools import product
    rmse = []
    mae = []
    nrmse = []
    r2 = []
    metrics_names = ['rmse', 'mae', 'nrmse', 'r2']
    print('Assumption 1: We assume that the data from the 2 groups are independent')

    print('\nAssumption 2: Are the data from each of the 2 groups follow a normal distribution?')
    normal = np.array([True]*4)
    for filepath in [fp1, fp2]:
        normal = normal & normality_test(filepath)
        ll = open(filepath).readlines()
        lines = ' '.join(ll)
        rmse.append([float(x) for x in re.findall('(?<=RMSE) [0-9]*.[0-9]*', lines)])
        mae.append([float(x) for x in re.findall('(?<=MAE) [0-9]*.[0-9]*', lines)])
        nrmse.append([1e2*float(x) for x in re.findall('(?<=NRMSD) [0-9]*.[0-9]*', lines)])
        r2.append([float(x) for x in re.findall('(?<=R2) [-]?[0-9]*.[0-9]*', lines)[::2]])
    metrics = [rmse, mae, nrmse, r2]

    print('\nAssumption 3: Do the populations have the same variance?')
    equal_var = []
    for i, var in enumerate(metrics):
        pvalue = bartlett(var[0], var[1]).pvalue
        if pvalue < 0.05:
            equal_var.append(False)
            txt = 'there is significant difference between the variances of the two sets of data'
        else:
            equal_var.append(True)
            txt = 'there is no significant difference between the variances of the two sets of data'
        print('{} F-test: p-value={} --- > {}'.format(metrics_names[i], pvalue, txt))

    print('\nUnpaired t-test for the two pops having equal mean')
    pvalues = []
    for i, var in enumerate(metrics):
        if normal[i]:
            test = 't-test'
            pvalue = ttest_ind(var[0], var[1], equal_var=equal_var).pvalue
        else:
            test = 'Mann-Whitney rank test'
            pvalue = mannwhitneyu(var[0], var[1]).pvalue
        if pvalue < 0.05:
            txt = 'there is significant difference between the means of the two populations'
        else:
            txt = 'there is no significant difference between the means of the two populations'
        pvalues.append(pvalue)
        print('{} {}: p-value={} --- > {}'.format(metrics_names[i], test, pvalue, txt))

    print('Bonferroni corection')
    from statsmodels.sandbox.stats.multicomp import multipletests
    res = multipletests(pvalues, method='bonferroni')
    print(res[0],res[1])


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