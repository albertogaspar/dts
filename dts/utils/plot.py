import os
from datetime import datetime
from matplotlib import pyplot as plt


def plot(x, samples_per_day=96, save_at=None):
    n_series = len(x)
    today = datetime.today()

    for i in range(n_series):
        y = x[i][::samples_per_day].reshape(-1)
        plt.plot(y[:samples_per_day*4])
    if save_at is not None:
        plt.savefig('{}_{}_SMALL'.format(save_at, today))
    plt.show()

    for i in range(n_series):
        y = x[i][::samples_per_day].reshape(-1)
        plt.plot(y)
    if save_at is not None:
        plt.savefig('{}_{}'.format(save_at,today))
    plt.show()
