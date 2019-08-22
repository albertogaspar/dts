from dts.datasets.gefcom2014 import *
from matplotlib import pyplot as plt

if __name__ == '__main__':

    df = load_dataset()

    exogenous = False
    detrend = False
    split_type = 'default'
    for is_train in [True, False]:
        data = load_data(fill_nan='median',
                         preprocessing=True,
                         split_type=split_type,
                         use_prebuilt=False,
                         is_train=is_train,
                         detrend=detrend)
        scaler, train, test, trend = data['scaler'], data['train'], data['test'], data['trend']

        # plt.plot(df[TARGET].values)
        # plt.plot(inverse_transform(train[:,:,0], scaler=scaler, trend=data['trend'][0])[0])
        # plt.show()

        save_data(data=data, split_type=split_type, exogenous_vars=exogenous, is_train=is_train, dataset_name=NAME)
        x = load_prebuilt_data(split_type=split_type, exogenous_vars=exogenous, is_train=is_train, detrend=detrend,
                               dataset_name=NAME)
        for k,v in x.items():
            try:
                print(k, v.shape)
            except:
                print(k)