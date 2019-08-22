from dts.datasets import gefcom2014, uci_single_households
from matplotlib import pyplot as plt

ds = 'gefcom'

if __name__ == '__main__':
    if ds == 'uci':
        dataset = uci_single_households
        df = dataset.load_dataset('median')
    else:
        dataset = gefcom2014
        df = dataset.load_dataset()

    split_type = 'simple'
    for detrend in [True]:
        for exogenous in [True, False]:
            for is_train in [True, False]:
                data = dataset.load_data(fill_nan='median',
                                         preprocessing=True,
                                         split_type=split_type,
                                         use_prebuilt=False,
                                         is_train=is_train,
                                         detrend=detrend,
                                         exogenous_vars=exogenous)
                scaler, train, test, trend = data['scaler'], data['train'], data['test'], data['trend']

                plt.plot(df[dataset.TARGET].values)
                plt.plot(dataset.inverse_transform(train, scaler=scaler, trend=trend)[0])
                plt.show()

                dataset.save_data(data=data, split_type=split_type, exogenous_vars=exogenous, is_train=is_train,
                                  dataset_name=dataset.NAME)
                x = dataset.load_prebuilt_data(split_type=split_type, exogenous_vars=exogenous, is_train=is_train,
                                               detrend=detrend, dataset_name=dataset.NAME)
                scaler, train, test, trend = x['scaler'], x['train'], x['test'], x['trend']
                for k, v in x.items():
                    try:
                        print(k, v.shape)
                    except:
                        print(k)
