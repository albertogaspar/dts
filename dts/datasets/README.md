# Datasets

## Available datasets

- **Individual household electric power consumption Data Set**: Measurements of electric power consumption in _one household_ with a one-minute sampling rate over a period of almost 4 years.
[Dataset & Description](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption).
- **GEFCom 2014**: hourly consumption data coming from ISO New England (aggregated consumption).
[Dataset & Description](http://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html), 
[Paper](https://www.sciencedirect.com/science/article/pii/S0169207016000133?via%3Dihub).
If you use the GEFCom2014 you should cite this [paper](https://www.sciencedirect.com/science/article/pii/S0169207016000133?via%3Dihub) to acknowledge the source.

## Load Data
With DTS you can model your input values in many diffrent ways and then feed them to your favourite 
deep learning architectures. E.g.: 
- you can decide to include **exogenous features** (like temperature readings) if they are available.

- you can decide to apply **detrending** to the time series (see `dts.datasets.*.apply_detrend` for more details).


Train-test split for a dataset are avialble through the `load_data` method (avaialble for every dataset in
`dts.datasets`):
```python
from dts.datasets import uci_single_households as dataset
data = dataset.load_data(fill_nan='median',
                         preprocessing=True,
                         split_type='simple',
                         use_prebuilt=False,
                         is_train=False,
                         detrend=False,
                         exogenous_vars=False)
``` 
- use_prebuilt: if True, load already splitted data files from disk. To save train-test data on disk
have a look at the example provided in `dts.examples.save_datasets.py`

Check out the documentation for further details about the needed arguments.


## Format your data 

Once you obtain the train-test data you can format them in different ways:
- RNN format (`dts.utils.split.get_rnn_inputs`)
- seq2seq format (`dts.utils.split.get_seq2seq_inputs`)


The following is an example on how to generate train data useful for fitting an RNN model:
```python
import numpy as np
from dts.utils.split import *

# Generate synthetic data
n_features = 10
X = np.random.uniform(0., 10., size=(10000, n_features))

# Split data in train-test (and validation if needed)
train_len = 6000
valid_len = 0
test_len = 4000
train, _, test = simple_split(X, train_len=train_len, valid_len=valid_len, test_len=test_len)

# Format data: 
# window_size is the look-back, we set it to 7 days here (assuming a resolution of 1 hour for the data) 
# horizon is the forecasting horizon, i.e. the number of steps we wants to predict. We set it to 1 day
X_train, y_train = get_rnn_inputs(train, window_size=24*7, horizon=24, multivariate_output=False, shuffle=False)
```
- X_train has shape _(n_train_samples, 24*7, n_features)_, 
- y_train has shape _(n_train_samples, 24*7)_. if multivariate_output is set to True then
y_train will have shape _(n_train_samples, 24, n_features)_. 

