# DTS - Deep Time-Series Forecasting

DTS is a [Keras](https://keras.io/) library that provides multiple deep architectures aimed at multi-step time-series forecasting.
The [Sacred](https://github.com/IDSIA/sacred) library is used to keep track of different experiments and allow their reproducibility. 

## Installation
DTS is compatible with Python 3.5+, and is tested on Ubuntu 16.04. 
To install dts from source:
```
git clone https://github.com/albertogaspar/dts.git
cd dts
pip install -e .
```
The installation of [MongoDB](https://www.mongodb.com/) is not highly recommended.

# What's in it & How to use

#### Time-Series Forecasting
The package includes several deep learning architectures that can be used for multi step-time series forecasting. 
The package provides also several utilities to cast the forecasting problem into a supervised machine learning problem. 
Specifically a sliding window approach is used: each model is given a time window of size n<sub>T</sub> and asked 
to output a prediction for the following n<sub>O</sub> timesteps (see Figure below).

<p align="center">
  <img src="./images/notation.png" width="70%" height="70%"/>
</p>

An example on how to generate _synthetic data_ useful for fitting a model:
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
X_train has shape _(n_train_samples, 24*7, n_features)_, y_train has shape _(n_train_samples, 24*7)_. if multivariate_output is set to True then
y_train will have shape _(n_train_samples, 24, n_features)_. 

With DTS you can model your input values in many diffrent ways and then feed them to your favourite 
deep learning architectures. E.g.: 
- you can decide to include **exogenous features** (like temperature readings) if they are available.

- you can decide to apply **detrending** to the time series (see `dts.datasets.*.apply_detrend` for more details).

#### Datasets
- **Individual household electric power consumption Data Set**: Measurements of electric power consumption in _one household_ with a one-minute sampling rate over a period of almost 4 years.
[Dataset & Description](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption).
- **GEFCom 2014**: hourly consumption data coming from ISO New England (aggregated consumption).
[Dataset & Description](http://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html), 
[Paper](https://www.sciencedirect.com/science/article/pii/S0169207016000133?via%3Dihub).
If you use the GEFCom2014 you should cite this [paper](https://www.sciencedirect.com/science/article/pii/S0169207016000133?via%3Dihub) to acknowledge the source.

To see a working example on one of this datasets check `dts.examples`.

#### Available architectures
Included architetures are:
- Feed Forward networks with and without residual connections.
- **Recurrent Neural Networks** (Elmann, LSTM, GRU) with different trainig procedure:
  - MIMO: a Dense Network is used to map the last state of the RNN to the output space of size n<sub>O</sub>. 
  The training and inference procedures are the same 
  - Recurrent: The RNN is trained to predict the next step, i.e. the output space during training has size 1. During inference, 
  the network is fed with (part of) the input plus it's own predictions in a recurrent fashion until an ouput vector of length 
  n<sub>O</sub> is obtained.
- **Seq2Seq**:

<p align="center">
  <img src="./images/seq2seq.png" width="60%" height="60%"/>
</p>

  different training procedure are available (see [Professor Forcing: A New Algorithm for Training Recurrent Networks](https://arxiv.org/abs/1610.09038) for more details)
  - Teacher Forcing and Self-Genearted Samples:
  
<p align="center">
  <img src="./images/seq2seqTFSG.png" width="60%" height="60%"/>
</p>

  - TODO: [Professor Forcing](https://arxiv.org/abs/1610.09038), [Scheduled Sampling](https://arxiv.org/abs/1506.03099) 
- **Temporal Convolutional Neural Networks**:
    
<p align="center">
  <img src="./images/TCN.png" width="70%" height="70%"/>
</p>

  - [Wavenet](https://arxiv.org/abs/1609.03499)
  - [TCN](https://arxiv.org/abs/1803.01271)
  - [Conditional TCN](https://arxiv.org/abs/1703.04691)

**Train a model**:

See `dts.examples`

**Load weights**:

If you want to load a model using pretrained weights just run the model setting the paramters load to the fullpath of 
the file containing the weights. The model will be initilized with this weights before training. 

#### Run Experiment
The main function for your model should always look similar to this one:
```python
if __name__ == '__main__':
    import os, yaml
    from dts import config
    from dts.utils.experiments import DTSExperiment, run_grid_search, run_single_experiment
    
    if grid_search:
        run_grid_search(
            experimentclass=DTSExperiment,
            parameters=yaml.load(open(os.path.join(config['config'], 'MODEL_CONFIG_FILENAME.yaml'))),
            db_name=config['db'],
            ex_name='EXPERIMENT_NAME',
            f_main=main,
            f_config=ex_config,
            f_metrics=log_metrics,
            cmd_args=vars(args),
            observer_type='mongodb')
    else:
        run_single_experiment(
            experimentclass=DTSExperiment,
            db_name=config['db'],
            ex_name='EXPERIMENT_NAME',
            f_main=main,
            f_config=ex_config,
            f_capture=log_metrics,
            cmd_args=vars(args),
            observer_type='mongodb')
```
- _grid_search_: defines whether or not you are searching for the best hyperparamters. 
If True, multiple experiments are runned, each with a different combination of hyperparamters. 
The process terminates when all possible combinations of hyperparamers have been explored. 
The hyperparamters should be define  as a yaml file in the config folder 
(see [How to write a config file]() for more details).
- for the other paramters please check the documentation at `dts.utils.experiments`

## Project Structure
- **dts**: contains models, utilities and example to train and test differnt deep learning models.
- **data**: contains raw data and .npz,.npy data (already preprocessed data). 
- **config**: yaml file to be used for grid search of hyperparamteres for all architectures.
- **weights**: conatins models' weights. If you use sacred using the mongodb observer then, the _artifactID_ field in each document 
contains the name of the trained model that achieved the presented performance. 
- **log**: If you use sacred without mongodb then all the relevant files are stored in this directory.

#### Sacred Collected Information
The animation below provides an intutive explanation of the information collected by Sacred (using MongoDB as Observer).
The example refers to a completed experiment of a TCN model trained on the Individual household electric power consumption Data Set 
(for brevity, 'uci'):

<p align="center">
  <img src="./images/sacred.gif" width="70%" height="70%"/>
</p>

When MongoDB is used as an Observer, the collected information for an experiment is stored in a document. 
In the above documents are visualized using [MongoDB Compass](https://www.mongodb.com/products/compass?lang=it-it) 

## Reference
This is the code used for [Deep Learning for Time Series Forecasting: The Electric Load Case](https://arxiv.org/abs/1907.09207).
Mind that the code has been changed a bit, thus you may notice some differences with the models described in the paper. 

If you find it interesting it please consider citing us:
```
@article{gasparin2019deep,
  title={Deep Learning for Time Series Forecasting: The Electric Load Case},
  author={Gasparin, Alberto and Lukovic, Slobodan and Alippi, Cesare},
  journal={arXiv preprint arXiv:1907.09207},
  year={2019}
}
```