from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from pymongo import MongoClient
from dts import config
from itertools import product
import os
import yaml


def main_wrapper(f_main, ex, f_ex_capture, curr_db_name, _run):
    """
    Wrapper for the main function of an experiment.
    Ensures that the DB do not already contain an experiment with the same config as this one.

    :param f_main: function
        updates the main experiment function arguments, calls it and save the
        experiment results and artifacts.
        f_main should have the following signature: f_main(ex, _run, f_log_metrics)
    :param ex: the experiment (an instance of the Experiment class)
    :param f_ex_capture: function
        The function that implements the metrics logging API with sacred
        (should be used with Lambda in keras but has problem right now. Thus it can be ignored)
    :param curr_db_name: str
        Name of the db in use
    :param _run: the run object for the current run
        For more details about the Run object see https://sacred.readthedocs.io/en/latest/experiment.html#run-the-experiment
    """
    client = MongoClient('localhost', 27017)
    print('db = ', curr_db_name)
    db = client[curr_db_name]
    duplicate_ex = check_for_completed_experiment(db, _run.config)
    if duplicate_ex is not None:
        raise ValueError('Aborting due to a duplicate experiment')
        # return f_main(ex, _run, f_ex_capture)
    else:
        return f_main(ex, _run, f_ex_capture)


def check_for_completed_experiment(db, config):
    return db['runs'].find_one({'config': config})


def log_metrics(_run, logs):
    """
    Implements the metrics logging API with SACRED
    """
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("mse", float(logs.get('mse')))
    _run.log_scalar("val_mse", float(logs.get('val_mse')))
    _run.result = float(logs.get('val_loss'))


class SacredWrapper():
    """
    Base class for a Sacred Experiment.
    """
    def __init__(self, f_main, f_config, f_capture,
                 observer_type='file',
                 mongo_url='mongodb://localhost:27017',
                 verbose=False):
        """
        :param f_main: function
            The main function for the experiment
        :param f_config: function or str
            The function where all the sacred parameters are init or
            a file with the config parameters
        :param f_capture: function
            The function that implements the metrics logging API with sacred
            (should be used with Lambda in keras but has problem right now. Thus it can be ignored)
        :param mongo_url: str
            The url for MongoDB
        :param verbose: bool
            If True logging is enabled
        """

        self.sacred_db_name()

        ex = Experiment(self.sacred_ex_name())
        ex.captured_out_filter = apply_backspaces_and_linefeeds

        if observer_type == 'mongodb':
            print('Connecting to MongoDB at {}:{}'.format(mongo_url, self.sacred_db_name()))
            ex.observers.append(MongoObserver.create(url=mongo_url, db_name=self.sacred_db_name()))
        elif observer_type == 'file':
            basedir = os.path.join(config['logs'], 'sacred')
            ex.observers.append(FileStorageObserver.create(basedir))
        else:
            raise ValueError('{} is not a valid type for a SACRED observer.'.format(observer_type))

        if hasattr(f_config, '__call__'):
            # init the experiment configuration using a function
            ex.config(f_config)
        elif isinstance(f_config, str):
            # init the experiment configuration usinga  file
            ex.add_config(f_config)
        elif isinstance(f_config, dict):
            # init the experiment configuration usinga  file
            ex.add_config(f_config)
        else:
            raise ValueError('You should provide either a fucntion or a config file for setting up an experiemet.'
                             'The given paramter has type {} which is not valid'.format(type(f_config)))

        # init the experiment logging (capture) method
        f_ex_capture = ex.capture(f_capture)

        # init the experiment main
        @ex.main
        def ex_main(_run):
            if observer_type == 'mongodb':
                return main_wrapper(f_main, ex, f_ex_capture, self.sacred_db_name(), _run)
            else:
                f_main(ex, _run, f_ex_capture)

        self.ex = ex

    def sacred_db_name(self):
        """
        This method should be override by child class.
        Returns the Mongo DB name for this set of experiments.
        """
        pass

    def sacred_ex_name(self):
        """
        This method should be override by child class.
        Returns the current experiment name.
        """
        pass


class DTSExperiment(SacredWrapper):
    """
    A General class that can be used for all kind of Experiments.
    """
    def __init__(self, ex_name, db_name, **kwargs):
        """
        :param ex_name: Name of the experiment
        :param db_name: Name of the db where the results will be stored
        :param kwargs:
        """
        self.ex_name = ex_name
        self.db_name = db_name
        super().__init__(**kwargs)

    def sacred_db_name(self):
        return self.db_name

    def sacred_ex_name(self):
        return self.ex_name


def run_grid_search(experimentclass, db_name, ex_name, f_main, f_metrics, f_config, observer_type):
    """
    Run multiple experiments exploring all the possible combinations of the given hyper-parameters.
    Each combination of parameters is an experiment and they will be stored as separate documents.
    Still, they all share the same experiment name.

    :param experimentclass: the wrapper class for the Sacred Experiment.
        Use DTSExperiemnt (see dts.experiment.DTSExperiment)
    :param db_name: str
        Name of the DB where all sort of information regarding the experiment should be stored.
        To be used only when observer_type is 'mongodb'.
    :param ex_name: str
        Experiment name
    :param f_main: the main function. Have a look in dts.examples to understand this better.
    :param f_config: str
        fullpath to the yaml file containing the parameters
    :param observer_type: 'mongodb' or 'file' depending what you want to use.
        If 'file' is used the results/logs are stored in the logs folder
        otherwise everything is stored in the DB.
    """
    parameters = yaml.load(open(f_config))
    keys = list(parameters.keys())
    values = list(parameters.values())
    for vals in product(*values):
        _run_params = dict(sorted(list(zip(keys, vals))))
        run_single_experiment(
            experimentclass=experimentclass,
            db_name=db_name,
            ex_name=ex_name,
            f_main=f_main,
            f_config=_run_params,
            f_metrics=f_metrics,
            observer_type=observer_type
        )


def run_single_experiment(experimentclass, db_name, ex_name, f_main, f_config, f_metrics, observer_type):
    """
    Run a single experiment.

    :param f_config: function
        The function where the standard values for the hyper-parameters are defined.
        The hyper-paramters that appears here overwrite the values defined in cmd_args.

    see run_grid_search for other params.
    """
    experiment = experimentclass(
        db_name=db_name,
        ex_name=ex_name,
        f_main=f_main,
        f_config=f_config,
        f_capture=f_metrics,
        observer_type=observer_type)
    experiment.ex.run()