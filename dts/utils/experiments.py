from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from pymongo import MongoClient
from dts import config
import os


def main_wrapper(f_main, ex, f_ex_capture, curr_db_name, _run):
    """
    f_main updates the main experiment function arguments, calls it and save the
    experiment results and artifacts.
    f_main should implement the following API:
        f_main(ex, _run, f_log_metrics)
    """
    client = MongoClient('localhost', 27017)
    print('db = ', curr_db_name)
    db = client[curr_db_name]
    duplicate_ex = check_for_completed_experiment(db, _run.config)
    if duplicate_ex is not None:
        # raise ValueError('Aborting due to a duplicate experiment')
        return f_main(ex, _run, f_ex_capture)
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
    def __init__(self, f_main, f_config, f_capture, cmd_args,
                 observer_type='file',
                 mongo_url='mongodb://localhost:27017',
                 verbose=False):
        """
        Base class for a Sacred Experiment.

        :param f_main: function
            The main function for the experiment
        :param f_config: function
            The function where all the sacred parameters are init
        :param f_capture: function
            The function that implements the metrics logging API with sacred
        :param cfg: dict
            A dict containing all the arguments for the given experiment
        :param mongo_url: str
            The url for MongoDB
        :param verbose: bool
            If True logging is enabled
        """
        self.cfg = cmd_args
#        self.is_debug = self.cfg['debug']

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

        # init the experiment configuration (params)
        ex.config(f_config)

        # init the experiment logging (capture) method
        f_ex_capture = ex.capture(f_capture)

        # init the experiment main
        @ex.main
        def ex_main(_run):
            return main_wrapper(f_main, ex, f_ex_capture, self.sacred_db_name(), _run)

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
        self.ex_name = ex_name
        self.db_name = db_name
        super().__init__(**kwargs)

    def sacred_db_name(self):
        return self.db_name

    def sacred_ex_name(self):
        return self.ex_name

